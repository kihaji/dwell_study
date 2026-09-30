# Downstream analytics (places, patterns, transitions, next location, colocation)

Code: `src/dwells/downstream/` (`places.py`, `patterns.py`, `transitions.py`, `nextloc.py`, `colocation.py`).
Evaluation driver: `src/dwells/harness/downstream_eval.py`. The scoring details are in [../metrics.md](../metrics.md).

## Summary

The detectors in this directory produce a DWELLS table (`id, dwell_id, start_ts, end_ts, duration_s, lat, lon, n_events, sigma_major, sigma_minor, azimuth`, see `src/dwells/schema.py:20-32`). The downstream modules consume only that table. They never see raw events, so the same analytics can run on the output of any detector:

| Module | Input | Output | Scope |
|---|---|---|---|
| `places.assign` | dwells | dwells + `place_id`, PLACES table | per id |
| `patterns.place_profiles`, `label_home_work` | dwells with `place_id` | per (id, place) profile, home/work labels | per id |
| `transitions.from_dwells`, `matrix`, `sequences` | dwells with `place_id` | TRANSITIONS, transition matrix, k-sequences | per id |
| `nextloc.Predictor`, `fit`, `evaluate` | (id, place_id, start_ts) visit sequence | PREDICTIONS (top-N), acc@k / MRR | per id (model tables keyed by id) |
| `colocation.same_time`, `diff_time` | dwells of all ids | COLOCATIONS | **cross-id** |

`place_id` is unique across ids (`places.py:126-134`), but places are **per-identifier**. Two ids at the same building get different place ids, so cross-id questions (colocation) work on dwell coordinates, not on place ids.

## Mapping to the analytic questions (project.md)

| Question | Answered by | Notes |
|---|---|---|
| Where does an identifier dwell? | detector output + `places.assign` / `places.place_table` | PLACES row: center, `radius_m`, `n_visits`, `total_dwell_s`, `first_seen`, `last_seen` |
| Is there a pattern to where they dwell? | `patterns.place_profiles` (hour/day-of-week histograms, `regularity`, `night_s`, `workhours_s`), `patterns.label_home_work` | Home/work inference is the main semantic output |
| Is there a pattern to their movement after a dwell? | `transitions.matrix` (P(to \| from), median travel time, median departure hour), `transitions.sequences` (A->B->C counts) | |
| Dwell data: start, end, how often they return, how many events | DWELLS (`start_ts`, `end_ts`, `n_events`); `place_profiles` (`n_visits`, `mean_dwell_s`, `events_per_dwell`, `n_days`, `return_interval_h_med`) | |
| Other identifiers dwelling there with them (same time) or at different times | `colocation.same_time` (3 spatial tests), `colocation.diff_time` | |
| Probable next locations after a dwell (top-N) | `nextloc.Predictor.predict_top`, `nextloc.evaluate` | 4 models with backoff |

---

## 1. Places: `places.py`

`assign(dwells, method, radius_m=150)` (`places.py:121-136`):
1. Sort dwells by `(id, start_ts)`, split the frame into per-id blocks (`:128`).
2. For each id, project the dwell centroids to a local equirectangular ENU frame anchored at the id's median dwell position (`:130`, `geo.to_enu`).
3. Cluster with `_cluster(method, ...)` (`:79-118`). If an id has a single dwell, that dwell is place 0 (`:81-82`).
4. Relabel to 0..k-1 (`np.unique(..., return_inverse=True)`), offset so ids never share a place id (`:132-134`).
5. Return the dwells with `place_id` added, plus `place_table(d)`.

`radius_m` is the place scale that every method uses (`downstream_eval` passes gps 150 m, adtech 250 m, cdr 1000 m, `downstream_eval.py:39`).

### 1.1 `dbscan` (`places.py:84-85`)
`DBSCAN(eps=radius_m, min_samples=1)` on ENU centroids. With `min_samples=1` every point is a core point, so the clusters are the connected components of the graph "centroid distance <= eps". This is **single-linkage clustering cut at eps**, and it chains: a line of dwells spaced < eps apart becomes one place of unbounded extent.

Origin: new_ellipses `cotravel_cointegration/place_follower.py` (branch `cotravel-research`, line 151: `DBSCAN(eps = cfg.place_eps_m, min_samples = 1)`, `place_eps_m = 150.0` at line 32). **Deviation:** the original clusters *all identifiers' dwell centroids together* ("shared place ids", docstring line 5), so an arrival event is keyed to a place that is shared across ids. Here DBSCAN runs per id, and place ids are per-id. The default eps (150 m) matches.

### 1.2 `hdbscan` (`places.py:86-92`)
1. `HDBSCAN(min_cluster_size=2, allow_single_cluster=True)` on the centroids. The default excess-of-mass selection is used, `min_samples` defaults to `min_cluster_size`.
2. Each noise dwell becomes its own singleton label (`:88-89`).
3. The label means (cx, cy) are merged with `DBSCAN(eps=radius_m, min_samples=1)` (`:90-92`). HDBSCAN density is relative and has no absolute scale: in testing it split one home by its 20 m jitter. `cluster_selection_epsilon` would do the same job, but it crashes under numpy 2.5 (docstring `:6-9`).

The result is HDBSCAN, then single-linkage at radius_m over cluster centers. So it inherits DBSCAN's chaining at the cluster-center level.

### 1.3 `gmm` (`places.py:93-101`)
For k = 1..min(n, 15), the code fits `GaussianMixture(k, covariance_type="spherical", reg_covar=(radius_m/2)^2, random_state=0)`. It keeps the model with the lowest BIC and returns `predict(X)` (hard MAP assignment).

- BIC = -2 ln L + p ln n. For spherical 2-D components, p = k·2 (means) + k (variances) + (k-1) (weights) (Schwarz 1978).
- `reg_covar` is added to every component variance, so each variance is at least (radius_m/2)^2. That is the per-axis variance of a uniform disk of radius radius_m, which gives an absolute floor on place size. Without the floor, a component could collapse onto a few identical centroids.
- Cost: up to 15 EM fits per id. It is deterministic because of `random_state=0`. With n small (< ~5), BIC tends to pick k=1 or k=n.

### 1.4 `h3` (`places.py:102-115`)
1. The resolution is `r = resolution_for(radius_m)`: the finest H3 resolution whose *average* edge is >= radius_m (`algorithms/h3_occupancy.py`). Examples: 150 m gives res 9 (edge 200.8 m), 250 m gives res 8 (531 m), 1000 m gives res 7 (1406 m).
2. Each dwell centroid is mapped to its cell. The code builds the graph whose vertices are the occupied cells and whose edges link occupied cells in each other's k=1 disk (`:112`). `grid_disk` includes the cell itself, so the edge list is never empty.
3. `scipy.sparse.csgraph.connected_components` runs on that graph. A place is a connected set of occupied, mutually adjacent cells (`:114-115`).

Using one cell per place split stays that straddle a boundary (docstring `:12-14`), hence the connected components. Like DBSCAN, this chains: a corridor of occupied cells becomes one place.

### 1.5 `mahalanobis` (new, `places.py:40-76`, `:116-117`)
Model: dwell i's reported centroid is z_i ~ N(mu_place, C_i), with

    C_i = Sigma_i(1-sigma centroid ellipse) + v I,     v = (radius_m / 2)^2          (_dwell_cov, :40-49)

Sigma_i comes from `ellipse_to_cov(sigma_major, sigma_minor, azimuth, p = 1 - e^{-1/2})`, so k = 1 and the axes are 1-sigma. Only `ua_mahalanobis`, `ua_segmentation` and `hmm_stay_move` report these axes. `_dwell_cov` is evaluated once on the whole input frame (`:125`). If all of its `sigma_major` values are null, then C_i = v I (`:42-44`). If only some are null, the nulls are filled with sigma = radius_m/2, which gives C_i = 2v I for those dwells (`:45-47`).

Sequential merge (`_mahalanobis_merge`, `:52-76`):
1. Process the dwells in ascending order of tr(C_i) (most precise first, `:53`).
2. Each existing place k keeps information sums I_k = sum_j C_j^{-1} and b_k = sum_j C_j^{-1} z_j. Its fused center is mu_k = I_k^{-1} b_k, with covariance P_k = I_k^{-1} (`:59-61`).
3. For the new dwell, compute d^2_k = (z_i - mu_k)^T (C_i + P_k)^{-1} (z_i - mu_k) for every place (`:62-64`). Note that C_i already contains v, so S = Sigma_i + v I + P_k.
4. If min_k d^2_k <= chi2_2(0.99) = 9.21 (`:117`), add C_i^{-1} and C_i^{-1} z_i to that place. Otherwise open a new place (`:70-75`).

Intuition: P_k shrinks like 1/n_visits, but C_i always contains v. So the effective gate radius never drops below sqrt(9.21 v) = 1.52 radius_m for precise dwells. For a coarse dwell (large Sigma_i) the gate widens, and the dwell joins the nearest compatible place instead of creating a spurious new one. Its tiny information weight barely moves that place's center.

Limitations: the result is order-dependent (greedy, no revisiting). Once a place exists its membership is never re-evaluated, and two places whose centers converge are never merged.

### 1.6 `place_table` (`places.py:139-149`)
For each (id, place_id):
- `lat`/`lon` is the arithmetic mean of the member dwell centroids. This holds for **every** method, including `mahalanobis`, whose fused center is not used here.
- `radius_m` is the maximum haversine distance from that mean to a member centroid (0 for a single visit).
- `n_visits` is the number of dwells, and `total_dwell_s` is the sum of their `duration_s`.
- `first_seen` = min `start_ts`, `last_seen` = max `end_ts`.

### Use of positional uncertainty (places)
Only `mahalanobis` uses it, through the reported dwell centroid ellipse. The other four methods cluster points and are shape-agnostic.

---

## 2. Patterns: `patterns.py`

### `_hour_overlap` (`patterns.py:17-27`)
This splits each dwell into local clock hours. Both endpoints are shifted by a **fixed** `utc_offset_h`, and `datetime_ranges(truncate(s, 1h), e, 1h)` is exploded. For each hour bucket h, sec = max(0, min(e, h+1h) - max(s, h)). `hour` is 0-23, and `dow` = polars `weekday()` - 1, which gives 0 = Monday.

### `place_profiles(dwells, utc_offset_h)` (`patterns.py:30-61`)
Per (id, place_id):

| Column | Definition | Line |
|---|---|---|
| `n_visits`, `total_dwell_s`, `mean_dwell_s` | count / sum / mean of `duration_s` | 32-34 |
| `events_per_dwell` | mean `n_events` | 34 |
| `n_days` | distinct local dates of `start_ts` | 35 |
| `return_interval_h_med` | median of successive `start_ts` differences within the place, in hours | 36 |
| `hour_hist` (24), `dow_hist` (7) | fractions of dwell *seconds* per local hour / weekday | 42-49 |
| `night_s` | seconds in local hours 22:00-06:00 | 51 |
| `workhours_s`, `work_days` | seconds on Mon-Fri in hours 9..16 inclusive (09:00-17:00), and the distinct dates with such seconds | 52-54 |
| `regularity` | 1 - H(hour_hist)/ln 24, where H = -sum p ln p | 59-61 |

`regularity` is 1 when all dwell time falls in one clock hour and 0 when time is spread uniformly over 24 hours. It is a normalized-entropy measure of the kind used for predictability of human mobility (Song et al. 2010), applied per place to the hour-of-day distribution.

### `label_home_work(profiles, min_work_days=3)` (`patterns.py:64-72`)
- **home** is the place with the largest `night_s` (> 0) per id.
- **work** is the place with the largest `workhours_s` among the places that are not home and have `work_days >= min_work_days`.

These are the standard "most night-time" and "most weekday-daytime" heuristics used for CDR/GPS home and work detection (e.g. Isaacman et al. 2011, who use a similar logic with regression weights). `group_by("id").first()` after a `sort` picks the maximum because polars keeps row order within a group.

### Limitations (patterns)
- The offset is fixed, so there is no DST handling. The generator uses fixed offsets too (`generator/build.py:28`), so this is consistent within the synthetic evaluation but wrong for real multi-season data.
- `n_days` counts start dates only: a dwell crossing midnight counts one day.
- `return_interval_h_med` is biased low when a detector fragments one stay into several dwells, because the intra-stay gaps become "returns". It is null for a single-visit place.
- A place whose dwells all have zero duration has a histogram total of 0. Dividing by 0 gives NaN histograms and a NaN `regularity` (`:47-48`).
- Night and work-hour windows are hard-coded (22-06, Mon-Fri 09-17).

---

## 3. Transitions: `transitions.py`

- `from_dwells` (`transitions.py:17-26`): each dwell is paired with the id's next dwell (`shift(-1).over("id")`), so `depart_ts` = this dwell's `end_ts` and `arrive_ts` = the next dwell's `start_ts`. The last dwell per id is dropped (`drop_nulls`). **Self-transitions are kept** (from_place == to_place). They occur when a stay was fragmented or when an intermediate stop was not detected.
- `matrix(tr, utc_offset_h, drop_self=False)` (`:29-38`): per (id, from, to) it computes `count`, `travel_s_med` (median arrive - depart) and `depart_hour_med` (median local departure hour). Then `prob = count / sum(count over (id, from))`, which is the maximum-likelihood first-order Markov transition matrix.
- `sequences(dwells, k=3)` (`:41-45`): counts length-k windows of consecutive `place_id` values per id.

### Limitations (transitions)
- `depart_hour_med` is a linear median of clock hours, not a circular statistic. Departures at 23h and 01h give 12h.
- `sequences` does **not** collapse consecutive repeats (A, A, B is a valid 3-sequence), while `nextloc` does collapse them (`nextloc.py:27-30`). The two modules therefore use different "visit sequence" definitions. `from_dwells` also keeps self-transitions unless `matrix(drop_self=True)` is used.

---

## 4. Next location: `nextloc.py`

Sequence construction (`_collapse`, `nextloc.py:27-30`): the visits are sorted by (id, start_ts), and consecutive duplicate `place_id` values are dropped. The kept row is the *first* of the run, so its `start_ts` is the arrival at that place.

Counts (`Predictor.update`, `:48-55`), per uid:
- `freq[uid][p]` counts visits to p.
- `c1[(uid, p_{i-1})][p_i]` counts first-order transitions.
- `c2[(uid, p_{i-2}, p_{i-1})][p_i]` counts second-order transitions.
- `ct[(uid, p_{i-1}, bucket(t_{i-1}))][p_i]` counts time-conditioned transitions. `bucket` (`:33-37`) = 6·[weekend] + floor(local hour / 4), which gives 12 buckets.

Ranking (`rank`, `:57-75`) is a **backoff chain**: candidate tables are concatenated in priority order, and places are listed by count within each table (`Counter.most_common`, ties broken by first insertion). The current place is always excluded.

| model | tables in order |
|---|---|
| `most_frequent` | freq |
| `markov1` | c1[current] -> freq |
| `markov2` | c2[prev, current] -> c1[current] -> freq |
| `markov1_time` | ct[current, bucket] -> c1[current] -> freq |

The candidates are only places the uid has already visited in training, so a new place is never predicted.

`predict_top(seq, n=3)` (`:77-86`): for each id it ranks from the id's last visited place and emits PREDICTIONS rows `(id, from_place, rank, to_place, prob)`.

`evaluate(seq, model, test_frac=0.3, top_n=(1,3,5))` (`:96-116`):
1. Per id, cut = round(len · (1 - test_frac)). The id is skipped if cut < 2 or cut >= len.
2. The model is trained on visits [0, cut) of every id. It is a static model and is not updated during the test.
3. For each test step i >= cut, the model ranks with the true history places[:i] and records the rank of places[i] (inf if absent).
4. acc@k = mean(rank <= k), MRR = mean(1/rank) (1/inf = 0), `n_test` = number of steps.

These are the standard order-k Markov predictors for location sequences (Gambs et al. 2012; Song et al. 2010 for the predictability bound).

### Discrepancies (nextloc)
1. **Time bucket is the arrival time, not the departure time.** The module docstring (`:8-10`) says the bucket uses the departure (visit end if available, else the next start). But `fit` (`:92`), `evaluate` (`:101`) and `predict_top` (`:81`) all pass `start_ts`, so `_bucket(times[i-1])` is the time of *arrival* at the current place. Because of `_collapse`, that is the first arrival in a run of repeats. For long stays (home overnight, work all day) arrival and departure fall in different buckets, so `markov1_time` conditions on something weaker than intended.
2. **`prob` in `predict_top` is always the markov1 frequency.** It is computed as c1[(uid, current)][p] / sum c1[(uid, current)] (`:83-85`) whatever the model. For `markov2`, `markov1_time` and `most_frequent` the probability does not match the ranking, and places reached by backoff (freq table) get prob 0.
3. `rank` excludes the current place even for `most_frequent`. That is consistent with collapsed sequences but means order-0 is "most frequent other place".

---

## 5. Colocation: `colocation.py`

### Candidate generation (`_candidates`, `colocation.py:39-47`)
1. `r = resolution_for(bucket_m)` gives an average edge e >= bucket_m.
2. Each dwell gets its cell. Each dwell is exploded to its k=1 disk (7 cells), and the disk is equi-joined to the dwells' own cells. Pairs are kept with i < j and different `id`.

Completeness: the minimum distance between points in two non-adjacent cells of a regular hexagonal tiling is one edge length e. So every pair closer than about e is a candidate. Real H3 cells vary in size around the average edge, so this holds only approximately. Pairs farther than e *may* still be candidates if they happen to be in adjacent cells. The cost grows with local density (pairs per bucket), not with n^2.

### Spatial tests (`_spatial_ok`, `:59-74`)
- `distance`: haversine(centroid_a, centroid_b) <= radius_m.
- `mahalanobis`: d^2 = dx^T (S_a + S_b)^{-1} dx <= 5.991, which is chi2_2(0.95), with dx in the ENU frame of dwell a. Each S = the 1-sigma centroid ellipse (`ellipse_to_cov` at p = 1 - e^{-1/2}) + v I, v = (radius_m/2)^2, and null sigmas are filled with radius_m/2 (`_cov`, `:50-56`).
- `h3`: same cell or neighbor cells (`h3.are_neighbor_cells`) at `resolution_for(radius_m / 2)`.

### `same_time(dwells, method, radius_m=100, min_overlap_s=900)` (`:85-103`)
1. The candidate bucket scale is radius_m. For `mahalanobis` with reported sigmas it is max(radius_m, 3 · q95(sigma_major)) (`:89-91`).
2. Keep pairs whose temporal overlap, min(end) - max(start), is >= min_overlap_s.
3. Apply the spatial test.
4. Orient each pair so id_a < id_b. The output location is the midpoint of the two centroids, overlap_start/end is the intersection, and kind = "same_time".

### `diff_time(dwells, radius_m=100, window_s=7 days)` (`:106-119`)
Candidates at scale radius_m. The code computes gap = max(start) - min(end) and keeps 0 < gap <= window_s, i.e. no overlap and the later dwell starts within window_s of the *end* of the earlier one. The distance test is then applied. Output rows keep `overlap_start` = later start and `overlap_end` = earlier end, so **overlap_end < overlap_start**, with `overlap_s = 0` and kind = "diff_time".

### Deviation from the original (ellipses/colocation.py)

| | original `ellipses/colocation.py` | this repo |
|---|---|---|
| dwell covariance | **mean** of the member events' covariances, `ellipse_to_covariance(semi_major, semi_minor, azimuth, confidence)` (`compute_dwell_covariance`, lines 414-449). This is the per-event scatter, not the uncertainty of the centroid | fused 1-sigma **centroid** ellipse (shrinks like 1/n) + within-place spread (radius_m/2)^2 |
| threshold | `DEFAULT_SPATIAL_THRESHOLD = 5.991` (line 20) | 5.991 |
| pair search | O(n^2) all-pairs numba loop over dwells (`_find_colocation_pairs_numba`, lines 165-258) | H3 bucket equi-join |
| overlap | `overlap_start < overlap_end` and >= `min_overlap_minutes` (default 15) | >= `min_overlap_s` (default 900) |
| default confidence | `DEFAULT_CONFIDENCE = 0.95` for events lacking one | not applicable (every event's own p is used upstream) |

### Discrepancies and limitations (colocation)
1. **Mahalanobis candidates can be incomplete.** Candidates are only guaranteed within about one edge (edge >= scale), but the gate radius can exceed that. With all sigmas null, each S = (r/2)^2 + (r/2)^2 = r^2/2 per axis, so S_a + S_b = r^2 I and the gate radius is sqrt(5.991)·r = 2.45 r, while the scale stays r. With precise reported sigmas the gate radius is about sqrt(5.991 · r^2/2) = 1.73 r, and scale = r. At r = 50 m (res 10, edge 75.9 m) a Monte Carlo check (300 random pairs 80-120 m apart, sigma null, all inside the 122 m gate) found 5 pairs never generated as candidates. At r = 100 (edge 200.8 m) the problem mostly disappears. The fix would be a scale >= gate radius.
2. **Inconsistent null-sigma handling.** `colocation._cov` fills null sigma with r/2 **and** adds v (2v per axis). `places._dwell_cov` uses v alone when all sigmas are null, and 2v only for partially null frames.
3. **h3 method scale.** Two points in adjacent hexes can be up to sqrt(13)·e ≈ 3.6 e apart. At r = 100 the test resolution is res 10 (e = 75.9 m), so accepted pairs can be about 273 m apart. The docstring's "~6 edges = 600 m" for edge >= r is imprecise (3.6 × 200.8 ≈ 721 m). Candidates are generated at edge >= r, so pairs accepted by the h3 test but farther than about one candidate edge may be missed.
4. **diff_time definition.** The docstring (`:11-13`) says "start within window_s of each other". The code uses the end-to-start gap. `metrics.colocation_truth` (`src/dwells/metrics.py:146-148`) defines diff_time as overlap_s <= 0, which includes touching intervals (gap = 0). The prediction requires gap > 0 strictly, so touching pairs can be truth-only. Pairs overlapping by 0 < overlap < min_overlap_s are neither same_time nor diff_time in both prediction and truth.
5. The same_time location is the midpoint of the two centroids and does not use uncertainty weighting. Episodes are dwell-pair rows: consecutive fragmented dwells give several rows for one real meeting.
6. There is no "place" identity across ids: each colocation is a pair of dwells, so "who else dwells *at this place*" needs a grouping step on top of this.

### Use of positional uncertainty (colocation)
Only `mahalanobis`, and only through the dwell centroid ellipses from uncertainty-aware detectors. For the other detectors it reduces to a fixed circular gate of 2.45 r (all sigma null).

---

## Complexity and Spark-readiness

| Function | Cost | Spark expression |
|---|---|---|
| `places.assign` dbscan / h3 | O(n log n) per id (n = dwells/id, small) | `groupBy("id").applyInPandas(...)`. H3 can also be done in SQL (h3 UDF + a connected-components pass per id) |
| hdbscan | O(n log n) to O(n^2) per id | `applyInPandas` |
| gmm | 15 EM fits per id | `applyInPandas`. The cost is dominated by the fits, which is fine because n is small |
| mahalanobis | O(n · places) per id, sequential | `applyInPandas` (order-dependent, not SQL) |
| `place_table`, `place_profiles`, `label_home_work` | linear | pure SQL: `sequence(date_trunc('hour', s), e, interval 1 hour)` + `explode`, group-bys, `row_number()` over (id order by night_s desc) |
| `transitions.*` | linear | SQL window functions `lead(place_id) over (partition by id order by start_ts)` |
| `nextloc.fit` | linear | group-by counts on (id, prev, cur[, bucket]). `rank` needs a per-id top-k (`row_number`) with backoff via `coalesce` of joins |
| `nextloc.evaluate` | linear in test steps | per id `applyInPandas` (a Python loop, not vectorized) |
| `colocation.same_time/diff_time` | ~ sum over buckets of pairs | **cross-id shuffle**: `explode(h3_grid_disk(cell, 1))` then an equi-join on cell with `i < j` and `id <> id_b`, a time-overlap filter, then the spatial UDF. It needs a global shuffle by cell, and hot cells (malls, stations) create skew. For diff_time with a 7-day window, pairs per hot cell grow quadratically |

Everything except colocation partitions by id with no global state. Colocation is the one global step.

---

## How `downstream_eval.py` uses these

The inputs are the tuned detections from the eval stage (shape `3_mixed` for uncertainty-aware detectors, shape `1` otherwise, `downstream_eval.py:78`). The truth dwells are filtered to `duration_s >= md` (default 900).

- **places**: runs every method at `PLACE_RADIUS[profile]`. It reports the ARI between detected and true place ids over matched dwells (IoU >= 0.5), averaged over ids with >= 3 matched dwells, and `place_count_ratio` = (detected places visited >= 2 times) / (true ones) (`:87-102`).
- **home_work** (dbscan places only): `label_home_work(place_profiles(...))`. A hit is a detected home or work center within `HIT_M` (gps 100 m, adtech 250 m, cdr 1000 m) of the agent's true home or work. Work is scored for `persona == "worker"` only (`:103-118`).
- **nextloc**: `evaluate` on (a) true sequences (model ceiling), (b) detected place sequences, and (c) detected dwells relabeled with the true place of their matched truth dwell (`:65-67`, `:119-129`).
- **colocation**: `same_time` for each method vs `metrics.colocation_truth`, plus a decoy false-positive rate on neighboring-household pairs that never truly co-dwell. It also runs a radius sweep (`COLOC_RADII`) for distance/mahalanobis and scores `diff_time` pairs (`:131-147`).

---

## References

- Ester, Kriegel, Sander, Xu (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD-96. https://cdn.aaai.org/KDD/1996/KDD96-037.pdf
- Campello, Moulavi, Sander (2013). Density-based clustering based on hierarchical density estimates. PAKDD. doi:10.1007/978-3-642-37456-2_14
- Schwarz (1978). Estimating the dimension of a model. Annals of Statistics 6(2). doi:10.1214/aos/1176344136
- Gambs, Killijian, del Prado Cortez (2012). Next place prediction using mobility Markov chains. MPM workshop. doi:10.1145/2181196.2181199
- Song, Qu, Blumm, Barabási (2010). Limits of predictability in human mobility. Science 327. doi:10.1126/science.1177170
- Isaacman et al. (2011). Identifying important places in people's lives from cellular network data. Pervasive. doi:10.1007/978-3-642-21726-5_9
- Crandall et al. (2010). Inferring social ties from geographic coincidences. PNAS 107(52). doi:10.1073/pnas.1006155107
- Eagle & Pentland (2006). Reality mining: sensing complex social systems. Pers. Ubiquit. Comput. 10. doi:10.1007/s00779-005-0046-3
- Uber H3 hexagonal hierarchical grid. https://h3geo.org
- Ported sources: `ellipses/colocation.py`; `new_ellipses` branch `cotravel-research` `cotravel_cointegration/place_follower.py`.
