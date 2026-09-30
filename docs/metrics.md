# Evaluation metrics and protocol

This document covers how detected dwells and colocations are scored against synthetic ground truth (`src/dwells/metrics.py`), how the accuracy harness tunes and evaluates detectors (`src/dwells/harness/run.py`), and how the downstream pipeline is scored (`src/dwells/harness/downstream_eval.py`). Line numbers refer to the working tree as of 2026-09-30.

---

## 1. Dwell scoring (`src/dwells/metrics.py`)

### 1.1 Interval convention

Every dwell is treated as a half-open time interval `[start_ts, end_ts)`, converted to float seconds (`_arr`, `metrics.py:21-26`). Two intervals overlap only if `ts < pe` and `te > ps` (`metrics.py:50`). Intervals that merely touch (one ends exactly when the other starts) do not overlap. Zero-length predictions (`start == end`) can never overlap anything, because the `ts < pe` and `te > ps` tests are strict.

### 1.2 Candidate generation: `overlap_pairs` (`metrics.py:29-61`)

This step finds all (prediction, truth) pairs of the same `id` whose intervals overlap, without an O(n_pred × n_truth) scan:

1. Per id, the truth indices are sorted by start time. The code also builds the **running maximum of the end times**, `tend_max = maximum.accumulate(te)` (`:35-38`). Truths may overlap each other (they do not in the generator), so the plain end times need not be sorted, but their running maximum always is.
2. For each prediction `[ps, pe)` (`:44-52`):
   - `hi = searchsorted(tstart, pe, "left")`: truths at index `>= hi` start at or after `pe`, so they cannot overlap.
   - `lo = searchsorted(tend_max, ps, "right")`: every truth before `lo` has its running-max end `<= ps`, so all of them end at or before `ps`.
   - The exact test `ts < pe & te > ps` is then applied to the slice `[lo, hi)`. This is needed because a truth inside the slice may still end early when truths overlap each other.
3. For each pair the function returns `inter_s` (intersection length) and `iou = inter / union`. It also returns `p_frac = inter / |P|` (the share of the prediction that lies inside the truth) and `t_frac = inter / |T|` (the share of the truth that is covered) (`:54-61`).

The cost is O((n_p + n_t) log n_t + number of pairs) per id.

### 1.3 Matching: temporal IoU only

```
IoU(P, T) = |P ∩ T| / |P ∪ T|      (lengths in seconds)
```

`match_dwells` (`metrics.py:64-74`) keeps the candidate pairs with `iou >= iou_min` (default 0.5). It sorts them by descending IoU and accepts a pair greedily when neither the prediction nor the truth has been used yet. The result is a one-to-one matching.

**Why location is not a match criterion** (module docstring, `metrics.py:3-8`). The centroid error of a correct detection depends on the source profile. For GPS it is metres. For CDR it is often 0.5–2 km, because the reported position is a sector centroid. For adtech it ranges up to km, from IP and cell fixes. A spatial gate tight enough for GPS would make every CDR dwell a false negative. A gate loose enough for CDR would be meaningless for GPS. Temporal identity ("was the device staying somewhere during this time") is what the detector is responsible for. Location quality is reported separately as the centroid error of matched pairs (§1.6). A consequence: a detection at the wrong place but at the right time counts as a true positive. With one agent per id and no concurrent true stays, that can only happen if the detector's centroid is wrong, so read F1 together with `centroid_err_m_*`.

**When greedy equals optimal.** Claim: if predictions are pairwise disjoint within an id, truths are pairwise disjoint within an id, and the threshold is *strictly* greater than 0.5, then each prediction has at most one candidate truth and vice versa. The candidate graph is then already a matching, so greedy, Hungarian (Kuhn 1955) and any other maximum matching coincide.

Proof. Suppose `IoU(P,T) > 1/2`. Since `|P ∪ T| >= max(|P|, |T|)`,
```
|P ∩ T| > |P ∪ T| / 2 >= |P| / 2    and    |P ∩ T| > |T| / 2.
```
If two disjoint truths `T1`, `T2` both had IoU > 1/2 with `P`, then `|P ∩ T1| + |P ∩ T2| > |P|`. That is impossible because `P ∩ T1` and `P ∩ T2` are disjoint subsets of `P`. The symmetric argument, using `|T|/2`, rules out two disjoint predictions sharing one truth. ∎

Caveats:

- The code uses `>=` (`metrics.py:67`), so IoU exactly 0.5 is admitted. Counter-example: `P = [0,2)`, `T1 = [0,1)`, `T2 = [1,2)` gives `IoU(P,T1) = IoU(P,T2) = 0.5`. Uniqueness then fails and the tie is broken by sort order. This is measure-zero with real-valued timestamps, but it is possible with synthetic round-number data.
- The argument needs disjoint predictions. Every detector in `src/dwells/algorithms/` emits non-overlapping dwells per id, but this is not enforced by `schema.validate_dwells`.
- The **lenient** matching at IoU >= 0.2 (`metrics.py:95`) is *not* guaranteed to be maximum-cardinality. A prediction can have several candidates, and greedy-by-IoU can block a pairing that a maximum matching would find. The resulting `recall_lenient` is a lower bound on the maximum-matching recall at 0.2.

### 1.4 Precision / recall / F1 (`metrics.py:77-80, 83-103`)

`dwell_metrics(pred, truth, min_duration_s, iou_min=0.5, min_obs_events=2)`:

- The truth is filtered to `duration_s >= min_duration_s` (`:85`). **Predictions are not filtered.** The detector is responsible for honouring `min_duration_s`. A detector that emits dwells shorter than `min_duration_s` (e.g. `kalman_dwell`, whose grid-based end time can be one step below the threshold) is charged a false positive for each one, unless it happens to match a slightly longer truth.
- `precision = matched / n_pred`, `recall = matched / n_truth`, `F1 = 2PR / (P + R)`. Each is 0 when its denominator is 0.

### 1.5 Observable recall and lenient scores

A truth dwell is **observable** when the sampled event stream contains at least `min_obs_events = 2` events inside it (`truth.n_events`, computed in `generator/build.py` from `truth_dwell_id`). A stay with 0–1 events cannot be detected by any method, so it is excluded from the denominator:

- `recall_observable = (# matched truths that are observable) / n_truth_observable` (`:91-94`).
- `f1_observable = F1(matched_observable, n_pred, n_obs)` (`:101`). Its precision numerator is also *matched observable truths*. A prediction matched to an unobservable truth (possible only if the detector bridges a truth with a single event, using neighbouring events) therefore counts as a false positive in `f1_observable`, even though it is a true positive in `f1`. In practice such matches are rare. Strictly, this makes `f1_observable` slightly pessimistic.
- `recall_lenient`, `f1_lenient`: the same scores at IoU >= 0.2 (see §1.3). They separate "found the stay but with poor boundaries" from "missed the stay".

### 1.6 Boundary and location errors (matched pairs at IoU >= 0.5; `metrics.py:104-119`)

| key | definition |
|---|---|
| `start_err_s_med` | median of \|pred.start − truth.start\| |
| `end_err_s_med` | median of \|pred.end − truth.end\| |
| `duration_bias_s_med` | median of (pred duration − truth duration). Negative means detections are too short. This is expected for "first/last event inside" semantics under sparse sampling |
| `centroid_err_m_med`, `centroid_err_m_p90` | haversine distance from the prediction centroid to the truth `lat/lon` |

The truth `lat/lon` is the **true place position** (`generator/build.py:75-82` joins `area.places`). The true in-place position of the agent varies within `radius_m` of the place, so even a perfect centroid has a non-zero error of up to the place radius. All error values are `None` when nothing matched.

Start/end semantics differ by detector: first/last event inside the stay, versus first fix outside it (trackintel, skmob, fastmob), versus grid times (kalman). This shows up directly in these medians. See each `docs/algorithms/<name>.md`.

### 1.7 Fragmentation, merging, time-based scores (`metrics.py:120-129`)

These use **all overlapping pairs**, not only matched ones:

- `fragmentation` = fraction of truths that have at least 2 predictions, each with at least 50% of its *own* duration inside that truth (`p_frac >= 0.5`). This means one stay was split into several.
- `merging` = fraction of predictions that cover at least 50% of each of at least 2 truths (`t_frac >= 0.5`). This means several stays were merged into one.
- `time_precision = Σ inter_s / Σ pred durations` and `time_recall = Σ inter_s / Σ truth durations`. These are duration-weighted and do not need a matching. If predictions (or truths) overlapped each other, the numerator would double-count. With disjoint sets it does not.

---

## 2. Colocation scoring (`src/dwells/metrics.py:135-175`)

### 2.1 Truth derivation: `colocation_truth` (`:135-153`)

Input: true dwells (the harness passes `truth` filtered to `duration_s >= md`).

1. Self-join on the **true `place_id`** with `id < id_b` (`:140`). Two ids are candidates only if they stay at the *same catalogued place*. Neighbouring buildings are different places, even at a few tens of metres.
2. `overlap_s = min(end) − max(start)`, which is negative when the stays are disjoint.
3. `same_time`: `overlap_s >= min_overlap_s` (default 900 s).
4. `diff_time`: `overlap_s <= 0` and `−overlap_s <= diff_time_window_s` (default 7 days). The gap between the end of the earlier stay and the start of the later one is at most the window. Touching intervals (`overlap_s == 0`) count as diff_time here. The predictor `downstream.colocation.diff_time` requires a strictly positive gap, a minor boundary mismatch.
5. Pairs with `0 < overlap_s < min_overlap_s` are in *neither* class.

### 2.2 `colocation_metrics(pred, truth, kind)` (`:161-175`)

- **Pair level.** Both sides are reduced to the *set* of unordered id pairs `{a, b}` that are ever colocated (of the given kind). The function computes `pair_precision`, `pair_recall` and `pair_f1` on these sets. It ignores how many episodes there were, when they happened and where.
- **Episode level** (same_time only). A truth episode is recalled if any predicted episode *of the same pair* overlaps it in time. A predicted episode is precise if it overlaps any truth episode of the same pair (`:169-174`). **There is no spatial check.** A prediction that co-locates a and b at the wrong place, at a time when they were truly together elsewhere, counts as correct. The diff_time kind gets pair-level scores only.

---

## 3. Detector evaluation protocol (`src/dwells/harness/run.py`)

### 3.1 Datasets

`generator/build.py:43-57` (`catalog()`) builds, for each area (`urban`, `suburban`) and each profile (`gps`, `cdr`, `adtech`):

| group | name | agents | days | seed |
|---|---|---|---|---|
| dev | `dev_{area}_{profile}` | 20 | 7 | 11 |
| eval | `eval_{area}_{profile}` | 100 | 21 | 1 |

Dev and eval share the **same OSM area and place catalog** (`load_area(spec.area)`). They differ in agents, seeds, routines and length. The held-out evaluation therefore tests generalisation to new people and new sampling noise, **not to new geography**. Scale datasets (`scale_*`) are only for benchmarking.

### 3.2 Profiles and sweep grid

`profile_of(name)` (`run.py:54-58`): the profile is taken from the `_gps`, `_cdr` or `_adtech` substring in the name. Public datasets map to `adtech` if the name contains `foursquare`, otherwise to `gps`, so `pol_50`, `staga` and `geolife*` use GPS parameters.

- `MIN_DURATIONS = (300, 900, 1800, 3600)` s (`:37`). This is the *definition* of a dwell and is never tuned.
- `GRID` (`:38-42`), tuned per profile:

| profile | `radius_m` | `max_gap_s` |
|---|---|---|
| gps | 50, 100, 250 | 1800, 7200 |
| cdr | 250, 500, 1000, 2000 | 7200, 21600, 43200 |
| adtech | 100, 250, 500, 1000 | 7200, 21600, 43200 |

Algorithm-specific parameters (`gate_p`, `penalty`, `k_break`, ...) stay at their module defaults. Only the three harness parameters are swept.

### 3.3 Shapes

`_shapes` (`run.py:111-112`): an algorithm with `USES_UNCERTAINTY = True` runs on all five shapes `("1", "2", "3_p68", "3_p95", "3_mixed")`. Otherwise it runs on shape `"1"` only. This is valid because all shapes share identical `lat/lon` (`io.shape_view`, see `docs/uncertainty.md`), so an error-blind algorithm gives the same output on every shape. Its shape-1 row stands for all shapes.

### 3.4 Sweep, then held-out eval

1. **sweep** (`sweep_tasks`, `:115-123`) runs every `(algo, dev dataset, shape, min_duration, radius, max_gap)`. `run_task` (`:80-108`) loads the events and calls `detect`. It validates the output and scores it with `dwell_metrics(d, truth, min_duration_s=params.min_duration_s)` at IoU 0.5. Results go to `results/sweep.parquet`.
2. **best_params** (`:126-133`) takes successful rows and averages `f1` over the dev datasets of a profile (both areas) for each `(algo, profile, shape, min_duration, params)`. It keeps the params with the highest mean F1 per `(algo, profile, shape, min_duration)`. Ties go to whichever row sorts first, which is arbitrary but deterministic for a given parquet. Tuning uses plain `f1`, not `f1_observable`.
3. **eval** (`eval_tasks`, `:136-143`): for every eval dataset, each best-params row of the matching profile is run once. Metrics go to `results/eval.parquet` and detections to `results/dwells/<dataset>/<algo>__<shape>__<md>.parquet`. The default glob is `eval_*`. Labeled public datasets can be added with `--datasets` (e.g. `pol_50` is present in the current results) and receive `gps` parameters.

### 3.5 Execution mechanics

- `task_key` is the SHA-1 of the task JSON (`:72-73`). `execute` (`:146-164`) skips keys that already succeeded in the output parquet, so runs are resumable and errored tasks are retried.
- Each task has a wall-clock timeout via `SIGALRM`: `TIMEOUT_S`, default 1800 s, overridable with `DWELLS_TIMEOUT_S` (`:43, 84-86`). A timeout is recorded as `error = "timeout>…"`. Timeouts and errors are excluded from `best_params`.
- The pool is a `ProcessPoolExecutor` with the **spawn** context. Forked children deadlock on polars' thread pool (`:156-157`). Tasks are sorted by dataset and shape so the per-worker `lru_cache` of loaded events is reused.
- `runtime_s` measures only the `detect` call.

`src/dwells/harness/bench.py` is separate. It measures runtime and peak RSS per algorithm, in isolated subprocesses, on scale ladders (`scale_gps_*`, `scale_adtech_*`, `scale_gps_single_1m`) with fixed parameters. It stops climbing a ladder after a timeout or failure. Results go to `results/bench.parquet`, and it computes no accuracy.

---

## 4. Downstream evaluation (`src/dwells/harness/downstream_eval.py`)

Inputs: tuned detections from the eval stage at `--md` (default 900 s). Uncertainty-aware algorithms use shape `3_mixed`; the others use shape `1` (`:78-79`). Scale constants by profile: `PLACE_RADIUS` (gps 150, adtech 250, cdr 1000 m), `HIT_M` (100 / 250 / 1000 m), `COLOC_RADII` (`:39-41`).

| stage | what is scored | code |
|---|---|---|
| `nextloc_truth` | acc@1/3/5 and MRR of each next-location model on the **true** place sequences (dwells >= md). This is the ceiling, independent of detection | `:65-67` |
| `places` | For each grouping method in `places.METHODS`: detections are matched to truth at IoU 0.5. For each id with >= 3 matched dwells, the adjusted Rand index (Hubert & Arabie 1985) between detected `place_id` and true `place_id` is computed and averaged over ids (0 if none). `place_count_ratio` = detected places with >= 2 visits / true places with >= 2 visits | `:87-102` |
| `home_work` (dbscan places only) | `patterns.label_home_work` on the detected places. A hit means the inferred place centroid is within `HIT_M` of the agent's true home or work place. Work is scored only for `persona == "worker"`. A missing label is a miss | `:103-118` |
| `nextloc_detected` | models trained and tested on detected place sequences (what an application would see) | `:120, 126` |
| `nextloc_mapped` | matched detected dwells relabelled with the *true* place of their matched truth. This isolates whether detection preserves the routine, independent of place clustering | `:121-129` |
| `colocation_same_time` | for each method in `colocation.METHODS` (at `PLACE_RADIUS`): pair and episode P/R/F1 vs `colocation_truth(tr_md)`, plus `decoy_fp_rate` | `:132-137` |
| `colocation_radius` | a sweep of `distance` and `mahalanobis` over `COLOC_RADII`, showing recall against decoy false positives | `:139-145` |
| `colocation_diff_time` | pair-level P/R/F1 of `diff_time` (7-day window) | `:146-147` |

**Decoys** (`:71-75`): pairs of agents in households flagged as next-door neighbours (`neighbor_of`). These agents live at adjacent buildings and never truly co-dwell. Pairs that *are* true same_time pairs (e.g. they meet elsewhere) are removed. `decoy_fp_rate` = the fraction of decoy pairs that the method reports as colocated. It measures whether a method can tell adjacent buildings apart.

Next-location evaluation (`nextloc.evaluate`) trains on each id's first 70% of collapsed visits. It scores each remaining step with the true history as context. `acc@k` is the fraction of steps whose true next place is ranked <= k. MRR is the mean of 1/rank, with 0 for unranked places (Voorhees 1999). See `docs/algorithms/downstream.md` for the models.

Output: `results/downstream.parquet`, long format `(dataset, algo, stage, method, metric, value)`.

---

## 5. Caveats and threats to validity

1. **Same geography in dev and eval.** Parameters tuned on dev may overfit the place density and sizes of the two OSM areas. The public labeled sets (`pol_50`, `staga`) are the only out-of-distribution check.
2. **Simulator-defined truth.** A "dwell" is whatever the simulator calls a stay: arrive/depart at a catalogued place. Short stops that the simulator does not model (traffic lights, drive-throughs) appear as false positives if detected.
3. **Temporal-only matching** rewards a detection at the right time even at the wrong place. Always read F1 alongside `centroid_err_m_*`.
4. **Predictions are not filtered by min_duration**, while truths are. This is deliberate, but boundary conventions (first fix outside vs last fix inside, grid snapping) shift durations by one sampling interval. That is enough to push near-threshold dwells across `min_duration_s` in either direction.
5. **Tuning on F1 only**, per profile and shape, lets an uncertainty-aware algorithm pick different `radius_m` values on different shapes. Shape comparisons are therefore "best achievable per shape", not "same parameters".
6. **Greedy matching** is exactly optimal only under the conditions in §1.3. The lenient (0.2) numbers are lower bounds.
7. **Episode-level colocation** has no spatial test, and pair-level colocation ignores timing. A method can score well on pairs while getting the episodes' locations wrong.
8. **Observable recall** uses `n_events >= 2`. That is the minimum for a *span*, but many detectors also require `min_events` of 2–3 and a duration of at least `min_duration_s` between the first and last event. A stay with 2 events 1 minute apart is "observable" but undetectable at `min_duration_s = 900`.
9. **Runtime** in the sweep includes JIT warm-up (numba `cache=True` mitigates this after the first run) and is measured under concurrent workers. Use `bench.py` for scaling claims.

## References

- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly* 2(1–2):83–97. doi:10.1002/nav.3800020109
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification* 2:193–218. doi:10.1007/BF01908075
- Voorhees, E. M. (1999). The TREC-8 Question Answering Track Report. *Proc. TREC-8*. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf (mean reciprocal rank)
