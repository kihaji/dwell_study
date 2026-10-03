# Downstream building blocks: places and colocation candidates

This branch keeps two building blocks under `src/dwells/downstream/`, both used by the
pattern-of-life pipeline (`src/dwells/pol/`, tables in [../outputs.md](../outputs.md)):

* `places.py`: groups an identifier's visits into recurring places (Mahalanobis merge).
* `colocation.py`: finds candidate pairs of visits by different identifiers, used by
  `dwells.pol.coloc`. It is kept separate on purpose, for a later deep dive.

Place profiles, rankings, transitions, next-place probabilities and anomalies are
computed in `src/dwells/pol/stats.py`; their definitions are in [../outputs.md](../outputs.md)
and, in plain language, [../analyst_guide.md](../analyst_guide.md). The alternatives
evaluated earlier live on `main`:
* place grouping by DBSCAN, HDBSCAN, GMM or H3;
* the `patterns`, `transitions` and `nextloc` modules;
* the Mahalanobis and H3 colocation matchers.

## 1. Places: `places.assign(visits, radius_m=150)`

1. Sort the visits by `(id, start_ts)` and split them into per-id blocks.
2. For each id, project the visit centroids into a local east/north frame (meters)
   anchored at the id's median visit position (`geo.to_enu`).
3. Merge sequentially (`_mahalanobis_merge`, below). An id with a single visit gets place 0.
4. Relabel to 0..k-1 and offset, so place ids are unique across ids. The input is
   returned with a `place_id` column.

### Model

Visit i's reported centroid is z_i ~ N(mu_place, C_i), with

    C_i = Sigma_i + v I,     v = (radius_m / 2)^2          (_dwell_cov)

* Sigma_i is the visit's 1σ centroid ellipse (`ellipse_to_cov(sigma_major, sigma_minor,
  azimuth, p = 1 − e^{−1/2})`, so k = 1).
* A missing ellipse is filled with σ = radius_m / 2.
* v is the per-axis variance of a position spread uniformly over a disk of radius
  radius_m: movement inside the place.

### Sequential merge (`_mahalanobis_merge`)

1. Process visits most precise first (ascending tr(C_i)).
2. Each existing place k keeps information sums I_k = Σ C_j⁻¹ and b_k = Σ C_j⁻¹ z_j. Its
   fused center is mu_k = I_k⁻¹ b_k, with covariance P_k = I_k⁻¹.
3. For the new visit compute d²_k = (z_i − mu_k)ᵀ (C_i + P_k)⁻¹ (z_i − mu_k) for every place.
4. If min_k d²_k ≤ χ²₂(0.99) = 9.21 (`GATE`), add C_i⁻¹ and C_i⁻¹ z_i to that place;
   otherwise open a new place.

**Intuition.** P_k shrinks as visits accumulate, but C_i always contains v, so for precise
visits the effective gate never drops below √(9.21 v) ≈ 1.52 × radius_m. For a coarse
visit (large Σ_i) the gate widens, so it joins the nearest compatible place instead of
creating a spurious one. Its small information weight barely moves that place's center.
This is why the method held up best on coarse sources (held-out ARI: CDR 0.91 vs 0.82
for DBSCAN, see [../tuning.md](../tuning.md)).

**Limitations.**
* The result depends on processing order (greedy, no revisiting).
* Place membership is never re-evaluated, and two places whose centers converge are
  never merged.

The place center, uncertainty and extent written to the `places` output table are
computed afterwards in `pol/stats.py` (`_place_centers`, `_place_extent`).

## 2. Colocation candidates: `colocation.py`

### Candidate search (`_neighborhoods`, `_same_time_pairs`)

1. Each visit is assigned the H3 cell at `resolution_for(bucket_m)`: the finest
   resolution whose average edge is ≥ bucket_m (`dwells/h3util.py`).
2. For every occupied cell, A = the visits in that cell and B = the visits in its k = 1
   disk (the cell plus its 6 neighbors), sorted by start time.
3. **Same time.** For each a in A, a time-sorted sweep over B yields only visits b of
   another id that overlap a by ≥ min_overlap_s (index pairs i < j, deduplicated).
   Memory is therefore bounded by real temporal overlaps. The earlier all-pairs-per-cell
   join exhausted memory on dense urban GPS. `pol.coloc` then keeps pairs whose
   centroids are ≤ radius_m apart.
4. **Different time** (`diff_time`). For each a, the visits b of other ids that start
   within [end_a, end_a + window_s] and lie within radius_m. Only the earliest such b per
   other id is kept, and one row per id pair (the shortest gap). overlap_start /
   overlap_end hold the gap interval (earlier end, later start).

**Completeness.** Points in two non-adjacent cells of a hexagonal tiling are at least
about one edge apart, so every pair closer than about bucket_m is a candidate (H3 cells
vary in size, so this holds approximately).

**Spark.** The same logic is an H3-cell explode (cell + ring), an equi-join on cell and
a time-range condition: a bucketed range join.

### Known limitations

* Next-door buildings closer than the positional precision cannot be told apart. On
  the synthetic data, neighbors (median 8–21 m apart) were flagged as colocated in 84–100%
  of cases even at a 15 m match radius (main-branch evaluation).
* The distance test does not use positional uncertainty.
* Same-time rows are visit pairs, so a meeting split into several visits yields
  several rows.

## Complexity and Spark-readiness

| function | cost | Spark expression |
|---|---|---|
| `places.assign` | O(n · places) per id, sequential (n = visits per id, small) | `groupBy("id").applyInPandas(...)` (order-dependent, not SQL) |
| colocation candidates | sweep per H3 neighborhood; memory ~ real overlaps | cell explode + equi-join + time-range filter |

## References

- Bar-Shalom & Fortmann (1988). *Tracking and Data Association*. Academic Press (gating with the Mahalanobis distance).
- Uber H3 hierarchical hexagonal grid: https://h3geo.org
- Crandall et al. (2010). Inferring social ties from geographic coincidences. PNAS 107(52). doi:10.1073/pnas.1006155107
