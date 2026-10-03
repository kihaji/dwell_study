# Downstream building block: places

This branch keeps one building block under `src/dwells/downstream/`: `places.py`, which
groups an identifier's visits into recurring places (Mahalanobis merge) for the
pattern-of-life pipeline (`src/dwells/pol/`, tables in [../outputs.md](../outputs.md)).
Colocation (identifiers sharing places) was removed from this branch for a separate,
focused study.

Place profiles, rankings, transitions, next-place probabilities and anomalies are
computed in `src/dwells/pol/stats.py`; their definitions are in [../outputs.md](../outputs.md)
and, in plain language, [../analyst_guide.md](../analyst_guide.md). The alternatives
evaluated earlier live on `main`:
* place grouping by DBSCAN, HDBSCAN, GMM or H3;
* the `patterns`, `transitions` and `nextloc` modules;
* colocation candidate search and matchers.

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

## Complexity and Spark-readiness

| function | cost | Spark expression |
|---|---|---|
| `places.assign` | O(n · places) per id, sequential (n = visits per id, small) | `groupBy("id").applyInPandas(...)` (order-dependent, not SQL) |

## References

- Bar-Shalom & Fortmann (1988). *Tracking and Data Association*. Academic Press (gating with the Mahalanobis distance).
