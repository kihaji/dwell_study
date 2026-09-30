# Dwell-detection algorithms: index

Every detector is a module in `src/dwells/algorithms/`, registered in
`src/dwells/algorithms/__init__.py:19-32` (`_PLANNED`), and exposes

```
NAME, FAMILY, USES_UNCERTAINTY, REFERENCE
detect(events, *, min_duration_s, radius_m, max_gap_s, **kw) -> DWELLS frame
```

`events` is a validated event frame of any shape (1, 2 or 3), sorted by `(id, timestamp)`
(`src/dwells/schema.py:98-121`). The output follows `schema.DWELLS`
(`src/dwells/schema.py:20-32`). Downstream analytics are documented in
[downstream.md](downstream.md), scoring in [../metrics.md](../metrics.md), and the error
model in [../uncertainty.md](../uncertainty.md).

## Index

| Name | Family | Uses uncertainty | Origin | One line |
|---|---|---|---|---|
| [li_centroid](li_centroid.md) | sequential | no | Port: new_ellipses `dwell_detection.py` | Running-mean centroid, break on distance > r or gap > g |
| [h3_cell_dwell](h3_cell_dwell.md) | grid | grid size only | Port: ellipses `hexbin.py` | 500 m sessions, then contiguous runs of one H3 cell sized from the mean error ellipse |
| [kalman_dwell](kalman_dwell.md) | sequential | yes (smoother) | Port: new_ellipses `series.py` + `place_follower.py` | CV Kalman/RTS smoother onto a 60 s grid, then greedy running-mean dwells on the grid |
| [trackintel_sliding](trackintel_sliding.md) | library | no | trackintel 1.4.2 | Li et al. anchor-radius sliding window |
| [skmob_stops](skmob_stops.md) | library | no | scikit-mobility 1.2.2 | Lachesis anchor radius, no trailing stay |
| [fastmob_stays](fastmob_stays.md) | library | no | fastmob 0.3.2 | Rust port of skmob stops with `include_last` |
| [movingpandas_stops](movingpandas_stops.md) | library | no | MovingPandas 0.23.0 | Diameter (minimum rotated rectangle) window |
| [infostop](infostop.md) | library | no | Infostop 0.1.9 | Running-median stationary groups + Infomap places; visits = label runs |
| [st_dbscan](st_dbscan.md) | density | no | Birant & Kut 2007 (own numba impl.) | DBSCAN with space AND time neighbourhoods, then time-contiguous runs |
| [st_hdbscan](st_hdbscan.md) | density | no | Campello et al. 2013 (sklearn HDBSCAN) | HDBSCAN on positions, time-contiguous runs, absolute radius trim |
| [ua_mahalanobis](ua_mahalanobis.md) | uncertainty | yes | New | Sequential information-filter stay with chi-square gating, coarse-point buffering, k-miss breaking |
| [ua_segmentation](ua_segmentation.md) | uncertainty | yes | New | Heteroscedastic Gaussian PELT (exact penalised segmentation) + chi-square acceptance |
| [hmm_stay_move](hmm_stay_move.md) | probabilistic | yes | New | Two-state HMM on displacements with Rayleigh emissions and CTMC transitions, Viterbi |
| [changepoint_pelt](changepoint_pelt.md) | probabilistic | no | ruptures PELT (L2) | Homoscedastic baseline for ua_segmentation |
| [h3_occupancy](h3_occupancy.md) | grid | yes (coarse filter) | New | Anchor cell + k=1 ring hysteresis, coarse events ignored |

"Uses uncertainty" is the module's `USES_UNCERTAINTY` flag. The harness runs algorithms
with `USES_UNCERTAINTY = False` on shape `1` only, because every shape has identical
lat/lon (`src/dwells/harness/run.py:111-112`, `src/dwells/io.py:23-43`).

## Taxonomy

- **Sequential (anchor or running centroid)**: walk each id in time order and keep one
  open cluster. They differ in the reference point: the first fix (trackintel, skmob,
  fastmob; Li et al. 2008 and Hariharan & Toyama 2004), the running mean (li_centroid,
  h3_cell_dwell sessions, kalman_dwell; Kang et al. 2004), the running median
  (Infostop), or the window's extent (MovingPandas diameter). They are O(n) per id,
  or O(n·k) with k the window length, and they are trivially per-id partitionable.
- **Density**: cluster positions (plus time) without a sequential assumption, then cut
  clusters into time-contiguous runs so that revisits become separate dwells
  (`common.contiguous_runs`, `src/dwells/algorithms/common.py:108-133`).
- **Uncertainty-aware (new)**: explicit Gaussian observation model
  `z_i ~ N(mu, C_i + v I)` with per-event error covariance `C_i` and a within-place
  spread `v = radius_m^2/4`. Membership uses Mahalanobis gates or likelihood costs
  instead of fixed metres, coarse events are prevented from defining dwell boundaries,
  and the centroid is the information-weighted (fused) mean with a reported 1-sigma
  ellipse.
- **Probabilistic / temporal**: segment the time series (change points, HMM state
  paths). `changepoint_pelt` is the homoscedastic baseline and `ua_segmentation` its
  heteroscedastic counterpart. `hmm_stay_move` models the displacement process.
- **Grid**: discretise space into H3 cells. `h3_cell_dwell` uses plain cell runs, and
  `h3_occupancy` adds anchor-ring hysteresis and coarse-event rejection. Cell ids are
  per-event maps, so this family maps best onto Spark SQL.

## Shared conventions (read once, applies to every doc)

- **Parameter contract.** Every `detect` takes `min_duration_s`, `radius_m` and
  `max_gap_s` and swallows extra keyword arguments (`**_`). The harness sweeps
  `min_duration_s ∈ {300, 900, 1800, 3600}` and a per-profile grid of radius x max_gap
  (`src/dwells/harness/run.py:37-42`). Each doc states how an algorithm interprets the
  three.
- **Local planar frame.** `common.prepare` (`src/dwells/algorithms/common.py:31-41`)
  converts positions to east/north metres with an equirectangular projection anchored
  at each id's median lat/lon (`src/dwells/geo.py:16-20`). The error grows
  with distance from the anchor: roughly 0.1% within 50 km, and the east scale uses
  `cos(lat0)`. For ids that span hundreds of km the metric radius is distorted. The
  ported originals used a per-pair midpoint latitude or haversine.
- **Error covariance.** `prepare` calls `uncertainty.with_covariance` with
  `default_sigma_m = 10` for shape 1 (see [../uncertainty.md](../uncertainty.md)).
  "Coarse" always means the RMS 1-sigma radius `sqrt((C_ee + C_nn)/2)` exceeds a
  threshold.
- **Gap segmentation.** `common.gap_segments` (`common.py:44-48`) gives a segment
  number that increments on a new id or when consecutive events are more than
  `max_gap_s` apart.
- **Labels to dwells.** `common.dwells_from_labels` (`common.py:51-77`): events with
  the same label form a dwell. `start_ts`/`end_ts` are the first/last *labelled*
  event (not the first fix outside). The dwell is kept if
  `end - start >= min_duration_s`. The centroid is the `mean` (default) or `median`
  of lat/lon, or `fused`: the inverse-covariance-weighted mean in a local frame, with
  the 1-sigma ellipse of the fused estimate in `sigma_major/sigma_minor/azimuth`
  (`common.py:80-99`, via `cov_to_ellipse(..., ONE_SIGMA_P)`, `ONE_SIGMA_P = 1 - e^{-1/2}`
  so k = 1). `extra_var` is added to each event's variance before fusing.
- **Contract tests.** `tests/test_algorithms.py` checks every detector on clean
  one-fix-per-minute tracks: three easy dwells, constant motion, gap split, a short
  stay dropped, and multiple ids. Known violations are listed as strict xfails in
  `KNOWN_LIMITATIONS` (`tests/test_algorithms.py:62-71`).
  `tests/test_uncertainty_algorithms.py` checks the `ua_*`, `hmm_*` and `h3_occ*`
  detectors on coarse-point handling: a coarse point must not break a dwell, must not
  extend it, and must not pull its centroid.

## References (general)

- Li, Q. et al. (2008). Mining user similarity based on location history. ACM SIGSPATIAL GIS. doi:10.1145/1463434.1463477
- Kang, J. H., Welbourne, W., Stewart, B., Borriello, G. (2004). Extracting places from traces of locations. WMASH '04. doi:10.1145/1024733.1024748
- Hariharan, R., Toyama, K. (2004). Project Lachesis: parsing and modeling location histories. GIScience 2004, LNCS 3234. doi:10.1007/978-3-540-30231-5_8
- Zheng, Y. et al. (2009). Mining interesting locations and travel sequences from GPS trajectories. WWW '09. doi:10.1145/1526709.1526816
- H3 hierarchical hexagonal grid: https://h3geo.org
