# Weighted scorecard

Computed by `uv run python -m dwells.harness.score` (method in its module docstring). Each component is scaled to [0, 1] across algorithms, then weighted. Change the weights in `src/dwells/harness/weights.toml` and the rubric in `src/dwells/harness/rubric.toml`, then re-run.

* **accuracy:** 0.6 x held-out synthetic F1 (md 15 + 30 min) + 0.2 x labeled public F1 + 0.2 x place ARI, divided by the best.
* **scalability:** 0.5 x log10 throughput (min-max scaled) + 0.5 x fraction of benchmark rungs completed.
* **uncertainty:** measured F1 gain from shape 1 to 3_mixed on CDR + adtech, divided by the best gain.
* **spark / extensibility / ease_of_use:** author rubric (1–5), below.

Weights: {'accuracy': 0.35, 'scalability': 0.2, 'uncertainty': 0.15, 'spark': 0.1, 'extensibility': 0.1, 'ease_of_use': 0.1}

| algo | score | s_accuracy | s_scalability | s_uncertainty | s_spark | s_extensibility | s_ease_of_use | f1_synthetic | f1_public | place_ari | f1_gain_from_error | throughput_eps | max_events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ua_segmentation | 0.905 | 1.0 | 0.886 | 0.787 | 0.8 | 1.0 | 0.8 | 0.621 | 0.744 | 0.937 | 0.117 | 1,782,375 | 16494494 |
| ua_mahalanobis | 0.899 | 0.976 | 0.981 | 0.541 | 1.0 | 1.0 | 0.8 | 0.604 | 0.733 | 0.912 | 0.08 | 6,898,067 | 16494494 |
| hmm_stay_move | 0.879 | 0.872 | 0.92 | 1.0 | 1.0 | 0.8 | 0.6 | 0.561 | 0.625 | 0.78 | 0.149 | 2,890,704 | 16494494 |
| h3_occupancy | 0.815 | 0.955 | 0.886 | 0.291 | 1.0 | 0.8 | 0.8 | 0.573 | 0.735 | 0.927 | 0.043 | 1,779,472 | 16494494 |
| li_centroid | 0.767 | 0.82 | 1.0 | 0.0 | 1.0 | 0.8 | 1.0 | 0.428 | 0.729 | 0.891 | 0.0 | 9,063,802 | 16494494 |
| st_dbscan | 0.728 | 0.856 | 0.944 | 0.0 | 0.8 | 0.8 | 0.8 | 0.498 | 0.618 | 0.918 | 0.0 | 4,111,727 | 16494494 |
| fastmob_stays | 0.716 | 0.792 | 0.997 | 0.0 | 0.8 | 0.6 | 1.0 | 0.428 | 0.736 | 0.783 | 0.0 | 8,670,185 | 16494494 |
| kalman_dwell | 0.657 | 0.71 | 0.609 | 0.713 | 0.6 | 0.6 | 0.6 | 0.399 | 0.66 | 0.659 | 0.106 | 34,588 | 16494494 |
| trackintel_sliding | 0.592 | 0.779 | 0.599 | 0.0 | 0.6 | 0.6 | 0.8 | 0.428 | 0.736 | 0.74 | 0.0 | 66,213 | 9367454 |
| h3_cell_dwell | 0.586 | 0.542 | 0.884 | 0.0 | 1.0 | 0.6 | 0.6 | 0.156 | 0.58 | 0.873 | -0.037 | 1,729,524 | 16494494 |
| skmob_stops | 0.536 | 0.795 | 0.687 | 0.0 | 0.4 | 0.4 | 0.4 | 0.421 | 0.769 | 0.785 | 0.0 | 230,880 | 9367454 |
| changepoint_pelt | 0.534 | 0.837 | 0.403 | 0.0 | 0.4 | 0.6 | 0.6 | 0.534 | 0.495 | 0.869 | 0.0 | 19,942 | 16494494 |
| infostop | 0.515 | 0.824 | 0.632 | 0.0 | 0.2 | 0.4 | 0.4 | 0.432 | 0.729 | 0.896 | 0.0 | 106,462 | 9367454 |
| st_hdbscan | 0.514 | 0.596 | 0.625 | 0.0 | 0.6 | 0.6 | 0.6 | 0.249 | 0.55 | 0.817 | 0.0 | 96,407 | 16494494 |
| movingpandas_stops | 0.512 | 0.784 | 0.389 | 0.0 | 0.4 | 0.6 | 0.6 | 0.48 | 0.468 | 0.87 | 0.0 | 7,369 | 1969661 |


## Rubric rationale

| algorithm | spark | extensibility | ease of use |
|---|---|---|---|
| li_centroid | 5: single O(n) numba scan per id; O(1) state | 4: tiny, readable kernel; no uncertainty hooks | 5: 3 intuitive params, no dependencies beyond numba |
| h3_cell_dwell | 5: per-event H3 map + per-id run scan; SQL-expressible | 3: resolution rule tied to ellipse size; hard to tune | 3: resolution chosen implicitly from error; fragile on precise data |
| kalman_dwell | 3: per id, but a dense time grid over the whole span: memory ~ span/dt x 16 floats | 3: state-space model is extensible but the greedy grid step is ad hoc | 3: grid step, acceleration noise and gap params interact |
| trackintel_sliding | 3: per-user sliding window; geopandas stack on executors | 3: rich framework (trips, tours, locations) but fixed internals | 4: clean install, good docs, minutes-based params |
| skmob_stops | 2: per-user python loop; slow on sparse data | 2: dormant project; internals not designed for extension | 2: breaks on shapely 2 / pandas 3 (shim needed), drops uid, loses trailing stay |
| fastmob_stays | 4: Rust kernel, batch over users, arrow-native; young dependency on executors | 3: broad API (places, next-location, co-presence) but Rust core | 5: pip install, polars/arrow in, very fast; must pass uid_col |
| movingpandas_stops | 2: per trajectory but quadratic in stay length and slow | 3: good trajectory toolkit; detector itself simple | 3: geopandas-heavy; no gap handling; splitting bug |
| infostop | 1: global Infomap clustering over all users' stationary events | 2: C++ core, dormant | 2: build dependency pin, non-deterministic, GPL-3.0 + AGPL infomap |
| st_dbscan | 4: per id; time-window neighbor scan O(n x window) | 4: neighborhood predicate easy to change (e.g. uncertainty-aware) | 4: familiar DBSCAN params |
| st_hdbscan | 3: per id sklearn HDBSCAN; memory heavy for large ids | 3: needs spatial trim + run splitting to behave | 3: sklearn epsilon bug under numpy 2.5; relative density surprises |
| ua_mahalanobis | 5: single O(n) numba scan per id; O(1) state | 5: explicit Gaussian model: gate, coarse rule, place spread all pluggable | 4: same 3 core params plus gate/coarse defaults that rarely need tuning |
| ua_segmentation | 4: per id PELT; near-linear with pruning, O(n) memory per segment | 5: any additive likelihood cost can be plugged in | 4: penalty parameter less intuitive than a radius |
| hmm_stay_move | 5: per id Viterbi, linear | 4: more states (e.g. transport modes) or emissions can be added | 3: priors (speed, mean stay/move) need care; sensitive to outliers |
| changepoint_pelt | 2: ruptures PELT is pure python and slow per id | 3: ruptures offers many cost models | 3: simple API but slow; penalty scale in m^2 |
| h3_occupancy | 5: per-event H3 map + neighbor lists, per-id run scan; SQL-friendly | 4: resolution / hysteresis / coarse rules are explicit | 4: radius picks the resolution; few params |
