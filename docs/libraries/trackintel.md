# trackintel 1.4.2

## Overview

| | |
|---|---|
| What | Framework for human-mobility analysis built around a fixed data model: positionfixes -> staypoints -> triplegs -> trips -> tours, plus locations |
| Maintainer | MIE Lab, ETH Zurich (Henry Martin, Ye Hong, Dominik Bucher and others). Actively maintained; last push 2026-09 |
| License | MIT |
| Maturity | Stable 1.x line with a peer-reviewed paper and a test suite. The most complete staypoint → location → trip model of the five libraries |
| Dependencies | Heavy: pandas, geopandas>=0.12, shapely, scikit-learn, osmnx, geoalchemy2, matplotlib, tqdm, similaritymeasures, joblib |
| API style | pandas/geopandas subclasses (`ti.Positionfixes`, `ti.Staypoints`, `ti.Locations`), exposed as methods or as `ti.preprocessing.*` functions. Requires a GeoDataFrame with `user_id`, `tracked_at` (tz-aware) and point geometry |

## Stay detection: `generate_staypoints(method="sliding")`

Source: `trackintel/preprocessing/positionfixes.py`: `generate_staypoints` -> `_generate_staypoints_sliding_user` -> `__create_new_staypoints`. Adapted from Li et al. (2008).

| Param | Default | Meaning |
|---|---|---|
| `dist_threshold` | 100 | Metres (haversine, R=6371000). Distance from the window's **anchor (first point)** |
| `time_threshold` | 5.0 | **Minutes**. Minimum duration |
| `gap_threshold` | 15.0 | **Minutes**. A gap between consecutive points larger than this restarts the window |
| `include_last` | False | Emit the stay that is still open at the end of a user's track |
| `exclude_duplicate_pfs` | True | Runs `pfs.drop_duplicates()` on whole rows (with a warning) |
| `n_jobs` | 1 | joblib parallelism over `groupby("user_id")` |
| `distance_metric` | "haversine" | The only accepted value |

How it works, per user, with points sorted stably by `tracked_at`:
1. `start = 0`. For each `curr = 1..n-1`:
   - If `tracked_at[curr] - tracked_at[curr-1] > gap_threshold`, set `start = curr` and **continue**. The open window is thrown away, not closed. A stay that had already lasted longer than `time_threshold` before the gap is lost. Verified: a 20-minute stay followed by a 60-minute gap, with time_threshold=10, produced no staypoint.
   - `d = haversine(pt[start], pt[curr])`. If `d >= dist_threshold`:
     - if `tracked_at[curr] - tracked_at[start] >= time_threshold`, emit a stay over the points `[start, curr)`.
     - `start = curr`. This happens whether or not a stay was emitted.
2. With `include_last`, if `tracked_at[last] - tracked_at[start] >= time_threshold`, the stay is emitted over `[start, n)`.

Semantics and gotchas:
- **Radius is measured to the first point, not to a centroid.** Slow drift (for example 20 m every 5 min) splits into back-to-back stays of about `dist_threshold` each (verified).
- **End time is the timestamp of the first point outside the stay** (`finished_at = tracked_at[curr]`), and the duration test uses that same point. So duration includes the travel time up to the next fix. A single fix followed by a far fix 3 h later gives a 3 h staypoint with one point (verified). With `include_last`, the end is the last fix.
- The time test is `>=` and the distance test is `>=`. Exactly 10 min with time_threshold=10 counts as a stay (verified). skmob uses `>` for both.
- **Centroid: `points.union_all()` followed by a circular mean of lon and an arithmetic mean of lat** (`angle_centroid_multipoints`, `preprocessing/util.py`). Because `union_all` merges identical coordinates, repeated fixes at the same point count once. The result is an unweighted mean of distinct positions. Verified: 7 fixes at two positions (4 and 3) gave the midpoint, not the 4:3 weighted mean. This matters for CDR or adtech data with repeated tower or snapped coordinates.
- `elevation`, if present, is aggregated by median. There is no count column. Get `n_events` from `pfs.staypoint_id.value_counts()`.
- Complexity: an O(n) Python loop per user with a scalar haversine call per point, plus a shapely union per stay and an `explode`/`join` to map `staypoint_id` back to points. Expect roughly 10^5–10^6 points per minute per core. Memory holds lists of point ids per stay.
- **Spark:** purely per-user sequential with bounded state (one anchor). `groupBy("id").applyInPandas` works.

## Places and downstream analysis

- `generate_locations(sp, method="dbscan", epsilon=100, num_samples=1, distance_metric="haversine", agg_level="user"|"dataset", activities_only=False)` (`preprocessing/staypoints.py`). Runs sklearn DBSCAN with a ball tree on radians and `eps = epsilon/6371000`.
  - `agg_level="user"`: one DBSCAN per user; ids are offset so they are globally unique. Per-user and partitionable.
  - `agg_level="dataset"`: one DBSCAN over all users' staypoints, which gives shared location ids across users. This is directly useful for colocation and different-time co-visitation, but it is a global operation (not partitionable).
  - Location `center` is the centroid of the dissolved (unioned) staypoints, so duplicate staypoint coordinates again collapse. `extent` is the convex hull buffered by `epsilon`. Noise (`num_samples>1`) gets `location_id = NA`.
- `merge_staypoints(sp, tpls, max_time_gap="10min")` merges consecutive staypoints at the same `location_id` when no tripleg lies between them. It fixes stays fragmented by jitter, but it breaks the pfs↔sp linkage.
- `generate_triplegs(pfs, sp, method="between_staypoints", gap_threshold=15)`, then `generate_trips(sp, tpls, gap_threshold=15)` and `generate_tours(...)` (`preprocessing/triplegs.py`, `trips.py`). These give movement after a dwell: origin and destination staypoints per trip.
- `create_activity_flag(sp, time_threshold=15)` flags staypoints with duration ≥ threshold.
- `location_identifier(sp, method="FREQ"|"OSNA", pre_filter=True)` (`analysis/location_identification.py`) labels home and work. FREQ ranks locations by summed duration; OSNA uses weekday rest/work/leisure time windows (Efstathiades 2015). `pre_filter_locations` sets data-sufficiency thresholds.
- `radius_gyration`, `jump_length` (`analysis/metrics.py`), `temporal_tracking_quality` (coverage per day, week or hour).
- Missing: next-location prediction, transition matrices and colocation. Those can be built from `trips` (`origin_staypoint_id` → `destination_staypoint_id` → `location_id`) and dataset-level `location_id`.

## Positional uncertainty

None. `Positionfixes` lists `accuracy` as an optional column (`model/positionfixes.py`), and `dataset_reader` fills it, but no algorithm reads it. Use accuracy only for pre-filtering or down-weighting outside the library.

## Compatibility (Py 3.12 / pandas 3.0.6 / shapely 2.1.2 / geopandas 1.2)

- Imports and runs cleanly with no warnings on the toy runs (uses `union_all`, which is shapely-2 native).
- `tracked_at` must be tz-aware; UTC round-trips correctly.
- `user_id` dtype is preserved, and string ids work.
- The heavy optional-in-spirit deps (osmnx, geoalchemy2) are hard requirements.

## Minimal usage

```python
import geopandas as gpd, polars as pl, trackintel as ti

def trackintel_stays(df: pl.DataFrame, min_duration_s, radius_m, max_gap_s, include_last=True):
    pdf = df.to_pandas().rename(columns={"id": "user_id", "timestamp": "tracked_at"})
    pdf["tracked_at"] = pdf["tracked_at"].dt.tz_convert("UTC")  # must be tz-aware
    gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf.lon, pdf.lat), crs="EPSG:4326")
    pfs, sp = ti.Positionfixes(gdf).generate_staypoints(
        method="sliding",
        dist_threshold=radius_m,               # metres, anchor-to-point
        time_threshold=min_duration_s / 60,    # MINUTES
        gap_threshold=max_gap_s / 60,          # MINUTES; pre-gap window is discarded
        include_last=include_last,
        n_jobs=1,
    )
    n = pfs.groupby("staypoint_id").size().rename("n_events")
    sp = sp.join(n)
    # end_ts = first fix AFTER the stay (Li et al.); duration includes departure travel
    return sp  # columns: user_id, started_at, finished_at, geom (+ n_events)

# places: sp, locs = sp.generate_locations(epsilon=100, num_samples=1, agg_level="dataset")
```

## Wrapper checklist
- Convert seconds to minutes for `time_threshold` and `gap_threshold`.
- Decide on the end-time convention. `finished_at` is the first outside fix. To get the last inside fix, take `pfs[pfs.staypoint_id==k].tracked_at.max()`.
- Recompute the centroid (plain mean or median over all member fixes) if you want count-weighted centroids.
- Pre-gap stays are dropped. To close at gaps instead, split tracks on gaps yourself (a synthetic `user_id` per segment) and set `gap_threshold` very large.

## References
- Martin, H., Hong, Y., Wiedemann, N., Bucher, D., Raubal, M. (2023). Trackintel: An open-source Python library for human mobility analysis. *Computers, Environment and Urban Systems* 101, 101938. https://doi.org/10.1016/j.compenvurbsys.2023.101938
- Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., Ma, W.-Y. (2008). Mining user similarity based on location history. *ACM SIGSPATIAL GIS '08*, 34. https://doi.org/10.1145/1463434.1463477
- Efstathiades, H. et al. (2015). Identification of key locations based on online social network activity. *ASONAM '15*. https://doi.org/10.1145/2808797.2808877
- Docs: https://trackintel.readthedocs.io/ · Repo: https://github.com/mie-lab/trackintel

## Wrapper in this repo

`src/dwells/algorithms/trackintel_sliding.py` (`NAME = "trackintel_sliding"`).

- Parameter mapping: `dist_threshold = radius_m`, `time_threshold = min_duration_s / 60`, `gap_threshold = max_gap_s / 60`, `include_last=True`, other arguments at their defaults.
- Gaps: each id is split on gaps > `max_gap_s` (`common.gap_segments`), and every segment goes to trackintel as its own `user_id`. A gap therefore closes the stay instead of discarding it, and `include_last` emits the stay that is open at each segment's end. Results are mapped back to the real id.
- Native semantics kept: the radius is measured from the first point with `>=` tests, and the end is the first fix outside the stay (the last fix for a trailing stay). The centroid is trackintel's circular mean of *distinct* member positions, and `n_events` comes from the library's pfs→staypoint membership.
- Cross-check: start, end and `n_events` match `fastmob_stays` exactly (1420 and 7047 dwells on dev_urban_gps at two parameter sets). Only the centroid differs.
- Throughput: 54k GPS events in 1.0 s (~55k ev/s), 890k GPS events (eval_urban_gps) in 15 s (~60k ev/s). On sparse CDR data it drops to ~3.6k ev/s (86k events in 24 s) because of per-user overhead across many short segments.
