# MovingPandas 0.23.0

## Overview

| | |
|---|---|
| What | Trajectory data structures (`Trajectory`, `TrajectoryCollection`) on top of GeoPandas, with splitters, generalizers, cleaners, smoothers, a stop detector, an aggregator (significant points and flows) and plotting |
| Maintainer | Anita Graser and contributors. Very active: last push 2026-09-30, about 1.4k stars |
| License | BSD-3-Clause |
| Maturity | Mature and well documented. Oriented toward trajectory exploration and visualisation more than large-scale batch analytics |
| Dependencies | Moderate: geopandas, shapely>=2, geopy, matplotlib. Optional: stonesoup (Kalman smoother), holoviews/folium/bokeh (viz) |
| API style | Object-oriented: build a `TrajectoryCollection` from a (Geo)DataFrame (`traj_id_col`, `t`, `x`, `y`), then call tool classes such as `TrajectoryStopDetector(tc).get_stop_points(...)`. Each trajectory holds a GeoDataFrame indexed by a DatetimeIndex |

## Stay detection: `TrajectoryStopDetector`

Source: `movingpandas/trajectory_stop_detector.py`, `TrajectoryStopDetector._process_traj`. This is a **diameter-based** detector, not an anchor-radius one.

| Param | Default | Meaning |
|---|---|---|
| `max_diameter` | required | **Metres** (geodesic, via geopy) for EPSG:4326, otherwise CRS units. It is a **diameter**, so use `2*radius_m` to match radius-based libraries |
| `min_duration` | required | `datetime.timedelta` |
| `n_processes` | 1 | multiprocessing `Pool` over trajectories |

How it works, per trajectory, iterating points in time order:
1. Append the point to a window. **While not stopped**, drop the oldest points while `len>2` and `t_last - t_first >= min_duration`. This keeps a trailing window just under `min_duration`.
2. If the window has more than one point: a quick reject when the geodesic bbox diagonal is ≥ `1.5*max_diameter`. Otherwise the window is "stopped" if the **diagonal of the minimum rotated rectangle** (`geometry_utils.mrr_diagonal`) is `< max_diameter` (strict). The rectangle is built in degree space, and the diagonal is measured geodesically between its corners.
3. When a stopped window becomes not-stopped, the stop is emitted if `ts[-2] - ts[0] >= min_duration`. The window then restarts from the current point.
4. At the end of the trajectory, an open stop is emitted if its duration is sufficient. **The last stay is always included.**

Semantics and gotchas:
- **End time = the last point inside the stop** (`ts[-2]`), not the first point outside. This is the only library besides Infostop to use that convention. Verified: stop 00:00–00:30, where the other libraries give 00:35.
- Start time = the first point of the window. For a stationary stretch that is the first stationary fix. When arriving, approach points within the diameter are included.
- A stop needs ≥2 points spanning ≥ `min_duration`. A single isolated fix is never a stop, which differs from the anchor-radius libraries.
- The diameter is over **all member points**, which suits noisy data better than an anchor radius because it has no first-point bias.
- **No gap handling at all.** Two fixes at the same place 3 h apart form a 3 h stop. Split on gaps first with `ObservationGapSplitter(tc).split(gap=timedelta(...))`. Unlike the other libraries, this gives gap-closes-stop semantics, because the stop ends at the last pre-gap fix.
- **Bug: `TrajectorySplitter._split_traj_collection` keeps a piece only if `get_length() > min_length` (strict, default 0).** A perfectly stationary segment (all identical coordinates) has length 0 and is **silently dropped** when splitting a `TrajectoryCollection`. Verified: B's 00:00–00:20 stationary piece vanished. This bites CDR-like data with repeated tower coordinates. Workaround: split single `Trajectory` objects (that path returns all pieces), or split by gap yourself and assign segment ids before building the collection.
- `get_stop_points` location = **median x and median y** of the stop segment's points. The output has `start_time, end_time, duration_s, traj_id`, and the index `stop_id = f"{traj_id}_{start_time}"`. It has no point count; use `get_stop_segments` and `len(seg.df)`.
- The trajectory index is time, and **duplicate timestamps are dropped (keep first)** per trajectory. Ids with fewer than 2 points are skipped entirely.
- **Timezone is stripped** (`TimeZoneWarning`). Times are naive local wall time, so re-localize to UTC.
- Complexity: per point, it builds Python lists and a shapely `MultiPoint` plus a minimum rotated rectangle over the whole window. While stopped the window grows without bound, so each step is O(k) (min/max plus MRR over k points). **This is O(k²) per stop** and very slow for long, dense stays such as a 1M-fix single id. Expect it to be the slowest library.
- **Spark:** per-trajectory sequential and partitionable by id, but the state grows with stop length.

## Places and downstream analysis

- `TrajectoryCollectionAggregator(tc, max_distance, min_distance, min_stop_duration, min_angle=45)` (`trajectory_aggregator.py`). It extracts "significant points" (turns and stops), clusters them with a grid-based `PointClusterer` (centroid clusters within `max_distance`), and outputs `get_clusters_gdf()` and `get_flows_gdf()` (flow counts between clusters). This gives a global place set and flows (Andrienko & Andrienko style), but it is **dataset-global** and not per id, and it has no visit semantics.
- `StopSplitter(tc).split(max_diameter, min_duration)` returns the moves between stops, which are the "movement after a dwell" segments.
- Also `ObservationGapSplitter`, `SpeedSplitter`, `TemporalSplitter`, `MinDistanceGeneralizer`, and cleaners (`OutlierCleaner`).
- There are no home/work, recurrence, next-location or colocation measures.

## Positional uncertainty

None in detection. `KalmanSmootherCV.smooth(process_noise_std, measurement_noise_std)` (it needs stonesoup, which is not installed) takes a **single scalar or per-axis** measurement noise, not per-point accuracy. It can be used to pre-smooth tracks.

## Compatibility (Py 3.12 / pandas 3 / shapely 2)

- It works, and requires shapely>=2.
- On import it warns about the missing optional stonesoup.
- `TimeZoneWarning` on tz-aware input.
- The splitter length bug above.

## Minimal usage

```python
from datetime import timedelta
import polars as pl, movingpandas as mpd

def movingpandas_stays(df: pl.DataFrame, min_duration_s, radius_m, max_gap_s):
    pdf = df.to_pandas()
    pdf["timestamp"] = pdf["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)  # avoid warning
    # gap-split ourselves (avoids the zero-length-segment drop in ObservationGapSplitter)
    pdf = pdf.sort_values(["id", "timestamp"])
    seg = (pdf.groupby("id")["timestamp"].diff().dt.total_seconds() > max_gap_s).groupby(pdf["id"]).cumsum()
    pdf["traj"] = pdf["id"].astype(str) + "|" + seg.astype(str)
    tc = mpd.TrajectoryCollection(pdf, traj_id_col="traj", t="timestamp", x="lon", y="lat", crs="EPSG:4326")
    sp = mpd.TrajectoryStopDetector(tc).get_stop_points(
        max_diameter=2 * radius_m,                       # DIAMETER in metres
        min_duration=timedelta(seconds=min_duration_s))
    sp["id"] = sp["traj_id"].str.rsplit("|", n=1).str[0]
    sp["start_ts"] = sp["start_time"].dt.tz_localize("UTC")
    sp["end_ts"] = sp["end_time"].dt.tz_localize("UTC")   # LAST fix inside the stop
    return sp
```

## Wrapper checklist
- Diameter = 2 × radius. Gap-split before detection (by hand). Re-localize UTC.
- The end-time convention is last-inside, so normalise it against the other libraries.
- Avoid it on very dense long stays (quadratic cost), or subsample first.

## References
- Graser, A. (2019). MovingPandas: Efficient structures for movement data in Python. *GI_Forum* 7(1), 54–68. https://doi.org/10.1553/giscience2019_01_s54
- Andrienko, N., Andrienko, G. (2011). Spatial generalization and aggregation of massive movement data. *IEEE TVCG* 17(2) (basis of the aggregator).
- Docs: https://movingpandas.readthedocs.io/ · Repo: https://github.com/movingpandas/movingpandas · Stop detection example: https://github.com/movingpandas/movingpandas-examples

## Wrapper in this repo

`src/dwells/algorithms/movingpandas_stops.py` (`NAME = "movingpandas_stops"`).

- Parameter mapping: `max_diameter = 2 * radius_m` (metres, geodesic) and `min_duration = timedelta(seconds=min_duration_s)`.
- Gaps: MovingPandas has none. Each id is split on gaps > `max_gap_s` by the wrapper, not with the splitters that drop zero-length pieces. Each segment becomes one `Trajectory` via `TrajectoryCollection(traj_id_col="seg")`, which does not apply the length filter when `min_length=0`. Timestamps are passed as naive UTC to avoid the `TimeZoneWarning`, and UTC is restored afterwards.
- Native semantics kept: the MRR-diagonal < diameter window, start = first window fix, end = **last fix inside**, and the open stop at a segment's end is emitted. Time ranges come from `get_stop_time_ranges`. The members are the events in `[start, end]`, and their median lat/lon is the centroid, which is the same as what `get_stop_points` computes but without its per-stop pandas `.at` loop. Duplicate timestamps, which MovingPandas drops, are counted in `n_events`.
- Throughput: the slowest library. 54k GPS events take 7.3 s (~7.3k ev/s), 890k GPS events take 114 s (~7.8k ev/s), and 86k CDR events take 20 s (~4.3k ev/s). A 50-combo sweep on 1M events is about 1.5–2 h single-core, so subsample ids or parallelise over segments (`n_processes`) if needed.
