# fastmob 0.3.2

## Overview

| | |
|---|---|
| What | A Rust-accelerated reimplementation of much of scikit-mobility, plus trackintel-style staypoints, locations and trips. Adds H3 clustering, Markov next-location prediction, RECAST co-presence and social-tie inference, motifs, and road-network tools |
| Maintainer | Gustavo Henrique Bruno dos Santos (gefgu), a single author. Repo created 2026-04, about 6 stars, releases every few weeks (v0.2 → v0.3.2 in 2026) |
| License | BSD-3-Clause |
| Maturity | **Very young (0.x).** The API is still moving (for example, `cluster` switched from DBSCAN to H3 connected components while keeping the DBSCAN-shaped signature). The test suite and benchmarks look serious, but expect breaking changes. Pin the version |
| Dependencies | Light: narwhals>=1, numpy, pyarrow>=14. Optional `geo` extra (geopandas, shapely, pyproj). Ships a 19 MB compiled `_core` extension (PyO3, rayon, h3o) |
| API style | **narwhals**: accepts pandas, polars or pyarrow frames and returns the same backend. Column names are auto-detected (see the gotcha below) or passed as `*_col=` overrides. It also has trackintel-like wrappers (`fastmob.Staypoints`, `Locations`, `Positionfixes`) |

## Stay detection: `fastmob.preprocessing.stay_locations`

Python wrapper: `fastmob/preprocessing/_stay_locations.py`. Kernel: `fastmob-core/src/preprocessing/stay_locations.rs` (`detect_stops_for_user_core`), read from the upstream repo at tag v0.3.1 (there is no v0.3.2 tag). Behaviour was checked empirically against 0.3.2. It is the skmob algorithm in Rust, with trackintel-like comparisons.

| Param | Default | Meaning |
|---|---|---|
| `spatial_radius_km` | 0.2 | **km**. Radius from the **anchor (first point)** |
| `minutes_for_a_stop` | 20.0 | **Minutes**. Minimum duration |
| `no_data_for_minutes` | 1e12 | **Minutes**. Gap threshold, off by default |
| `include_last` | **True** | Close the still-open stay at the end of the track. The default differs from trackintel's |
| `leaving_time` | True | Adds `leaving_datetime` |
| `min_speed_kmh` | None | Trims trailing points whose anchor-relative speed is too high |
| `uid_col, lat_col, lng_col, datetime_col` | auto | Column overrides |

How it works, per user (rayon-parallel over users, input sorted by uid then time internally):
1. Drop rows with non-finite lat, lng or time, and **drop consecutive exact duplicates** (same lat, lng and t).
2. The anchor is point 0. For each next point: if `t - t_prev > gap`, reset the anchor to this point and `continue`. The open window is **discarded** (verified: a 20-minute pre-gap stay was lost).
3. `dr = haversine_km(anchor, pt)`. If `dr >= radius` (or this is the last point and `include_last`): if `t - t0 >= min_minutes`, emit `median(lat), median(lon)` over the members `[segment_start, i]` (the exiting point excluded), with `entry = t0` and `leaving = t`. Then reset the anchor.

Semantics and gotchas:
- **`leaving_datetime` is the first point outside the radius**, as in skmob and trackintel. With `include_last` it is the last fix.
- Comparisons are `>=` on both distance and time, which matches trackintel, not skmob's strict `>` (verified at exactly 10 min).
- Centroid = coordinate-wise **median** of member fixes (`median_slice_in_place`).
- **Output timestamps lose their timezone.** A UTC-aware input returns naive `datetime[ns]` for both `timestamp` and `leaving_datetime`, because times go through epoch milliseconds (verified with polars and pandas). Re-attach UTC. Sub-millisecond precision is truncated.
- **The uid column is auto-detected only from `["uid", "user", "user_id"]`. A column named `id` is ignored and the whole frame is treated as ONE user,** which merges tracks across ids (verified). Always pass `uid_col="id"`. `lon` and `timestamp` are auto-detected.
- The output has no `n_events` and no member-row mapping.
- Complexity: an O(n) Rust loop per user plus an O(k) median per stop. It releases the GIL and runs parallel across users. Orders of magnitude faster than the pure-Python versions.
- **Spark:** per-user sequential with bounded state. Partitionable by id (`applyInPandas` or `mapInArrow`, since the core takes Arrow arrays).

## Places and downstream analysis

- `preprocessing.cluster(stops, cluster_radius_km=0.1, min_samples=1, h3_resolution=None, uid_col=...)` (`preprocessing/_cluster.py`, kernel `h3_cluster.rs`). This is **not DBSCAN**. Each stop is assigned to an H3 cell. Cells with at least `min_samples` stops are active, and active cells that are grid-disk-1 neighbours are unioned (union-find) into clusters. Labels are ranked by stop count (0 = most visited) and -1 is noise. The radius maps to a resolution: 0.025→12, 0.05→11, 0.1→10, 0.2→9, 0.5→8, 1.0→7. Clustering is per user. Chaining across adjacent cells can make clusters arbitrarily large.
- `Staypoints.generate_user_locations(epsilon_km, min_samples)` wraps `cluster` and computes location centres as the **mean** of member stays. `Staypoints.generate_global_locations(h3_resolution=9)` gives **shared cross-user locations = H3 cell id**. This is trivially partitionable and useful for colocation.
- `identify_locations(locations, staypoints, method="freq")` labels home (most night dwell) and work (most weekday-daytime dwell). There are also `home_location`, `work_location`, `homes_per_location`.
- Recurrence and patterns: `location_frequency`, `number_of_visits`, `recency_rank`, `frequency_rank`, `regularity`, `waiting_times`, `daily_motifs`, `motif_distribution`, `activity_transition_matrix`, `daily_activity_distribution`, and the entropies (`real_entropy`, `trajectory_predictability`).
- Movement after a dwell: `individual_mobility_network` (edge counts), `od_matrix`, `Positionfixes.generate_triplegs`, trips and tours.
- **Top-N next location:** `fastmob.models.NextLocationPredictor(order=k, backoff=True)`, an order-k Markov model per user over `location_id` sequences with `predict_proba(top_k)` and `evaluate(top_k, n_holdout)`. It can serve directly as a baseline for plan step 22.
- **Colocation:** `fastmob.social.co_presence_graph_from_staypoints(staypoints, locations, min_minutes_for_encounter=5)` builds a daily interval-overlap graph of staypoints at the same global location (RECAST). `recast_from_staypoints` classifies ties (friends, acquaintances and so on). This covers same-time co-location. Different-time co-visitation must be derived from global location ids.

## Positional uncertainty

None. No accuracy or ellipse inputs anywhere (grep for accuracy/uncertainty finds only prediction accuracy).

## Compatibility (Py 3.12 / pandas 3 / polars 1.44)

- Installs as a wheel. Works with pandas 3 and polars. No warnings seen.
- Timezone is stripped on output and `id` is not auto-detected as uid (see above).
- `cluster` accepts `n_jobs` but ignores it.

## Minimal usage

```python
import polars as pl
from fastmob.preprocessing import stay_locations

def fastmob_stays(df: pl.DataFrame, min_duration_s, radius_m, max_gap_s, include_last=True):
    st = stay_locations(
        df, uid_col="id", lat_col="lat", lng_col="lon", datetime_col="timestamp",  # MUST pass uid_col
        spatial_radius_km=radius_m / 1000,
        minutes_for_a_stop=min_duration_s / 60,
        no_data_for_minutes=max_gap_s / 60,
        include_last=include_last,
    )
    return st.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC").alias("start_ts"),
        pl.col("leaving_datetime").dt.replace_time_zone("UTC").alias("end_ts"),  # first fix outside
    )
# places: fastmob.preprocessing.cluster(stays, cluster_radius_km=0.1, uid_col="id", datetime_col="start_ts")
```

## Wrapper checklist
- Pass `uid_col` explicitly. Re-attach UTC. Convert to minutes and km.
- Consecutive exact duplicates are removed before detection. Count `n_events` yourself from `[start, end)`.
- Pre-gap windows are discarded, so split on gaps first if gap-closing semantics are wanted.

## References
- Repo: https://github.com/gefgu/fastmob · Docs: https://gefgu.github.io/fastmob/
- Hariharan & Toyama (2004), Project Lachesis (algorithm origin, via skmob). http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf
- Pappalardo et al. (2022), scikit-mobility, *JSS* 103(4) (the API it mirrors).
- H3: https://h3geo.org/ ; RECAST: Vaz de Melo, P. O. S. et al. (2015). RECAST: Telling apart social and random relationships in dynamic networks. *Performance Evaluation* 87.
