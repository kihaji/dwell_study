# scikit-mobility (skmob) 1.2.2

## Overview

| | |
|---|---|
| What | General mobility-analysis library covering preprocessing (filter, compress, stop detection, clustering), individual and collective measures, generative models (EPR, gravity, radiation), privacy risk, and tessellation |
| Maintainer | scikit-mobility org (Pappalardo, Simini, Barlacchi, Pellungrini and others). **Effectively dormant**: last push 2024-05, and 1.2.2 is the last release |
| License | BSD-3-Clause ("new BSD") |
| Maturity | Widely cited (JSS paper). Older code base, pre-pandas-2 idioms |
| Dependencies | Heavy: pandas, geopandas, python-igraph, powerlaw, statsmodels, folium, geojson, h3, scikit-learn, requests, tqdm. It also imports **fiona** (`skmob/io/file.py`) without declaring it |
| API style | `skmob.TrajDataFrame` (a pandas DataFrame subclass with columns `uid, lat, lng, datetime`), plus module functions in `skmob.preprocessing.*` and `skmob.measures.*` |

Note: in 1.2.2 the detector is `skmob.preprocessing.detection.stops` (and the `TrajDataFrame` method `stops`). Later docs and fastmob call the same algorithm `stay_locations`.

## Stay detection: `detection.stops`

Source: `skmob/preprocessing/detection.py`: `stops` -> `_stops_trajectory` -> `_stops_array`. Lachesis-style (Hariharan & Toyama 2004), same family as Li et al.

| Param | Default | Meaning |
|---|---|---|
| `spatial_radius_km` | 0.2 | **km**. Radius around the **anchor (first point)**. Overrides `stop_radius_factor` unless it is None |
| `stop_radius_factor` | 0.5 | Used only when `spatial_radius_km=None`: radius = factor × the `compress()` radius stored in `tdf.parameters` |
| `minutes_for_a_stop` | 20.0 | **Minutes**. Minimum duration |
| `no_data_for_minutes` | 1e12 | **Minutes**. Gap threshold. **Off by default**, so stays span arbitrary gaps |
| `leaving_time` | True | Adds a `leaving_datetime` column |
| `min_speed_kmh` | None | Trims trailing points whose anchor-relative average speed exceeds this value |

How it works, per user, sorted by `(uid, datetime)`:
1. The anchor is `(lat_0, lon_0, t_0)` of point 0, and `sum_lat/lon` collect the members.
2. For each next point `(lat, lon, t)`:
   - If `t - t_prev > no_data_for_minutes`, the anchor resets to this point and the member lists are cleared. The open window is **discarded, not closed** (verified).
   - `Dr = haversine_km(anchor, pt)` (R = `gislib.earthradius`), `Dt = t - t_0`.
   - If `Dr > radius` (strict): if `Dt > minutes_for_a_stop` (strict), emit a stop with `lat/lng = median(members)` (lat and lon medians taken independently), `datetime = t_0`, `leaving_datetime = t`. Either way, reset the anchor to the current point.
   - Append the point to the members.
3. **No include_last.** A stop still open at the end of a track is never emitted (verified: the final 35-minute stay of the toy user was dropped).

Semantics and gotchas:
- **`leaving_datetime` is the first point outside the radius**, and duration is measured to it. As in trackintel, a single fix followed by a far fix 3 h later gives a 3 h stop (verified).
- Strict `>` on distance and time. Exactly 10 min with `minutes_for_a_stop=10` is **not** a stop, while trackintel and fastmob accept it (verified).
- Centroid = coordinate-wise **median** of member fixes. Duplicates count.
- The output has no member-point mapping and no `n_events`. The other columns are copied from the last inside row.
- With `min_speed_kmh`, "speed" is displacement from the anchor divided by elapsed time, not point-to-point speed. The trimming index arithmetic (`sum_lat[:-j]` against `sum_t[-j+1]`) is quirky. Avoid it.
- Complexity: an O(n) pure-Python loop with Python lists, plus an O(k) median per stop. Slowest of the sequential implementations.
- **Spark:** per-user sequential with bounded state, so it is partitionable by id.

### Pandas 3 bug: the uid column is dropped
`stops` and `cluster` use `tdf.groupby([uid], group_keys=False, as_index=False).apply(...)`. Under pandas 3 the grouping column is excluded from the frame passed to `apply`, so **the output has no `uid` column** (verified: output columns were `lat, lng, datetime, leaving_datetime`). A later `clustering.cluster` then sees a single user and clusters stops across users. **Workaround:** loop over ids yourself, pass a uid-less `TrajDataFrame` per id, and re-attach the id (see the snippet).

## Places and downstream analysis

- `clustering.cluster(tdf, cluster_radius_km=0.1, min_samples=1)` (`preprocessing/clustering.py`). Per-user DBSCAN, haversine, ball tree, `eps = r/6371.0088`. Labels are **re-ranked by visit frequency** (0 = most visited). **Bug:** with `min_samples>1`, noise label -1 raises `KeyError: -1` in `_cluster_array` (`[c2mfl[k] for k in labels]`, verified). Only `min_samples=1` is safe. It is subject to the same pandas-3 uid loss.
- `compression.compress(tdf, spatial_radius_km=0.2)` merges consecutive points within the radius into median points and stores the radius for `stop_radius_factor`.
- Measures (`measures/individual.py`) operate on **exact `(lat,lng)` tuples**, so replace stop coordinates with cluster centroids before calling them:
  - `home_location(start_night='22:00', end_night='07:00')` returns the most frequent night location.
  - `location_frequency`, `number_of_visits`, `number_of_locations`, `visits_per_location`, `recency_rank`, `frequency_rank` cover recurrence.
  - `waiting_times`, `jump_lengths`, `radius_of_gyration`, `k_radius_of_gyration`, `max_distance_from_home` cover dwell and movement statistics.
  - `individual_mobility_network(self_loops=False)` gives origin→destination edge counts between consecutive locations. This is the basis for transition matrices and top-N next location (no predictor is provided).
  - `random_entropy`, `uncorrelated_entropy`, `real_entropy` measure predictability and regularity.
- Collective (`measures/collective.py`): `visits_per_location`, `homes_per_location`, `visits_per_time_unit`, location entropies. No colocation or co-presence.

## Positional uncertainty

None in the algorithms. `io.file.load_google_timeline(min_accuracy_meters=...)` only filters on accuracy at load time.

## Compatibility (Py 3.12 / pandas 3 / shapely 2)

- `skmob/tessellation/tilers.py` imports `shapely.ops.cascaded_union`, which was removed in shapely 2 and is pulled in by `import skmob`. Workaround: run `import shapely.ops as so; so.cascaded_union = so.unary_union` before `import skmob`.
- It needs `fiona` installed (undeclared).
- **pandas 3 groupby.apply drops `uid`** (see above). This is the most dangerous issue because it fails silently.
- There are `SyntaxWarning: invalid escape sequence` warnings in `measures/individual.py` docstrings under Python 3.12. They are harmless.
- `datetime` stays tz-aware (UTC preserved) in the output.

## Minimal usage

```python
import shapely.ops as so; so.cascaded_union = so.unary_union  # before importing skmob
import pandas as pd, polars as pl, skmob
from skmob.preprocessing import detection

def skmob_stays(df: pl.DataFrame, min_duration_s, radius_m, max_gap_s):
    pdf = df.to_pandas()
    out = []
    for uid, g in pdf.groupby("id", sort=False):          # per-id loop: avoids pandas-3 uid loss
        tdf = skmob.TrajDataFrame(g[["lat", "lon", "timestamp"]],
                                  latitude="lat", longitude="lon", datetime="timestamp")
        st = detection.stops(tdf,
                             spatial_radius_km=radius_m / 1000,     # km, anchor radius
                             minutes_for_a_stop=min_duration_s / 60,
                             no_data_for_minutes=max_gap_s / 60,    # default is "never"
                             leaving_time=True)
        out.append(st.assign(id=uid))
    stays = pd.concat(out, ignore_index=True)
    # start=datetime (first fix), end=leaving_datetime (first fix OUTSIDE); no last open stay
    return stays.rename(columns={"datetime": "start_ts", "leaving_datetime": "end_ts", "lng": "lon"})
```

## Wrapper checklist
- Loop per id (pandas 3). Pass km and minutes, and always set `no_data_for_minutes`.
- The last open stay is lost. To recover it, append a sentinel far-away fix at `t_last + epsilon`, or handle the tail yourself.
- Compute `n_events` by re-joining fixes into `[start, end)` per id.
- Use `cluster(min_samples=1)` only, or do DBSCAN yourself.

## References
- Pappalardo, L., Simini, F., Barlacchi, G., Pellungrini, R. (2022). scikit-mobility: A Python library for the analysis, generation, and risk assessment of mobility data. *Journal of Statistical Software* 103(4). https://doi.org/10.18637/jss.v103.i04 (arXiv:1907.07062)
- Hariharan, R., Toyama, K. (2004). Project Lachesis: Parsing and modeling location histories. *GIScience 2004*, LNCS 3234, 106–124. http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf
- Zheng, Y. (2015). Trajectory data mining: an overview. *ACM TIST* 6(3).
- Docs: https://scikit-mobility.github.io/scikit-mobility/ · Repo: https://github.com/scikit-mobility/scikit-mobility

## Wrapper in this repo

`src/dwells/algorithms/skmob_stops.py` (`NAME = "skmob_stops"`).

- Parameter mapping: `spatial_radius_km = radius_m / 1000`, `minutes_for_a_stop = min_duration_s / 60`, `no_data_for_minutes = max_gap_s / 60`, `leaving_time=True`, `min_speed_kmh=None`.
- Gaps: each id is split on gaps > `max_gap_s`, and the segment number is passed as skmob's `uid` in **one** `stops` call. The pandas-3 uid loss is handled by carrying a copy of the segment column: `_stops_array` copies the other columns of each stop's last inside row into the output.
- `import shapely.ops as so; so.cascaded_union = so.unary_union` runs before `import skmob`.
- Native semantics kept: an anchor (first point) radius with strict `>` tests, end = `leaving_datetime` (first fix outside), and a coordinate-wise median centroid. `n_events` counts the id's events in `[start, end)`.
- **No include_last.** The stay open at the end of every segment (before every gap and at the end of the track) is never emitted, and the wrapper does not add one. As a result it fails 4 of the 5 contract tests (easy dwells, long-gap split, short-stay, multi-id: each loses its final stay). These are listed as known limitations. On real data it loses one stay per segment compared with trackintel/fastmob (1386 vs 1420 on dev_urban_gps). All other stays match trackintel's start, end and `n_events` exactly.
- Throughput: 54k GPS events in 0.4 s (~145k ev/s), 890k GPS events in 4.7 s (~190k ev/s). Sparse CDR data runs at ~4.6k ev/s (skmob's groupby.apply costs ~3 ms per segment).
