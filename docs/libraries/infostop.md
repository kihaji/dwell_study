# Infostop 0.1.9

## Overview

| | |
|---|---|
| What | Stop-location detection that combines (1) sequential detection of stationary events with (2) network community detection (Infomap) over the stationary-event medians. The output is a **per-fix location label** (stop-location id, or -1 for moving) |
| Maintainer | Ulf Aslak and Laura Alessandretti (DTU / Copenhagen). **Dormant**: last push 2025-01, no releases since 0.1.x |
| License | **GPL-3.0-or-later** according to the LICENSE file, while the PyPI classifier says MIT (conflicting metadata). It depends on **infomap==1.0.6, which is AGPL-3.0-or-later**. This is a real licensing concern for embedding in a downstream application |
| Maturity | Research code with a paper and a small API. `compute_label_area`, `compute_label_counts` and `predict` raise `NotImplementedError` |
| Dependencies | numpy, scikit-learn (BallTree), infomap (C++ build), pybind11, folium, tqdm, scipy. It ships a compiled `cpputils` extension |
| API style | numpy: `Infostop(...).fit_predict(array_or_list_of_arrays)`, where each array is `[lat, lon, t]` of shape (N, 3) sorted by time. There are no DataFrames, so all tabular glue is ours |

## Stay detection

Sources: `infostop/models.py` `Infostop.fit_predict`; `cpputils/main.cpp` `get_stationary_events` (C++, upstream repo); `infostop/utils.py` `query_neighbors`, `infomap_communities`, `label_network`; `infostop/postprocess.py` `compute_intervals`.

| Param | Default | Meaning |
|---|---|---|
| `r1` | 10 | Metres (haversine). Max distance of a fix from the **running median** of the current stationary group |
| `r2` | 10 | Metres. Link distance between stationary-event medians in the place network |
| `min_staying_time` | 300 | Time units of column 2 (use seconds). Minimum stop duration |
| `max_time_between` | 86400 | Max gap between consecutive fixes inside a stop. **A larger gap closes the group.** Must be > `min_staying_time` |
| `min_size` | 2 | Minimum fixes per stationary event (must be >1) |
| `label_singleton` | True | Stationary events with no neighbour within `r2` get their own label (False → -1) |
| `min_spacial_resolution` | 0 | Rounds medians (in degrees) before dedup to speed up clustering |
| `distance_metric` | "haversine" | or "euclidean" |
| `weighted`, `weight_exponent` | False, 1 | Edge weights `max(count) * d^-exp` |

Step 1: stationary events, per user, sequential (C++):
- The group starts at fix `i0`. For each next fix, compute `d = haversine(fix, (median(lat_group), median(lon_group)))` and `dt = t_i - t_{i-1}`.
- If `d <= r1 and dt <= max_time_between`, add the fix. The medians are maintained with sorted vectors (`insert_ordered`, O(k)).
- Otherwise, close the group. It is kept if `i - i0 >= min_size` and `t_{i-1} - t_{i0} >= min_staying_time`, and it is represented by its median lat/lon. Then start a new group at `i`.
- The final group gets the same test, so **the last stay is always considered**.

Step 2: dedupe medians, then use a BallTree to find neighbours within `r2` and build the network. Two-level Infomap over all stationary events gives place labels. With a **list of per-user arrays, the network is built over all users jointly, so labels are shared cross-user places**.

Step 3: labels are mapped back to fixes. Non-stationary fixes get -1.

Semantics and gotchas:
- **Membership is measured against the running median**, not the first fix. This makes it robust to a noisy first fix, with no anchor bias.
- **A gap closes the stop instead of discarding it**, the opposite of trackintel, skmob and fastmob. Verified: a 20-minute pre-gap stay was kept.
- **Duration = first to last fix inside the group.** Travel time to the next fix is not counted. End time = the last inside fix (verified 00:00–00:30, like movingpandas).
- The output is **per-fix labels, not stays**. Consecutive events with the same label are one visit. Stay intervals must be derived.
- **Bug in `postprocess.compute_intervals`:** the final group is appended only `if loc_prev == -1`, so **the last stop of every trace is dropped** (verified: A's final 35-minute stay was missing). Don't use it; derive intervals yourself (see the snippet).
- The stay centroid comes from `compute_label_medians()`, which gives the median of **stationary-event medians per place label** (a place centre, not a visit centre). For a per-visit centroid, take the median of the member fixes yourself.
- Infomap is stochastic, and Infostop exposes no seed (the Infomap default is used). Labels can change with input order or content.
- It raises `Exception("No stop events found")` if no user has a stationary event. It asserts strictly `-90<lat<90` and `-180<lon<180`, sorted timestamps, and no NaN.
- Complexity: step 1 is O(n·k) because of sorted-vector insertion, so it is quadratic in fixes per event (a concern for very dense 1M-fix stays). Steps 2–3 are a BallTree radius query plus Infomap over the unique event medians of **all users together**, which is super-linear and global.
- **Spark:** step 1 is per user. Step 2 with shared labels is a global operation. Calling `fit_predict` per id (single array) gives per-id places and full partitionability, but loses cross-id place identity (re-cluster globally later for colocation).

## Places and downstream analysis

Infostop *is* a place detector. The labels are recurring stop locations, per user or shared across users in multi-user mode. That makes multi-user mode directly useful for colocation and co-visitation: the same label for different ids is the same place. It has nothing for home/work, transitions or next location, and `visualize.plot_map` gives folium maps. Build transitions and top-N from the label sequences.

## Positional uncertainty

None. It could be adapted: `r1` membership could become Mahalanobis, and edge weights (`weighted=True`) could use uncertainty. Both need a fork.

## Compatibility (Py 3.12 / numpy 2 / scipy 1.18)

- It needs infomap 1.0.6 built from sdist. That build requires `setuptools<81` (our `pyproject.toml` has `[tool.uv.extra-build-dependencies] infomap = ["setuptools<81"]`).
- `from scipy.spatial.qhull import QhullError` raises a `DeprecationWarning` and **will break in SciPy 2.0**.
- The wheel installs top-level packages `cpputils` and **`tests`**, which pollutes the site-packages namespace and can shadow a local `tests` package on imports.
- Works with numpy 2.5 on the toy runs.

## Minimal usage

```python
import numpy as np, polars as pl
from infostop import Infostop

def infostop_stays(df: pl.DataFrame, min_duration_s, radius_m, max_gap_s, place_radius_m=None, shared=True):
    df = df.sort(["id", "timestamp"]).with_columns(t=pl.col("timestamp").dt.epoch("s").cast(pl.Float64))
    parts = df.partition_by("id", maintain_order=True)
    traces = [p.select("lat", "lon", "t").to_numpy() for p in parts]
    model = Infostop(r1=radius_m, r2=place_radius_m or radius_m,
                     min_staying_time=min_duration_s, max_time_between=max_gap_s)  # need max_gap > min_dur
    labels = model.fit_predict(traces if shared else traces[0])   # list => shared cross-id labels
    lab = pl.Series("place", np.concatenate(labels if shared else [labels]))
    df = pl.concat(parts).with_columns(lab)
    gap = pl.col("t").diff().over("id") > max_gap_s
    run = ((pl.col("place") != pl.col("place").shift().over("id")) | gap).fill_null(True).cum_sum()
    return (df.with_columns(run=run).filter(pl.col("place") >= 0)
              .group_by("id", "run", "place", maintain_order=True)
              .agg(start_ts=pl.col("timestamp").min(), end_ts=pl.col("timestamp").max(),  # last fix inside
                   lat=pl.col("lat").median(), lon=pl.col("lon").median(), n_events=pl.len()))
```

The snippet was checked on toy data: it recovers all four stays, including the final one that `compute_intervals` drops.

## Wrapper checklist
- Epoch seconds as column 2, sorted per id. Keep `max_time_between > min_staying_time`.
- Derive stays from label runs yourself (compute_intervals bug). The end time is last-inside.
- Choose per-id versus shared mode explicitly. It matters for both scaling and colocation.
- Treat GPL/AGPL licensing as a scorecard item.

## References
- Aslak, U., Alessandretti, L. (2020). Infostop: Scalable stop-location detection in multi-user mobility data. arXiv:2003.14370. https://arxiv.org/abs/2003.14370
- Rosvall, M., Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. *PNAS* 105(4), 1118–1123 (Infomap). https://www.mapequation.org/
- Repo: https://github.com/ulfaslak/infostop · Docs: https://infostop.readthedocs.io/

## Wrapper in this repo

`src/dwells/algorithms/infostop.py` (`NAME = "infostop"`).

- Parameter mapping: `r1 = r2 = radius_m`, `min_staying_time = min_duration_s`, `max_time_between = max_gap_s`, and the defaults `min_size=2` and `label_singleton=True`. The library asserts `max_gap_s > min_duration_s`, so a sweep must respect that.
- Gaps: each id is split on gaps > `max_gap_s`, and all segments go to **one** `fit_predict` call as a list of `[lat, lon, t_seconds]` arrays. This is multi-user mode, so place labels are shared across ids.
- Visits are built from label runs: consecutive fixes with the same label within a segment form one visit (`compute_intervals` is not used because it drops the last stop). The centroid is the median of the visit's fixes, start and end are the first and last fix inside, and `n_events` is the run length.
- "No stop events found" is caught and returns an empty frame. Infomap is unseeded; only run merging (adjacent stationary events with the same place label) depends on it.
- Throughput: 54k GPS events in 0.09 s (~570k ev/s), 890k GPS events in 4.4 s (~200k ev/s, super-linear because of the global BallTree + Infomap), and 86k CDR events in 0.6 s.
