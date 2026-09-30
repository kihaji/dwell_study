"""MovingPandas ``TrajectoryStopDetector`` (diameter-based stop detection).

Each id is split on gaps > ``max_gap_s`` here (MovingPandas has no gap handling;
its collection splitters drop zero-length pieces) and every segment becomes one
``Trajectory`` of a ``TrajectoryCollection``. ``max_diameter = 2 * radius_m``.
Native semantics kept: a stop is a window whose minimum-rotated-rectangle
diagonal is < ``max_diameter`` spanning >= ``min_duration``; start = first fix
of the window, end = last fix inside the stop; the open stop at the end of a
segment is always emitted. Stop time ranges come from ``get_stop_time_ranges``;
member fixes are the events in [start, end] and the centroid is their
coordinate-wise median (what ``get_stop_points`` computes, without its per-stop
pandas overhead). Duplicate timestamps, which MovingPandas drops, are counted.

Throughput: ~7k events/s on dev_urban_gps (54k events in 7.3 s; a geodesic bbox +
minimum rotated rectangle per fix over the growing window, O(k^2) per stop).
"""

from __future__ import annotations

from datetime import timedelta

import movingpandas as mpd
import numpy as np
import polars as pl

from dwells import schema
from dwells.algorithms.common import dwells_from_labels, gap_segments

NAME = "movingpandas_stops"
FAMILY = "library"
USES_UNCERTAINTY = False
REFERENCE = ("MovingPandas 0.23.0 TrajectoryStopDetector; Graser 2019, GI_Forum 7(1), "
             "doi:10.1553/giscience2019_01_s54")


def detect(events: pl.DataFrame, *, min_duration_s: float = 300, radius_m: float = 100,
           max_gap_s: float = 1800, **_) -> pl.DataFrame:
    ev = events.with_columns(gap_segments(events, max_gap_s))
    pdf = ev.select("seg", "lat", "lon", pl.col("timestamp").dt.replace_time_zone(None)).to_pandas()
    tc = mpd.TrajectoryCollection(pdf, traj_id_col="seg", t="timestamp", x="lon", y="lat", crs="EPSG:4326")
    ranges = mpd.TrajectoryStopDetector(tc).get_stop_time_ranges(
        max_diameter=2 * radius_m, min_duration=timedelta(seconds=min_duration_s))
    if not ranges:
        return schema.empty(schema.DWELLS)
    t = ev["timestamp"].dt.replace_time_zone(None).to_numpy()
    seg = ev["seg"].to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    seg_start = dict(zip(seg[bounds[:-1]], bounds[:-1]))
    seg_end = dict(zip(seg[bounds[:-1]], bounds[1:]))
    labels = np.full(len(t), -1, np.int64)
    for k, r in enumerate(ranges):
        a, b = seg_start[r.traj_id], seg_end[r.traj_id]
        lo = a + np.searchsorted(t[a:b], np.datetime64(r.t_0), "left")
        hi = a + np.searchsorted(t[a:b], np.datetime64(r.t_n), "right")
        labels[lo:hi] = k
    return dwells_from_labels(events, labels, 0, centroid="median")
