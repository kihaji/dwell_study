"""Infostop stop-location detection (sequential stationary events + Infomap places).

Each id is split on gaps > ``max_gap_s`` here and all segments are passed to one
``fit_predict`` call as a list of ``[lat, lon, t]`` traces (multi-user mode, so
place labels are shared across ids); ``r1 = r2 = radius_m``,
``min_staying_time = min_duration_s``, ``max_time_between = max_gap_s`` (the
library requires ``max_gap_s > min_duration_s``). Native semantics kept: a fix
joins the stationary group if within ``r1`` of the group's running median and
within ``max_time_between`` of the previous fix; >= 2 fixes; start/end = first/
last fix inside (the last stay is kept). Infostop outputs per-fix place labels;
a visit is a run of consecutive fixes with the same label within a segment
(built here because ``postprocess.compute_intervals`` drops the last stop).
Centroid = coordinate-wise median of the visit's fixes. Infomap is unseeded.

Throughput (GPS): ~570k events/s on 54k events, ~200k events/s on 890k events
(super-linear: one global BallTree + Infomap over all stationary events).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from infostop import Infostop

from dwells import schema
from dwells.algorithms.common import dwells_from_labels, gap_segments

NAME = "infostop"
FAMILY = "library"
USES_UNCERTAINTY = False
REFERENCE = "Infostop 0.1.9 (infomap 1.0.6); Aslak & Alessandretti 2020, arXiv:2003.14370"


def detect(events: pl.DataFrame, *, min_duration_s: float = 300, radius_m: float = 100,
           max_gap_s: float = 1800, **_) -> pl.DataFrame:
    seg = gap_segments(events, max_gap_s).to_numpy()
    xyt = np.column_stack([events["lat"].to_numpy(), events["lon"].to_numpy(),
                           events["timestamp"].dt.epoch("us").to_numpy() / 1e6])
    bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    model = Infostop(r1=radius_m, r2=radius_m, min_staying_time=min_duration_s, max_time_between=max_gap_s)
    try:
        place = np.concatenate(model.fit_predict([xyt[a:b] for a, b in zip(bounds[:-1], bounds[1:])]))
    except Exception as e:  # raised when no trace has a stationary event
        if "No stop events found" not in str(e):
            raise
        return schema.empty(schema.DWELLS)
    new_run = np.r_[True, (place[1:] != place[:-1]) | (seg[1:] != seg[:-1])]
    labels = np.where(place >= 0, np.cumsum(new_run), -1)
    return dwells_from_labels(events, labels, min_duration_s, centroid="median")
