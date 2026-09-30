"""H3 cell dwell (port of ellipses/hexbin.py segment_sessions + compute_cell_dwell_times).

1. Sessions: per id, a new session starts when the gap to the previous event
   exceeds ``max_gap_s`` or the event is > ``session_radius_m`` from the session's
   running-mean centroid (original defaults 60 min / 500 m).
2. Each event is mapped to an H3 cell. The resolution is chosen once for the whole
   dataset, as in ``select_h3_resolution``: the finest resolution whose edge length is
   >= 1.16 x the average semi-major axis, so an average error ellipse fits in a cell.
   Without error fields the average semi-major is replaced by ``radius_m``.
3. A dwell is a contiguous run of events in the same cell within a session whose
   span (last - first) is >= ``min_duration_s``.
Uncertainty only sizes the grid; it does not affect membership. Cell-boundary
flicker splits a stay into several runs (a known weakness of the approach).
"""

from __future__ import annotations

import h3
import numpy as np
import polars as pl

from dwells.algorithms.common import dwells_from_labels, prepare
from dwells.algorithms.li_centroid import _kernel as _running_centroid
from dwells.uncertainty import CEP50_K

NAME = "h3_cell_dwell"
FAMILY = "grid"
USES_UNCERTAINTY = True
REFERENCE = "Port of ellipses/hexbin.py (segment_sessions, compute_cell_dwell_times, select_h3_resolution)"

EDGE_M = [h3.average_hexagon_edge_length(r, "m") for r in range(16)]


def select_resolution(avg_semi_major_m: float) -> int:
    need = 1.16 * avg_semi_major_m
    for r in range(15, -1, -1):
        if EDGE_M[r] >= need:
            return r
    return 0


def _avg_semi_major(events: pl.DataFrame, fallback: float) -> float:
    if "semiMajor" in events.columns:
        return float(events["semiMajor"].mean())
    if "horizontalAccuracy" in events.columns:
        # CEP50 -> 95% circle radius, comparable to a typical 95% semi-major
        return float(events["horizontalAccuracy"].mean() / CEP50_K * 2.4477)
    return fallback


def detect(events: pl.DataFrame, *, min_duration_s: float = 1800, radius_m: float = 100,
           max_gap_s: float = 3600, session_radius_m: float = 500, **_) -> pl.DataFrame:
    a = prepare(events)
    # sessions = running-centroid clusters with no size/duration filter
    sessions = _running_centroid(a.starts, a.t, a.x, a.y, session_radius_m, max_gap_s, 1, 0.0)
    res = select_resolution(_avg_semi_major(events, radius_m))
    cells = np.array([h3.str_to_int(h3.latlng_to_cell(la, lo, res))
                      for la, lo in zip(events["lat"].to_numpy(), events["lon"].to_numpy())], dtype=np.uint64)
    new_run = np.r_[True, (sessions[1:] != sessions[:-1]) | (cells[1:] != cells[:-1])]
    labels = np.cumsum(new_run) - 1
    return dwells_from_labels(events, labels, min_duration_s)
