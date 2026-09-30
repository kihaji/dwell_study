"""Sequential running-centroid stay detection (port of new_ellipses/dwell_detection.py).

Walk each id's events in time order. The open cluster is broken when the next
event is more than ``max_gap_s`` after the previous one or more than ``radius_m``
from the cluster's running *mean* centroid. A closed cluster is a dwell if it has
at least ``min_events`` events and (last - first) >= ``min_duration_s``.
Same family as ellipses/colocation.py ``_find_dwell_periods_numba`` (defaults
there: 60 min gap, 500 m radius). Li et al. (2008) differ by anchoring the radius
at the first point instead of the running centroid.
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl

from dwells.algorithms.common import dwells_from_labels, prepare

NAME = "li_centroid"
FAMILY = "sequential"
USES_UNCERTAINTY = False
REFERENCE = "Port of new_ellipses dwell_detection.py (DwellConfig); cf. Li et al. 2008, doi:10.1145/1463434.1463477"


@numba.njit(cache=True)
def _kernel(starts, t, x, y, radius, max_gap, min_events, min_duration):
    n = len(t)
    labels = np.full(n, -1, np.int64)
    nxt = 0
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        c0 = a
        sx, sy = x[a], y[a]
        for i in range(a + 1, b + 1):
            close = i == b
            if not close:
                cnt = i - c0
                d = np.hypot(x[i] - sx / cnt, y[i] - sy / cnt)
                close = (t[i] - t[i - 1] > max_gap) or (d > radius)
            if close:
                if i - c0 >= min_events and t[i - 1] - t[c0] >= min_duration:
                    labels[c0:i] = nxt
                    nxt += 1
                if i < b:
                    c0 = i
                    sx, sy = x[i], y[i]
            else:
                sx += x[i]
                sy += y[i]
    return labels


def detect(events: pl.DataFrame, *, min_duration_s: float = 300, radius_m: float = 100,
           max_gap_s: float = 1800, min_events: int = 3, **_) -> pl.DataFrame:
    a = prepare(events)
    labels = _kernel(a.starts, a.t, a.x, a.y, radius_m, max_gap_s, min_events, min_duration_s)
    return dwells_from_labels(events, labels, min_duration_s)
