"""ST-DBSCAN (Birant & Kut 2007) applied per id for stay detection.

Two events are neighbors when they are within ``radius_m`` in space AND within
``eps_t_s`` in time (default ``max_gap_s``). Core events have >= ``min_samples``
neighbors (including themselves); clusters grow through core events as in DBSCAN.
Clusters are then cut into time-contiguous runs (common.contiguous_runs) and runs
shorter than ``min_duration_s`` are dropped. Implemented in numba: neighbor search
uses the time ordering (forward window scan), so cost ~ O(n * events per eps_t).
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl

from dwells.algorithms.common import contiguous_runs, dwells_from_labels, prepare

NAME = "st_dbscan"
FAMILY = "density"
USES_UNCERTAINTY = False
REFERENCE = "Birant & Kut 2007, ST-DBSCAN, doi:10.1016/j.datak.2006.01.013"


@numba.njit(cache=True)
def _st_dbscan(starts, t, x, y, eps_s, eps_t, min_samples):
    n = len(t)
    labels = np.full(n, -1, np.int64)
    nxt = 0
    r2 = eps_s * eps_s
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        # neighbor counts via symmetric forward scan
        cnt = np.ones(b - a, np.int64)
        for i in range(a, b):
            j = i + 1
            while j < b and t[j] - t[i] <= eps_t:
                if (x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2 <= r2:
                    cnt[i - a] += 1
                    cnt[j - a] += 1
                j += 1
        core = cnt >= min_samples
        stack = np.empty(b - a, np.int64)
        for i in range(a, b):
            if labels[i] >= 0 or not core[i - a]:
                continue
            labels[i] = nxt
            top = 0
            stack[top] = i
            top += 1
            while top > 0:
                top -= 1
                p = stack[top]
                if not core[p - a]:
                    continue
                lo = p
                while lo > a and t[p] - t[lo - 1] <= eps_t:
                    lo -= 1
                q = lo
                while q < b and t[q] - t[p] <= eps_t:
                    if labels[q] < 0 and (x[q] - x[p]) ** 2 + (y[q] - y[p]) ** 2 <= r2:
                        labels[q] = nxt
                        stack[top] = q
                        top += 1
                    q += 1
            nxt += 1
    return labels


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, eps_t_s: float | None = None, min_samples: int = 3, **_) -> pl.DataFrame:
    a = prepare(events)
    lab = _st_dbscan(a.starts, a.t, a.x, a.y, radius_m, eps_t_s or max_gap_s, min_samples)
    runs = contiguous_runs(a.starts, a.t, lab, max_gap_s, min_samples)
    return dwells_from_labels(events, runs, min_duration_s)
