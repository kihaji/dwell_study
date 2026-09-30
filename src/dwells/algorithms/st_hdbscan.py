"""ST-HDBSCAN: HDBSCAN clustering with temporal run splitting, per id.

By default HDBSCAN (Campello et al. 2013; scikit-learn) clusters positions only
with excess-of-mass selection ('leaf' over-fragmented stays). cluster_selection_epsilon
is not used: scikit-learn 1.8's epsilon_search raises TypeError under numpy 2.5.
Time enters by cutting each cluster into time-contiguous
runs (common.contiguous_runs) and dropping runs shorter than ``min_duration_s``.
Passing ``time_scale_s`` adds t * radius_m / time_scale_s as a third feature
(classic ST-HDBSCAN); in testing that elongated stays along the time axis and
merged them with approach/departure points, so it is off by default. HDBSCAN
needs no global eps, so it adapts to varying density within one id. Because its
density is relative, runs are trimmed to events within ``radius_m`` of the run's
median center (common.trim_runs), and runs that lose most events are rejected.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN

from dwells.algorithms.common import contiguous_runs, trim_runs, dwells_from_labels, prepare

NAME = "st_hdbscan"
FAMILY = "density"
USES_UNCERTAINTY = False
REFERENCE = "Campello, Moulavi & Sander 2013 (HDBSCAN), doi:10.1007/978-3-642-37456-2_14; sklearn.cluster.HDBSCAN"


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, time_scale_s: float | None = None, min_cluster_size: int = 3,
           selection: str = "eom",
           **_) -> pl.DataFrame:
    a = prepare(events)
    ts = time_scale_s or max_gap_s / 2  # EXPERIMENT: classic space-time features on by default
    labels = np.full(len(a.t), -1, np.int64)
    off = 0
    for k in range(len(a.starts) - 1):
        s, e = a.starts[k], a.starts[k + 1]
        if e - s < min_cluster_size:
            continue
        X = np.c_[a.x[s:e], a.y[s:e]]
        if ts:
            X = np.c_[X, (a.t[s:e] - a.t[s]) * radius_m / ts]
        lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size,
                      cluster_selection_method=selection, allow_single_cluster=True,
                      copy=True).fit_predict(X)
        labels[s:e] = np.where(lab >= 0, lab + off, -1)
        off += max(lab.max() + 1, 0)
    runs = contiguous_runs(a.starts, a.t, labels, max_gap_s, min_cluster_size)
    # HDBSCAN density is relative: it attaches approach points to stays and calls evenly
    # sampled motion a cluster, so enforce an absolute stay extent
    runs = trim_runs(a.x, a.y, runs, radius_m)
    return dwells_from_labels(events, runs, min_duration_s)
