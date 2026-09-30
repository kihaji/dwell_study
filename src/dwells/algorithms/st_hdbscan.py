"""ST-HDBSCAN: HDBSCAN clustering in space-time with temporal run splitting, per id.

HDBSCAN (Campello et al. 2013; scikit-learn) clusters the features
(x, y, t * radius_m / time_scale_s), time_scale_s = max_gap_s / 2 by default (classic
ST-HDBSCAN), with excess-of-mass selection ('leaf' over-fragmented stays) and
allow_single_cluster. time_scale_s=0 clusters positions only: better on CDR, worse on
GPS (branch exp/st-hdbscan-spacetime). cluster_selection_epsilon is not used because
scikit-learn 1.8's epsilon_search raises TypeError under numpy 2.5.

Clusters are cut into time-contiguous runs (common.contiguous_runs). HDBSCAN density
is relative (no absolute scale): it attaches approach points to stays and calls evenly
sampled motion a cluster, so runs are trimmed to events within radius_m of their
median center (common.trim_runs) and runs losing most events are rejected. Runs
shorter than min_duration_s are dropped.
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
    ts = max_gap_s / 2 if time_scale_s is None else time_scale_s  # 0 -> spatial only
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
