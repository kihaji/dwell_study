"""Change-point segmentation baseline with the ``ruptures`` library (PELT, L2 cost).

Per gap-free segment, ruptures.Pelt(model="l2") partitions the (east, north)
series into pieces minimizing sum of squared deviations + penalty * #pieces, with
penalty = ``penalty_factor * radius_m^2`` (a split must explain more than about
penalty_factor points' worth of radius-sized deviation). Pieces whose RMS spread is
<= radius_m and whose span is >= min_duration_s become dwells.

This is the homoscedastic counterpart of ua_segmentation: identical in spirit but
blind to per-event error, which isolates what the error weighting contributes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import ruptures as rpt

from dwells.algorithms.common import dwells_from_labels, gap_segments, prepare

NAME = "changepoint_pelt"
FAMILY = "probabilistic"
USES_UNCERTAINTY = False
REFERENCE = "ruptures (Truong, Oudre & Vayatis 2020, doi:10.1016/j.sigpro.2019.107299); PELT (Killick et al. 2012)"


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, penalty_factor: float = 5.0, jump: int = 1, min_events: int = 2,
           **_) -> pl.DataFrame:
    a = prepare(events)
    seg = gap_segments(events, max_gap_s).to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    labels = np.full(len(seg), -1, np.int64)
    nxt = 0
    pen = penalty_factor * radius_m**2
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < min_events:
            continue
        X = np.c_[a.x[s:e], a.y[s:e]]
        cps = rpt.Pelt(model="l2", min_size=2, jump=jump).fit(X).predict(pen=pen) if e - s > 3 else [e - s]
        prev = 0
        for cp in cps:
            idx = np.arange(s + prev, s + cp)
            rms = np.sqrt(((X[prev:cp] - X[prev:cp].mean(0)) ** 2).sum(1).mean())
            if len(idx) >= min_events and rms <= radius_m and a.t[idx[-1]] - a.t[idx[0]] >= min_duration_s:
                labels[idx] = nxt
                nxt += 1
            prev = cp
    return dwells_from_labels(events, labels, min_duration_s)
