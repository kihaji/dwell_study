"""Uncertainty-aware optimal segmentation (penalized change-point detection, new).

Each gap-free segment of an id's track is partitioned into pieces minimizing

    sum_pieces cost(piece) + penalty * (#pieces)

with the heteroscedastic Gaussian cost of a stationary piece
    cost = 1/2 * sum_i (z_i - mu)^T W_i (z_i - mu),   W_i = (C_i + v I)^-1,
where mu is the information-weighted mean of the piece, C_i the event's error
covariance and v = radius_m^2 / 4 the within-place spread (as in ua_mahalanobis).
Prefix sums of W, W z and z^T W z give each piece's cost in O(1), and PELT pruning
(Killick et al. 2012) keeps the search near-linear. The cost is a negative
log-likelihood, so this is maximum-a-posteriori segmentation under a
constant-hazard change-point prior (penalty = -log prior odds of a change).

Pieces become dwells when they have >= ``min_events`` informative events, span >=
``min_duration_s`` and fit the stationary model (reduced chi-square of the
residuals <= ``max_chi2_per_event``). Coarse events carry little weight so they
neither create change points nor pull the centroid; like ua_mahalanobis, they do
not define a dwell's start/end.
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl

from dwells.algorithms.common import dwells_from_labels, gap_segments, prepare

NAME = "ua_segmentation"
FAMILY = "uncertainty"
USES_UNCERTAINTY = True
REFERENCE = ("New (this repo). PELT: Killick, Fearnhead & Eckley 2012, doi:10.1080/01621459.2012.737745; "
             "Bayesian change point: Adams & MacKay 2007")


@numba.njit(cache=True)
def _cost(S, a, b):
    """Half chi-square of the best stationary fit to events [a, b) from prefix sums."""
    Iee = S[b, 0] - S[a, 0]
    Inn = S[b, 1] - S[a, 1]
    Ien = S[b, 2] - S[a, 2]
    bx = S[b, 3] - S[a, 3]
    by = S[b, 4] - S[a, 4]
    q = S[b, 5] - S[a, 5]
    D = Iee * Inn - Ien * Ien
    if D <= 0:
        return 0.0
    # b^T I^-1 b
    fit = (Inn * bx * bx - 2 * Ien * bx * by + Iee * by * by) / D
    return 0.5 * max(q - fit, 0.0)


@numba.njit(cache=True)
def _pelt(S, n, penalty):
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, np.int64)
    cand = np.empty(n + 1, np.int64)
    nc = 1
    cand[0] = 0
    for t in range(1, n + 1):
        best, arg = np.inf, 0
        for j in range(nc):
            s = cand[j]
            v = F[s] + _cost(S, s, t) + penalty
            if v < best:
                best, arg = v, s
        F[t] = best
        last[t] = arg
        # prune: s can never be optimal again if F[s] + cost(s, t) > F[t]
        k = 0
        for j in range(nc):
            s = cand[j]
            if F[s] + _cost(S, s, t) <= F[t]:
                cand[k] = s
                k += 1
        cand[k] = t
        nc = k + 1
    # backtrack change points
    cps = []
    t = n
    while t > 0:
        cps.append(t)
        t = last[t]
    return cps[::-1]


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, penalty: float = 10.0, max_chi2_per_event: float = 3.0,
           coarse_factor: float = 3.0, min_events: int = 2, default_sigma_m: float = 10.0,
           **_) -> pl.DataFrame:
    a = prepare(events, default_sigma_m)
    v = radius_m**2 / 4
    de, dn = a.cee + v, a.cnn + v
    dd = de * dn - a.cen**2
    wee, wnn, wen = dn / dd, de / dd, -a.cen / dd
    q = wee * a.x**2 + 2 * wen * a.x * a.y + wnn * a.y**2
    comp = np.c_[wee, wnn, wen, wee * a.x + wen * a.y, wen * a.x + wnn * a.y, q]
    informative = np.sqrt((a.cee + a.cnn) / 2) <= coarse_factor * radius_m
    seg = gap_segments(events, max_gap_s).to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    labels = np.full(len(seg), -1, np.int64)
    nxt = 0
    for s, e in zip(bounds[:-1], bounds[1:]):
        S = np.vstack([np.zeros((1, 6)), np.cumsum(comp[s:e], axis=0)])
        cps = _pelt(S, e - s, penalty)
        prev = 0
        for cp in cps:
            idx = np.arange(s + prev, s + cp)
            inf_idx = idx[informative[idx]]
            if len(inf_idx) >= min_events:
                chi = 2 * _cost(S, prev, cp) / max(len(idx) - 1, 1)
                t0, t1 = a.t[inf_idx[0]], a.t[inf_idx[-1]]
                if chi <= max_chi2_per_event and t1 - t0 >= min_duration_s:
                    # the dwell spans its informative events; coarse ones inside are attached
                    inside = idx[(a.t[idx] >= t0) & (a.t[idx] <= t1)]
                    labels[inside] = nxt
                    nxt += 1
            prev = cp
    return dwells_from_labels(events, labels, min_duration_s, centroid="fused",
                              default_sigma_m=default_sigma_m, extra_var=v)
