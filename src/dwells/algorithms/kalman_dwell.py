"""Kalman/RTS-smoothed dwell detection (port of new_ellipses cotravel_cointegration:
series.py ``kalman_series_for`` + place_follower.py ``_dwells``).

1. Per id, split on gaps > ``max_gap_s``. Within each segment run a constant-velocity
   Kalman filter + RTS smoother whose measurement noise is each event's own error
   covariance, and output the smoothed position on a regular ``dt_s`` grid.
2. Greedy dwell search on the grid: starting at each valid step, extend while the
   next smoothed position is within ``radius_m`` of the running mean; accept runs of
   at least ``min_duration_s / dt_s`` steps (original defaults 150 m / 600 s / 60 s).
Uncertainty enters through the smoother (precise fixes pull the track, coarse ones
barely move it). Dwell start/end are grid times; the centroid is the mean smoothed
position.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from dwells import geo, schema
from dwells.algorithms.common import prepare

NAME = "kalman_dwell"
FAMILY = "sequential"
USES_UNCERTAINTY = True
REFERENCE = "Port of new_ellipses (cotravel-research) series.py _kalman_rts + place_follower.py _dwells"


@njit(cache=True)
def _kalman_rts(ev_t, ev_obs, z_e, z_n, r_ee, r_en, r_nn, q):
    """Constant-velocity KF + RTS smoother (verbatim port). ev_obs[i] >= 0 is an observation
    index, -1 a grid-output-only step. Returns smoothed state (m, 4)."""
    m = len(ev_t)
    xp = np.zeros((m, 4)); Pp = np.zeros((m, 4, 4)); xf = np.zeros((m, 4)); Pf = np.zeros((m, 4, 4))  # noqa: E702
    F = np.eye(4); Q = np.zeros((4, 4))  # noqa: E702
    x = np.zeros(4); P = np.zeros((4, 4))  # noqa: E702
    first = ev_obs[0]
    x[0] = z_e[first]; x[1] = z_n[first]  # noqa: E702
    P[0, 0] = r_ee[first]; P[0, 1] = r_en[first]; P[1, 0] = r_en[first]; P[1, 1] = r_nn[first]  # noqa: E702
    P[2, 2] = 100.0; P[3, 3] = 100.0  # noqa: E702  velocity unknown: sd 10 m/s
    t_prev = ev_t[0]
    for i in range(m):
        dt = ev_t[i] - t_prev
        t_prev = ev_t[i]
        if i > 0:
            F[0, 2] = dt; F[1, 3] = dt  # noqa: E702
            dt2 = dt * dt; dt3 = dt2 * dt  # noqa: E702
            Q[0, 0] = q * dt3 / 3; Q[1, 1] = q * dt3 / 3; Q[0, 2] = q * dt2 / 2; Q[2, 0] = q * dt2 / 2  # noqa: E702
            Q[1, 3] = q * dt2 / 2; Q[3, 1] = q * dt2 / 2; Q[2, 2] = q * dt; Q[3, 3] = q * dt  # noqa: E702
            x = F @ x
            P = F @ P @ F.T + Q
        xp[i] = x; Pp[i] = P  # noqa: E702
        j = ev_obs[i]
        if j >= 0 and i > 0:
            s00 = P[0, 0] + r_ee[j]; s01 = P[0, 1] + r_en[j]; s11 = P[1, 1] + r_nn[j]  # noqa: E702
            det = s00 * s11 - s01 * s01
            i00 = s11 / det; i01 = -s01 / det; i11 = s00 / det  # noqa: E702
            ye = z_e[j] - x[0]; yn = z_n[j] - x[1]  # noqa: E702
            K = np.zeros((4, 2))
            for a in range(4):
                K[a, 0] = P[a, 0] * i00 + P[a, 1] * i01
                K[a, 1] = P[a, 0] * i01 + P[a, 1] * i11
            for a in range(4):
                x[a] += K[a, 0] * ye + K[a, 1] * yn
            KH = np.zeros((4, 4))
            for a in range(4):
                KH[a, 0] = K[a, 0]; KH[a, 1] = K[a, 1]  # noqa: E702
            IKH = np.eye(4) - KH
            R = np.zeros((2, 2)); R[0, 0] = r_ee[j]; R[0, 1] = r_en[j]; R[1, 0] = r_en[j]; R[1, 1] = r_nn[j]  # noqa: E702
            P = IKH @ P @ IKH.T + K @ R @ K.T
        xf[i] = x; Pf[i] = P  # noqa: E702
    xs = np.zeros((m, 4))
    xs[m - 1] = xf[m - 1]
    for i in range(m - 2, -1, -1):
        dt = ev_t[i + 1] - ev_t[i]
        F[0, 2] = dt; F[1, 3] = dt  # noqa: E702
        C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
    return xs


@njit(cache=True)
def _dwells(e, n, min_len, radius):
    """Greedy dwell detection on a regular series (verbatim port, all steps valid)."""
    m = len(e)
    starts = np.zeros(m, dtype=np.int64); ends = np.zeros(m, dtype=np.int64)  # noqa: E702
    ce = np.zeros(m); cn = np.zeros(m)  # noqa: E702
    k = 0
    i = 0
    while i < m:
        se = e[i]; sn = n[i]; c = 1  # noqa: E702
        j = i + 1
        while j < m:
            me = se / c; mn = sn / c  # noqa: E702
            if (e[j] - me) ** 2 + (n[j] - mn) ** 2 > radius * radius:
                break
            se += e[j]; sn += n[j]; c += 1  # noqa: E702
            j += 1
        if j - i >= min_len:
            starts[k] = i; ends[k] = j; ce[k] = se / c; cn[k] = sn / c  # noqa: E702
            k += 1
            i = j
        else:
            i += 1
    return starts[:k], ends[:k], ce[:k], cn[:k]


def detect(events: pl.DataFrame, *, min_duration_s: float = 600, radius_m: float = 150,
           max_gap_s: float = 900, dt_s: float = 60.0, accel_sigma: float = 0.2,
           default_sigma_m: float = 10.0, **_) -> pl.DataFrame:
    a = prepare(events, default_sigma_m)
    lat, lon = events["lat"].to_numpy(), events["lon"].to_numpy()
    ids = events["id"].to_numpy()
    min_len = max(int(round(min_duration_s / dt_s)), 1)
    rows = []
    for k in range(len(a.starts) - 1):
        s, e = a.starts[k], a.starts[k + 1]
        t = a.t[s:e]
        lat0, lon0 = np.median(lat[s:e]), np.median(lon[s:e])
        brk = np.r_[0, np.flatnonzero(np.diff(t) > max_gap_s) + 1, len(t)]
        for p, q in zip(brk[:-1], brk[1:]):
            if q - p < 2:
                continue
            ga, gb = int(np.ceil(t[p] / dt_s)), int(np.floor(t[q - 1] / dt_s)) + 1
            grid_t = np.arange(ga, gb) * dt_s
            ev_t = np.concatenate([t[p:q], grid_t])
            ev_obs = np.concatenate([np.arange(s + p, s + q), np.full(len(grid_t), -1)]).astype(np.int64)
            order = np.lexsort((ev_obs < 0, ev_t))
            ev_t, ev_obs = ev_t[order], ev_obs[order]
            xs = _kalman_rts(ev_t, ev_obs, a.x, a.y, a.cee, a.cen, a.cnn, accel_sigma**2)
            g = xs[ev_obs < 0]
            st, en, ce, cn = _dwells(g[:, 0], g[:, 1], min_len, radius_m)
            for i0, i1, x, y in zip(st, en, ce, cn):
                t0, t1 = grid_t[i0], grid_t[i1 - 1]
                if t1 - t0 < min_duration_s - dt_s:
                    continue
                cnt = int(((t >= t0) & (t <= t1)).sum())
                la, lo = geo.from_enu(x, y, lat0, lon0)  # same per-id frame as prepare()
                rows.append((ids[s], t0, t1, la, lo, cnt))
    if not rows:
        return schema.empty(schema.DWELLS)
    df = pl.DataFrame(rows, schema=["id", "t0", "t1", "lat", "lon", "n_events"], orient="row")
    us = lambda c: (pl.col(c) * 1e6).cast(pl.Int64).cast(pl.Datetime("us", "UTC"))  # noqa: E731
    return schema.validate_dwells(df.with_columns(start_ts=us("t0"), end_ts=us("t1")).drop("t0", "t1"))

