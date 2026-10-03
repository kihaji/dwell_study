"""Group an id's dwells into recurring places (Mahalanobis merge).

Sequential information-weighted merge, per id: a dwell joins the place whose fused
center is within chi2_2(0.99) Mahalanobis distance given both uncertainties (the
dwell's 1-sigma ellipse when the detector reports one, plus within-place spread
(radius_m / 2)^2); otherwise it starts a new place. Dwells are processed most-precise
first. Chosen over DBSCAN, HDBSCAN, GMM and H3 grouping on held-out data (see
docs/tuning.md, "Places"); those alternatives live on the main branch.
Returns the dwells with a ``place_id`` column (unique across ids).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import chi2

from dwells import geo
from dwells.uncertainty import ellipse_to_cov

ONE_SIGMA_P = 1 - np.exp(-0.5)
GATE = chi2.ppf(0.99, 2)


def _dwell_cov(d: pl.DataFrame, radius_m: float):
    v = (radius_m / 2) ** 2
    # a missing centroid ellipse counts as 1-sigma radius_m / 2
    a = d["sigma_major"].fill_null(radius_m / 2).to_numpy()
    b = d["sigma_minor"].fill_null(radius_m / 2).to_numpy()
    az = d["azimuth"].fill_null(0.0).to_numpy()
    cee, cnn, cen = ellipse_to_cov(a, b, az, ONE_SIGMA_P)
    return cee + v, cnn + v, cen


def _mahalanobis_merge(x, y, cee, cnn, cen, gate):
    order = np.argsort(cee + cnn)
    labels = np.full(len(x), -1, np.int64)
    P = []  # per place: [Iee, Inn, Ien, bx, by]
    for i in order:
        best, bd = -1, np.inf
        for k, (Iee, Inn, Ien, bx, by) in enumerate(P):
            D = Iee * Inn - Ien**2
            pee, pnn, pen = Inn / D, Iee / D, -Ien / D
            mx, my = pee * bx + pen * by, pen * bx + pnn * by
            see, snn, sen = cee[i] + pee, cnn[i] + pnn, cen[i] + pen
            dx, dy = x[i] - mx, y[i] - my
            d2 = (snn * dx * dx - 2 * sen * dx * dy + see * dy * dy) / (see * snn - sen**2)
            if d2 < bd:
                best, bd = k, d2
        det = cee[i] * cnn[i] - cen[i] ** 2
        w = np.array([cnn[i], cee[i], -cen[i]]) / det
        upd = np.array([w[0], w[1], w[2], w[0] * x[i] + w[2] * y[i], w[2] * x[i] + w[1] * y[i]])
        if best >= 0 and bd <= gate:
            P[best] = [p + u for p, u in zip(P[best], upd)]
            labels[i] = best
        else:
            P.append(list(upd))
            labels[i] = len(P) - 1
    return labels


def assign(dwells: pl.DataFrame, radius_m: float = 150) -> pl.DataFrame:
    d = dwells.sort("id", "start_ts")
    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    ids = d["id"].to_numpy()
    cov = _dwell_cov(d, radius_m)
    place = np.full(d.height, -1, np.int64)
    off = 0
    bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)] if d.height else np.array([0])
    for s, e in zip(bounds[:-1], bounds[1:]):
        x, y = geo.to_enu(lat[s:e], lon[s:e], np.median(lat[s:e]), np.median(lon[s:e]))
        lab = _mahalanobis_merge(x, y, *(c[s:e] for c in cov), GATE) if e - s > 1 else np.zeros(1, np.int64)
        lab = np.unique(lab, return_inverse=True)[1]
        place[s:e] = lab + off
        off += lab.max() + 1
    return d.with_columns(place_id=pl.Series(place, dtype=pl.Int64))
