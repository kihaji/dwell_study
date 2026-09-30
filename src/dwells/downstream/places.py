"""Group an id's dwells into recurring places.

Methods (all per id; ``radius_m`` is the place scale):
* dbscan       DBSCAN(eps=radius_m, min_samples=1) on dwell centroids
               (port of new_ellipses place_follower.py, eps=150 m there).
* hdbscan      HDBSCAN(min_cluster_size=2); unclustered dwells become singleton places,
               then clusters whose centers are within radius_m are merged (HDBSCAN has
               no absolute scale and split one home by its 20 m jitter in testing;
               cluster_selection_epsilon would do this but crashes under numpy 2.5).
* gmm          Gaussian mixture, number of components chosen by BIC, spherical
               covariances regularized to (radius_m / 2)^2.
* h3           H3 cells at the finest resolution with edge >= radius_m; a place is a
               connected set of occupied neighboring cells (a single cell per place
               split stays that straddle a cell boundary in testing).
* mahalanobis  new: sequential information-weighted merge. A dwell joins the place
               whose fused center is within chi2_2(0.99) Mahalanobis distance given
               both uncertainties (dwell 1-sigma ellipse, when the detector reports
               one, plus within-place spread (radius_m / 2)^2); otherwise it starts a
               new place. Dwells are processed most-precise first.
Returns (dwells + place_id, places table in the PLACES schema). place_id is unique
across ids.
"""

from __future__ import annotations

import h3
import numpy as np
import polars as pl
from scipy.stats import chi2
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture

from dwells import geo, schema
from dwells.uncertainty import ellipse_to_cov

METHODS = ("dbscan", "hdbscan", "gmm", "h3", "mahalanobis")
ONE_SIGMA_P = 1 - np.exp(-0.5)


def _dwell_cov(d: pl.DataFrame, radius_m: float):
    v = (radius_m / 2) ** 2
    # a missing centroid ellipse counts as 1-sigma radius_m / 2 (as in colocation._cov)
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


def _cluster(method, x, y, d_id, radius_m, cov, lat, lon):
    n = len(x)
    if n == 1:
        return np.zeros(1, np.int64)
    X = np.c_[x, y]
    if method == "dbscan":
        return DBSCAN(eps=radius_m, min_samples=1).fit_predict(X)
    if method == "hdbscan":
        lab = HDBSCAN(min_cluster_size=2, allow_single_cluster=True, copy=True).fit_predict(X)
        noise = lab < 0
        lab[noise] = lab.max() + 1 + np.arange(noise.sum())
        cx = np.bincount(lab, x) / np.bincount(lab)
        cy = np.bincount(lab, y) / np.bincount(lab)
        return DBSCAN(eps=radius_m, min_samples=1).fit_predict(np.c_[cx, cy])[lab]
    if method == "gmm":
        best, best_bic = None, np.inf
        for k in range(1, min(n, 15) + 1):
            g = GaussianMixture(k, covariance_type="spherical", reg_covar=(radius_m / 2) ** 2,
                                random_state=0).fit(X)
            bic = g.bic(X)
            if bic < best_bic:
                best, best_bic = g, bic
        return best.predict(X)
    if method == "h3":
        from dwells.algorithms.h3_occupancy import resolution_for

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        r = resolution_for(radius_m)
        cells = [h3.latlng_to_cell(a, b, r) for a, b in zip(lat, lon)]
        uniq, inv = np.unique(cells, return_inverse=True)
        pos = {c: i for i, c in enumerate(uniq)}
        edges = [(i, pos[nb]) for i, c in enumerate(uniq) for nb in h3.grid_disk(c, 1) if nb in pos]
        u, v = zip(*edges)
        _, comp = connected_components(coo_matrix((np.ones(len(u)), (u, v)), shape=(len(uniq),) * 2))
        return comp[inv]
    if method == "mahalanobis":
        return _mahalanobis_merge(x, y, *cov, chi2.ppf(0.99, 2))
    raise ValueError(method)


def assign(dwells: pl.DataFrame, method: str = "dbscan", radius_m: float = 150) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = dwells.sort("id", "start_ts")
    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    ids = d["id"].to_numpy()
    cov = _dwell_cov(d, radius_m)
    place = np.full(d.height, -1, np.int64)
    off = 0
    bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)] if d.height else np.array([0])
    for s, e in zip(bounds[:-1], bounds[1:]):
        x, y = geo.to_enu(lat[s:e], lon[s:e], np.median(lat[s:e]), np.median(lon[s:e]))
        lab = _cluster(method, x, y, ids[s], radius_m, tuple(c[s:e] for c in cov), lat[s:e], lon[s:e])
        lab = np.unique(lab, return_inverse=True)[1]
        place[s:e] = lab + off
        off += lab.max() + 1
    d = d.with_columns(place_id=pl.Series(place, dtype=pl.Int64))
    return d, place_table(d)


def place_table(d: pl.DataFrame) -> pl.DataFrame:
    g = d.group_by("id", "place_id").agg(
        pl.col("lat").mean(), pl.col("lon").mean(),
        pl.col("lat").alias("_lats"), pl.col("lon").alias("_lons"),
        pl.len().alias("n_visits"), pl.col("duration_s").sum().alias("total_dwell_s"),
        pl.col("start_ts").min().alias("first_seen"), pl.col("end_ts").max().alias("last_seen"),
    )
    rad = [float(np.max(geo.haversine_m(np.array(la), np.array(lo), c1, c2))) if len(la) else 0.0
           for la, lo, c1, c2 in zip(g["_lats"].to_list(), g["_lons"].to_list(), g["lat"], g["lon"])]
    g = g.with_columns(radius_m=pl.Series(rad)).drop("_lats", "_lons")
    return g.select([pl.col(c).cast(t) for c, t in schema.PLACES.items()]).sort("id", "place_id")
