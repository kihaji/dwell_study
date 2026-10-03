"""Positional-uncertainty normalization.

Every event's error is converted to a 2x2 covariance (m^2) in local east/north
coordinates, assuming a bivariate normal error model:

* Dataset 2 ``horizontalAccuracy`` is a circular radius at probability p, given per event
  by the optional ``accuracyProbability`` column (absent = 0.5, i.e. CEP50):
  sigma = radius / k(p), so CEP50 / sqrt(2 ln 2).
* Dataset 3 ellipse at probability p: the ellipse is the k-sigma contour with
  k = sqrt(-2 ln(1 - p)) (chi-square, 2 dof), so sigma_axis = axis / k.
  ``azimuth`` is the semi-major axis bearing in degrees clockwise from north.
* Dataset 1 has no error; callers supply ``default_sigma_m``.

Covariance columns: ``cov_ee`` (east var), ``cov_nn`` (north var), ``cov_en``.
``sigma_m`` is the RMS 1-sigma radius sqrt((cov_ee + cov_nn) / 2), a scalar summary.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

CEP50_K = math.sqrt(2 * math.log(2))  # 1.17741


def k_factor(p):
    return np.sqrt(-2.0 * np.log1p(-np.asarray(p, dtype=float)))


def ellipse_to_cov(a, b, azimuth_deg, p):
    k = k_factor(p)
    sa, sb = np.asarray(a) / k, np.asarray(b) / k
    th = np.radians(azimuth_deg)
    # unit vector of the major axis in (east, north)
    ue, un = np.sin(th), np.cos(th)
    cee = sa**2 * ue**2 + sb**2 * un**2
    cnn = sa**2 * un**2 + sb**2 * ue**2
    cen = (sa**2 - sb**2) * ue * un
    return cee, cnn, cen


def cov_to_ellipse(cee, cnn, cen, p):
    """Inverse of ellipse_to_cov: returns (semi_major, semi_minor, azimuth in [0, 180))."""
    cee, cnn, cen = (np.asarray(v, dtype=float) for v in (cee, cnn, cen))
    tr, det = cee + cnn, cee * cnn - cen**2
    disc = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0.0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0.0)
    k = k_factor(p)
    # major-axis eigenvector direction; bearing = atan2(east, north)
    az = np.degrees(0.5 * np.arctan2(2 * cen, cnn - cee)) % 180.0
    return np.sqrt(l1) * k, np.sqrt(l2) * k, az


def mahalanobis2(dx, dy, cee, cnn, cen):
    det = cee * cnn - cen**2
    return (cnn * dx**2 - 2 * cen * dx * dy + cee * dy**2) / det


def with_covariance(events: pl.DataFrame, default_sigma_m: float = 30.0,
                    source_sigma: dict[str, float] | None = None) -> pl.DataFrame:
    """Append cov_ee, cov_nn, cov_en, sigma_m columns derived from whatever error fields exist.

    Without accuracy fields every event gets ``default_sigma_m``, or, when the events carry
    ``location_source`` and ``source_sigma`` maps that label, the label's assumed 1-sigma.
    Stated accuracy always wins over the label."""
    cols = set(events.columns)
    n = events.height
    if {"semiMajor", "semiMinor", "azimuth", "probability"} <= cols:
        cee, cnn, cen = ellipse_to_cov(
            events["semiMajor"].to_numpy(),
            events["semiMinor"].to_numpy(),
            events["azimuth"].to_numpy(),
            events["probability"].to_numpy(),
        )
    elif "horizontalAccuracy" in cols:
        r = events["horizontalAccuracy"].to_numpy()
        p = events["accuracyProbability"].to_numpy() if "accuracyProbability" in cols else 0.5
        s2 = (r / k_factor(p)) ** 2
        cee, cnn, cen = s2, s2.copy(), np.zeros(n)
    else:
        sig = np.full(n, float(default_sigma_m))
        if source_sigma and "location_source" in cols:
            lab = events["location_source"].to_numpy()
            for k, v in source_sigma.items():
                sig[lab == k] = v
        cee = sig**2
        cnn, cen = cee.copy(), np.zeros(n)
    return events.with_columns(
        cov_ee=pl.Series(cee, dtype=pl.Float64),
        cov_nn=pl.Series(cnn, dtype=pl.Float64),
        cov_en=pl.Series(cen, dtype=pl.Float64),
        sigma_m=pl.Series(np.sqrt((cee + cnn) / 2), dtype=pl.Float64),
    )
