"""Map geometry: uncertainty ellipses and convex hulls as closed [lon, lat] rings."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from dwells import geo
from dwells.uncertainty import k_factor


def ellipse_ring(lat: float, lon: float, sigma_major: float, sigma_minor: float, azimuth: float,
                 p: float = 0.95, n: int = 48) -> list[list[float]]:
    """Ellipse containing the center with probability p (1-sigma axes in meters,
    azimuth = major-axis bearing clockwise from north)."""
    k = float(k_factor(p))
    a, b = sigma_major * k, sigma_minor * k
    th = np.radians(azimuth)
    t = np.linspace(0, 2 * np.pi, n)
    u, v = a * np.cos(t), b * np.sin(t)
    x = u * np.sin(th) + v * np.cos(th)
    y = u * np.cos(th) - v * np.sin(th)
    rlat, rlon = geo.from_enu(x, y, lat, lon)
    ring = np.c_[rlon, rlat].tolist()
    ring[-1] = ring[0]
    return ring


def hull_ring(lats: np.ndarray, lons: np.ndarray, min_m: float = 5.0) -> list[list[float]]:
    """Convex hull of points; degenerate inputs (1-2 points or collinear) become a small
    circle of radius min_m around their mean so the polygon is always drawable."""
    lat0, lon0 = float(np.mean(lats)), float(np.mean(lons))
    x, y = geo.to_enu(np.asarray(lats), np.asarray(lons), lat0, lon0)
    pts = np.unique(np.round(np.c_[x, y], 2), axis=0)
    if len(pts) >= 3:
        try:
            h = ConvexHull(pts)
            ring = pts[h.vertices]
            rlat, rlon = geo.from_enu(ring[:, 0], ring[:, 1], lat0, lon0)
            out = np.c_[rlon, rlat].tolist()
            return out + [out[0]]
        except QhullError:
            pass
    return ellipse_ring(lat0, lon0, min_m, min_m, 0.0, p=1 - np.exp(-0.5), n=16)
