"""Spherical/planar helpers. All inputs are WGS84 decimal degrees; outputs meters."""

from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_enu(lat, lon, lat0, lon0):
    """Local equirectangular east/north meters around (lat0, lon0). Accurate to <0.1% within ~50 km."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y


def from_enu(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_R)
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon
