"""The detector must exploit error information (coarse fixes, precision)."""

import numpy as np
import polars as pl
import pytest

from dwells import schema
from dwells.algorithms import ua_mahalanobis
from test_algorithms import PARAMS, T0, track


def _with_points(ev, rows):
    """Insert extra events: (minute, dx_m, dy_m, horizontalAccuracy_m)."""
    from dwells.geo import from_enu

    lat, lon = from_enu(np.array([r[1] for r in rows], float), np.array([r[2] for r in rows], float), 40.0, -75.0)
    extra = pl.DataFrame({
        "id": "a", "lat": lat, "lon": lon,
        "timestamp": [int((T0 + r[0] * 60) * 1e6) for r in rows],
        "horizontalAccuracy": [float(r[3]) for r in rows],
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    return schema.validate_events(pl.concat([ev, extra.select(ev.columns)]))


@pytest.fixture
def ua():
    return ua_mahalanobis


def test_coarse_point_does_not_break_dwell(ua):
    # 60-minute stay; at minute 30 a cell-tower fix 800 m away with 1.5 km accuracy
    ev = _with_points(track([("stay", 0, 0, 60)]), [(30.5, 800, 0, 1500)])
    d = ua.detect(ev, **PARAMS)
    assert d.height == 1
    assert d["duration_s"][0] >= 55 * 60


def test_coarse_point_does_not_extend_dwell(ua):
    # a coarse fix consistent with the stay but 3 h later must not stretch it
    ev = _with_points(track([("stay", 0, 0, 30)]), [(210, 300, 0, 5000)])
    d = ua.detect(ev, **dict(PARAMS, max_gap_s=6 * 3600))
    assert d.height == 1
    assert d["duration_s"][0] <= 35 * 60


def test_coarse_points_do_not_pull_centroid(ua):
    ev = _with_points(track([("stay", 0, 0, 40)]), [(10.5, 1200, 0, 3000), (20.5, 1200, 0, 3000)])
    d = ua.detect(ev, **PARAMS)
    from dwells.geo import haversine_m

    assert d.height == 1
    assert haversine_m(40.0, -75.0, d["lat"][0], d["lon"][0]) < 20


def test_uses_precision_when_centroid_reported(ua):
    d = ua.detect(track([("stay", 0, 0, 40)]), **PARAMS)
    if d["sigma_major"].is_null().all():
        pytest.skip("algorithm does not report centroid uncertainty")
    assert d["sigma_major"][0] < 100
