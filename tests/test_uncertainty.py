import math

import numpy as np
import polars as pl
import pytest

from dwells import geo, uncertainty as U


def test_haversine_known_distance():
    # 1 degree of latitude ~ 111.195 km on the mean-radius sphere
    d = geo.haversine_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111_195, rel=1e-3)


def test_enu_roundtrip():
    lat0, lon0 = 32.84, -83.63
    lat = np.array([32.841, 32.85, 32.83])
    lon = np.array([-83.631, -83.62, -83.64])
    x, y = geo.to_enu(lat, lon, lat0, lon0)
    lat2, lon2 = geo.from_enu(x, y, lat0, lon0)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    # small offsets agree with haversine
    np.testing.assert_allclose(np.hypot(x, y), geo.haversine_m(lat0, lon0, lat, lon), rtol=2e-3)


def test_k_factor_values():
    # bivariate normal: P(inside k-sigma ellipse) = 1 - exp(-k^2/2)
    assert U.k_factor(0.95) == pytest.approx(2.4477, abs=1e-4)
    assert U.k_factor(0.68) == pytest.approx(1.5096, abs=1e-4)
    assert U.k_factor(1 - math.exp(-0.5)) == pytest.approx(1.0)


def test_cep50_is_k_at_half():
    assert U.k_factor(0.5) == pytest.approx(U.CEP50_K) == pytest.approx(1.1774, abs=1e-4)


def test_ellipse_cov_roundtrip():
    a, b, az, p = 120.0, 40.0, 30.0, 0.95
    cee, cnn, cen = U.ellipse_to_cov(a, b, az, p)
    a2, b2, az2 = U.cov_to_ellipse(cee, cnn, cen, p)
    assert (a2, b2, az2) == pytest.approx((a, b, az), rel=1e-9)


def test_ellipse_azimuth_is_clockwise_from_north():
    # azimuth 0 -> semi-major along north, azimuth 90 -> along east
    cee, cnn, _ = U.ellipse_to_cov(100, 10, 0, 0.95)
    assert cnn > cee
    cee, cnn, _ = U.ellipse_to_cov(100, 10, 90, 0.95)
    assert cee > cnn


@pytest.mark.parametrize("p", [0.5, 0.68, 0.95])
def test_monte_carlo_coverage(p):
    rng = np.random.default_rng(0)
    cee, cnn, cen = U.ellipse_to_cov(80.0, 25.0, 115.0, p)
    cov = np.array([[cee, cen], [cen, cnn]])
    pts = rng.multivariate_normal([0, 0], cov, size=200_000)
    d2 = U.mahalanobis2(pts[:, 0], pts[:, 1], cee, cnn, cen)
    inside = d2 <= U.k_factor(p) ** 2
    assert inside.mean() == pytest.approx(p, abs=0.005)


def _events(shape):
    base = {
        "id": ["a", "a"],
        "lat": [32.84, 32.8401],
        "lon": [-83.63, -83.63],
        "timestamp": [0, 60_000_000],
    }
    if shape == 2:
        base["horizontalAccuracy"] = [11.774, 23.548]
    if shape == 3:
        base |= {"semiMajor": [100.0, 50.0], "semiMinor": [50.0, 50.0], "azimuth": [0.0, 45.0],
                 "probability": [0.95, 0.68]}
    return pl.DataFrame(base).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_with_covariance_shape1_uses_default_sigma():
    out = U.with_covariance(_events(1), default_sigma_m=30.0)
    assert out["cov_ee"].to_list() == pytest.approx([900.0, 900.0])
    assert out["cov_en"].to_list() == pytest.approx([0.0, 0.0])


def test_with_covariance_shape2():
    out = U.with_covariance(_events(2))
    assert out["cov_ee"].to_list() == pytest.approx([100.0, 400.0], rel=1e-3)


def test_with_covariance_shape3_normalizes_probability():
    out = U.with_covariance(_events(3))
    s = 50.0 / U.k_factor(0.68)
    assert out["cov_ee"][1] == pytest.approx(s * s)
    s_major = 100.0 / U.k_factor(0.95)
    assert out["cov_nn"][0] == pytest.approx(s_major**2)
    assert "sigma_m" in out.columns


def test_with_covariance_shape2_accuracy_probability():
    # a 68% / 95% radius is converted with that probability's k-factor, not CEP50's
    ev = _events(2).with_columns(horizontalAccuracy=pl.Series([15.096, 24.477]),
                                 accuracyProbability=pl.Series([0.68, 0.95]))
    out = U.with_covariance(ev)
    assert out["sigma_m"].to_list() == pytest.approx([10.0, 10.0], rel=1e-3)
    assert out["cov_en"].to_list() == pytest.approx([0.0, 0.0])


def test_with_covariance_shape2_defaults_to_cep50():
    ev = _events(2)
    with_p = U.with_covariance(ev.with_columns(accuracyProbability=pl.lit(0.5)))
    assert with_p["cov_ee"].to_list() == pytest.approx(U.with_covariance(ev)["cov_ee"].to_list())
