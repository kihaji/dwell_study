import numpy as np
import polars as pl
import pytest

from dwells.harness import perturb


@pytest.fixture
def master():
    n = 200
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "id": ["a"] * 100 + ["b"] * 100,
        "timestamp": np.r_[np.arange(100), np.arange(100)] * 60_000_000,
        "lat": 40 + rng.normal(0, 1e-4, n), "lon": -75 + rng.normal(0, 1e-4, n),
        "true_lat": np.full(n, 40.0), "true_lon": np.full(n, -75.0), "source": "gps",
        "sig_a": np.full(n, 10.0), "sig_b": np.full(n, 5.0), "az": np.zeros(n),
        "p_mixed": np.full(n, 0.95), "truth_dwell_id": pl.Series([0] * n, dtype=pl.Int64),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_error_scale(master):
    out = perturb.apply(master, "error_scale", 2.0, seed=0)
    assert out["sig_a"].to_list() == [20.0] * 200
    assert out["lat"].equals(master["lat"])


def test_duplicates_adds_rows_and_keeps_order(master):
    out = perturb.apply(master, "duplicates", 0.1, seed=0)
    assert out.height == pytest.approx(220, abs=12)
    assert out.sort("id", "timestamp").equals(out)


def test_stale_repeats_previous_fix(master):
    out = perturb.apply(master, "stale", 0.2, seed=0)
    assert out.height == master.height
    same = (out["lat"] == out["lat"].shift(1)) & (out["id"] == out["id"].shift(1))
    assert 25 <= same.sum() <= 55


def test_outliers_move_positions_but_keep_reported_error(master):
    out = perturb.apply(master, "outliers", 0.1, seed=0, dist_m=500)
    moved = (out["lat"] - master["lat"]).abs() + (out["lon"] - master["lon"]).abs() > 1e-3
    assert 10 <= moved.sum() <= 35
    assert out["sig_a"].equals(master["sig_a"])


def test_thin_and_truncate_and_jitter(master):
    assert perturb.apply(master, "thin", 0.5, seed=0).height == pytest.approx(100, abs=20)
    tr = perturb.apply(master, "truncate", 3, seed=0)
    assert (tr["lat"] * 1000 - (tr["lat"] * 1000).round()).abs().max() < 1e-6
    j = perturb.apply(master, "clock_jitter", 30, seed=0)
    assert j.height == master.height and j.sort("id", "timestamp").equals(j)


def test_unknown_perturbation(master):
    with pytest.raises(ValueError):
        perturb.apply(master, "nope", 1, seed=0)


def test_ip_confident_only_changes_ip_reports(master):
    m = master.with_columns(source=pl.Series(["ip", "gps"] * 100))
    out = perturb.apply(m, "ip_confident", 50.0, seed=0)
    assert out.filter(pl.col("source") == "ip")["sig_a"].unique().to_list() == [50.0]
    assert out.filter(pl.col("source") == "gps")["sig_a"].unique().to_list() == [10.0]
