import numpy as np
import polars as pl
import pytest

from dwells import io, schema
from dwells import uncertainty as U
from dwells.generator import sampling
from dwells.generator.area import synthetic_grid
from dwells.generator.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def area():
    return synthetic_grid(n=20, seed=1)


@pytest.fixture(scope="module")
def sim(area):
    return simulate(area, SimConfig(n_agents=12, n_days=7, seed=3, meet_rate_per_week=3.0))


def test_stays_are_ordered_and_linked(sim):
    for aid, st in sim.stays.group_by("id"):
        st = st.sort("arrive")
        a, d = st["arrive"].to_numpy(), st["depart"].to_numpy()
        assert np.all(a < d)
        assert np.all(d[:-1] <= a[1:])  # travel between stays
        # consecutive stays are at different places (merged otherwise)
        p = st["place_id"].to_numpy()
        assert np.all(p[1:] != p[:-1])


def test_household_shares_home(sim):
    ag = sim.agents
    multi = ag.group_by("household").agg(pl.col("home").n_unique(), pl.len()).filter(pl.col("len") > 1)
    assert multi.height > 0
    assert (multi["home"] == 1).all()


def test_meetings_create_joint_stays(sim):
    assert sim.meetings.height > 0
    m = sim.meetings.row(0, named=True)
    for aid in (m["id_a"], m["id_b"]):
        s = sim.stays.filter(
            (pl.col("id") == aid) & (pl.col("place_id") == m["place_id"])
            & (pl.col("arrive") < m["end"]) & (pl.col("depart") > m["start"])
        )
        assert s.height >= 1


def test_workers_have_work_stays(sim):
    workers = sim.agents.filter(pl.col("persona") == "worker")["id"].to_list()
    assert workers
    w = sim.stays.join(sim.agents.select("id", "work"), on="id").filter(
        pl.col("id").is_in(workers) & (pl.col("place_id") == pl.col("work"))
    )
    assert w.height >= 3 * len(workers)  # at least 3 of 5 weekdays


def test_deterministic(area):
    a = simulate(area, SimConfig(n_agents=4, n_days=2, seed=9))
    b = simulate(area, SimConfig(n_agents=4, n_days=2, seed=9))
    assert a.stays.equals(b.stays)


@pytest.mark.parametrize("profile", ["gps", "cdr", "adtech"])
def test_sampling_produces_valid_events(area, sim, profile):
    ev = sampling.sample(sim, area, profile, seed=0)
    assert ev.height > 0
    for shape in io.SHAPES:
        out = schema.validate_events(io.shape_view(ev, shape))
        assert schema.event_shape(out) == int(shape[0])


def test_gps_truth_dwell_positions_inside_place(area, sim):
    ev = sampling.sample(sim, area, "gps", seed=0)
    inside = ev.filter(pl.col("truth_dwell_id").is_not_null()).join(
        sim.stays.select("id", pl.col("dwell_id").alias("truth_dwell_id"), "place_id"),
        on=["id", "truth_dwell_id"],
    ).join(area.places.select("place_id", pl.col("lat").alias("plat"), pl.col("lon").alias("plon"), "radius_m"), on="place_id")
    from dwells.geo import haversine_m

    d = haversine_m(inside["true_lat"], inside["true_lon"], inside["plat"], inside["plon"])
    assert np.all(d <= inside["radius_m"].to_numpy() * 1.003 + 0.1)  # haversine vs ENU


def test_gps_error_coverage_calibrated(area, sim):
    ev = sampling.sample(sim, area, "gps", seed=1, calibration_noise=0.0, outlier_rate=0.0)
    ev = io.shape_view(ev, "3_p95", keep_truth=True)
    ev = U.with_covariance(ev)
    tl = ev.select("true_lat", "true_lon").to_numpy()
    from dwells.geo import to_enu

    x, y = to_enu(ev["lat"].to_numpy(), ev["lon"].to_numpy(), *area.center)
    tx, ty = to_enu(tl[:, 0], tl[:, 1], *area.center)
    d2 = U.mahalanobis2(x - tx, y - ty, ev["cov_ee"].to_numpy(), ev["cov_nn"].to_numpy(), ev["cov_en"].to_numpy())
    assert (d2 <= U.k_factor(0.95) ** 2).mean() == pytest.approx(0.95, abs=0.02)


def test_adtech_is_sparse_and_heterogeneous(area, sim):
    g = sampling.sample(sim, area, "gps", seed=0)
    a = sampling.sample(sim, area, "adtech", seed=0)
    assert a.height < g.height / 3
    per_id_spread = a.group_by("id").agg((pl.col("sig_a").max() / pl.col("sig_a").min()).alias("r"))
    assert per_id_spread["r"].median() > 20


def test_mixed_probability_values(area, sim):
    ev = io.shape_view(sampling.sample(sim, area, "adtech", seed=0), "3_mixed")
    assert set(ev["probability"].unique().to_list()) == {0.68, 0.95}
