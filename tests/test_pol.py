"""Pattern-of-life output tables: schemas and cross-table invariants."""

import numpy as np
import polars as pl
import pytest

from dwells import io, schema
from dwells.generator import sampling
from dwells.generator.area import synthetic_grid
from dwells.generator.simulate import SimConfig, simulate
from dwells.pol import pipeline
from dwells.pol import schema as S


@pytest.fixture(scope="module")
def result():
    area = synthetic_grid(n=20, seed=1)
    sim = simulate(area, SimConfig(n_agents=4, n_days=14, seed=3, meet_rate_per_week=2.0))
    ev = schema.validate_events(io.shape_view(sampling.sample(sim, area, "gps", seed=0), "2"))
    return pipeline.analyze(ev, profile="gps", geocoder=None)


def test_all_tables_present_with_schema(result):
    for name, spec in S.TABLES.items():
        df = result[name]
        missing = set(spec) - set(df.columns)
        assert not missing, (name, missing)


def test_event_assignment_links(result):
    ev, vis, pl_ = result["events"], result["visits"], result["places"]
    member = ev.filter(pl.col("visit_id").is_not_null())
    assert member.height > 0
    assert set(member["visit_id"].unique()) <= set(vis["visit_id"])
    assert set(vis["place_id"].unique()) <= set(pl_["place_id"])
    # every visit's member events lie within its estimated span
    j = member.join(vis.select("visit_id", "start_ts", "end_ts"), on="visit_id")
    assert (j["timestamp"] >= j["start_ts"]).all() and (j["timestamp"] <= j["end_ts"]).all()


def test_visit_fields(result):
    v = result["visits"]
    assert (v["end_ts"] >= v["start_ts"]).all()
    assert (v["n_informative"] <= v["n_events"]).all()
    assert v["confidence"].is_between(0, 100).all()
    assert v["weekday"].is_between(0, 6).all() and v["start_hour"].is_between(0, 23).all()
    ring = v["ellipse_ring"][0].to_list()
    assert ring[0] == ring[-1] and len(ring) > 8


def test_place_profiles_and_scores(result):
    p = result["places"]
    assert p["score"].is_between(0, 100).all()
    for c in ("score_evidence", "score_precision", "score_regularity"):
        assert p[c].is_between(0, 1).all()
    assert all(len(h) == 24 for h in p["hour_presence_s"])
    assert all(len(h) == 7 for h in p["weekday_visits"])
    assert all(len(h) == 168 for h in p["weekday_hour_presence_s"])
    # rankings are 1..n within each id
    for c in ("rank_duration", "rank_score", "rank_visits"):
        for _, g in p.group_by("id"):
            assert sorted(g[c].to_list()) == list(range(1, g.height + 1))
    assert (p.group_by("id").agg(pl.col("time_share").sum())["time_share"] - 1).abs().max() < 1e-6


def test_home_label_found(result):
    homes = result["places"].filter(pl.col("label") == "primary")
    assert homes["id"].n_unique() == result["places"]["id"].n_unique()


def test_transitions_and_next_places(result):
    nxt = result["next_places"]
    sums = nxt.group_by("id", "from_place").agg(pl.col("prob").sum())
    assert (sums["prob"] <= 1 + 1e-9).all()
    assert (nxt["prob_low"] <= nxt["prob"] + 1e-12).all()
    tr = result["transitions"]
    assert (tr["from_place"] != tr["to_place"]).all()


def test_presence_bounded(result):
    pr = result["presence"]
    tot = pr.group_by("id", "date", "hour").agg(pl.col("seconds").sum())
    assert (tot["seconds"] <= 3600 + 1e-6).all()


def test_quality_per_id(result):
    q = result["quality"]
    assert q.height == result["events"]["id"].n_unique()
    assert q["coarse_frac"].is_between(0, 1).all()


def test_write_and_read_roundtrip(result, tmp_path):
    pipeline.write(result, tmp_path, meta={"dataset": "test"})
    back = pipeline.read(tmp_path)
    assert back["places"].height == result["places"].height
    assert (tmp_path / "meta.json").exists()


def test_wilson_lower_bound():
    from dwells.pol.stats import wilson_low

    assert wilson_low(np.array([5]), np.array([10]))[0] == pytest.approx(0.2366, abs=1e-3)
    assert wilson_low(np.array([0]), np.array([4]))[0] == 0.0
