"""Pattern-of-life output tables: schemas and cross-table invariants."""

import numpy as np
import polars as pl
import pytest

from dwells import schema
from dwells.geo import from_enu
from dwells.pol import pipeline
from dwells.pol import schema as S

T0 = 1_709_510_400  # 2024-03-04 00:00 UTC, a Monday
H = 3600


def _routine(uid, home, work, gym, seed):
    """14 days: home overnight, work 08-17 on weekdays, gym 18-19 Mon/Wed/Fri;
    one GPS fix every 5 min (8 m jitter), every 1 min while traveling."""
    rng = np.random.default_rng(seed)
    plan = []  # (start_h, end_h, (x, y)) stays in hours since T0
    for d in range(14):
        b = d * 24
        if d % 7 < 5:
            plan += [(b, b + 7.5, home), (b + 8, b + 17, work)]
            if d % 7 in (0, 2, 4):
                plan += [(b + 17.5, b + 19, gym), (b + 19.5, b + 24, home)]
            else:
                plan += [(b + 17.5, b + 24, home)]
        else:
            plan += [(b, b + 24, home)]
    t, x, y = [], [], []
    for k, (s0, s1, (px, py)) in enumerate(plan):
        ts = np.arange(s0 * H, s1 * H, 300.0)
        t += list(ts); x += list(px + rng.normal(0, 8, len(ts))); y += list(py + rng.normal(0, 8, len(ts)))
        if k + 1 < len(plan) and plan[k + 1][2] != (px, py):  # straight-line trip to the next stay
            nx, ny = plan[k + 1][2]
            tt = np.arange(s1 * H, plan[k + 1][0] * H, 60.0)
            f = (tt - s1 * H) / max((plan[k + 1][0] - s1) * H, 1)
            t += list(tt); x += list(px + (nx - px) * f); y += list(py + (ny - py) * f)
    lat, lon = from_enu(np.array(x), np.array(y), 40.0, -75.0)
    return pl.DataFrame({"id": uid, "lat": lat, "lon": lon, "timestamp": ((T0 + np.array(t)) * 1e6).astype(np.int64),
                         "horizontalAccuracy": 9.4})


@pytest.fixture(scope="module")
def result():
    ev = pl.concat([_routine("a", (0, 0), (3000, 500), (1200, -900), 1),
                    _routine("b", (-2000, 1500), (3000, 500), (-500, 2500), 2),
                    _routine("c", (4000, -3000), (-1000, -1000), (3500, -2500), 3)])
    ev = schema.validate_events(ev.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))))
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
