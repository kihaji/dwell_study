import numpy as np
import polars as pl
import pytest

from dwells import schema
from dwells.downstream import colocation, nextloc, patterns, places, transitions
from dwells.geo import from_enu

H = 3600
T0 = 1_709_532_000  # 2024-03-04 06:00 UTC (Monday 00:00 local, utc-6)


def dw(rows):
    """rows: (id, start_h, end_h, x_m, y_m)."""
    ids, s, e, x, y = zip(*rows)
    lat, lon = from_enu(np.array(x, float), np.array(y, float), 40.0, -75.0)
    df = pl.DataFrame({"id": ids, "start_ts": [int((T0 + a * H) * 1e6) for a in s],
                       "end_ts": [int((T0 + b * H) * 1e6) for b in e], "lat": lat, "lon": lon,
                       "n_events": 5})
    return schema.validate_dwells(df)


# home (0,0) -> work (3000,0) -> home, 5 weekdays; one gym visit
WEEK = []
for d in range(5):
    b = d * 24
    WEEK += [("a", b + 0, b + 7.5, 5 * d, 0), ("a", b + 8, b + 17, 3000, 10 * d), ("a", b + 17.5, b + 24 - 0.01, 0, 5)]
WEEK += [("a", 4 * 24 + 18, 4 * 24 + 19, 1500, 1500)]
WEEK.sort(key=lambda r: r[1])


@pytest.mark.parametrize("method", places.METHODS)
def test_places_group_repeated_visits(method):
    d = dw([r for r in WEEK if r[3] != 1500])
    out, pl_tab = places.assign(d, method=method, radius_m=150)
    assert out["place_id"].null_count() == 0
    # home and work are two places
    assert out["place_id"].n_unique() == 2
    assert set(pl_tab.columns) >= set(schema.PLACES)


def test_places_uncertainty_merge_uses_sigma():
    d = dw([("a", 0, 1, 0, 0), ("a", 2, 3, 400, 0)]).with_columns(
        sigma_major=pl.Series([300.0, 300.0]), sigma_minor=pl.Series([300.0, 300.0]), azimuth=pl.Series([0.0, 0.0]))
    out, _ = places.assign(d, method="mahalanobis", radius_m=50)
    assert out["place_id"].n_unique() == 1  # consistent given 300 m uncertainty
    out, _ = places.assign(d.with_columns(sigma_major=pl.lit(5.0), sigma_minor=pl.lit(5.0)), method="mahalanobis", radius_m=50)
    assert out["place_id"].n_unique() == 2


def test_patterns_home_work():
    d, _ = places.assign(dw(WEEK), method="dbscan", radius_m=150)
    prof = patterns.place_profiles(d, utc_offset_h=-6)
    labels = patterns.label_home_work(prof)
    home = labels.filter(pl.col("label") == "home")
    work = labels.filter(pl.col("label") == "work")
    assert home.height == 1 and work.height == 1
    hp = prof.filter(pl.col("place_id") == home["place_id"][0])
    assert hp["n_visits"][0] >= 5
    assert len(hp["hour_hist"][0]) == 24


def test_transitions_and_matrix():
    d, _ = places.assign(dw(WEEK), method="dbscan", radius_m=150)
    tr = transitions.from_dwells(d)
    assert set(tr.columns) >= set(schema.TRANSITIONS)
    m = transitions.matrix(tr)
    top = m.sort("count", descending=True).row(0, named=True)
    assert top["count"] >= 4 and top["from_place"] != top["to_place"]


def test_nextloc_markov_beats_nothing():
    seq = pl.DataFrame({"id": ["a"] * 40, "place_id": [0, 1, 0, 2] * 10,
                        "start_ts": [int((T0 + i * H) * 1e6) for i in range(40)]}).with_columns(
        pl.col("start_ts").cast(schema.TS))
    for model in nextloc.MODELS:
        res = nextloc.evaluate(seq, model=model, test_frac=0.3, top_n=(1, 3))
        assert 0 <= res["acc@1"] <= 1
    # after 1 -> always 0; after 2 -> always 0; after 0 -> 1 or 2 alternately (order 2 resolves)
    assert nextloc.evaluate(seq, model="markov2", test_frac=0.3)["acc@1"] == pytest.approx(1.0)
    assert nextloc.evaluate(seq, model="markov1", test_frac=0.3)["acc@3"] == pytest.approx(1.0)
    pred = nextloc.fit(seq.head(28), "markov1").predict_top(seq.head(28), n=2)
    assert set(pred.columns) >= set(schema.PREDICTIONS)


def test_colocation_same_time_distance_and_decoy():
    d = dw([("a", 0, 2, 0, 0), ("b", 1, 3, 20, 0),     # together
            ("c", 0, 2, 400, 0),                        # neighbor building 400 m away
            ("d", 5, 6, 0, 0)])                         # same place later
    for method in colocation.METHODS:
        c = colocation.same_time(d, method=method, radius_m=100, min_overlap_s=900)
        pairs = set(map(tuple, c.select("id_a", "id_b").rows()))
        assert pairs == {("a", "b")}, method
    dt = colocation.diff_time(d, radius_m=100, window_s=86400)
    assert ("a", "d") in set(map(tuple, dt.select("id_a", "id_b").rows()))
