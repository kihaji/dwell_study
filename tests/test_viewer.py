"""Shared viewer logic (app/views.py, app/map_data.py) used by the Dash app and the
marimo notebook: rows, selection handling and map layer data."""

import sys
from pathlib import Path

import polars as pl
import pytest
from test_pol import _routine

from dwells import schema
from dwells.pol import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import map_data  # noqa: E402
import views  # noqa: E402


@pytest.fixture(scope="module")
def t():
    ev = pl.concat([_routine("a", (0, 0), (3000, 500), (1200, -900), 1)])
    ev = schema.validate_events(ev.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))))
    return pipeline.analyze(ev, profile="gps")


def test_default_selection_is_top_place(t):
    top = t["places"].sort("rank_duration")["place_id"][0]
    assert views.default_selection(t) == {"place_id": top, "visit_id": None}
    empty = {k: v.head(0) for k, v in t.items()}
    assert views.default_selection(empty) == {"place_id": None, "visit_id": None}


def test_selection_from_pick():
    assert views.selection_from_pick({"place_id": 3, "visit_id": 7}) == {"place_id": 3, "visit_id": 7}
    assert views.selection_from_pick({"place_id": 3}) == {"place_id": 3, "visit_id": None}
    assert views.selection_from_pick({"time": "x", "place_id": None}) is None  # an event outside any place
    assert views.selection_from_pick(None) is None


def test_rank_rows(t):
    for by in views.RANK_COLS:
        rows = views.rank_rows(t, by)
        assert 0 < len(rows) <= 10 and [r["#"] for r in rows] == list(range(1, len(rows) + 1))
    first = views.rank_rows(t, "total time")[0]
    assert first["place_id"] == views.default_selection(t)["place_id"]


def test_table_rows(t):
    pid = views.default_selection(t)["place_id"]
    assert len(views.places_rows(t)) == t["places"].height
    one = views.visits_rows(t, pid, all_places=False)
    assert len(one) == t["visits"].filter(pl.col("place_id") == pid).height
    assert len(views.visits_rows(t, pid, all_places=True)) == t["visits"].height
    assert len(views.next_rows(t, pid)) == t["next_places"].filter(pl.col("from_place") == pid).height
    assert all(r["severity"] >= 0.5 for r in views.anomaly_rows(t, 0.5))
    assert len(views.quality_items(t)) == 14


def test_place_summary(t):
    pid = views.default_selection(t)["place_id"]
    s = views.place_summary(t, pid)
    assert s["title"].startswith("P1") and "visits on" in s["stats"]
    assert views.place_summary(t, None) is None


def test_map_rows(t):
    pid = views.default_selection(t)["place_id"]
    r = map_data.rows(t, ["ellipse", "hull", "member"], pid, None)
    assert len(r["events"]) == t["events"].height
    assert len(r["places"]) == len(r["ellipses"]) == t["places"].height
    assert len(r["hulls"]) == t["visits"].filter(pl.col("place_id") == pid).height
    members = t["events"].filter(pl.col("place_id") == pid).height
    assert len(r["member_links"]) == len(r["event_error"]) == members
    assert {a["to_place"] for a in r["arcs"]} == set(t["next_places"].filter(pl.col("from_place") == pid)["to_place"])
    none = map_data.rows(t, [], None, None)
    assert none["hulls"] == none["ellipses"] == none["member_links"] == none["arcs"] == []


def test_selection_from_points():
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure([go.Bar(x=[1, 2], customdata=[[5, 50], [6, 60]]), go.Scatter(x=[0], customdata=[[7]])])
    assert views.selection_from_points([{"curveNumber": 0, "pointIndex": 1}], fig) == {"place_id": 6, "visit_id": 60}
    assert views.selection_from_points([{"curveNumber": 1, "pointIndex": 0}], fig) == {"place_id": 7, "visit_id": None}
    assert views.selection_from_points([{"customdata": [9, 90]}], fig) == {"place_id": 9, "visit_id": 90}
    assert views.selection_from_points([], fig) is None


def test_movement_graph(t):
    pytest.importorskip("plotly")
    import figures

    pid = views.default_selection(t)["place_id"]
    f = figures.movement_graph(t, pid)
    nodes = [tr for tr in f.data if tr.name == "places"][0]
    keep = set(t["places"].filter(pl.col("rank_duration") <= 15)["place_id"].to_list()) | {pid}
    assert {c[0] for c in nodes.customdata} == keep
    tr = t["transitions"].filter(pl.col("from_place").is_in(list(keep)) & pl.col("to_place").is_in(list(keep)))
    assert len([a for a in f.layout.annotations if a.showarrow]) == tr.height
    assert views.selection_from_points([{"curveNumber": list(f.data).index(nodes), "pointIndex": 0}], f)["place_id"] in keep


def test_map_bounds(t):
    from dwells.geo import haversine_m

    pid = views.default_selection(t)["place_id"]
    (w, s), (e, n) = map_data.bounds(t, pid, pad_m=500)
    r = t["places"].filter(pl.col("place_id") == pid).row(0, named=True)
    assert w < r["lon"] < e and s < r["lat"] < n
    assert haversine_m(s, r["lon"], n, r["lon"]) == pytest.approx(1000, rel=0.01)
    (w2, s2), (e2, n2) = map_data.bounds(t, None)
    assert w2 <= t["places"]["lon"].min() and e2 >= t["places"]["lon"].max()
