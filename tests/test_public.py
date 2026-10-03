import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from dwells import io
from dwells.public import (
    cluster_places,
    label_stops,
    nominal_error,
    staga,
    write_dataset,
)

STAGA_GPS = """ts,x,y,longitude,latitude,motion_score,diary_label
2000-01-01 00:00:00+01:00,0,0,10.0,44.0,1.0,stop
2000-01-01 00:01:30+01:00,0,0,10.00001,44.00001,1.0,stop
2000-01-01 00:03:00+01:00,0,0,10.01,44.0,9.0,trip
2000-01-01 00:10:00+01:00,0,0,10.02,44.0,1.0,stop
2000-05-06 10:00:00+02:00,0,0,10.00002,44.0,1.0,stop
"""
STAGA_DIARY = """start_time,end_time
2000-01-01 00:00:00+01:00,2000-01-01 00:02:00+01:00
2000-01-01 00:09:59.500000+01:00,2000-01-01 00:20:00+01:00
2000-05-06 09:59:00+02:00,2000-05-06 10:30:00+02:00
2000-05-06 11:00:00+02:00,2000-05-06 11:30:00+02:00
"""


def _ts(*a):
    return datetime(*a, tzinfo=UTC)


def test_cluster_places_groups_within_radius():
    lat = np.array([44.0, 44.0002, 44.01, 44.0001])  # 0, ~22 m, ~1.1 km, ~11 m
    lon = np.array([10.0, 10.0, 10.0, 10.0])
    p = cluster_places(lat, lon, 50.0)
    assert p[0] == p[1] == p[3] != p[2]


def test_label_stops_inclusive_and_per_id():
    ev = pl.DataFrame({"id": ["a", "a", "a", "b"],
                       "timestamp": [_ts(2024, 1, 1, 0, 0), _ts(2024, 1, 1, 0, 5), _ts(2024, 1, 1, 0, 6),
                                     _ts(2024, 1, 1, 0, 1)]})
    stops = pl.DataFrame({"id": ["a"], "dwell_id": [7], "start_ts": [_ts(2024, 1, 1, 0, 0)],
                          "end_ts": [_ts(2024, 1, 1, 0, 5)]})
    out = label_stops(ev, stops)
    assert out["truth_dwell_id"].to_list() == [7, 7, None, None]
    assert out["truth_dwell_id"].dtype == pl.Int64


def test_nominal_error_columns():
    ev = pl.DataFrame({"id": ["a", "a"], "timestamp": [_ts(2024, 1, 1), _ts(2024, 1, 2)],
                       "lat": [1.0, 2.0], "lon": [3.0, 4.0]})
    out = nominal_error(ev, "x", seed=0)
    assert out["true_lat"].to_list() == [1.0, 2.0]
    assert (out["sig_a"] == 8.0).all() and (out["sig_b"] == 8.0).all()
    assert set(out["p_mixed"].unique().to_list()) <= {0.68, 0.95}
    assert (out["source"] == "x").all()


def test_staga_parse_and_truth():
    ev = staga.parse_gps(STAGA_GPS)
    assert ev["timestamp"][0] == _ts(1999, 12, 31, 23, 0)
    assert ev["timestamp"][4] == _ts(2000, 5, 6, 8, 0)  # +02:00 -> UTC
    stops = staga.parse_diary(STAGA_DIARY)
    assert stops["start_ts"][1] == datetime(1999, 12, 31, 23, 9, 59, 500000, tzinfo=UTC)
    ev, dw = staga.to_dataset(ev, stops)
    assert ev["truth_dwell_id"].to_list() == [0, 0, None, 1, 2]
    # labels agree with the per-point diary label
    assert ev.select((pl.col("truth_dwell_id").is_not_null() == (pl.col("diary_label") == "stop")).all()).item()
    assert dw["n_events"].to_list() == [2, 1, 1, 0]
    assert dw["duration_s"].to_list() == [120, 600, 1860, 1800]
    # stops 0 and 2 are ~1-2 m apart -> same place; stop 1 is ~1.6 km away; stop 3 has no events
    p = dw["place_id"].to_list()
    assert p[0] == p[2] != p[1] and p[3] is None
    assert dw["lat"][3] is None
    assert {"id", "dwell_id", "place_id", "category", "start_ts", "end_ts", "duration_s",
            "lat", "lon", "radius_m", "n_events"} <= set(dw.columns)


def test_write_dataset_loads_all_shapes(tmp_path):
    ev, dw = staga.to_dataset(staga.parse_gps(STAGA_GPS), staga.parse_diary(STAGA_DIARY))
    meta = write_dataset(tmp_path / "s", ev, {"source_url": "x"}, dwells=dw)
    assert meta["stats"]["events"] == 5
    assert (tmp_path / "s" / "truth_dwells.parquet").exists()
    for shape in io.SHAPES:
        out = io.load_events(tmp_path / "s", shape)
        assert out.height == 5
    assert json.loads((tmp_path / "s" / "meta.json").read_text())["source_url"] == "x"
    # master frame keeps only the canonical columns
    assert pl.read_parquet(tmp_path / "s" / "events.parquet").columns == [
        "id", "timestamp", "lat", "lon", "true_lat", "true_lon", "source", "sig_a", "sig_b", "az",
        "p_mixed", "truth_dwell_id"]

