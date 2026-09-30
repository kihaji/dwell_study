import io as _io
import zipfile
from datetime import datetime, timezone

import numpy as np
import polars as pl

from dwells import geo, io
from dwells.public import cluster_places, foursquare, geolife, label_stops, nominal_error, pol, staga, write_dataset

UTC = timezone.utc

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


def test_pol_localizes_and_labels():
    tr = pl.DataFrame({"tick": [0, 1, 2, 3], "agent_id": [5, 5, 5, 5],
                       "time": [datetime(2024, 1, 1, 0, m) for m in range(4)],
                       "latitude": [33.7] * 4, "longitude": [-84.4] * 4})
    gt = pl.DataFrame({"tick": [0], "agent_id": [5], "startTime": [datetime(2024, 1, 1, 0, 0)],
                       "endTime": [datetime(2024, 1, 1, 0, 2)], "latitude": [33.7], "longitude": [-84.4],
                       "loc_id_per_user": [3], "num_visits": [1]})
    ev, dw = pol.to_dataset(tr, gt)
    assert ev["id"].to_list() == ["5"] * 4
    assert ev["timestamp"][0] == _ts(2024, 1, 1, 5, 0)  # EST -> UTC
    assert ev["truth_dwell_id"].to_list() == [0, 0, 0, None]
    assert dw["place_id"].to_list() == [3] and dw["n_events"].to_list() == [3]
    assert dw["start_ts"][0] == _ts(2024, 1, 1, 5, 0) and dw["duration_s"][0] == 120


def _plt(rows):
    head = "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,My Track,0,0,2,8421376\n0\n"
    return (head + "".join(f"{la},{lo},0,492,39744.12,{d},{t}\n" for la, lo, d, t in rows)).encode()


def test_geolife_reads_zip(tmp_path):
    z = tmp_path / "g.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("Geolife Trajectories 1.3/Data/000/Trajectory/a.plt",
                   _plt([(39.9, 116.3, "2008-10-23", "02:53:04"), (39.91, 116.31, "2008-10-23", "02:53:10")]))
        f.writestr("Geolife Trajectories 1.3/Data/007/Trajectory/b.plt",
                   _plt([(40.0, 116.0, "2009-01-01", "00:00:00")]))
        f.writestr("Geolife Trajectories 1.3/Data/007/labels.txt", "Start Time\tEnd Time\tTransportation Mode\n")
    ev = geolife.read_zip(z)
    assert ev.height == 3
    assert ev["id"].to_list() == ["000", "000", "007"]
    assert ev["timestamp"][0] == _ts(2008, 10, 23, 2, 53, 4)
    assert ev["lat"][1] == 39.91


FSQ = (
    "470\t49bbd6c0f964a520f4531fe3\t4bf58dd8d48988d127951735\tArts & Crafts Store\t40.7198\t-74.0025\t-240\t"
    "Tue Apr 03 18:00:09 +0000 2012\n"
    "470\t4c5cc7b485a1e21e00d35711\t4bf58dd8d48988d103941735\tHome (private)\t40.7161\t-73.8830\t-240\t"
    "Wed Apr 04 01:02:24 +0000 2012\n"
    "69\t4c5cc7b485a1e21e00d35711\t4bf58dd8d48988d103941735\tCaf\xe9\t40.7161\t-73.8830\t-300\t"
    "Sat Nov 10 12:00:00 +0000 2012\n"
).encode("latin-1")


def test_foursquare_parse_and_error():
    raw = foursquare.parse(FSQ)
    assert raw["timestamp"][0] == _ts(2012, 4, 3, 18, 0, 9)
    assert raw["category"][2] == "Caf\xe9"
    ev, visits = foursquare.to_dataset(raw, seed=1)
    assert ev["true_lat"].to_list() == [40.7198, 40.7161, 40.7161]
    d = geo.haversine_m(ev["lat"], ev["lon"], ev["true_lat"], ev["true_lon"])
    assert (d > 0).all()
    assert (ev["sig_a"] >= ev["sig_b"]).all() and (ev["sig_b"] > 0).all()
    assert set(ev["source"].to_list()) <= {"gps", "wifi", "cell"}
    ev2, _ = foursquare.to_dataset(raw, seed=1)
    assert ev.equals(ev2)
    assert visits.columns == ["id", "venue_id", "timestamp", "lat", "lon", "category"]
    assert visits["lat"].to_list() == [40.7198, 40.7161, 40.7161]


def test_write_dataset_loads_all_shapes(tmp_path):
    ev, dw = staga.to_dataset(staga.parse_gps(STAGA_GPS), staga.parse_diary(STAGA_DIARY))
    meta = write_dataset(tmp_path / "s", ev, {"source_url": "x"}, dwells=dw)
    assert meta["stats"]["events"] == 5
    assert (tmp_path / "s" / "truth_dwells.parquet").exists()
    for shape in io.SHAPES:
        out = io.load_events(tmp_path / "s", shape)
        assert out.height == 5
    assert io.load_meta(tmp_path / "s")["source_url"] == "x"
    # master frame keeps only the canonical columns
    assert pl.read_parquet(tmp_path / "s" / "events.parquet").columns == [
        "id", "timestamp", "lat", "lon", "true_lat", "true_lon", "source", "sig_a", "sig_b", "az",
        "p_mixed", "truth_dwell_id"]

