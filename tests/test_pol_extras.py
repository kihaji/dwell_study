import numpy as np
import polars as pl
import pytest

from dwells.geo import from_enu
from dwells.pol import coloc, geocode, geometry


def test_offline_geocoder_nearest_address_and_name(tmp_path):
    lat, lon = from_enu(np.array([0.0, 30.0, 400.0]), np.array([0.0, 0.0, 0.0]), 40.0, -75.0)
    idx = pl.DataFrame({"lat": lat, "lon": lon, "name": ["Cafe Uno", None, "Far Away"],
                        "housenumber": [None, "12", None], "street": [None, "Main St", None],
                        "city": [None, "Springfield", None], "postcode": [None] * 3,
                        "category": ["amenity=cafe", "building=house", "shop=x"]})
    p = tmp_path / "g.parquet"
    idx.write_parquet(p)
    g = geocode.OfflineOSMGeocoder([p])
    r = g.reverse(40.0, -75.0)
    assert r.name == "Cafe Uno" and r.address == "12 Main St, Springfield"
    assert r.distance_m == pytest.approx(0, abs=1)
    far = g.reverse(*from_enu(np.array([5000.0]), np.array([0.0]), 40.0, -75.0))
    assert far.address is None and far.name is None


def test_hull_ring_degenerate_points_become_small_circle():
    ring = geometry.hull_ring(np.array([40.0, 40.0]), np.array([-75.0, -75.0]))
    assert ring[0] == ring[-1] and len(ring) >= 8


def _visits(rows):
    """rows: (id, visit_id, place_id, start_h, end_h, x_m)."""
    t0 = 1_700_000_000
    ids, vid, pid, s, e, x = zip(*rows)
    lat, lon = from_enu(np.array(x, float), np.zeros(len(x)), 40.0, -75.0)
    return pl.DataFrame({"id": ids, "visit_id": vid, "place_id": pid, "lat": lat, "lon": lon,
                         "start_ts": [int((t0 + h * 3600) * 1e6) for h in s],
                         "end_ts": [int((t0 + h * 3600) * 1e6) for h in e],
                         "sigma_major": 5.0, "sigma_minor": 5.0, "azimuth": 0.0}).with_columns(
        pl.col("start_ts", "end_ts").cast(pl.Datetime("us", "UTC")))


def test_colocations_reference_visits():
    v = _visits([("a", 0, 10, 0, 2, 0), ("b", 1, 20, 1, 3, 20), ("c", 2, 30, 5, 6, 10), ("d", 3, 40, 0, 2, 900)])
    c = coloc.colocations(v, radius_m=100)
    same = c.filter(pl.col("kind") == "same_time")
    assert same.select("id_a", "id_b", "visit_id_a", "visit_id_b").rows() == [("a", "b", 0, 1)]
    assert same["overlap_s"][0] == 3600
    diff = c.filter(pl.col("kind") == "diff_time")
    assert {("a", "c"), ("b", "c")} <= set(diff.select("id_a", "id_b").rows())
    assert diff["visit_id_a"].null_count() == 0
