import numpy as np
import polars as pl
import pytest

from dwells.geo import from_enu
from dwells.pol import geocode, geometry


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
