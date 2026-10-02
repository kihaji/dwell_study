"""Reverse geocoding of place centers (pluggable).

Providers expose ``reverse(lat, lon) -> GeoResult``:

* OfflineOSMGeocoder (default): nearest OpenStreetMap address / named feature from a
  local index (parquet) built once per area, either downloaded from Overpass
  (``build_index_for_area``) or read from a local OSM XML extract
  (``build_index_from_xml``) for air-gapped use. Queries are fully offline.
* NominatimGeocoder (optional): online reverse geocoding via a Nominatim server
  (public server policy: <= 1 request/s, identify via User-Agent), cached on disk.

Results describe what is *near* the place center; with coarse data the true building
can differ, so ``distance_m`` (center to the matched feature) is always reported.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from dwells import geo
from dwells.generator.area import AREAS, DATA_DIR

ADDR_TAGS = ["name", "addr:housenumber", "addr:street", "addr:city", "addr:postcode",
             "amenity", "shop", "office", "leisure", "building", "tourism"]


@dataclass
class GeoResult:
    address: str | None
    name: str | None
    category: str | None
    distance_m: float | None
    source: str


def _category(r: dict) -> str | None:
    for k in ("amenity", "shop", "office", "leisure", "tourism"):
        if r.get(k):
            return f"{k}={r[k]}"
    return f"building={r['building']}" if r.get("building") else None


def _features_to_index(feats) -> pl.DataFrame:
    feats = feats[feats.geometry.notna()]
    proj = feats.to_crs(feats.estimate_utm_crs())
    cent = proj.geometry.centroid.to_crs(4326)
    cols = [c for c in ADDR_TAGS if c in feats.columns]
    recs = feats[cols].astype(object).where(feats[cols].notna(), None).to_dict("records")
    df = pl.DataFrame({
        "lat": cent.y.to_numpy(), "lon": cent.x.to_numpy(),
        "name": [r.get("name") for r in recs],
        "housenumber": [r.get("addr:housenumber") for r in recs],
        "street": [r.get("addr:street") for r in recs],
        "city": [r.get("addr:city") for r in recs],
        "postcode": [r.get("addr:postcode") for r in recs],
        "category": [_category(r) for r in recs],
    }, schema_overrides={c: pl.String for c in ("name", "housenumber", "street", "city", "postcode", "category")})
    return df.filter(pl.col("name").is_not_null() | pl.col("street").is_not_null() | pl.col("category").is_not_null())


def build_index_for_area(area: str) -> Path:
    """Download address/name features for a study area from Overpass (needs internet once)."""
    import osmnx as ox

    cfg = AREAS[area]
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA_DIR / "osm" / "_cache")
    tags = {"addr:housenumber": True, "name": True, "amenity": True, "shop": True, "office": True}
    feats = ox.features_from_point(cfg["center"], tags=tags, dist=cfg["dist_m"])
    out = DATA_DIR / "osm" / area / "geocode.parquet"
    _features_to_index(feats).write_parquet(out)
    return out


def build_index_from_xml(osm_xml: str | Path, out: str | Path) -> Path:
    """Build an index from a local .osm XML extract (fully offline)."""
    import osmnx as ox

    feats = ox.features_from_xml(osm_xml, tags={"addr:housenumber": True, "name": True, "amenity": True,
                                                "shop": True, "office": True})
    _features_to_index(feats).write_parquet(out)
    return Path(out)


class OfflineOSMGeocoder:
    def __init__(self, index_paths: list[str | Path], max_distance_m: float = 150.0, name_distance_m: float = 60.0):
        idx = pl.concat([pl.read_parquet(p) for p in index_paths if Path(p).exists()], how="diagonal_relaxed")
        self.df = idx
        self.lat0, self.lon0 = float(idx["lat"].mean()), float(idx["lon"].mean())
        x, y = geo.to_enu(idx["lat"].to_numpy(), idx["lon"].to_numpy(), self.lat0, self.lon0)
        self.tree = cKDTree(np.c_[x, y])
        self.rows = idx.to_dicts()
        self.max_d, self.name_d = max_distance_m, name_distance_m

    @classmethod
    def for_repo_areas(cls, **kw):
        return cls([DATA_DIR / "osm" / a / "geocode.parquet" for a in AREAS], **kw)

    def reverse(self, lat: float, lon: float) -> GeoResult:
        x, y = geo.to_enu(np.array([lat]), np.array([lon]), self.lat0, self.lon0)
        dist, ix = self.tree.query(np.c_[x, y][0], k=min(25, len(self.rows)), distance_upper_bound=self.max_d)
        cand = [(d, self.rows[i]) for d, i in zip(np.atleast_1d(dist), np.atleast_1d(ix)) if np.isfinite(d)]
        if not cand:
            return GeoResult(None, None, None, None, "osm-offline")
        addr = next(((d, r) for d, r in cand if r["street"]), None)
        named = next(((d, r) for d, r in cand if r["name"] and d <= self.name_d), None)
        best = named or addr or cand[0]
        address = None
        if addr:
            r = addr[1]
            address = " ".join(x for x in (r["housenumber"], r["street"]) if x) + (f", {r['city']}" if r["city"] else "")
        return GeoResult(address, named[1]["name"] if named else None, best[1]["category"], float(best[0]),
                         "osm-offline")


class NominatimGeocoder:
    def __init__(self, url: str = "https://nominatim.openstreetmap.org", user_agent: str = "dwells-research",
                 cache_path: str | Path = DATA_DIR / "osm" / "nominatim_cache.json", min_interval_s: float = 1.0):
        self.url, self.ua, self.min_dt = url.rstrip("/"), user_agent, min_interval_s
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self._last = 0.0

    def reverse(self, lat: float, lon: float) -> GeoResult:
        key = f"{lat:.5f},{lon:.5f}"
        if key not in self.cache:
            wait = self.min_dt - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            q = urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18})
            req = urllib.request.Request(f"{self.url}/reverse?{q}", headers={"User-Agent": self.ua})
            with urllib.request.urlopen(req, timeout=20) as r:
                j = json.load(r)
            self._last = time.monotonic()
            a = j.get("address", {})
            road = " ".join(x for x in (a.get("house_number"), a.get("road")) if x)
            city = a.get("city") or a.get("town") or a.get("village")
            res = GeoResult(", ".join(x for x in (road, city) if x) or j.get("display_name"), j.get("name") or None,
                            f"{j.get('category')}={j.get('type')}" if j.get("category") else None,
                            float(geo.haversine_m(lat, lon, float(j["lat"]), float(j["lon"]))) if "lat" in j else None,
                            "nominatim")
            self.cache[key] = asdict(res)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return GeoResult(**self.cache[key])
