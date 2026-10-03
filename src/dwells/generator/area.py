"""Study areas: a road graph for travel plus categorized places (POIs/buildings).

Two real OSM areas are defined (dense urban core, suburban) and cached as parquet
under ``data/osm/<name>/``. ``synthetic_grid`` builds a small offline area for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from dwells import geo

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

AREAS = {
    # Dense core: Chicago Loop / River North / South Loop. Small blocks, tall
    # mixed-use buildings, many nearby POIs -> hard case for separating dwells.
    "urban": {"center": (41.8830, -87.6310), "dist_m": 2500, "speed_kph": 18.0},
    # Suburban/small city: Macon, GA (continuity with the prior ellipses repos).
    "suburban": {"center": (32.8407, -83.6324), "dist_m": 7000, "speed_kph": 45.0},
}

CATEGORIES = ("home", "work", "food", "shop", "grocery", "gym", "worship", "park", "fuel", "school")


@dataclass
class Area:
    name: str
    center: tuple[float, float]
    node_lat: np.ndarray
    node_lon: np.ndarray
    graph: sparse.csr_matrix  # edge weights = travel seconds
    places: pl.DataFrame  # place_id, category, lat, lon, radius_m, node

    def paths_from(self, sources: np.ndarray):
        """Shortest travel-time trees from each source node: (dist[s, n], pred[s, n])."""
        return dijkstra(self.graph, directed=True, indices=sources, return_predecessors=True)

    def nearest_node(self, lat, lon) -> np.ndarray:
        from scipy.spatial import cKDTree

        x, y = geo.to_enu(self.node_lat, self.node_lon, *self.center)
        qx, qy = geo.to_enu(np.asarray(lat), np.asarray(lon), *self.center)
        return cKDTree(np.c_[x, y]).query(np.c_[qx, qy])[1]


def synthetic_grid(n: int = 20, spacing_m: float = 150.0, seed: int = 0,
                   center=(40.0, -75.0), speed_kph: float = 30.0) -> Area:
    """n x n street grid with places scattered near intersections (offline, for tests)."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = (jj.ravel() - n / 2) * spacing_m
    y = (ii.ravel() - n / 2) * spacing_m
    lat, lon = geo.from_enu(x, y, *center)
    idx = np.arange(n * n).reshape(n, n)
    u = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    v = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    t = spacing_m / (speed_kph / 3.6)
    g = sparse.coo_matrix((np.full(2 * len(u), t), (np.r_[u, v], np.r_[v, u])), shape=(n * n, n * n))
    cats = rng.choice(list(CATEGORIES), size=n * n * 2,
                      p=[0.45, 0.15, 0.1, 0.07, 0.05, 0.04, 0.03, 0.04, 0.03, 0.04])
    px = rng.uniform(x.min(), x.max(), len(cats))
    py = rng.uniform(y.min(), y.max(), len(cats))
    plat, plon = geo.from_enu(px, py, *center)
    area = Area("grid", center, lat, lon, g.tocsr(), pl.DataFrame())
    places = pl.DataFrame({
        "place_id": np.arange(len(cats)), "category": cats, "lat": plat, "lon": plon,
        "radius_m": rng.uniform(8, 40, len(cats)),
    })
    area.places = places.with_columns(node=pl.Series(area.nearest_node(plat, plon)))
    return area


# ---------------------------------------------------------------- OSM loading

def _categorize(tags: dict) -> str | None:
    b = tags.get("building")
    a = tags.get("amenity")
    s = tags.get("shop")
    le = tags.get("leisure")
    if a in ("restaurant", "cafe", "fast_food", "bar", "pub", "food_court"):
        return "food"
    if a == "fuel":
        return "fuel"
    if a == "place_of_worship":
        return "worship"
    if a in ("school", "kindergarten", "university", "college"):
        return "school"
    if s in ("supermarket", "convenience", "greengrocer", "grocery"):
        return "grocery"
    if s:
        return "shop"
    if le in ("fitness_centre", "sports_centre"):
        return "gym"
    if le == "park":
        return "park"
    if tags.get("office") or a in ("hospital", "clinic", "bank", "townhall", "courthouse", "police"):
        return "work"
    if b in ("house", "residential", "apartments", "detached", "semidetached_house",
             "terrace", "bungalow", "dormitory"):
        return "home"
    if b in ("commercial", "office", "industrial", "retail", "warehouse", "hospital",
             "government", "civic", "hotel"):
        return "work"
    if b == "yes":
        return "building_yes"
    return None


def download_area(name: str) -> None:
    import osmnx as ox

    cfg = AREAS[name]
    out = DATA_DIR / "osm" / name
    out.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA_DIR / "osm" / "_cache")
    G = ox.graph_from_point(cfg["center"], dist=cfg["dist_m"], network_type="drive")
    G = ox.truncate.largest_component(G, strongly=True)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()
    edges = edges.reset_index()
    pl.DataFrame({"osmid": nodes["osmid"].to_numpy("int64"), "lat": nodes["y"].to_numpy(float),
                  "lon": nodes["x"].to_numpy(float)}).write_parquet(out / "nodes.parquet")
    pl.DataFrame({"u": edges["u"].to_numpy("int64"), "v": edges["v"].to_numpy("int64"),
                  "length_m": edges["length"].to_numpy(float)}).write_parquet(out / "edges.parquet")

    tags = {"building": True, "amenity": True, "shop": True, "leisure": ["park", "fitness_centre", "sports_centre"], "office": True}
    feats = ox.features_from_point(cfg["center"], tags=tags, dist=cfg["dist_m"])
    feats = feats[feats.geometry.notna()]
    proj = feats.to_crs(feats.estimate_utm_crs())
    area_m2 = proj.geometry.area.to_numpy()
    cent = proj.geometry.centroid.to_crs(4326)
    cols = [c for c in ("building", "amenity", "shop", "leisure", "office") if c in feats.columns]
    recs = feats[cols].astype(object).where(feats[cols].notna(), None).to_dict("records")
    cats = [_categorize(r) for r in recs]
    df = pl.DataFrame({"category": cats, "lat": cent.y.to_numpy(), "lon": cent.x.to_numpy(),
                       "area_m2": area_m2}, strict=False).filter(pl.col("category").is_not_null())
    df.write_parquet(out / "features.parquet")


def load_area(name: str) -> Area:
    cfg = AREAS[name]
    d = DATA_DIR / "osm" / name
    if not (d / "features.parquet").exists():
        download_area(name)
    nodes = pl.read_parquet(d / "nodes.parquet")
    edges = pl.read_parquet(d / "edges.parquet")
    feats = pl.read_parquet(d / "features.parquet")

    index = {o: i for i, o in enumerate(nodes["osmid"].to_list())}
    u = np.array([index[x] for x in edges["u"].to_list()])
    v = np.array([index[x] for x in edges["v"].to_list()])
    t = edges["length_m"].to_numpy() / (cfg["speed_kph"] / 3.6)
    n = nodes.height
    g = _min_csr(u, v, t, n)

    area_m2 = feats["area_m2"].to_numpy()
    radius = np.where(area_m2 > 0, np.sqrt(area_m2 / np.pi), 15.0)
    # Unlabeled small buildings are treated as houses, large ones as workplaces.
    cats = feats["category"].to_numpy().astype(object)
    small = area_m2 < 400
    cats = np.where(cats == "building_yes", np.where(small, "home", "work"), cats)
    places = pl.DataFrame({
        "category": cats.astype(str),
        "lat": feats["lat"], "lon": feats["lon"],
        "radius_m": np.clip(radius, 8.0, 150.0),
    }).with_row_index("place_id").with_columns(pl.col("place_id").cast(pl.Int64))
    area = Area(name, cfg["center"], nodes["lat"].to_numpy(), nodes["lon"].to_numpy(), g, places)
    area.places = places.with_columns(node=pl.Series(area.nearest_node(places["lat"], places["lon"])))
    return area


def _min_csr(u, v, w, n) -> sparse.csr_matrix:
    """CSR adjacency keeping the minimum weight over parallel edges."""
    order = np.lexsort((w, v, u))
    u, v, w = u[order], v[order], w[order]
    keep = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
    return sparse.csr_matrix((w[keep], (u[keep], v[keep])), shape=(n, n))
