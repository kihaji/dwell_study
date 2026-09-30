"""trackintel sliding-window staypoints (``generate_staypoints(method="sliding")``).

Each id is split on gaps > ``max_gap_s`` here and every segment is passed to
trackintel as its own ``user_id`` (``gap_threshold = max_gap_s`` too), so a gap
closes a stay instead of trackintel's native discard. Native semantics kept:
radius measured from the window's first point (``>=`` tests on distance and
time), ``include_last=True`` (the library option) so the open stay at the end of
a segment is emitted, end = first fix outside the stay (last fix for the trailing
stay), centroid = trackintel's circular mean of *distinct* member positions.
``n_events`` comes from the library's positionfix -> staypoint membership.
"""

from __future__ import annotations

import geopandas as gpd
import polars as pl
import trackintel as ti

from dwells import schema
from dwells.algorithms.common import gap_segments

NAME = "trackintel_sliding"
FAMILY = "library"
USES_UNCERTAINTY = False
REFERENCE = ("trackintel 1.4.2 generate_staypoints(method='sliding'); Martin et al. 2023, "
             "doi:10.1016/j.compenvurbsys.2023.101938; Li et al. 2008, doi:10.1145/1463434.1463477")


def detect(events: pl.DataFrame, *, min_duration_s: float = 300, radius_m: float = 100,
           max_gap_s: float = 1800, **_) -> pl.DataFrame:
    ev = events.select("id", "lat", "lon", "timestamp").with_columns(gap_segments(events, max_gap_s))
    pdf = ev.select(user_id="seg", tracked_at="timestamp", lat="lat", lon="lon").to_pandas()
    gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf.lon, pdf.lat), crs="EPSG:4326")
    pfs, sp = ti.Positionfixes(gdf).generate_staypoints(
        method="sliding", dist_threshold=radius_m, time_threshold=min_duration_s / 60,
        gap_threshold=max_gap_s / 60, include_last=True)
    if len(sp) == 0:
        return schema.empty(schema.DWELLS)
    counts = pfs.groupby("staypoint_id").size()
    out = pl.DataFrame({
        "seg": sp["user_id"].to_numpy(),
        "start_ts": sp["started_at"], "end_ts": sp["finished_at"],
        "lat": sp.geometry.y.to_numpy(), "lon": sp.geometry.x.to_numpy(),
        "n_events": counts.reindex(sp.index).to_numpy(),
    })
    seg_id = ev.select("seg", "id").unique("seg")
    return schema.validate_dwells(out.join(seg_id, on="seg").drop("seg"))
