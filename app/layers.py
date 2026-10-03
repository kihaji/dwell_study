"""deck.gl layers for the selected identifier (deckgl-dash); data from map_data."""

from __future__ import annotations

import polars as pl
from deckgl_dash.layers import ArcLayer, LineLayer, PolygonLayer, ScatterplotLayer, TextLayer, TileLayer

import map_data

# attribution required by the OSM tile usage policy is shown under the map
BASEMAP = TileLayer(id="basemap", data=map_data.OSM_TILES, min_zoom=0, max_zoom=19, tile_size=256)


def build(t: dict[str, pl.DataFrame], extent: list[str], sel_place: int | None, sel_visit: int | None,
          day_range: tuple | None = None) -> list:
    d = map_data.rows(t, extent, sel_place, sel_visit, day_range)
    layers = [BASEMAP, ScatterplotLayer(id="events", data=d["events"], get_position="@@=p", get_radius=3,
                                        radius_min_pixels=2, radius_max_pixels=6, get_fill_color="@@=c", pickable=True)]
    if "member" in extent:
        layers.append(ScatterplotLayer(id="event-error", data=d["event_error"], get_position="@@=p", get_radius="@@=s",
                                       filled=False, stroked=True, get_line_color=[90, 90, 90, 120],
                                       line_width_min_pixels=1))
        layers.append(LineLayer(id="member-links", data=d["member_links"], get_source_position="@@=s",
                                get_target_position="@@=t", get_color=[60, 60, 60, 110], get_width=1, width_min_pixels=1))
    if "hull" in extent:
        layers.append(PolygonLayer(id="hulls", data=d["hulls"], get_polygon="@@=poly", get_fill_color="@@=c",
                                   get_line_color="@@=l", line_width_min_pixels=1.5, pickable=True))
    if "ellipse" in extent:
        layers.append(PolygonLayer(id="ellipses", data=d["ellipses"], get_polygon="@@=poly", filled=False, stroked=True,
                                   get_line_color="@@=l", line_width_min_pixels=2, pickable=True))
    layers.append(ScatterplotLayer(id="places", data=d["places"], get_position="@@=p", get_radius="@@=r",
                                   radius_units="pixels", get_fill_color="@@=c", stroked=True,
                                   get_line_color=[255, 255, 255, 255], line_width_min_pixels=2, pickable=True))
    layers.append(TextLayer(id="place-labels", data=d["place_labels"], get_position="@@=p", get_text="@@=name",
                            get_size=12, get_color=[20, 20, 20, 255], get_pixel_offset=[0, -18], background=True))
    if sel_place is not None and t["next_places"].height:
        layers.append(ArcLayer(id="next-arcs", data=d["arcs"], get_source_position="@@=s", get_target_position="@@=t",
                               get_source_color=[30, 30, 30, 220], get_target_color=[214, 39, 40, 230],
                               get_width="@@=w", pickable=True))
        layers.append(TextLayer(id="arc-labels", data=d["arc_labels"], get_position="@@=p", get_text="@@=label",
                                get_size=12, get_color=[180, 0, 0, 255], get_pixel_offset=[0, 16]))
    return layers
