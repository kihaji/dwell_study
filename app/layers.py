"""deck.gl layers for the selected identifier."""

from __future__ import annotations

import polars as pl
from deckgl_dash.layers import ArcLayer, LineLayer, PolygonLayer, ScatterplotLayer, TextLayer, TileLayer

from data import hex_rgb, place_colors, place_label

# OpenStreetMap standard tiles; their usage policy requires visible attribution (shown under the map)
BASEMAP = TileLayer(id="basemap", data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    min_zoom=0, max_zoom=19, tile_size=256)


def build(t: dict[str, pl.DataFrame], extent: list[str], sel_place: int | None, sel_visit: int | None,
          day_range: tuple | None = None) -> list:
    places, visits, ev = t["places"], t["visits"], t["events"]
    if day_range:
        lo, hi = day_range
        ev = ev.filter(pl.col("timestamp").dt.date().is_between(lo, hi))
        visits = visits.filter(pl.col("start_ts").dt.date().is_between(lo, hi) | pl.col("end_ts").dt.date().is_between(lo, hi))
    colors = place_colors(places)
    layers = [BASEMAP]

    focus = visits.filter(pl.col("visit_id") == sel_visit) if sel_visit is not None else (
        visits.filter(pl.col("place_id") == sel_place) if sel_place is not None else visits)

    # raw events: members colored by place, others grey
    pts = [{"p": [r["lon"], r["lat"]], "s": r["sigma_m"], "visit_id": r["visit_id"], "place_id": r["place_id"],
            "c": hex_rgb(colors.get(r["place_id"], "#616161"), 200 if r["place_id"] is not None else 90),
            "time": str(r["timestamp"])[:19], "acc_m": round(r["sigma_m"], 1),
            "source": r.get("location_source") or "unknown"}
           for r in ev.iter_rows(named=True)]
    layers.append(ScatterplotLayer(id="events", data=pts, get_position="@@=p", get_radius=3, radius_min_pixels=2,
                                   radius_max_pixels=6, get_fill_color="@@=c", pickable=True))
    if "member" in extent:
        fv = set(focus["visit_id"].to_list())
        mem = [p for p in pts if p["visit_id"] in fv]
        cent = {r["visit_id"]: [r["lon"], r["lat"]] for r in focus.iter_rows(named=True)}
        layers.append(ScatterplotLayer(id="event-error", data=mem, get_position="@@=p", get_radius="@@=s",
                                       filled=False, stroked=True, get_line_color=[90, 90, 90, 120],
                                       line_width_min_pixels=1))
        layers.append(LineLayer(id="member-links", data=[{"s": p["p"], "t": cent[p["visit_id"]]} for p in mem],
                                get_source_position="@@=s", get_target_position="@@=t",
                                get_color=[60, 60, 60, 110], get_width=1, width_min_pixels=1))
    if "hull" in extent:
        rows = focus if sel_place is not None or sel_visit is not None else places
        layers.append(PolygonLayer(
            id="hulls", data=[{"poly": r["hull_ring"], "place_id": r["place_id"], "visit_id": r.get("visit_id"),
                               "c": hex_rgb(colors.get(r["place_id"], "#616161"), 50),
                               "l": hex_rgb(colors.get(r["place_id"], "#616161"), 220)} for r in rows.iter_rows(named=True)],
            get_polygon="@@=poly", get_fill_color="@@=c", get_line_color="@@=l", line_width_min_pixels=1.5,
            pickable=True))
    if "ellipse" in extent:
        rows = places if sel_visit is None else focus
        layers.append(PolygonLayer(
            id="ellipses", data=[{"poly": r["ellipse_ring"], "place_id": r["place_id"],
                                  "l": hex_rgb(colors.get(r["place_id"], "#616161"), 255)} for r in rows.iter_rows(named=True)],
            get_polygon="@@=poly", filled=False, stroked=True, get_line_color="@@=l", line_width_min_pixels=2,
            pickable=True))

    # place centers sized by time share, with labels
    pc = [{"p": [r["lon"], r["lat"]], "place_id": r["place_id"], "c": hex_rgb(colors[r["place_id"]], 230),
           "r": 6 + 30 * r["time_share"] ** 0.5, "name": place_label(r), "score": round(r["score"], 1),
           "visits": r["n_visits"]} for r in places.iter_rows(named=True)]
    layers.append(ScatterplotLayer(id="places", data=pc, get_position="@@=p", get_radius="@@=r", radius_units="pixels",
                                   get_fill_color="@@=c", stroked=True, get_line_color=[255, 255, 255, 255],
                                   line_width_min_pixels=2, pickable=True))
    top = [p for p in pc if p["name"].split(" · ")[0] in {f"P{i}" for i in range(1, 11)}]
    layers.append(TextLayer(id="place-labels", data=top, get_position="@@=p", get_text="@@=name", get_size=12,
                            get_color=[20, 20, 20, 255], get_pixel_offset=[0, -18], background=True))

    # next-place arcs from the selected place
    if sel_place is not None and t["next_places"].height:
        pos = {r["place_id"]: [r["lon"], r["lat"]] for r in places.iter_rows(named=True)}
        arcs = [{"s": pos[sel_place], "t": pos[r["to_place"]], "w": 2 + 14 * r["prob"],
                 "label": f"{r['prob']:.0%} (≥{r['prob_low']:.0%})", "to_place": r["to_place"], "place_id": r["to_place"]}
                for r in t["next_places"].filter(pl.col("from_place") == sel_place).iter_rows(named=True)
                if r["to_place"] in pos and sel_place in pos]
        layers.append(ArcLayer(id="next-arcs", data=arcs, get_source_position="@@=s", get_target_position="@@=t",
                               get_source_color=[30, 30, 30, 220], get_target_color=hex_rgb("#d62728", 230),
                               get_width="@@=w", pickable=True))
        layers.append(TextLayer(id="arc-labels", data=[{"p": a["t"], "label": a["label"]} for a in arcs],
                                get_position="@@=p", get_text="@@=label", get_size=12, get_color=[180, 0, 0, 255],
                                get_pixel_offset=[0, 16]))
    return layers
