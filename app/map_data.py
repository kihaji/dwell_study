"""Map layer data for the selected identifier, shared by the Dash app (app/layers.py,
deckgl-dash) and the marimo notebook (deckgl-marimo). Each entry is a list of row dicts;
positions are [lon, lat], colors RGBA lists."""

from __future__ import annotations

import math

import polars as pl

from data import hex_rgb, place_colors, place_label

# OpenStreetMap standard tiles; their usage policy requires visible attribution
OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = "© OpenStreetMap contributors"


def rows(t: dict[str, pl.DataFrame], extent: list[str], sel_place: int | None, sel_visit: int | None,
         day_range: tuple | None = None) -> dict[str, list[dict]]:
    """``extent`` holds the chosen overlays: "ellipse", "hull", "member"."""
    places, visits, ev = t["places"], t["visits"], t["events"]
    if day_range:
        lo, hi = day_range
        ev = ev.filter(pl.col("timestamp").dt.date().is_between(lo, hi))
        visits = visits.filter(pl.col("start_ts").dt.date().is_between(lo, hi) | pl.col("end_ts").dt.date().is_between(lo, hi))
    colors = place_colors(places)
    out = {k: [] for k in ("events", "event_error", "member_links", "hulls", "ellipses", "places", "place_labels",
                           "arcs", "arc_labels")}

    focus = visits.filter(pl.col("visit_id") == sel_visit) if sel_visit is not None else (
        visits.filter(pl.col("place_id") == sel_place) if sel_place is not None else visits)

    # raw events: members colored by place, others grey
    out["events"] = [{"p": [r["lon"], r["lat"]], "s": r["sigma_m"], "visit_id": r["visit_id"], "place_id": r["place_id"],
                      "c": hex_rgb(colors.get(r["place_id"], "#616161"), 200 if r["place_id"] is not None else 90),
                      "time": str(r["timestamp"])[:19], "acc_m": round(r["sigma_m"], 1),
                      "source": r.get("location_source") or "unknown"}
                     for r in ev.iter_rows(named=True)]
    if "member" in extent:
        fv = set(focus["visit_id"].to_list())
        mem = [p for p in out["events"] if p["visit_id"] in fv]
        cent = {r["visit_id"]: [r["lon"], r["lat"]] for r in focus.iter_rows(named=True)}
        out["event_error"] = mem
        out["member_links"] = [{"s": p["p"], "t": cent[p["visit_id"]]} for p in mem]
    if "hull" in extent:
        hrows = focus if sel_place is not None or sel_visit is not None else places
        out["hulls"] = [{"poly": r["hull_ring"], "place_id": r["place_id"], "visit_id": r.get("visit_id"),
                         "c": hex_rgb(colors.get(r["place_id"], "#616161"), 50),
                         "l": hex_rgb(colors.get(r["place_id"], "#616161"), 220)} for r in hrows.iter_rows(named=True)]
    if "ellipse" in extent:
        erows = places if sel_visit is None else focus
        out["ellipses"] = [{"poly": r["ellipse_ring"], "place_id": r["place_id"],
                            "l": hex_rgb(colors.get(r["place_id"], "#616161"), 255)} for r in erows.iter_rows(named=True)]

    # place centers sized by time share, with labels for the top 10
    out["places"] = [{"p": [r["lon"], r["lat"]], "place_id": r["place_id"], "c": hex_rgb(colors[r["place_id"]], 230),
                      "r": 6 + 30 * r["time_share"] ** 0.5, "name": place_label(r), "score": round(r["score"], 1),
                      "visits": r["n_visits"]} for r in places.iter_rows(named=True)]
    out["place_labels"] = [p for p in out["places"] if p["name"].split(" · ")[0] in {f"P{i}" for i in range(1, 11)}]

    # next-place arcs from the selected place
    if sel_place is not None and t["next_places"].height:
        pos = {r["place_id"]: [r["lon"], r["lat"]] for r in places.iter_rows(named=True)}
        out["arcs"] = [{"s": pos[sel_place], "t": pos[r["to_place"]], "w": 2 + 14 * r["prob"],
                        "label": f"{r['prob']:.0%} (≥{r['prob_low']:.0%})", "to_place": r["to_place"],
                        "place_id": r["to_place"]}
                       for r in t["next_places"].filter(pl.col("from_place") == sel_place).iter_rows(named=True)
                       if r["to_place"] in pos and sel_place in pos]
        out["arc_labels"] = [{"p": a["t"], "label": a["label"]} for a in out["arcs"]]
    return out


def view(t: dict[str, pl.DataFrame], sel_place: int | None) -> dict:
    """Map center / zoom: the selected place, else all places."""
    p = t["places"]
    focus = p.filter(pl.col("place_id") == sel_place) if sel_place is not None else p
    if focus.height:
        return {"longitude": float(focus["lon"].mean()), "latitude": float(focus["lat"].mean()), "zoom": 14}
    return {"longitude": -83.63, "latitude": 32.84, "zoom": 12}


def bounds(t: dict[str, pl.DataFrame], sel_place: int | None, pad_m: float = 500) -> list[list[float]]:
    """[[west, south], [east, north]] around the selected place (else all places), padded
    by ``pad_m``; for map widgets that recenter by fitting bounds."""
    p = t["places"]
    focus = p.filter(pl.col("place_id") == sel_place) if sel_place is not None else p
    if focus.height == 0:
        v = view(t, None)
        focus = pl.DataFrame({"lon": [v["longitude"]], "lat": [v["latitude"]]})
    dlat = pad_m / 111_195.0  # metres per degree of latitude (mean Earth radius)
    dlon = dlat / max(math.cos(math.radians(float(focus["lat"].mean()))), 1e-6)
    return [[float(focus["lon"].min()) - dlon, float(focus["lat"].min()) - dlat],
            [float(focus["lon"].max()) + dlon, float(focus["lat"].max()) + dlat]]
