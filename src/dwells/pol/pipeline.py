"""Pattern-of-life pipeline: events -> visits -> places -> patterns (+ quality, anomalies).

    from dwells.pol import pipeline
    tables = pipeline.analyze(events, profile="mixed")
    pipeline.write(tables, "results/pol/my_dataset", meta={...})

``events`` is a validated events frame (any input shape, see dwells.schema). Output
tables and columns are defined in dwells.pol.schema and documented in docs/outputs.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from dwells import geo
from dwells.algorithms import ua_mahalanobis as ua
from dwells.downstream import places as places_mod
from dwells.pol import geometry, stats
from dwells.pol import schema as S
from dwells.pol.quality import quality
from dwells.presets import DEFAULT_MIN_DURATION_S, params_for

PLACE_RADIUS_M = 150  # place-grouping scale (Mahalanobis merge; best ARI across sources)


def analyze(events: pl.DataFrame, profile: str = "mixed", params: dict | None = None,
            min_duration_s: float = DEFAULT_MIN_DURATION_S, place_radius_m: float = PLACE_RADIUS_M,
            geocoder=None) -> dict[str, pl.DataFrame]:
    params = params or params_for(profile, min_duration_s)
    det, labels, coarse, sigma = ua.detect_with_members(events, **params)

    # --- visits with global ids, grouped into places
    det = det.sort("id", "start_ts").with_columns(visit_id=pl.int_range(pl.len()).cast(pl.Int64))
    if det.height:
        grouped = places_mod.assign(det, radius_m=place_radius_m)
        det = det.join(grouped.select("visit_id", "place_id"), on="visit_id")
    else:
        det = det.with_columns(place_id=pl.lit(None, dtype=pl.Int64))
    lab2visit = dict(zip(det["dwell_label"].to_list(), det["visit_id"].to_list()))
    visit_of_event = np.array([lab2visit.get(int(lb), -1) if lb >= 0 else -1 for lb in labels], np.int64)

    # --- event assignment table
    src = pl.col("location_source") if "location_source" in events.columns else pl.lit(None, dtype=pl.String)
    ev = events.select("id", "timestamp", "lat", "lon", src.alias("location_source")).with_columns(
        event_idx=pl.int_range(pl.len()).over("id").cast(pl.Int64),
        sigma_m=pl.Series(sigma), coarse=pl.Series(coarse),
        visit_id=pl.Series(visit_of_event).replace(-1, None).cast(pl.Int64),
    ).join(det.select("visit_id", "place_id"), on="visit_id", how="left", maintain_order="left")

    visits = _visit_table(det, ev)
    presence = stats.hour_overlap(visits)
    active = ev.group_by("id").agg(pl.col("timestamp").dt.date().n_unique().alias("active_days"))
    places = stats.place_table(visits, ev, presence, active) if visits.height else S.conform(pl.DataFrame(), S.PLACES)
    places = _place_geometry(places, ev)
    if geocoder is not None and places.height:
        places = _geocode(places, geocoder)
    trans, nxt = stats.transitions(visits) if visits.height else (pl.DataFrame(), pl.DataFrame())
    anom = stats.anomalies(visits, places, active) if visits.height else pl.DataFrame()
    flags = anom.group_by("visit_id").agg(pl.col("kind").sort().alias("anomalies")) if anom.height else None
    if flags is not None:
        visits = visits.drop("anomalies").join(flags, on="visit_id", how="left", maintain_order="left").with_columns(
            pl.col("anomalies").fill_null(pl.lit([], dtype=pl.List(pl.String))))
    q = quality(events, ev, visits)
    out = {
        "events": ev, "visits": visits, "places": places, "transitions": trans, "next_places": nxt,
        "presence": presence.group_by("id", "date", "hour", "place_id").agg(pl.col("seconds").sum()),
        "anomalies": anom, "quality": q,
    }
    return {k: S.conform(v, S.TABLES[k]) for k, v in out.items()}


def _visit_table(det: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    if det.height == 0:
        return S.conform(pl.DataFrame(), S.VISITS)
    mem = ev.filter(pl.col("visit_id").is_not_null())
    agg = mem.group_by("visit_id").agg(
        pl.len().cast(pl.Int64).alias("n_events_member"), (~pl.col("coarse")).sum().cast(pl.Int64).alias("n_informative"),
        pl.col("lat").filter(~pl.col("coarse")).alias("_lats"), pl.col("lon").filter(~pl.col("coarse")).alias("_lons"))
    v = det.join(agg, on="visit_id", how="left").sort("id", "start_ts")
    ext, hulls, ellipses = [], [], []
    for r in v.iter_rows(named=True):
        la, lo = np.asarray(r["_lats"] or [r["lat"]]), np.asarray(r["_lons"] or [r["lon"]])
        ext.append(float(np.sqrt(np.mean(geo.haversine_m(la, lo, r["lat"], r["lon"]) ** 2))))
        hulls.append(geometry.hull_ring(la, lo))
        ellipses.append(geometry.ellipse_ring(r["lat"], r["lon"], r["sigma_major"] or 5.0, r["sigma_minor"] or 5.0,
                                              r["azimuth"] or 0.0))
    span = (pl.col("last_event_ts") - pl.col("first_event_ts")).dt.total_seconds()
    v = v.with_columns(
        extent_m=pl.Series(ext), hull_ring=pl.Series(hulls), ellipse_ring=pl.Series(ellipses),
        n_events=pl.col("n_events_member"),
        weekday=(pl.col("start_ts").dt.weekday() - 1).cast(pl.Int8),
        start_hour=pl.col("start_ts").dt.hour().cast(pl.Int8),
        prev_place_id=pl.col("place_id").shift(1).over("id"),
        next_place_id=pl.col("place_id").shift(-1).over("id"),
        coverage=(span / pl.col("duration_s").clip(1)).clip(0, 1),
    ).with_columns(
        # detection confidence of one visit (0-100): enough informative fixes, a precise
        # centroid, and observations covering most of the estimated stay
        confidence=100 * (0.5 * (1 - (-pl.col("n_informative") / 5).exp())
                          + 0.25 / (1 + pl.col("sigma_major").fill_null(50) / 50)
                          + 0.25 * pl.col("coverage")),
        anomalies=pl.lit([], dtype=pl.List(pl.String)),
    )
    return v.drop("_lats", "_lons")


def _place_geometry(places: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    if places.height == 0:
        return places
    pts = ev.filter(pl.col("place_id").is_not_null() & ~pl.col("coarse")).group_by("id", "place_id").agg("lat", "lon")
    pts = {(r["id"], r["place_id"]): (r["lat"], r["lon"]) for r in pts.iter_rows(named=True)}
    hull, ell = [], []
    for r in places.iter_rows(named=True):
        la, lo = pts.get((r["id"], r["place_id"]), ([r["lat"]], [r["lon"]]))
        hull.append(geometry.hull_ring(np.asarray(la), np.asarray(lo)))
        ell.append(geometry.ellipse_ring(r["lat"], r["lon"], max(r["sigma_major"], 1.0), max(r["sigma_minor"], 1.0),
                                         r["azimuth"]))
    return places.with_columns(hull_ring=pl.Series(hull), ellipse_ring=pl.Series(ell))


def _geocode(places: pl.DataFrame, geocoder) -> pl.DataFrame:
    res = [geocoder.reverse(r["lat"], r["lon"]) for r in places.select("lat", "lon").iter_rows(named=True)]
    return places.with_columns(
        address=pl.Series([g.address for g in res], dtype=pl.String),
        place_name=pl.Series([g.name for g in res], dtype=pl.String),
        place_category=pl.Series([g.category for g in res], dtype=pl.String),
        geocode_distance_m=pl.Series([g.distance_m for g in res], dtype=pl.Float64),
        geocode_source=pl.Series([g.source for g in res], dtype=pl.String),
    )


def write(tables: dict[str, pl.DataFrame], out_dir: str | Path, meta: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        v.write_parquet(out / f"{k}.parquet")
    (out / "meta.json").write_text(json.dumps((meta or {}) | {"written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                              indent=2, default=str))


def read(out_dir: str | Path) -> dict[str, pl.DataFrame]:
    out = Path(out_dir)
    return {p.stem: pl.read_parquet(p) for p in out.glob("*.parquet")}
