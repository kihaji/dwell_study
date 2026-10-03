"""Output table schemas of the pattern-of-life pipeline (see docs/outputs.md).

All times are UTC. Rings are closed lists of [lon, lat] pairs (GeoJSON order), ready
for deck.gl PolygonLayer / PathLayer. Hour profiles are indexed 0-23 (UTC hour),
weekday 0 = Monday, weekday_hour index = weekday * 24 + hour.
"""

from __future__ import annotations

import polars as pl

TS = pl.Datetime("us", "UTC")
RING = pl.List(pl.List(pl.Float64))

EVENTS = {
    "id": pl.String, "event_idx": pl.Int64, "timestamp": TS, "lat": pl.Float64, "lon": pl.Float64,
    "location_source": pl.String, "sigma_m": pl.Float64, "coarse": pl.Boolean, "visit_id": pl.Int64, "place_id": pl.Int64,
}

VISITS = {
    "id": pl.String, "visit_id": pl.Int64, "place_id": pl.Int64,
    "start_ts": TS, "end_ts": TS, "duration_s": pl.Int64,
    "first_event_ts": TS, "last_event_ts": TS,
    "arrive_after": TS, "arrive_by": TS, "depart_after": TS, "depart_by": TS,
    "n_events": pl.Int64, "n_informative": pl.Int64,
    "lat": pl.Float64, "lon": pl.Float64,
    "sigma_major": pl.Float64, "sigma_minor": pl.Float64, "azimuth": pl.Float64,
    "extent_m": pl.Float64, "weekday": pl.Int8, "start_hour": pl.Int8,
    "prev_place_id": pl.Int64, "next_place_id": pl.Int64,
    "confidence": pl.Float64, "anomalies": pl.List(pl.String),
    "ellipse_ring": RING, "hull_ring": RING,
}

PLACES = {
    "id": pl.String, "place_id": pl.Int64, "lat": pl.Float64, "lon": pl.Float64,
    "sigma_major": pl.Float64, "sigma_minor": pl.Float64, "azimuth": pl.Float64, "extent_m": pl.Float64,
    "n_visits": pl.Int64, "n_days": pl.Int64, "n_events": pl.Int64, "total_dwell_s": pl.Int64,
    "median_dwell_s": pl.Float64, "first_seen": TS, "last_seen": TS, "time_share": pl.Float64,
    "hour_presence_s": pl.List(pl.Float64), "hour_events": pl.List(pl.Int64),
    "weekday_visits": pl.List(pl.Int64), "weekday_hour_presence_s": pl.List(pl.Float64),
    "regularity": pl.Float64, "score": pl.Float64,
    "score_evidence": pl.Float64, "score_precision": pl.Float64, "score_regularity": pl.Float64,
    "rank_duration": pl.Int64, "rank_score": pl.Int64, "rank_visits": pl.Int64,
    "label": pl.String,
    "address": pl.String, "place_name": pl.String, "place_category": pl.String,
    "geocode_distance_m": pl.Float64, "geocode_source": pl.String,
    "ellipse_ring": RING, "hull_ring": RING,
}

TRANSITIONS = {
    "id": pl.String, "from_place": pl.Int64, "to_place": pl.Int64, "n": pl.Int64,
    "prob": pl.Float64, "prob_low": pl.Float64, "median_travel_s": pl.Float64,
    "typical_depart_hour": pl.Int8, "last_ts": TS,
}

NEXT_PLACES = {
    "id": pl.String, "from_place": pl.Int64, "rank": pl.Int64, "to_place": pl.Int64,
    "prob": pl.Float64, "prob_low": pl.Float64, "n_from": pl.Int64,
}

PRESENCE = {"id": pl.String, "date": pl.Date, "hour": pl.Int8, "place_id": pl.Int64, "seconds": pl.Float64}

ANOMALIES = {"id": pl.String, "visit_id": pl.Int64, "place_id": pl.Int64, "kind": pl.String,
             "detail": pl.String, "severity": pl.Float64}

QUALITY = {
    "id": pl.String, "shape": pl.Int8, "n_events": pl.Int64, "first_ts": TS, "last_ts": TS,
    "span_days": pl.Float64, "active_days": pl.Int64, "events_per_active_day": pl.Float64,
    "hour_coverage": pl.Float64, "gap_p50_s": pl.Float64, "gap_p95_s": pl.Float64, "gap_max_s": pl.Float64,
    "sigma_p50_m": pl.Float64, "sigma_p95_m": pl.Float64, "coarse_frac": pl.Float64,
    "duplicate_frac": pl.Float64, "stale_frac": pl.Float64, "speed_outlier_frac": pl.Float64,
    "visits": pl.Int64, "places": pl.Int64, "events_in_visits_frac": pl.Float64,
    "source_coverage": pl.Float64, "source_mismatch_frac": pl.Float64,
}

TABLES = {"events": EVENTS, "visits": VISITS, "places": PLACES, "transitions": TRANSITIONS,
          "next_places": NEXT_PLACES, "presence": PRESENCE, "anomalies": ANOMALIES, "quality": QUALITY}


def conform(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    """Select and cast to the spec (missing columns become typed nulls)."""
    return df.select([(pl.col(c) if c in df.columns else pl.lit(None)).cast(t).alias(c) for c, t in spec.items()])
