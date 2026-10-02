"""Per-identifier input data-quality diagnostics (see docs/data_contract.md)."""

from __future__ import annotations

import numpy as np
import polars as pl

from dwells import geo, schema


def quality(events: pl.DataFrame, assigned: pl.DataFrame, visits: pl.DataFrame,
            speed_limit_ms: float = 60.0) -> pl.DataFrame:
    """``events`` = validated input; ``assigned`` = the events output table (sigma_m,
    coarse, visit_id, place_id) in the same row order."""
    e = assigned.with_columns(
        gap=pl.col("timestamp").diff().over("id").dt.total_seconds().cast(pl.Float64),
        same_pos=(pl.col("lat") == pl.col("lat").shift(1).over("id")) & (pl.col("lon") == pl.col("lon").shift(1).over("id")),
        same_t=pl.col("timestamp") == pl.col("timestamp").shift(1).over("id"),
    )
    lat, lon = e["lat"].to_numpy(), e["lon"].to_numpy()
    plat, plon = e["lat"].shift(1).fill_null(e["lat"]).to_numpy(), e["lon"].shift(1).fill_null(e["lon"]).to_numpy()
    dist = geo.haversine_m(lat, lon, plat, plon)
    sig = e["sigma_m"].to_numpy()
    psig = e["sigma_m"].shift(1).fill_null(e["sigma_m"]).to_numpy()
    gap = e["gap"].fill_null(np.inf).to_numpy()
    # an implausible jump: faster than speed_limit_ms and farther than both fixes' error allows
    jump = (dist / np.maximum(gap, 1.0) > speed_limit_ms) & (dist > 3 * (sig + psig)) & np.isfinite(gap)
    e = e.with_columns(jump=pl.Series(jump))
    q = e.group_by("id", maintain_order=True).agg(
        pl.len().alias("n_events"), pl.col("timestamp").min().alias("first_ts"), pl.col("timestamp").max().alias("last_ts"),
        pl.col("timestamp").dt.date().n_unique().alias("active_days"),
        pl.col("timestamp").dt.truncate("1h").n_unique().alias("_hours"),
        pl.col("gap").quantile(0.5).alias("gap_p50_s"), pl.col("gap").quantile(0.95).alias("gap_p95_s"),
        pl.col("gap").max().alias("gap_max_s"),
        pl.col("sigma_m").quantile(0.5).alias("sigma_p50_m"), pl.col("sigma_m").quantile(0.95).alias("sigma_p95_m"),
        pl.col("coarse").mean().alias("coarse_frac"),
        (pl.col("same_t") & pl.col("same_pos")).mean().alias("duplicate_frac"),
        (~pl.col("same_t") & pl.col("same_pos")).mean().alias("stale_frac"),
        pl.col("jump").mean().alias("speed_outlier_frac"),
        pl.col("visit_id").is_not_null().mean().alias("events_in_visits_frac"),
    )
    span_h = (q["last_ts"] - q["first_ts"]).dt.total_seconds() / 3600
    q = q.with_columns(
        span_days=pl.Series((span_h / 24).to_numpy()),
        hour_coverage=(pl.col("_hours") / pl.Series(np.maximum(np.ceil(span_h.to_numpy()), 1))).clip(0, 1),
        events_per_active_day=pl.col("n_events") / pl.col("active_days"),
        shape=pl.lit(schema.event_shape(events), dtype=pl.Int8),
    ).drop("_hours")
    vp = visits.group_by("id").agg(pl.len().alias("visits"), pl.col("place_id").n_unique().alias("places"))
    return q.join(vp, on="id", how="left").with_columns(pl.col("visits", "places").fill_null(0))
