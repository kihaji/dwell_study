"""Canonical table schemas shared by the generator, algorithms, harness and viewer."""

from __future__ import annotations

import polars as pl

TS = pl.Datetime("us", "UTC")

EVENTS_BASE = {"id": pl.String, "lat": pl.Float64, "lon": pl.Float64, "timestamp": TS}
# Optional per-event provenance of the fix (Android/iOS provider terms plus adtech ones;
# OpenRTB geo.type 1 -> gps, 2 -> ip, 3 -> user_provided). Null = unknown.
LOCATION_SOURCES = ("unknown", "gps", "network", "fused", "ip", "user_provided")

EVENTS_SHAPE2 = {"horizontalAccuracy": pl.Float64}
EVENTS_SHAPE3 = {
    "semiMajor": pl.Float64,
    "semiMinor": pl.Float64,
    "azimuth": pl.Float64,
    "probability": pl.Float64,
}

# Detected (or true) dwells. sigma_* are 1-sigma axes (m) of the dwell centroid
# uncertainty when the algorithm can estimate it, else null.
DWELLS = {
    "id": pl.String,
    "dwell_id": pl.Int64,
    "start_ts": TS,
    "end_ts": TS,
    "duration_s": pl.Int64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "n_events": pl.Int64,
    "sigma_major": pl.Float64,
    "sigma_minor": pl.Float64,
    "azimuth": pl.Float64,
}

def empty(spec: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=spec)


def event_shape(df: pl.DataFrame) -> int:
    cols = set(df.columns)
    if set(EVENTS_SHAPE3) <= cols:
        return 3
    if set(EVENTS_SHAPE2) <= cols:
        return 2
    return 1


def _require(df: pl.DataFrame, spec: dict, name: str) -> None:
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _check(df: pl.DataFrame, expr: pl.Expr, msg: str) -> None:
    if df.select((~expr).any()).item():
        raise ValueError(msg)


def validate_events(df: pl.DataFrame) -> pl.DataFrame:
    """Check ranges, cast to canonical dtypes, sort by (id, timestamp)."""
    _require(df, EVENTS_BASE, "events")
    shape = event_shape(df)
    spec = dict(EVENTS_BASE)
    if shape == 2:
        spec |= EVENTS_SHAPE2
    if shape == 3:
        spec |= EVENTS_SHAPE3
    df = df.with_columns([pl.col(c).cast(t) for c, t in spec.items()])
    if "location_source" in df.columns:
        df = df.with_columns(pl.col("location_source").cast(pl.String).fill_null("unknown"))
        _check(df, pl.col("location_source").is_in(LOCATION_SOURCES),
               f"location_source must be one of {LOCATION_SOURCES}")
    _check(df, pl.col("lat").is_between(-90, 90), "lat out of range")
    _check(df, pl.col("lon").is_between(-180, 180), "lon out of range")
    if shape == 2:
        _check(df, pl.col("horizontalAccuracy") > 0, "horizontalAccuracy must be > 0")
    if shape == 3:
        _check(df, pl.col("semiMinor") > 0, "semiMinor must be > 0")
        _check(df, pl.col("semiMajor") >= pl.col("semiMinor"), "semiMajor < semiMinor")
        _check(df, pl.col("azimuth").is_between(0, 180), "azimuth outside [0, 180]")
        _check(
            df,
            (pl.col("probability") > 0) & (pl.col("probability") < 1),
            "probability outside (0, 1)",
        )
    return df.sort("id", "timestamp")


def validate_dwells(df: pl.DataFrame) -> pl.DataFrame:
    """Fill derived/optional columns and cast to the DWELLS schema."""
    required = ["id", "start_ts", "end_ts", "lat", "lon", "n_events"]
    _require(df, {c: None for c in required}, "dwells")
    df = df.with_columns(pl.col("start_ts", "end_ts").cast(TS))
    if "duration_s" not in df.columns:
        df = df.with_columns(
            duration_s=((pl.col("end_ts") - pl.col("start_ts")).dt.total_seconds())
        )
    if "dwell_id" not in df.columns:
        df = df.sort("id", "start_ts").with_columns(
            dwell_id=pl.int_range(pl.len()).over("id")
        )
    for c in ("sigma_major", "sigma_minor", "azimuth"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    extra = [c for c in df.columns if c not in DWELLS]
    return df.select([pl.col(c).cast(t) for c, t in DWELLS.items()] + extra).sort(
        "id", "start_ts"
    )
