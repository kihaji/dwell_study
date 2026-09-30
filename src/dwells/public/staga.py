"""STAGA (Spang et al. 2022): one subject, 126 days of GPS with diary stop intervals.

Truth dwells are the 689 diary stop intervals; every GPS point inside one is labelled
``stop`` in the raw file, so ``truth_dwell_id`` reproduces the per-point diary label.
Timestamps carry +01:00/+02:00 offsets and are converted to UTC. The dwell position is
the median of its GPS points (null if it has none); ``place_id`` groups dwell positions
within 50 m (leader clustering).
"""

from __future__ import annotations

import io as _io
from pathlib import Path

import polars as pl

from dwells.public import (
    OUT,
    RAW,
    cluster_places,
    count_events,
    download,
    label_stops,
    nominal_error,
    write_dataset,
)

NAME = "staga"
GPS_URL = "https://osf.io/download/y84dv/"
DIARY_URL = "https://osf.io/download/z5txn/"
FMT = "%Y-%m-%d %H:%M:%S%.f%:z"


def parse_gps(text: str) -> pl.DataFrame:
    return pl.read_csv(_io.StringIO(text)).select(
        pl.lit(NAME).alias("id"), pl.col("ts").str.to_datetime(FMT, time_zone="UTC").alias("timestamp"),
        pl.col("latitude").alias("lat"), pl.col("longitude").alias("lon"), "diary_label")


def parse_diary(text: str) -> pl.DataFrame:
    d = pl.read_csv(_io.StringIO(text)).select(
        pl.col("start_time").str.to_datetime(FMT, time_zone="UTC").alias("start_ts"),
        pl.col("end_time").str.to_datetime(FMT, time_zone="UTC").alias("end_ts")).sort("start_ts")
    return d.select(pl.lit(NAME).alias("id"), pl.int_range(pl.len(), dtype=pl.Int64).alias("dwell_id"),
                    "start_ts", "end_ts")


def to_dataset(ev: pl.DataFrame, stops: pl.DataFrame, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    ev = nominal_error(label_stops(ev, stops), NAME, seed)
    pos = ev.filter(pl.col("truth_dwell_id").is_not_null()).group_by(
        pl.col("truth_dwell_id").alias("dwell_id")).agg(pl.col("lat", "lon").median())
    dw = count_events(ev, stops.join(pos, on="dwell_id", how="left").sort("dwell_id"))
    has = dw["lat"].is_not_null()
    place = pl.Series("place_id", [None] * dw.height, dtype=pl.Int64)
    place = place.scatter(has.arg_true(), cluster_places(dw.filter(has)["lat"].to_numpy(),
                                                         dw.filter(has)["lon"].to_numpy(), 50.0))
    dw = dw.with_columns(
        place, category=pl.lit(None, dtype=pl.String), radius_m=pl.lit(None, dtype=pl.Float64),
        duration_s=(pl.col("end_ts") - pl.col("start_ts")).dt.total_seconds(),
    ).select("id", "dwell_id", "place_id", "category", "start_ts", "end_ts", "duration_s",
             "lat", "lon", "radius_m", "n_events")
    return ev, dw


def ingest(raw: Path = RAW / NAME, out_root: Path = OUT) -> dict:
    gps = download(GPS_URL, raw / "STAGA gps aggregated.csv")
    diary = download(DIARY_URL, raw / "STAGA diary.csv")
    ev, dw = to_dataset(parse_gps(gps.read_text()), parse_diary(diary.read_text()))
    meta = {
        "name": NAME, "source_url": [GPS_URL, DIARY_URL], "landing": "https://osf.io/34sft/",
        "license": "CC BY 4.0",
        "citation": "Spang et al., ISPRS Archives XLVIII-4/W1-2022, doi:10.5194/isprs-archives-XLVIII-4-W1-2022-443-2022",
        "transformations": [
            "ts (+01/+02 offsets) converted to UTC; dates are shifted by the authors to start 2000-01-01",
            "coordinates are the authors' anonymized (displaced) positions; used as-is",
            "no accuracy in source: nominal circular 1-sigma 8 m, positions not perturbed (true = observed)",
            "truth_dwells = diary stop intervals; truth_dwell_id = interval containing the point (closed)",
            "dwell lat/lon = median of its GPS points (null if none); place_id = 50 m leader clustering",
        ],
        "min_truth_dwell_s": int(dw["duration_s"].min()),
        "stop_points": int(ev["truth_dwell_id"].is_not_null().sum()),
    }
    return write_dataset(out_root / NAME, ev, meta, dwells=dw)
