"""PoL staypoint benchmark (arXiv 2607.19312): 500 synthetic agents, Atlanta, 1-min ticks.

The trajectory ``time`` is naive; agents are home overnight at local clock times, so it
is read as America/New_York (EST in Jan 2024, UTC-5) and converted to UTC. Ground-truth
staypoints (``startTime``/``endTime``, whole minutes) are closed intervals; ``place_id``
is the benchmark's per-agent ``loc_id_per_user``. The clean file has no positional
noise, so a nominal 8 m 1-sigma is attached without perturbing positions.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dwells.public import (
    OUT,
    RAW,
    count_events,
    download,
    label_stops,
    nominal_error,
    write_dataset,
)

NAME = "pol"
TRAJ_URL = "https://osf.io/download/4jr3g/"
GT_URL = "https://raw.githubusercontent.com/amirih/staypoint/main/data/v1/ground_truth.parquet"
TZ = "America/New_York"


def _utc(c: str) -> pl.Expr:
    return pl.col(c).dt.cast_time_unit("us").dt.replace_time_zone(TZ).dt.convert_time_zone("UTC")


def to_dataset(tr: pl.DataFrame, gt: pl.DataFrame, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    ev = tr.select(pl.col("agent_id").cast(pl.String).alias("id"), _utc("time").alias("timestamp"),
                   pl.col("latitude").alias("lat"), pl.col("longitude").alias("lon"))
    dw = gt.select(pl.col("agent_id").cast(pl.String).alias("id"), _utc("startTime").alias("start_ts"),
                   _utc("endTime").alias("end_ts"), pl.col("latitude").alias("lat"),
                   pl.col("longitude").alias("lon"), pl.col("loc_id_per_user").cast(pl.Int64).alias("place_id"),
                   ).sort("id", "start_ts").with_columns(dwell_id=pl.int_range(pl.len(), dtype=pl.Int64).over("id"))
    ev = nominal_error(label_stops(ev, dw), NAME, seed)
    dw = count_events(ev, dw).with_columns(
        category=pl.lit(None, dtype=pl.String), radius_m=pl.lit(None, dtype=pl.Float64),
        duration_s=(pl.col("end_ts") - pl.col("start_ts")).dt.total_seconds(),
    ).select("id", "dwell_id", "place_id", "category", "start_ts", "end_ts", "duration_s",
             "lat", "lon", "radius_m", "n_events")
    return ev, dw


def ingest(raw: Path = RAW / NAME, out_root: Path = OUT) -> dict:
    tr = pl.read_parquet(download(TRAJ_URL, raw / "trajectories_noiselevel0_dropoutlevel0.parquet"))
    gt = pl.read_parquet(download(GT_URL, raw / "ground_truth.parquet"))
    ev, dw = to_dataset(tr, gt)
    meta = {
        "name": NAME, "source_url": [TRAJ_URL, GT_URL], "landing": "https://osf.io/v8e4g/",
        "code": "https://github.com/amirih/staypoint",
        "license": "CC BY-NC-ND 4.0 (per the paper; OSF has no license field). Local research use only; "
                   "do not redistribute derived data.",
        "citation": "Staypoint Detection from Noisy Trajectory Data [Experiment Paper], arXiv 2607.19312 (2026)",
        "transformations": [
            "variant: noiselevel0_dropoutlevel0 (clean) only",
            f"naive time read as {TZ} and converted to UTC (+5 h in Jan 2024)",
            "id = str(agent_id); nominal circular 1-sigma 8 m, positions not perturbed (true = observed)",
            ("truth_dwells = ground_truth.parquet staypoints (closed [startTime, endTime]); "
             "place_id = loc_id_per_user (per-agent ids); dwell lat/lon from ground truth"),
        ],
        "min_truth_dwell_s": int(dw["duration_s"].min()),
    }
    return write_dataset(out_root / NAME, ev, meta, dwells=dw)
