"""Dataset I/O and shape views.

A synthetic dataset directory holds ``events.parquet`` (master frame with truth and
reported error, see generator/sampling.py) plus ``truth_*.parquet`` tables.
``shape_view`` projects the master frame onto one of the five evaluated shapes;
all shapes share identical lat/lon so shape comparisons isolate the error info.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from dwells import schema
from dwells.uncertainty import CEP50_K, k_factor

SHAPES = ("1", "2", "3_p68", "3_p95", "3_mixed")
TRUTH_COLS = ("true_lat", "true_lon", "source", "truth_dwell_id")


def shape_view(master: pl.DataFrame, shape: str, keep_truth: bool = False) -> pl.DataFrame:
    base = ["id", "lat", "lon", "timestamp"]
    extra = [c for c in TRUTH_COLS if keep_truth and c in master.columns]
    if shape == "1":
        return master.select(base + extra)
    if shape == "2":
        # CEP50 of the circle with the same RMS radius as the reported ellipse
        rms = ((pl.col("sig_a") ** 2 + pl.col("sig_b") ** 2) / 2).sqrt()
        return master.select(base + [(rms * CEP50_K).alias("horizontalAccuracy")] + extra)
    p = {"3_p68": pl.lit(0.68), "3_p95": pl.lit(0.95), "3_mixed": pl.col("p_mixed")}[shape]
    k = (-2.0 * (1.0 - p).log()).sqrt()
    return master.select(
        base
        + [
            (pl.col("sig_a") * k).alias("semiMajor"),
            (pl.col("sig_b") * k).alias("semiMinor"),
            pl.col("az").alias("azimuth"),
            p.alias("probability"),
        ]
        + extra
    )


def load_events(dataset_dir: str | Path, shape: str = "1", keep_truth: bool = False,
                ids: list[str] | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(Path(dataset_dir) / "events.parquet")
    if ids is not None:
        lf = lf.filter(pl.col("id").is_in(ids))
    return schema.validate_events(shape_view(lf.collect(), shape, keep_truth))


def load_truth(dataset_dir: str | Path, table: str) -> pl.DataFrame:
    return pl.read_parquet(Path(dataset_dir) / f"truth_{table}.parquet")


def load_meta(dataset_dir: str | Path) -> dict:
    return json.loads((Path(dataset_dir) / "meta.json").read_text())


_ = k_factor  # shape math above mirrors uncertainty.k_factor
