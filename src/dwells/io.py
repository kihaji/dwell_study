"""Dataset I/O and shape views.

A synthetic dataset directory holds ``events.parquet`` (master frame with truth and
reported error, see generator/sampling.py) plus ``truth_*.parquet`` tables.
``shape_view`` projects the master frame onto one of the five evaluated shapes;
all shapes share identical lat/lon so shape comparisons isolate the error info.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dwells import schema
from dwells.uncertainty import CEP50_K

SHAPES = ("1", "2", "3_p68", "3_p95", "3_mixed")
TRUTH_COLS = ("true_lat", "true_lon", "source", "truth_dwell_id")


# generator report source -> location_source label; "stale" (a cached fix re-reported)
# keeps the label of the report it repeats, as cached locations do on devices
GENERATOR_SOURCE = {"gps": "gps", "gps_outlier": "gps", "wifi": "network", "cell": "network",
                    "cdr": "network", "ip": "ip"}


def location_source(master: pl.DataFrame) -> pl.Expr:
    lab = pl.col("source").replace_strict(GENERATOR_SOURCE, default=None)
    return lab.forward_fill().over("id").fill_null("unknown").alias("location_source")


def shape_view(master: pl.DataFrame, shape: str, keep_truth: bool = False,
               with_source: bool = False) -> pl.DataFrame:
    """Project the master frame onto a dataset shape; ``with_source`` adds the optional
    ``location_source`` column derived from the generator's per-report source."""
    if with_source:
        return shape_view(master, shape, keep_truth).with_columns(
            master.select(location_source(master))["location_source"])
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
                ids: list[str] | None = None, with_source: bool = False) -> pl.DataFrame:
    lf = pl.scan_parquet(Path(dataset_dir) / "events.parquet")
    if ids is not None:
        lf = lf.filter(pl.col("id").is_in(ids))
    return schema.validate_events(shape_view(lf.collect(), shape, keep_truth, with_source))


def load_truth(dataset_dir: str | Path, table: str) -> pl.DataFrame:
    return pl.read_parquet(Path(dataset_dir) / f"truth_{table}.parquet")



def dataset_dirs() -> dict[str, Path]:
    """All repo datasets (synthetic + public) that have an events.parquet."""
    from dwells.generator.area import DATA_DIR

    out = {}
    for root in (DATA_DIR / "synthetic", DATA_DIR / "public"):
        if root.exists():
            out |= {p.name: p for p in sorted(root.iterdir()) if (p / "events.parquet").exists()}
    return out


def profile_of(name: str) -> str:
    """Source profile (gps / cdr / adtech) implied by a dataset name."""
    for p in ("gps", "cdr", "adtech"):
        if f"_{p}" in name:
            return p
    return "adtech" if "foursquare" in name else "gps"
