"""Public dataset ingestion (see docs/public_datasets.md, "Ingestion").

Each loader converts one raw download under ``data/public/raw/<name>/`` into a dataset
directory ``data/public/<name>/`` with the same master ``events.parquet`` layout the
generator writes (so ``io.load_events`` / ``io.shape_view`` work unchanged), optional
``truth_*.parquet`` tables and a ``meta.json``.

Real GPS has no reported accuracy, so ``nominal_error`` attaches a circular nominal
1-sigma (default 8 m) without perturbing positions: ``true_lat/true_lon`` equal the
observed ``lat/lon``.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from dwells import geo, io, schema
from dwells.generator import sampling
from dwells.generator.area import DATA_DIR

RAW = DATA_DIR / "public" / "raw"
OUT = DATA_DIR / "public"
NOMINAL_GPS_SIGMA_M = 8.0
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
MASTER = ["id", "timestamp", "lat", "lon", "true_lat", "true_lon", "source", "sig_a", "sig_b", "az",
          "p_mixed", "truth_dwell_id"]


def download(url: str, dest: Path) -> Path:
    """Fetch url to dest unless dest already exists (idempotent)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)
    return dest


def p_mixed(rng, n):
    return np.where(rng.random(n) < 0.5, 0.68, 0.95)


def nominal_error(ev: pl.DataFrame, source: str, seed: int, sigma: float = NOMINAL_GPS_SIGMA_M) -> pl.DataFrame:
    """Observed position is taken as truth; attach a circular nominal 1-sigma."""
    rng = np.random.default_rng(seed)
    return ev.with_columns(
        true_lat=pl.col("lat"), true_lon=pl.col("lon"), source=pl.lit(source),
        sig_a=pl.lit(sigma), sig_b=pl.lit(sigma), az=pl.lit(0.0),
        p_mixed=pl.Series(p_mixed(rng, ev.height)),
    )


def label_stops(ev: pl.DataFrame, stops: pl.DataFrame) -> pl.DataFrame:
    """truth_dwell_id = dwell_id of the stop whose closed [start_ts, end_ts] contains the event."""
    st = stops.select("id", "dwell_id", pl.col("start_ts").alias("timestamp"), "end_ts").sort("timestamp")
    out = ev.sort("timestamp").join_asof(st, on="timestamp", by="id", strategy="backward",
                                            check_sortedness=False)
    return out.with_columns(
        truth_dwell_id=pl.when(pl.col("timestamp") <= pl.col("end_ts")).then(pl.col("dwell_id")).cast(pl.Int64)
    ).drop("dwell_id", "end_ts").sort("id", "timestamp")


def count_events(ev: pl.DataFrame, dwells: pl.DataFrame) -> pl.DataFrame:
    n = ev.filter(pl.col("truth_dwell_id").is_not_null()).group_by(
        "id", pl.col("truth_dwell_id").alias("dwell_id")).len("n_events")
    return dwells.join(n, on=["id", "dwell_id"], how="left").with_columns(
        n_events=pl.col("n_events").fill_null(0).cast(pl.Int64))


def cluster_places(lat, lon, radius_m: float = 50.0) -> np.ndarray:
    """Leader clustering: each point joins the nearest existing leader within radius_m, else leads a new place."""
    x, y = geo.to_enu(lat, lon, float(np.mean(lat)), float(np.mean(lon)))
    leaders: list[int] = []
    out = np.empty(len(x), dtype=np.int64)
    for i in range(len(x)):
        if leaders:
            d = np.hypot(x[leaders] - x[i], y[leaders] - y[i])
            j = int(np.argmin(d))
            if d[j] <= radius_m:
                out[i] = j
                continue
        leaders.append(i)
        out[i] = len(leaders) - 1
    return out


def write_dataset(out: Path, ev: pl.DataFrame, meta: dict, **tables: pl.DataFrame) -> dict:
    """Write master events + truth_<name> tables + meta.json; check every shape loads."""
    out.mkdir(parents=True, exist_ok=True)
    if "truth_dwell_id" not in ev.columns:
        ev = ev.with_columns(truth_dwell_id=pl.lit(None, dtype=pl.Int64))
    ev = ev.select(MASTER).with_columns(
        pl.col("timestamp").cast(schema.TS), pl.col("id").cast(pl.String),
        pl.col("truth_dwell_id").cast(pl.Int64)).sort("id", "timestamp")
    ev.write_parquet(out / "events.parquet")
    for name, t in tables.items():
        t.write_parquet(out / f"truth_{name}.parquet")
    for shape in io.SHAPES:
        io.load_events(out, shape)
    meta = {**meta, "stats": sampling.describe(ev).to_dicts()[0],
            **{f"truth_{k}": t.height for k, t in tables.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta
