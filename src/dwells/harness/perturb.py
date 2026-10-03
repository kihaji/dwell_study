"""Input-defect injection for robustness experiments.

Each perturbation mimics a defect an upstream cleaning/transformation pipeline can
introduce or fail to remove. They act on the master events frame (positions plus
reported 1-sigma axes sig_a/sig_b, see generator/sampling.py), before a shape view
is taken, so every dataset shape sees the same defect.

| name          | level meaning                              | real-world cause |
|---------------|--------------------------------------------|------------------|
| error_scale   | factor applied to reported error           | accuracy semantics mix-up (68% or 95% radius labelled CEP50 -> 1.28 / 2.08; CEP50 labelled 68% -> 0.78), unit bugs |
| error_noise   | lognormal sigma of per-event error factor  | poorly calibrated accuracy estimates |
| duplicates    | fraction of rows duplicated exactly        | merges of overlapping feeds, retries |
| stale         | fraction of fixes replaced by the previous | cached / last-known location re-reported (adtech) |
| outliers      | fraction displaced by dist_m (error kept)  | multipath, wrong geolocation with a confident accuracy |
| clock_jitter  | timestamp noise sd (seconds)               | device clock drift, server-receipt timestamps |
| thin          | fraction of events kept                    | lower sampling rate, aggressive deduplication |
| truncate      | decimals kept in lat/lon                   | coordinate rounding (3 dp ~ 110 m, 4 dp ~ 11 m) |
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dwells import geo

PERTURBATIONS = ("error_scale", "error_noise", "duplicates", "stale", "outliers", "clock_jitter", "thin", "truncate")


def apply(master: pl.DataFrame, name: str, level: float, seed: int = 0, dist_m: float = 500.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = master.height
    m = master
    if name == "error_scale":
        m = m.with_columns(pl.col("sig_a") * level, pl.col("sig_b") * level)
    elif name == "error_noise":
        f = pl.Series(np.exp(rng.normal(0, level, n)))
        m = m.with_columns(pl.col("sig_a") * f, pl.col("sig_b") * f)
    elif name == "duplicates":
        m = pl.concat([m, m.filter(pl.Series(rng.random(n) < level))])
    elif name == "stale":
        pick = pl.Series(rng.random(n) < level)
        cols = ["lat", "lon", "sig_a", "sig_b", "az"]
        same_id = pl.col("id") == pl.col("id").shift(1)
        m = m.with_columns([pl.when(pick & same_id).then(pl.col(c).shift(1)).otherwise(pl.col(c)).alias(c) for c in cols])
    elif name == "outliers":
        pick = rng.random(n) < level
        ang = rng.uniform(0, 2 * np.pi, n)
        lat, lon = m["lat"].to_numpy(), m["lon"].to_numpy()
        dx, dy = np.where(pick, dist_m * np.cos(ang), 0.0), np.where(pick, dist_m * np.sin(ang), 0.0)
        nlat, nlon = geo.from_enu(dx, dy, lat, lon)
        m = m.with_columns(lat=pl.Series(nlat), lon=pl.Series(nlon))
    elif name == "clock_jitter":
        us = (rng.normal(0, level, n) * 1e6).astype(np.int64)
        m = m.with_columns((pl.col("timestamp").dt.epoch("us") + pl.Series(us)).cast(pl.Datetime("us", "UTC")).alias("timestamp"))
    elif name == "thin":
        m = m.filter(pl.Series(rng.random(n) < level))
    elif name == "truncate":
        m = m.with_columns(pl.col("lat").round(int(level)), pl.col("lon").round(int(level)))
    else:
        raise ValueError(f"unknown perturbation {name!r}; choose from {PERTURBATIONS}")
    return m.sort("id", "timestamp", maintain_order=True)
