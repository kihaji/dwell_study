"""Shared helpers: array preparation and label -> dwells aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from dwells import geo, schema
from dwells.uncertainty import cov_to_ellipse, with_covariance

ONE_SIGMA_P = 1 - np.exp(-0.5)  # probability whose k-factor is 1 (ellipse axes = 1-sigma)


@dataclass
class Arrays:
    """Flat per-event arrays for an events frame sorted by (id, timestamp)."""

    ids: np.ndarray       # int32 code per event
    starts: np.ndarray    # index of first event of each id (len n_ids + 1, last = n)
    t: np.ndarray         # seconds since epoch
    x: np.ndarray         # east meters (per-id local frame)
    y: np.ndarray         # north meters
    cee: np.ndarray       # east variance (m^2)
    cnn: np.ndarray
    cen: np.ndarray


def prepare(events: pl.DataFrame, default_sigma_m: float = 10.0) -> Arrays:
    ev = with_covariance(events, default_sigma_m)
    codes = ev["id"].rle_id().to_numpy().astype(np.int32)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(codes)]
    lat, lon = ev["lat"].to_numpy(), ev["lon"].to_numpy()
    # per-id local ENU frame anchored at the id's median position
    lat0 = np.repeat([np.median(lat[a:b]) for a, b in zip(starts[:-1], starts[1:])], np.diff(starts))
    lon0 = np.repeat([np.median(lon[a:b]) for a, b in zip(starts[:-1], starts[1:])], np.diff(starts))
    x, y = geo.to_enu(lat, lon, lat0, lon0)
    return Arrays(codes, starts, ev["timestamp"].dt.epoch("us").to_numpy() / 1e6, x, y,
                  ev["cov_ee"].to_numpy(), ev["cov_nn"].to_numpy(), ev["cov_en"].to_numpy())


def dwells_from_labels(events: pl.DataFrame, labels: np.ndarray, default_sigma_m: float = 10.0,
                       extra_var: float = 0.0) -> pl.DataFrame:
    """Aggregate events with label >= 0 into dwells (labels unique across ids).

    Start/end are the first/last member event times; the centroid is the inverse-
    covariance (fused) mean with its 1-sigma ellipse. ``extra_var`` (m^2) is added to each
    event's east/north variance before fusing (spread of true positions within a place).
    The label is returned as ``dwell_label``.
    """
    ev = events.with_columns(label=pl.Series(labels, dtype=pl.Int64)).filter(pl.col("label") >= 0)
    if ev.height == 0:
        return schema.empty(schema.DWELLS).with_columns(dwell_label=pl.lit(None, dtype=pl.Int64))
    out = ev.group_by("id", "label", maintain_order=True).agg(
        pl.col("timestamp").min().alias("start_ts"), pl.col("timestamp").max().alias("end_ts"),
        pl.len().alias("n_events"))
    out = out.join(_fused(ev, default_sigma_m, extra_var), on="label")
    return schema.validate_dwells(out.rename({"label": "dwell_label"}))


def _fused(ev: pl.DataFrame, default_sigma_m: float, extra_var: float = 0.0) -> pl.DataFrame:
    ev = with_covariance(ev, default_sigma_m).with_columns(
        pl.col("cov_ee") + extra_var, pl.col("cov_nn") + extra_var)
    det = pl.col("cov_ee") * pl.col("cov_nn") - pl.col("cov_en") ** 2
    ev = ev.with_columns(iee=pl.col("cov_nn") / det, inn=pl.col("cov_ee") / det, ien=-pl.col("cov_en") / det,
                         lat0=pl.col("lat").mean().over("label"), lon0=pl.col("lon").mean().over("label"))
    x, y = geo.to_enu(ev["lat"].to_numpy(), ev["lon"].to_numpy(), ev["lat0"].to_numpy(), ev["lon0"].to_numpy())
    ev = ev.with_columns(x=pl.Series(x), y=pl.Series(y))
    g = ev.group_by("label").agg(
        pl.col("iee").sum().alias("I_ee"), pl.col("inn").sum().alias("I_nn"), pl.col("ien").sum().alias("I_en"),
        (pl.col("iee") * pl.col("x") + pl.col("ien") * pl.col("y")).sum().alias("bx"),
        (pl.col("ien") * pl.col("x") + pl.col("inn") * pl.col("y")).sum().alias("by"),
        pl.col("lat0").first(), pl.col("lon0").first(),
    )
    D = g["I_ee"] * g["I_nn"] - g["I_en"] ** 2
    fee, fnn, fen = g["I_nn"] / D, g["I_ee"] / D, -g["I_en"] / D
    fx, fy = fee * g["bx"] + fen * g["by"], fen * g["bx"] + fnn * g["by"]
    lat, lon = geo.from_enu(fx.to_numpy(), fy.to_numpy(), g["lat0"].to_numpy(), g["lon0"].to_numpy())
    a, b, az = cov_to_ellipse(fee.to_numpy(), fnn.to_numpy(), fen.to_numpy(), ONE_SIGMA_P)
    return pl.DataFrame({"label": g["label"], "lat": lat, "lon": lon, "sigma_major": a, "sigma_minor": b, "azimuth": az})
