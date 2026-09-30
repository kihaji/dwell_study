"""Shared helpers for detectors: array preparation, gap segmentation, label -> dwells."""

from __future__ import annotations

from dataclasses import dataclass

import numba
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


def gap_segments(events: pl.DataFrame, max_gap_s: float) -> pl.Series:
    """Segment number per event: increments on a new id or a gap > max_gap_s."""
    gap = events["timestamp"].diff().dt.total_seconds().fill_null(0) > max_gap_s
    new_id = events["id"] != events["id"].shift(1)
    return (gap | new_id.fill_null(True)).cum_sum().alias("seg")


def dwells_from_labels(events: pl.DataFrame, labels: np.ndarray, min_duration_s: float,
                       centroid: str = "mean", min_events: int = 1,
                       default_sigma_m: float = 10.0, extra_var: float = 0.0) -> pl.DataFrame:
    """Aggregate events with label >= 0 into dwells (labels must be unique across ids).

    Start/end are the first/last event times. ``centroid``: mean | median | fused
    (inverse-covariance weighted, which also yields a 1-sigma centroid ellipse);
    ``extra_var`` (m^2) is added to each event's east/north variance before fusing,
    e.g. the spread of true positions within a place.
    """
    ev = events.with_columns(label=pl.Series(labels, dtype=pl.Int64)).filter(pl.col("label") >= 0)
    if ev.height == 0:
        return schema.empty(schema.DWELLS)
    agg = [pl.col("timestamp").min().alias("start_ts"), pl.col("timestamp").max().alias("end_ts"),
           pl.len().alias("n_events")]
    if centroid == "median":
        agg += [pl.col("lat").median(), pl.col("lon").median()]
    else:
        agg += [pl.col("lat").mean(), pl.col("lon").mean()]
    out = ev.group_by("id", "label", maintain_order=True).agg(agg)
    if centroid == "fused":
        out = out.drop("lat", "lon").join(_fused(ev, default_sigma_m, extra_var), on="label")
    out = out.filter(
        ((pl.col("end_ts") - pl.col("start_ts")).dt.total_seconds() >= min_duration_s)
        & (pl.col("n_events") >= min_events)
    )
    return schema.validate_dwells(out.drop("label"))


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


def per_id_pandas(events: pl.DataFrame):
    """Yield (id, pandas frame) per id (for libraries needing per-user calls)."""
    for (k,), g in events.group_by("id", maintain_order=True):
        yield k, g.to_pandas()


@numba.njit(cache=True)
def contiguous_runs(starts, t, labels, max_gap_s, max_noise_run):
    """Split cluster labels into time-contiguous runs (unique ids, -1 = none).

    A run of cluster c continues across other events only while they are noise
    (label -1), at most ``max_noise_run`` of them in a row, and the time since the
    previous c event is <= max_gap_s. Any event of a different cluster ends it.
    """
    n = len(t)
    out = np.full(n, -1, np.int64)
    nxt = -1
    for k in range(len(starts) - 1):
        cur, last_t, noise = -1, 0.0, 0
        for i in range(starts[k], starts[k + 1]):
            c = labels[i]
            if c < 0:
                noise += 1
                if noise > max_noise_run:
                    cur = -1
                continue
            if c != cur or t[i] - last_t > max_gap_s:
                nxt += 1
                cur = c
            out[i] = nxt
            last_t, noise = t[i], 0
    return out



def trim_runs(x: np.ndarray, y: np.ndarray, runs: np.ndarray, radius_m: float,
              min_keep: float = 0.5) -> np.ndarray:
    """Robust spatial trim of labeled runs.

    Events farther than radius_m from their run's median center are unlabeled
    (removes approach/departure points that density methods attach to stays); a run
    losing more than (1 - min_keep) of its events is dropped entirely (e.g. an evenly
    sampled trajectory that a relative-density method called a cluster).
    """
    out = runs.copy()
    idx = np.flatnonzero(runs >= 0)
    if len(idx) == 0:
        return out
    order = idx[np.argsort(runs[idx], kind="stable")]
    r = runs[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(r)) + 1, len(r)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        g = order[a:b]
        d = np.hypot(x[g] - np.median(x[g]), y[g] - np.median(y[g]))
        far = d > radius_m
        out[g[far]] = -1
        if (~far).mean() < min_keep:
            out[g] = -1
    return out
