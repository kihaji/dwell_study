"""Scoring detected dwells / colocations against ground truth.

Dwell matching: within each id, predicted and true dwells are paired by temporal
IoU (intersection-over-union of [start, end)); pairs with IoU >= iou_min are
assigned one-to-one greedily by descending IoU. For iou_min >= 0.5 and disjoint
intervals the assignment is unique, so greedy equals optimal. Location error is
reported for matched pairs but is not a matching criterion (coarse sources like
CDR would otherwise never match).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dwells.geo import haversine_m

_US = 1e-6


def _arr(df: pl.DataFrame):
    return (
        df["id"].to_numpy(),
        df["start_ts"].dt.epoch("us").to_numpy() * _US,
        df["end_ts"].dt.epoch("us").to_numpy() * _US,
    )


def overlap_pairs(pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """All (pred_idx, truth_idx) with positive temporal overlap in the same id."""
    pi, ps, pe = _arr(pred)
    ti, ts, te = _arr(truth)
    out_p, out_t = [], []
    t_groups = {}
    for k in np.unique(ti):
        idx = np.flatnonzero(ti == k)
        idx = idx[np.argsort(ts[idx])]
        t_groups[k] = (idx, ts[idx], np.maximum.accumulate(te[idx]))
    for k in np.unique(pi):
        if k not in t_groups:
            continue
        pidx = np.flatnonzero(pi == k)
        tidx, tstart, tend_max = t_groups[k]
        # truths starting before pred end and (running-max) ending after pred start
        hi = np.searchsorted(tstart, pe[pidx], side="left")
        lo = np.searchsorted(tend_max, ps[pidx], side="right")
        for p, a, b in zip(pidx, lo, hi):
            if b > a:
                cand = tidx[a:b]
                cand = cand[(ts[cand] < pe[p]) & (te[cand] > ps[p])]
                out_p.extend([p] * len(cand))
                out_t.extend(cand.tolist())
    p_arr, t_arr = np.asarray(out_p, dtype=np.int64), np.asarray(out_t, dtype=np.int64)
    inter = np.minimum(pe[p_arr], te[t_arr]) - np.maximum(ps[p_arr], ts[t_arr])
    union = np.maximum(pe[p_arr], te[t_arr]) - np.minimum(ps[p_arr], ts[t_arr])
    return pl.DataFrame({
        "p": p_arr, "t": t_arr, "inter_s": inter,
        "iou": np.divide(inter, union, out=np.zeros_like(inter), where=union > 0),
        "p_frac": inter / np.maximum(pe[p_arr] - ps[p_arr], 1e-9),
        "t_frac": inter / np.maximum(te[t_arr] - ts[t_arr], 1e-9),
    })


def match_dwells(pred: pl.DataFrame, truth: pl.DataFrame, iou_min: float = 0.5,
                 pairs: pl.DataFrame | None = None) -> pl.DataFrame:
    pairs = overlap_pairs(pred, truth) if pairs is None else pairs
    cand = pairs.filter(pl.col("iou") >= iou_min).sort("iou", descending=True)
    used_p, used_t, keep = set(), set(), []
    for p, t in zip(cand["p"].to_list(), cand["t"].to_list()):
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
            keep.append((p, t))
    return pl.DataFrame(keep, schema={"p": pl.Int64, "t": pl.Int64}, orient="row").join(cand, on=["p", "t"], how="left")


def _prf(matched, n_pred, n_truth):
    p = matched / n_pred if n_pred else 0.0
    r = matched / n_truth if n_truth else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def dwell_metrics(pred: pl.DataFrame, truth: pl.DataFrame, min_duration_s: float = 0,
                  iou_min: float = 0.5, min_obs_events: int = 2) -> dict:
    truth = truth.filter(pl.col("duration_s") >= min_duration_s)
    pred = pred.select("id", "start_ts", "end_ts", "lat", "lon")
    pairs = overlap_pairs(pred, truth)
    m = match_dwells(pred, truth, iou_min, pairs)
    n_pred, n_truth = pred.height, truth.height
    precision, recall, f1 = _prf(m.height, n_pred, n_truth)
    obs = truth["n_events"].to_numpy() >= min_obs_events
    n_obs = int(obs.sum())
    matched_t = m["t"].to_numpy()
    rec_obs = float(obs[matched_t].sum() / n_obs) if n_obs else 0.0
    lm = match_dwells(pred, truth, 0.2, pairs)
    _, rec_l, f1_l = _prf(lm.height, n_pred, n_truth)

    res = {
        "n_pred": n_pred, "n_truth": n_truth, "n_truth_observable": n_obs, "matched": m.height,
        "precision": precision, "recall": recall, "f1": f1, "recall_observable": rec_obs,
        "f1_observable": _prf(int(obs[matched_t].sum()), n_pred, n_obs)[2] if n_obs else 0.0,
        "recall_lenient": rec_l, "f1_lenient": f1_l,
    }
    if m.height:
        _, ps, pe = _arr(pred)
        _, ts, te = _arr(truth)
        p, t = m["p"].to_numpy(), m["t"].to_numpy()
        cerr = haversine_m(pred["lat"].to_numpy()[p], pred["lon"].to_numpy()[p],
                           truth["lat"].to_numpy()[t], truth["lon"].to_numpy()[t])
        res |= {
            "start_err_s_med": float(np.median(np.abs(ps[p] - ts[t]))),
            "end_err_s_med": float(np.median(np.abs(pe[p] - te[t]))),
            "duration_bias_s_med": float(np.median((pe[p] - ps[p]) - (te[t] - ts[t]))),
            "centroid_err_m_med": float(np.median(cerr)),
            "centroid_err_m_p90": float(np.quantile(cerr, 0.9)),
        }
    else:
        res |= {k: None for k in ("start_err_s_med", "end_err_s_med", "duration_bias_s_med",
                                  "centroid_err_m_med", "centroid_err_m_p90")}
    inside = pairs.filter(pl.col("p_frac") >= 0.5).group_by("t").len()
    covers = pairs.filter(pl.col("t_frac") >= 0.5).group_by("p").len()
    tot_p = (pred["end_ts"] - pred["start_ts"]).dt.total_seconds().sum()
    tot_t = truth["duration_s"].sum()
    res |= {
        "fragmentation": inside.filter(pl.col("len") >= 2).height / n_truth if n_truth else 0.0,
        "merging": covers.filter(pl.col("len") >= 2).height / n_pred if n_pred else 0.0,
        "time_precision": float(pairs["inter_s"].sum() / tot_p) if tot_p else 0.0,
        "time_recall": float(pairs["inter_s"].sum() / tot_t) if tot_t else 0.0,
    }
    return res
