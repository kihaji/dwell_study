"""Evaluate the post-detection pipeline against synthetic ground truth.

    uv run python -m dwells.harness.downstream_eval [--datasets glob,glob] [--md 900]

Inputs: tuned detections from the eval stage (results/dwells/<dataset>/<algo>__<shape>__<md>.parquet;
uncertainty-aware algorithms use shape 3_mixed, others shape 1). For every detector:

* places      for each grouping method: adjusted Rand index between detected place ids
              and true place ids over matched dwells (IoU >= 0.5), averaged over ids;
              plus n_places / n_true_places (visited >= 2 times).
* home_work   home/work inferred from detected places vs the agent's true home/work:
              hit if within ``hit_m`` (gps 100 m, adtech 250 m, cdr 1000 m).
* nextloc     acc@1/3/5 + MRR for each model on (a) true place sequences (model ceiling),
              (b) detected place sequences (what an app sees), (c) detected dwells mapped
              to the true place of their matched truth dwell (does detection preserve
              the true routine?).
* colocation  same_time pair/episode P/R/F1 per method vs truth derived from true dwells
              >= md; false-positive rate on next-door-neighbor decoy pairs; diff_time pair
              P/R/F1; a radius sweep (colocation_radius) shows recall vs decoy false positives.
Results -> results/downstream.parquet (long format: dataset, algo, stage, method, metric, value).
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

from dwells import algorithms, geo
from dwells.downstream import colocation, nextloc, patterns, places
from dwells.generator.build import UTC_OFFSET
from dwells.harness.run import RESULTS, dataset_dirs, profile_of
from dwells.metrics import colocation_metrics, colocation_truth, match_dwells

PLACE_RADIUS = {"gps": 150, "adtech": 250, "cdr": 1000}
HIT_M = {"gps": 100, "adtech": 250, "cdr": 1000}
COLOC_RADII = {"gps": (15, 30, 60, 150), "adtech": (30, 60, 150, 250), "cdr": (250, 500, 1000)}


def _rows(out, ds, algo, stage, method, metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and v is not None:
            out.append({"dataset": ds, "algo": algo, "stage": stage, "method": method, "metric": k, "value": float(v)})


def _truth_seq(truth, md):
    return truth.filter(pl.col("duration_s") >= md).select("id", "place_id", "start_ts")


def evaluate_dataset(ds: str, md: int, out: list):
    d = dataset_dirs()[ds]
    prof = profile_of(ds)
    area = "urban" if "urban" in ds else "suburban"
    off = UTC_OFFSET[area]
    truth = pl.read_parquet(d / "truth_dwells.parquet")
    agents = pl.read_parquet(d / "truth_agents.parquet")
    tplaces = pl.read_parquet(d / "truth_places.parquet")
    radius = PLACE_RADIUS[prof]
    tr_md = truth.filter(pl.col("duration_s") >= md)

    # model ceiling on true sequences (independent of detection)
    for model in nextloc.MODELS:
        _rows(out, ds, "_truth", "nextloc_truth", model, nextloc.evaluate(_truth_seq(truth, md), model, utc_offset_h=off))

    ct = colocation_truth(tr_md, min_overlap_s=900, diff_time_window_s=7 * 86400)
    # decoy pairs: members of neighboring households that never truly co-dwell
    hh = agents.select("id", "household", "neighbor_of")
    nb = hh.filter(pl.col("neighbor_of") >= 0).join(hh.select(pl.col("id").alias("id_b"), pl.col("household").alias("neighbor_of")), on="neighbor_of")
    decoys = {tuple(sorted(p)) for p in nb.select("id", "id_b").rows()}
    true_pairs = {tuple(sorted(p)) for p in ct.filter(pl.col("kind") == "same_time").select("id_a", "id_b").rows()}
    decoys -= true_pairs

    for algo in algorithms.names():
        shape = "3_mixed" if algorithms.get(algo).USES_UNCERTAINTY else "1"
        f = RESULTS / "dwells" / ds / f"{algo}__{shape}__{md}.parquet"
        if not f.exists():
            continue
        det = pl.read_parquet(f)
        if det.height == 0:
            continue
        tp = tr_md["place_id"].to_numpy()

        for method in places.METHODS:
            dp, _ = places.assign(det, method=method, radius_m=radius)
            dp = dp.sort("id", "start_ts")
            # match on the sorted frame so indices line up
            mm = match_dwells(dp, tr_md)
            pid = dp["place_id"].to_numpy()
            ids = dp["id"].to_numpy()
            aris = []
            for k in np.unique(ids[mm["p"].to_numpy()]) if mm.height else []:
                sel = ids[mm["p"].to_numpy()] == k
                if sel.sum() >= 3:
                    aris.append(adjusted_rand_score(tp[mm["t"].to_numpy()[sel]], pid[mm["p"].to_numpy()[sel]]))
            n_true = tr_md.group_by("id", "place_id").len().filter(pl.col("len") >= 2).height
            n_det = dp.group_by("id", "place_id").len().filter(pl.col("len") >= 2).height
            _rows(out, ds, algo, "places", method, {"ari": np.mean(aris) if aris else 0.0,
                                                     "place_count_ratio": n_det / max(n_true, 1)})
            if method != "dbscan":
                continue
            # home / work
            hw = patterns.label_home_work(patterns.place_profiles(dp, off))
            cent = places.place_table(dp).select("id", "place_id", "lat", "lon")
            hw = hw.join(cent, on=["id", "place_id"])
            for label, col in (("home", "home"), ("work", "work")):
                ta = agents.filter(pl.col(col).is_not_null()).select("id", pl.col(col).alias("pid")).join(
                    tplaces.select(pl.col("place_id").alias("pid"), pl.col("lat").alias("tlat"), pl.col("lon").alias("tlon")), on="pid")
                if label == "work":
                    ta = ta.join(agents.filter(pl.col("persona") == "worker").select("id"), on="id")
                j = ta.join(hw.filter(pl.col("label") == label), on="id", how="left")
                dist = geo.haversine_m(j["lat"].fill_null(0).to_numpy(), j["lon"].fill_null(0).to_numpy(),
                                       j["tlat"].to_numpy(), j["tlon"].to_numpy())
                hit = (~j["lat"].is_null().to_numpy()) & (dist <= HIT_M[prof])
                _rows(out, ds, algo, "home_work", label, {"accuracy": hit.mean() if len(hit) else 0.0})
            # next location on detected and truth-mapped sequences
            seq_det = dp.select("id", "place_id", "start_ts")
            mapped = None
            if mm.height:
                p_idx, t_idx = mm["p"].to_numpy(), mm["t"].to_numpy()
                mapped = dp[p_idx].with_columns(place_id=pl.Series(tp[t_idx], dtype=pl.Int64))
            for model in nextloc.MODELS:
                _rows(out, ds, algo, "nextloc_detected", model, nextloc.evaluate(seq_det, model, utc_offset_h=off))
                if mapped is not None:
                    _rows(out, ds, algo, "nextloc_mapped", model,
                          nextloc.evaluate(mapped.select("id", "place_id", "start_ts"), model, utc_offset_h=off))

        # colocation
        for method in colocation.METHODS:
            pc = colocation.same_time(det, method=method, radius_m=radius, min_overlap_s=900)
            cm = colocation_metrics(pc, ct, "same_time")
            pp = {tuple(sorted(p)) for p in pc.select("id_a", "id_b").rows()}
            cm["decoy_fp_rate"] = len(pp & decoys) / len(decoys) if decoys else 0.0
            _rows(out, ds, algo, "colocation_same_time", method, cm)
        # radius sweep: recall vs. neighbor-decoy false positives
        for method in ("distance", "mahalanobis"):
            for r in COLOC_RADII[prof]:
                pc = colocation.same_time(det, method=method, radius_m=r, min_overlap_s=900)
                cm = colocation_metrics(pc, ct, "same_time")
                pp = {tuple(sorted(p)) for p in pc.select("id_a", "id_b").rows()}
                cm["decoy_fp_rate"] = len(pp & decoys) / len(decoys) if decoys else 0.0
                _rows(out, ds, algo, "colocation_radius", f"{method}@{r}", cm)
        pdt = colocation.diff_time(det, radius_m=radius, window_s=7 * 86400)
        _rows(out, ds, algo, "colocation_diff_time", "distance", colocation_metrics(pdt, ct, "diff_time"))
        print(f"  {ds} {algo} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="eval_*")
    ap.add_argument("--md", type=int, default=900)
    ap.add_argument("--fresh", action="store_true", help="ignore existing results instead of resuming")
    a = ap.parse_args()
    prev = RESULTS / "downstream.parquet"
    out: list = pl.read_parquet(prev).to_dicts() if prev.exists() and not a.fresh else []
    done = {r["dataset"] for r in out}
    for ds in dataset_dirs():
        if ds in done:
            continue  # resume: datasets are written whole after they finish
        if any(fnmatch.fnmatch(ds, x) for x in a.datasets.split(",")) and (dataset_dirs()[ds] / "truth_agents.parquet").exists():
            evaluate_dataset(ds, a.md, out)
            pl.DataFrame(out).write_parquet(RESULTS / "downstream.parquet")
    Path(RESULTS).mkdir(exist_ok=True)
    pl.DataFrame(out).write_parquet(RESULTS / "downstream.parquet")


if __name__ == "__main__":
    main()
