"""Tuning and robustness harness for ua_mahalanobis.

    uv run python -m dwells.harness.tune baseline
    uv run python -m dwells.harness.tune sweep --profile gps --grid radius_m=25,50,100 gate_p=0.95,0.99
    uv run python -m dwells.harness.tune robustness

Datasets: ``dev_{area}_{profile}`` for tuning, ``eval_{area}_{profile}`` (different
agents and seeds) for held-out validation; areas = urban, suburban. Scores are the
mean over both areas. Results are appended to results/ua/<stage>.parquet.
Run heavy stages under a kernel memory cap on WSL2, e.g.
``systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 uv run ...``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import polars as pl

from dwells import io, schema
from dwells.algorithms import ua_mahalanobis as ua
from dwells.harness import perturb
from dwells.metrics import dwell_metrics
from dwells.presets import params_for

RESULTS = Path(__file__).resolve().parents[3] / "results" / "ua"
AREAS = ("urban", "suburban")
PROFILES = ("gps", "cdr", "adtech")
KEEP = ("f1", "precision", "recall", "recall_observable", "f1_lenient", "start_err_s_med", "end_err_s_med",
        "duration_bias_s_med", "centroid_err_m_med", "fragmentation", "merging", "time_precision", "time_recall",
        "n_pred", "n_truth")


@lru_cache(maxsize=4)
def _master(ds: str) -> pl.DataFrame:
    return pl.read_parquet(io.dataset_dirs()[ds] / "events.parquet")


@lru_cache(maxsize=4)
def _truth(ds: str) -> pl.DataFrame:
    return io.load_truth(io.dataset_dirs()[ds], "dwells")


def run_one(ds: str, shape: str, params: dict, perturbation: str | None = None, level: float | None = None,
            seed: int = 0, detect=None, with_source: bool = False) -> dict:
    m = _master(ds)
    if perturbation:
        m = perturb.apply(m, perturbation, level, seed=seed)
    ev = schema.validate_events(io.shape_view(m, shape, with_source=with_source))
    t = time.perf_counter()
    d = (detect or ua.detect)(ev, **params)
    rt = time.perf_counter() - t
    met = dwell_metrics(schema.validate_dwells(d), _truth(ds), min_duration_s=params["min_duration_s"])
    return {k: met.get(k) for k in KEEP} | {"runtime_s": rt, "n_events": ev.height}


def evaluate(profile: str, split: str, shape: str, params: dict, **kw) -> dict:
    rows = [run_one(f"{split}_{a}_{profile}", shape, params, **kw) for a in AREAS]
    out = {k: (sum(r[k] for r in rows if r[k] is not None) / max(1, sum(r[k] is not None for r in rows)))
           for k in rows[0]}
    return out


def _task(job: dict) -> dict:
    res = evaluate(job["profile"], job["split"], job["shape"], job["params"],
                   perturbation=job.get("perturbation"), level=job.get("level"),
                   with_source=job.get("with_source", False))
    return {**{k: v for k, v in job.items() if k != "params"}, "params": json.dumps(job["params"], sort_keys=True)} | res


def run_jobs(jobs: list[dict], stage: str, workers: int = 8) -> pl.DataFrame:
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_task, jobs))
    df = pl.DataFrame(rows, infer_schema_length=None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{stage}.parquet"
    if out.exists():
        df = pl.concat([pl.read_parquet(out), df], how="diagonal_relaxed")
    tmp = out.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    tmp.replace(out)
    return df


def _num(x: str):
    try:
        return float(x) if "." in x else int(x)
    except ValueError:
        return x


def _parse_grid(items: list[str]) -> dict[str, list]:
    grid = {}
    for it in items:
        k, v = it.split("=")
        grid[k] = [_num(x) for x in v.split(",")]
    return grid


ROBUSTNESS = {
    "error_scale": [0.5, 0.78, 1.28, 2.08, 4.0],
    "error_noise": [0.25, 0.5, 1.0],
    "duplicates": [0.1, 0.5, 1.0],
    "stale": [0.05, 0.15, 0.3],
    "outliers": [0.01, 0.05, 0.1],
    "clock_jitter": [10, 60, 300],
    "thin": [0.5, 0.25, 0.1],
    "truncate": [4, 3],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "sweep", "robustness"])
    ap.add_argument("--profile", default="all")
    ap.add_argument("--split", default=None, help="dev | eval (default: dev for sweep, eval otherwise)")
    ap.add_argument("--shape", default="3_mixed")
    ap.add_argument("--md", type=int, default=900)
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--preset", default=None, help="use this preset for every profile (e.g. mixed)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    profiles = PROFILES if a.profile == "all" else a.profile.split(",")
    split = a.split or ("dev" if a.stage == "sweep" else "eval")
    jobs = []
    for prof in profiles:
        base = params_for(a.preset or prof, a.md)
        if a.stage == "baseline":
            for shape in io.SHAPES:
                jobs.append({"profile": prof, "split": split, "shape": shape, "params": base, "tag": a.tag,
                             "preset": a.preset or prof})
        elif a.stage == "sweep":
            grid = _parse_grid(a.grid)
            for combo in itertools.product(*grid.values()):
                jobs.append({"profile": prof, "split": split, "shape": a.shape, "tag": a.tag,
                             "params": base | dict(zip(grid, combo))})
        else:
            jobs.append({"profile": prof, "split": split, "shape": a.shape, "params": base, "tag": a.tag,
                         "perturbation": None, "level": None})
            for name, levels in ROBUSTNESS.items():
                for lv in levels:
                    jobs.append({"profile": prof, "split": split, "shape": a.shape, "params": base, "tag": a.tag,
                                 "perturbation": name, "level": float(lv)})
    df = run_jobs(jobs, a.stage, a.workers)
    print(df.tail(len(jobs)).select([c for c in ("profile", "shape", "perturbation", "level", "params", "f1",
                                                 "precision", "recall", "start_err_s_med", "end_err_s_med")
                                     if c in df.columns]))


if __name__ == "__main__":
    main()
