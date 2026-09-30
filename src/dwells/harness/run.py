"""Accuracy harness: parameter sweeps on dev datasets, tuned evaluation on eval datasets.

    uv run python -m dwells.harness.run sweep [--algos a,b] [--datasets glob] [--workers N]
    uv run python -m dwells.harness.run eval  [--algos a,b] [--datasets glob] [--workers N]

sweep: every (algorithm, dev dataset, shape, min_duration, radius, max_gap) -> results/sweep.parquet
eval:  for each (algorithm, profile, shape, min_duration) the (radius, max_gap) with the
       best dev F1 is run on the matching eval datasets -> results/eval.parquet and
       predicted dwells under results/dwells/<dataset>/.
Algorithms with USES_UNCERTAINTY = False see identical lat/lon in every shape, so
they run on shape "1" only and their results stand for all shapes.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import polars as pl

from dwells import algorithms, io, schema
from dwells.generator.area import DATA_DIR
from dwells.metrics import dwell_metrics

RESULTS = Path(__file__).resolve().parents[3] / "results"
MIN_DURATIONS = (300, 900, 1800, 3600)
GRID = {  # per source profile: radius_m x max_gap_s tuning grid
    "gps": {"radius_m": (50, 100, 250), "max_gap_s": (1800, 7200)},
    "cdr": {"radius_m": (250, 500, 1000, 2000), "max_gap_s": (7200, 21600, 43200)},
    "adtech": {"radius_m": (100, 250, 500, 1000), "max_gap_s": (7200, 21600, 43200)},
}
TIMEOUT_S = int(os.environ.get("DWELLS_TIMEOUT_S", 1800))


def dataset_dirs() -> dict[str, Path]:
    out = {}
    for root in (DATA_DIR / "synthetic", DATA_DIR / "public"):
        if root.exists():
            out |= {p.name: p for p in sorted(root.iterdir()) if (p / "events.parquet").exists()}
    return out


def profile_of(name: str) -> str:
    for p in ("gps", "cdr", "adtech"):
        if f"_{p}" in name:
            return p
    return "adtech" if "foursquare" in name else "gps"


@lru_cache(maxsize=8)
def _events(ds: str, shape: str) -> pl.DataFrame:
    return io.load_events(dataset_dirs()[ds], shape)


@lru_cache(maxsize=8)
def _truth(ds: str) -> pl.DataFrame | None:
    p = dataset_dirs()[ds] / "truth_dwells.parquet"
    return pl.read_parquet(p) if p.exists() else None


def task_key(t: dict) -> str:
    return hashlib.sha1(json.dumps(t, sort_keys=True).encode()).hexdigest()[:16]


def _alarm(signum, frame):
    raise TimeoutError


def run_task(task: dict, save_dir: Path | None = None) -> dict:
    row = dict(task) | {"key": task_key(task), "params": json.dumps(task["params"], sort_keys=True)}
    try:
        algo = algorithms.get(task["algo"])
        ev = _events(task["dataset"], task["shape"])
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(TIMEOUT_S)
        t = time.perf_counter()
        d = schema.validate_dwells(algo.detect(ev, **task["params"]))
        row["runtime_s"] = time.perf_counter() - t
        signal.alarm(0)
        row["n_events"] = ev.height
        truth = _truth(task["dataset"])
        if truth is not None:
            row |= dwell_metrics(d, truth, min_duration_s=task["params"]["min_duration_s"])
        if save_dir is not None:
            p = save_dir / task["dataset"]
            p.mkdir(parents=True, exist_ok=True)
            d.write_parquet(p / f"{task['algo']}__{task['shape']}__{task['params']['min_duration_s']}.parquet")
        row["error"] = None
    except TimeoutError:
        row["error"] = f"timeout>{TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001 - record and continue
        row["error"] = f"{type(e).__name__}: {e}"[:500]
        traceback.print_exc()
    finally:
        signal.alarm(0)
    row.pop("params_obj", None)
    return row | {"params": row["params"]}


def _shapes(algo_name: str) -> list[str]:
    return list(io.SHAPES) if algorithms.get(algo_name).USES_UNCERTAINTY else ["1"]


def sweep_tasks(algos, datasets) -> list[dict]:
    tasks = []
    for a, ds in itertools.product(algos, datasets):
        g = GRID[profile_of(ds)]
        for shape in _shapes(a):
            for md, r, gap in itertools.product(MIN_DURATIONS, g["radius_m"], g["max_gap_s"]):
                tasks.append({"algo": a, "dataset": ds, "shape": shape,
                              "params": {"min_duration_s": md, "radius_m": r, "max_gap_s": gap}})
    return tasks


def best_params(sweep: pl.DataFrame) -> pl.DataFrame:
    s = sweep.filter(pl.col("error").is_null()).with_columns(
        profile=pl.col("dataset").map_elements(profile_of, return_dtype=pl.String),
        md=pl.col("params").str.json_path_match("$.min_duration_s").cast(pl.Int64),
    )
    # average F1 over the dev datasets of a profile (both areas), then pick best params
    g = s.group_by("algo", "profile", "shape", "md", "params").agg(pl.col("f1").mean())
    return g.sort("f1", descending=True).group_by("algo", "profile", "shape", "md").first()


def eval_tasks(algos, datasets, sweep: pl.DataFrame) -> list[dict]:
    bp = best_params(sweep)
    tasks = []
    for ds in datasets:
        prof = profile_of(ds)
        for r in bp.filter(pl.col("algo").is_in(algos) & (pl.col("profile") == prof)).iter_rows(named=True):
            tasks.append({"algo": r["algo"], "dataset": ds, "shape": r["shape"], "params": json.loads(r["params"])})
    return tasks


def execute(tasks: list[dict], out: Path, workers: int, save_dir: Path | None = None) -> pl.DataFrame:
    done = pl.read_parquet(out) if out.exists() else None
    if done is not None:
        keys = set(done.filter(pl.col("error").is_null())["key"].to_list())
        done = done.filter(pl.col("key").is_in(keys))
        tasks = [t for t in tasks if task_key(t) not in keys]
    print(f"{len(tasks)} tasks -> {out}", flush=True)
    rows = []
    # group tasks by dataset so each worker's cache is reused
    tasks.sort(key=lambda t: (t["dataset"], t["shape"], t["algo"]))
    # spawn, not fork: polars' thread pool (already used by the parent) deadlocks forked children
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(run_task, t, save_dir) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
                _flush(done, rows, out)
    return _flush(done, rows, out)


def _flush(done, rows, out: Path) -> pl.DataFrame:
    new = pl.DataFrame(rows, infer_schema_length=None) if rows else None
    parts = [x for x in (done, new) if x is not None and x.height]
    df = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "eval"])
    ap.add_argument("--algos", default="all")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    algos = algorithms.names() if a.algos == "all" else a.algos.split(",")
    pat = a.datasets or ("dev_*" if a.stage == "sweep" else "eval_*")
    datasets = [d for d in dataset_dirs() if fnmatch.fnmatch(d, pat)]
    if a.stage == "sweep":
        execute(sweep_tasks(algos, datasets), RESULTS / "sweep.parquet", a.workers)
    else:
        sweep = pl.read_parquet(RESULTS / "sweep.parquet")
        execute(eval_tasks(algos, datasets, sweep), RESULTS / "eval.parquet", a.workers, RESULTS / "dwells")


if __name__ == "__main__":
    main()
