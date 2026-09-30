"""Scaling benchmark: runtime and peak memory vs. events, one isolated process per run.

    uv run python -m dwells.harness.bench [--algos a,b] [--ladders gps,adtech,single] [--timeout 900]

Ladders (suburban synthetic data, see generator/build.py):
  gps     scale_gps_{1,8,80,800}         ~10k -> 9.4M events, dense per id (many events/id)
  adtech  scale_adtech_{20,...,18000}    ~20k -> 16.5M events, sparse (many ids)
  single  scale_gps_single_1m            one id with ~1M events (per-id scaling)
Runs execute sequentially (clean timings). An algorithm stops climbing a ladder after
a timeout or failure. Results -> results/bench.parquet. Detection only; loading
time and the RSS after loading are reported separately.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import polars as pl

from dwells import algorithms, io
from dwells.harness.run import RESULTS, dataset_dirs

LADDERS = {
    "gps": ["scale_gps_1", "scale_gps_8", "scale_gps_80", "scale_gps_800"],
    "adtech": ["scale_adtech_20", "scale_adtech_200", "scale_adtech_2000", "scale_adtech_18000"],
    "single": ["scale_gps_single_1m"],
}
PARAMS = {
    "gps": {"min_duration_s": 900, "radius_m": 100, "max_gap_s": 7200},
    "single": {"min_duration_s": 900, "radius_m": 100, "max_gap_s": 7200},
    "adtech": {"min_duration_s": 900, "radius_m": 250, "max_gap_s": 21600},
}


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def one(algo: str, dataset: str, shape: str, params: dict) -> dict:
    t = time.perf_counter()
    ev = io.load_events(dataset_dirs()[dataset], shape)
    load_s = time.perf_counter() - t
    rss_loaded = _rss_mb()
    mod = algorithms.get(algo)
    mod.detect(ev.head(500), **params)  # warm-up: JIT compilation / lazy imports excluded
    t = time.perf_counter()
    d = mod.detect(ev, **params)
    return {"load_s": load_s, "runtime_s": time.perf_counter() - t, "rss_loaded_mb": rss_loaded,
            "peak_rss_mb": _rss_mb(), "n_events": ev.height, "n_ids": ev["id"].n_unique(), "n_dwells": d.height}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="all")
    ap.add_argument("--ladders", default="gps,adtech,single")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--one", nargs=4, help=argparse.SUPPRESS)  # algo dataset shape params_json (child mode)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(a.one[0], a.one[1], a.one[2], json.loads(a.one[3]))))
        return
    algos = algorithms.names() if a.algos == "all" else a.algos.split(",")
    out = RESULTS / "bench.parquet"
    rows = pl.read_parquet(out).to_dicts() if out.exists() else []
    done = {(r["algo"], r["dataset"]) for r in rows}
    for ladder in a.ladders.split(","):
        for algo in algos:
            shape = "2" if algorithms.get(algo).USES_UNCERTAINTY else "1"
            for ds in LADDERS[ladder]:
                if (algo, ds) in done:
                    if any(r["algo"] == algo and r["dataset"] == ds and r["error"] for r in rows):
                        break
                    continue
                cmd = [sys.executable, "-W", "ignore", "-m", "dwells.harness.bench", "--one", algo, ds, shape,
                       json.dumps(PARAMS[ladder])]
                row = {"algo": algo, "ladder": ladder, "dataset": ds, "shape": shape, "error": None}
                t = time.perf_counter()
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                    if p.returncode == 0:
                        row |= json.loads(p.stdout.strip().splitlines()[-1])
                    else:
                        row["error"] = (p.stderr.strip().splitlines() or ["?"])[-1][:300]
                except subprocess.TimeoutExpired:
                    row["error"] = f"timeout>{a.timeout}s"
                row["wall_s"] = time.perf_counter() - t
                rows.append(row)
                pl.DataFrame(rows, infer_schema_length=None).write_parquet(out)
                print(f"{algo:20s} {ds:22s} {row.get('runtime_s', float('nan')):9.2f}s "
                      f"rss={row.get('peak_rss_mb', float('nan')):8.0f}MB {row['error'] or ''}", flush=True)
                if row["error"]:
                    break


if __name__ == "__main__":
    main()
