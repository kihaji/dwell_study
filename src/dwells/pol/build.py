"""Batch: run the pattern-of-life pipeline on repo datasets and write output tables.

    uv run python -m dwells.pol.build [--datasets "eval_*,dev_*"] [--shape 3_mixed] [--profile auto]

Writes results/pol/<dataset>/{events,visits,places,transitions,next_places,presence,
anomalies,quality,colocations}.parquet + meta.json. Places are reverse-geocoded with the
offline OSM index when one exists for the dataset's study area.
On WSL2 run under a memory cap: systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 ...
"""

from __future__ import annotations

import argparse
import fnmatch
import time
from pathlib import Path

from dwells import io
from dwells.generator.area import DATA_DIR
from dwells.pol import coloc, pipeline
from dwells.pol.geocode import OfflineOSMGeocoder
from dwells.presets import params_for

OUT = Path(__file__).resolve().parents[3] / "results" / "pol"


def geocoder_for(dataset: str):
    for area in ("urban", "suburban"):
        p = DATA_DIR / "osm" / area / "geocode.parquet"
        if f"_{area}_" in dataset and p.exists():
            return OfflineOSMGeocoder([p])
    return None


def build(dataset: str, shape: str = "3_mixed", profile: str = "auto", min_duration_s: float = 900) -> dict:
    t = time.time()
    prof = io.profile_of(dataset) if profile == "auto" else profile
    ev = io.load_events(io.dataset_dirs()[dataset], shape)
    params = params_for(prof, min_duration_s)
    tables = pipeline.analyze(ev, params=params, geocoder=geocoder_for(dataset))
    tables["colocations"] = coloc.colocations(tables["visits"])
    meta = {"dataset": dataset, "shape": shape, "profile": prof, "params": params,
            "rows": {k: v.height for k, v in tables.items()}, "seconds": round(time.time() - t, 1)}
    pipeline.write(tables, OUT / dataset, meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="eval_*,dev_*,staga")
    ap.add_argument("--shape", default="3_mixed")
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--md", type=float, default=900)
    a = ap.parse_args()
    for ds in io.dataset_dirs():
        if any(fnmatch.fnmatch(ds, p) for p in a.datasets.split(",")):
            m = build(ds, a.shape, a.profile, a.md)
            print(f"{ds:24s} {m['rows']} ({m['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
