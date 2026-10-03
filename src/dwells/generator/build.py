"""Build the synthetic dataset catalog under data/synthetic/.

    uv run python -m dwells.generator.build [--only NAME_SUBSTRING] [--list]

Each dataset directory contains:
  events.parquet         master events (all shapes derivable, see io.shape_view)
  truth_dwells.parquet   true stays incl. n_events observed during each
  truth_places.parquet   places visited (global place ids)
  truth_agents.parquet   agent personas/households/home/work
  truth_meetings.parquet scheduled friend meetings
  meta.json              generation config + summary stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import polars as pl

from dwells.generator import sampling
from dwells.generator.area import DATA_DIR, load_area
from dwells.generator.simulate import SimConfig, simulate

UTC_OFFSET = {"urban": -6.0, "suburban": -5.0}


@dataclass
class Spec:
    name: str
    area: str
    profile: str
    n_agents: int
    n_days: int
    seed: int = 0
    gps_opts: dict = field(default_factory=dict)
    group: str = "eval"


def catalog() -> list[Spec]:
    specs = []
    for area in ("urban", "suburban"):
        for prof in sampling.PROFILES:
            specs.append(Spec(f"dev_{area}_{prof}", area, prof, 20, 7, seed=11, group="dev"))
            specs.append(Spec(f"eval_{area}_{prof}", area, prof, 100, 21, seed=1, group="eval"))
    return specs


def build(spec: Spec, out_root=None) -> dict:
    out = (out_root or DATA_DIR / "synthetic") / spec.name
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    area = load_area(spec.area)
    cfg = SimConfig(n_agents=spec.n_agents, n_days=spec.n_days, seed=spec.seed,
                    utc_offset_h=UTC_OFFSET[spec.area])
    sim = simulate(area, cfg)
    ev = sampling.sample(sim, area, spec.profile, seed=spec.seed, gps_opts=spec.gps_opts)
    ev.write_parquet(out / "events.parquet")

    counts = ev.filter(pl.col("truth_dwell_id").is_not_null()).group_by(
        "id", pl.col("truth_dwell_id").alias("dwell_id")).len("n_events")
    places = area.places.select("place_id", "category", "lat", "lon", "radius_m")
    us = lambda c: (pl.col(c) * 1e6).cast(pl.Int64).cast(pl.Datetime("us", "UTC"))  # noqa: E731
    dwells = (
        sim.stays.join(places, on="place_id", how="left")
        .join(counts, on=["id", "dwell_id"], how="left")
        .with_columns(start_ts=us("arrive"), end_ts=us("depart"),
                      duration_s=(pl.col("depart") - pl.col("arrive")).cast(pl.Int64),
                      n_events=pl.col("n_events").fill_null(0).cast(pl.Int64))
        .select("id", "dwell_id", "place_id", "category", "start_ts", "end_ts", "duration_s",
                "lat", "lon", "radius_m", "n_events")
    )
    dwells.write_parquet(out / "truth_dwells.parquet")
    places.filter(pl.col("place_id").is_in(dwells["place_id"].unique().implode())).write_parquet(
        out / "truth_places.parquet")
    sim.agents.write_parquet(out / "truth_agents.parquet")
    sim.meetings.with_columns(start=us("start"), end=us("end")).write_parquet(out / "truth_meetings.parquet")

    stats = sampling.describe(ev).to_dicts()[0]
    meta = {"spec": asdict(spec), "sim": asdict(cfg), "stats": stats,
            "truth_dwells": dwells.height,
            "events_sha1": hashlib.sha1(ev.select("id", "timestamp", "lat", "lon").write_ipc(None).getvalue()).hexdigest(),
            "build_seconds": round(time.time() - t, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on dataset name")
    ap.add_argument("--group", default=None, help="dev | eval")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for s in catalog():
        if (a.only and a.only not in s.name) or (a.group and s.group != a.group):
            continue
        if a.list:
            print(s.name, s.area, s.profile, s.n_agents, s.n_days)
            continue
        m = build(s)
        print(f"{s.name:28s} events={m['stats']['events']:>10,} truth_dwells={m['truth_dwells']:>7,} "
              f"({m['build_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
