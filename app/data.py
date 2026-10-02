"""Data access for the app: precomputed pipeline outputs plus live re-runs for tuning."""

from __future__ import annotations

import json
from functools import lru_cache

import polars as pl

from dwells import io
from dwells.pol import pipeline
from dwells.pol.build import OUT, geocoder_for

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
           "#fabebe", "#008080"]
OTHER = "#9e9e9e"


def datasets() -> list[str]:
    return sorted(p.name for p in OUT.iterdir() if (p / "places.parquet").exists()) if OUT.exists() else []


@lru_cache(maxsize=4)
def load(dataset: str) -> dict[str, pl.DataFrame]:
    return pipeline.read(OUT / dataset)


@lru_cache(maxsize=4)
def meta(dataset: str) -> dict:
    return json.loads((OUT / dataset / "meta.json").read_text())


def ids(dataset: str) -> list[dict]:
    q = load(dataset)["quality"].sort("visits", descending=True)
    return [{"label": f"{r['id']}  ({r['visits']} visits, {r['places']} places)", "value": r["id"]}
            for r in q.iter_rows(named=True)]


_live: dict[str, dict[str, pl.DataFrame]] = {}


def live_key(dataset: str, uid: str, params: dict) -> str:
    return json.dumps([dataset, uid, params], sort_keys=True)


def run_live(dataset: str, uid: str, params: dict) -> str:
    """Re-run the pipeline for one id with custom parameters; returns a cache key."""
    key = live_key(dataset, uid, params)
    if key not in _live:
        m = meta(dataset)
        ev = io.load_events(io.dataset_dirs()[dataset], m["shape"], ids=[uid])
        _live[key] = pipeline.analyze(ev, params=params, geocoder=geocoder_for(dataset))
        if len(_live) > 16:
            _live.pop(next(iter(_live)))
    return key


def tables(dataset: str, uid: str, live: str | None = None) -> dict[str, pl.DataFrame]:
    """All tables filtered to one id (from a live re-run when ``live`` is a cache key)."""
    src = _live.get(live) if live else None
    src = src or load(dataset)
    out = {}
    for k, v in src.items():
        if k == "colocations":
            out[k] = v.filter((pl.col("id_a") == uid) | (pl.col("id_b") == uid))
        elif "id" in v.columns:
            out[k] = v.filter(pl.col("id") == uid)
        else:
            out[k] = v
    return out


def place_colors(places: pl.DataFrame) -> dict[int, str]:
    return {r["place_id"]: (PALETTE[r["rank_duration"] - 1] if r["rank_duration"] <= len(PALETTE) else OTHER)
            for r in places.iter_rows(named=True)}


def place_label(r: dict) -> str:
    name = r.get("place_name") or r.get("address")
    tag = {"primary": "Primary anchor", "secondary": "Secondary anchor"}.get(r.get("label") or "", "")
    base = f"P{r['rank_duration']}"
    return " · ".join(x for x in (base, tag, name) if x)


def hex_rgb(h: str, a: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


