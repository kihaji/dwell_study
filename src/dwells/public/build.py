"""Download and convert the public datasets under data/public/.

    uv run python -m dwells.public.build [--only NAME]

Raw downloads go to data/public/raw/<name>/ (skipped if present); outputs to
data/public/<dataset>/ (see dwells.public).
"""

from __future__ import annotations

import argparse
import time

from dwells.public import foursquare, geolife, pol, staga

LOADERS = {"staga": staga.ingest, "pol": pol.ingest, "foursquare": foursquare.ingest, "geolife": geolife.ingest}


def subset(src: str, dst: str, n_ids: int) -> None:
    """Copy a dataset keeping only its first n_ids ids (for evaluating slow algorithms)."""
    import json

    import polars as pl

    from dwells.generator.area import DATA_DIR

    s, d = DATA_DIR / "public" / src, DATA_DIR / "public" / dst
    d.mkdir(parents=True, exist_ok=True)
    ev = pl.scan_parquet(s / "events.parquet")
    ids = ev.select("id").unique().sort("id").head(n_ids).collect()["id"].implode()
    ev.filter(pl.col("id").is_in(ids)).collect().write_parquet(d / "events.parquet")
    for f in s.glob("truth_*.parquet"):
        pl.read_parquet(f).filter(pl.col("id").is_in(ids)).write_parquet(d / f.name)
    from dwells.generator.sampling import describe

    meta = json.loads((s / "meta.json").read_text()) if (s / "meta.json").exists() else {}
    sub = pl.read_parquet(d / "events.parquet")
    meta |= {"name": dst, "subset_of": src, "n_ids": n_ids, "stats": describe(sub).to_dicts()[0]}
    for f in d.glob("truth_*.parquet"):
        meta[f.stem] = pl.scan_parquet(f).select(pl.len()).collect().item()
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=sorted(LOADERS))
    ap.add_argument("--subset", nargs=3, metavar=("SRC", "DST", "N_IDS"), help="write an id subset and exit")
    a = ap.parse_args()
    if a.subset:
        subset(a.subset[0], a.subset[1], int(a.subset[2]))
        return
    for name, fn in LOADERS.items():
        if a.only and a.only != name:
            continue
        t = time.time()
        metas = fn()
        for m in metas if isinstance(metas, list) else [metas]:
            truth = {k: v for k, v in m.items() if k.startswith("truth_")}
            print(f"{m['name']:16s} events={m['stats']['events']:>11,} ids={m['stats']['ids']:>6,} {truth} "
                  f"({time.time() - t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
