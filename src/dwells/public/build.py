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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=sorted(LOADERS))
    a = ap.parse_args()
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
