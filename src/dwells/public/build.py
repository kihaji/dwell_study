"""Download and convert the public STAGA dataset (labeled stops) under data/public/.

    uv run python -m dwells.public.build

Raw downloads go to data/public/raw/staga/ (skipped if present); output to
data/public/staga/ (see dwells.public). Other public loaders (GeoLife, PoL benchmark,
Foursquare) live on the main branch.
"""

from __future__ import annotations

import time

from dwells.public import staga


def main():
    t = time.time()
    m = staga.ingest()
    truth = {k: v for k, v in m.items() if k.startswith("truth_")}
    print(f"{m['name']:16s} events={m['stats']['events']:>11,} ids={m['stats']['ids']:>6,} {truth} "
          f"({time.time() - t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
