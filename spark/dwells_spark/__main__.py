"""CLI: python -m dwells_spark --input cleaned.parquet --out results/<upload> [--profile mixed]"""

import argparse
import json
import os

from dwells.presets import PRESETS, params_for
from dwells_spark import job


def main():
    ap = argparse.ArgumentParser(prog="python -m dwells_spark", description=__doc__)
    ap.add_argument("--input", required=True, help="cleaned events parquet (file or directory)")
    ap.add_argument("--out", required=True, help="output directory (shared filesystem)")
    ap.add_argument("--profile", default="mixed", choices=sorted(PRESETS))
    ap.add_argument("--min-duration", type=float, default=900, help="dwell definition, seconds")
    ap.add_argument("--geocode-index", help="offline OSM geocode index parquet")
    ap.add_argument("--dataset", help="name recorded in meta.json")
    ap.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    a = ap.parse_args()
    spark = job.session(a.master)
    meta = job.run(spark, a.input, a.out, params_for(a.profile, a.min_duration), a.geocode_index,
                   meta={"dataset": a.dataset, "profile": a.profile})
    print(json.dumps(meta["rows"]), f"{meta['seconds']}s")
    spark.stop()


if __name__ == "__main__":
    main()
