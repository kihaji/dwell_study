"""GeoLife 1.3 (Microsoft Research Asia): 182 users of dense GPS, mostly Beijing, 2007-2012.

``.plt`` date/time columns are GMT (user guide), used as UTC. No stay labels. Rows with
out-of-range coordinates are dropped. Two datasets: ``geolife_top20`` (the 20 users
with the most points) and ``geolife_all``.
"""

from __future__ import annotations

import io as _io
import zipfile
from pathlib import Path

import polars as pl

from dwells.public import OUT, RAW, download, nominal_error, write_dataset

URL = ("https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/"
       "Geolife%20Trajectories%201.3.zip")
COLS = ["lat", "lon", "zero", "alt_ft", "days", "date", "time"]


def read_zip(path: Path) -> pl.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.endswith(".plt"):
                continue
            uid = n.split("/")[-3]
            df = pl.read_csv(_io.BytesIO(z.read(n)), skip_rows=6, has_header=False, new_columns=COLS,
                             infer_schema=False)
            frames.append(df.select(pl.lit(uid).alias("id"), "lat", "lon", "date", "time"))
    ev = pl.concat(frames)
    return ev.select(
        "id", (pl.col("date") + " " + pl.col("time")).str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
        .dt.cast_time_unit("us").alias("timestamp"),
        pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64),
    ).sort("id", "timestamp")


def ingest(raw: Path = RAW / "geolife", out_root: Path = OUT) -> list[dict]:
    ev = read_zip(download(URL, raw / "geolife13.zip"))
    ok = pl.col("lat").is_between(-90, 90) & pl.col("lon").is_between(-180, 180)
    dropped = int(ev.select((~ok).sum()).item())
    ev = ev.filter(ok)
    top = ev.group_by("id").len().sort("len", "id", descending=[True, False]).head(20)["id"]
    base = {
        "source_url": URL, "landing": "https://www.microsoft.com/en-us/download/details.aspx?id=52367",
        "license": "Microsoft Research terms: free for research; cite Zheng et al. WWW 2008, WWW 2009, "
                   "IEEE Data Eng. Bull. 2010",
        "transformations": [
            ".plt date + time (GMT per user guide) used as UTC; id = user folder (000-181)",
            f"dropped {dropped} rows with lat/lon out of range",
            "no accuracy in source: nominal circular 1-sigma 8 m, positions not perturbed (true = observed)",
            "no truth tables (labels.txt transport modes not used)",
        ],
    }
    metas = []
    for name, sub in (("geolife_top20", ev.filter(pl.col("id").is_in(top.implode()))), ("geolife_all", ev)):
        extra = {"ids": sorted(top.to_list())} if name == "geolife_top20" else {}
        metas.append(write_dataset(out_root / name, nominal_error(sub, "geolife", seed=0),
                                   {"name": name, **base, **extra}))
    return metas
