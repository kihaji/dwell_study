"""Foursquare NYC check-ins (TSMC2014, Yang et al. 2015), made adtech-like.

Check-in coordinates are *venue* positions. Each check-in becomes one event whose
reported point is the venue perturbed by synthetic heterogeneous error: per ping a
location source gps/wifi/cell is drawn with the generator's adtech mixture weights
(ip/stale excluded) and lognormal 1-sigmas, an ellipse is drawn as in
``generator.sampling``, and the offset is sampled from it. ``true_lat/true_lon`` keep
the venue. ``utcTimestamp`` is already UTC (``+0000``), so ``timezoneOffset`` (local
offset, minutes) is not needed and is dropped.
"""

from __future__ import annotations

import io as _io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from dwells import geo
from dwells.generator import sampling
from dwells.public import OUT, RAW, download, p_mixed, write_dataset

NAME = "foursquare_nyc"
URL = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
MEMBER = "dataset_tsmc2014/dataset_TSMC2014_NYC.txt"
COLS = ["userId", "venueId", "venueCategoryId", "venueCategory", "lat", "lon", "timezoneOffset", "utcTimestamp"]
KINDS, WEIGHTS = np.array(["gps", "wifi", "cell"]), np.array([0.45, 0.30, 0.17]) / 0.92


def parse(data: bytes) -> pl.DataFrame:
    df = pl.read_csv(_io.StringIO(data.decode("latin-1")), separator="\t", has_header=False,
                     new_columns=COLS, quote_char=None, schema_overrides={"userId": pl.String})
    return df.select(
        pl.col("userId").alias("id"), pl.col("venueId").alias("venue_id"),
        pl.col("utcTimestamp").str.to_datetime("%a %b %d %H:%M:%S %z %Y", time_zone="UTC")
        .dt.cast_time_unit("us").alias("timestamp"),
        pl.col("lat").cast(pl.Float64), pl.col("lon").cast(pl.Float64),
        pl.col("venueCategory").alias("category"),
    ).sort("id", "timestamp")


def to_dataset(raw: pl.DataFrame, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    n = raw.height
    kind = rng.choice(KINDS, size=n, p=WEIGHTS)
    sig = np.select([kind == "gps", kind == "wifi"],
                    [rng.lognormal(np.log(8), 0.4, n), rng.lognormal(np.log(35), 0.4, n)],
                    rng.lognormal(np.log(900), 0.5, n))
    sa, sb, az = sampling._rand_axes(rng, sig, np.where(kind == "cell", 3.0, 1.5))
    ex, ey = sampling._draw(rng, sa, sb, az)
    tlat, tlon = raw["lat"].to_numpy(), raw["lon"].to_numpy()
    lat, lon = geo.from_enu(ex, ey, tlat, tlon)
    ev = raw.select("id", "timestamp").with_columns(
        lat=lat, lon=lon, true_lat=tlat, true_lon=tlon, source=pl.Series(kind.astype(str)),
        sig_a=np.maximum(sa, sb), sig_b=np.minimum(sa, sb), az=az % 180, p_mixed=p_mixed(rng, n),
        truth_dwell_id=pl.lit(None, dtype=pl.Int64))
    return ev, raw.select("id", "venue_id", "timestamp", "lat", "lon", "category")


def ingest(raw: Path = RAW / "foursquare", out_root: Path = OUT) -> dict:
    with zipfile.ZipFile(download(URL, raw / "dataset_tsmc2014.zip")) as z:
        ev, visits = to_dataset(parse(z.read(MEMBER)))
    meta = {
        "name": NAME, "source_url": URL, "landing": "https://sites.google.com/site/yangdingqi/home/foursquare-dataset",
        "license": "research use; cite Yang et al., IEEE TSMC: Systems 45(1), 2015",
        "transformations": [
            "NYC file only; utcTimestamp (+0000) parsed as UTC; id = str(userId)",
            "true_lat/true_lon = venue coordinates; lat/lon = venue + synthetic error (seed 0)",
            ("per-ping source gps/wifi/cell with p = 0.45/0.30/0.17 renormalized; 1-sigma lognormal "
             "median 8 / 35 / 900 m (s = 0.4 / 0.4 / 0.5); axis ratio U(1, 1.5) (cell U(1, 3)); "
             "offset drawn from the reported ellipse (honest error, no calibration noise)"),
            "truth_visits = raw check-ins (venue position, category); no truth dwells",
        ],
    }
    return write_dataset(out_root / NAME, ev, meta, visits=visits)
