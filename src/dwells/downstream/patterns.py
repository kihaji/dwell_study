"""Temporal patterns of dwelling at places: when, how long, how regularly.

``place_profiles`` gives, per (id, place): visit count, total/mean dwell time, events
per dwell, distinct days, median return interval, local hour-of-day and day-of-week
histograms of dwell time (fractions), night / weekday-work-hours dwell seconds and a
regularity score (1 - normalized entropy of the hour histogram; 1 = always the same
hours). ``label_home_work`` infers home (most night time) and work (most weekday
09-17 time at another place on >= 3 distinct days).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _hour_overlap(d: pl.DataFrame, utc_offset_h: float) -> pl.DataFrame:
    """Explode dwells into local clock hours with seconds of overlap per hour."""
    off = pl.duration(hours=utc_offset_h)
    x = d.select("id", "place_id", "dwell_id", (pl.col("start_ts") + off).alias("s"), (pl.col("end_ts") + off).alias("e"))
    x = x.with_columns(h=pl.datetime_ranges(pl.col("s").dt.truncate("1h"), pl.col("e"), "1h")).explode("h")
    x = x.with_columns(
        sec=(pl.min_horizontal(pl.col("e"), pl.col("h") + pl.duration(hours=1))
             - pl.max_horizontal(pl.col("s"), pl.col("h"))).dt.total_seconds().clip(0),
        hour=pl.col("h").dt.hour(), dow=pl.col("h").dt.weekday() - 1,
    )
    return x


def place_profiles(dwells: pl.DataFrame, utc_offset_h: float = 0.0) -> pl.DataFrame:
    d = dwells.sort("id", "start_ts")
    base = d.group_by("id", "place_id").agg(
        pl.len().alias("n_visits"), pl.col("duration_s").sum().alias("total_dwell_s"),
        pl.col("duration_s").mean().alias("mean_dwell_s"), pl.col("n_events").mean().alias("events_per_dwell"),
        (pl.col("start_ts") + pl.duration(hours=utc_offset_h)).dt.date().n_unique().alias("n_days"),
        (pl.col("start_ts").diff().dt.total_seconds() / 3600).median().alias("return_interval_h_med"),
    )
    hrs = _hour_overlap(d, utc_offset_h)
    hh = hrs.group_by("id", "place_id", "hour").agg(pl.col("sec").sum())
    dd = hrs.group_by("id", "place_id", "dow").agg(pl.col("sec").sum())

    def hist(df, key, n):
        full = df.pivot(on=key, index=["id", "place_id"], values="sec", sort_columns=True).fill_null(0)
        cols = [str(i) for i in range(n)]
        for c in cols:
            if c not in full.columns:
                full = full.with_columns(pl.lit(0.0).alias(c))
        tot = pl.sum_horizontal(cols)
        return full.select("id", "place_id", pl.concat_list([pl.col(c) / tot for c in cols]).alias(f"{key}_hist"))

    night = hrs.filter((pl.col("hour") >= 22) | (pl.col("hour") < 6)).group_by("id", "place_id").agg(pl.col("sec").sum().alias("night_s"))
    work = hrs.filter((pl.col("dow") < 5) & pl.col("hour").is_between(9, 16)).group_by("id", "place_id").agg(
        pl.col("sec").sum().alias("workhours_s"),
        pl.col("h").dt.date().n_unique().alias("work_days"))
    out = (base.join(hist(hh, "hour", 24), on=["id", "place_id"], how="left")
           .join(hist(dd, "dow", 7), on=["id", "place_id"], how="left")
           .join(night, on=["id", "place_id"], how="left").join(work, on=["id", "place_id"], how="left")
           .with_columns(pl.col("night_s", "workhours_s", "work_days").fill_null(0)))
    ent = [(-np.sum([p * np.log(p) for p in h if p > 0]) / np.log(24)) if h is not None else None
           for h in out["hour_hist"].to_list()]
    return out.with_columns(regularity=1 - pl.Series(ent, dtype=pl.Float64)).sort("id", "place_id")


def label_home_work(profiles: pl.DataFrame, min_work_days: int = 3) -> pl.DataFrame:
    home = profiles.filter(pl.col("night_s") > 0).sort("night_s", descending=True).group_by("id").first().select(
        "id", "place_id", pl.lit("home").alias("label"))
    work = (profiles.join(home.select("id", pl.col("place_id").alias("home_pid")), on="id", how="left")
            .filter((pl.col("place_id") != pl.col("home_pid")) | pl.col("home_pid").is_null())
            .filter(pl.col("work_days") >= min_work_days)
            .sort("workhours_s", descending=True).group_by("id").first()
            .select("id", "place_id", pl.lit("work").alias("label")))
    return pl.concat([home, work]).sort("id", "label")
