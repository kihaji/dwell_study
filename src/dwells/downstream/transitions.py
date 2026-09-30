"""Movement after a dwell: transitions between an id's places.

``from_dwells`` pairs each dwell with the next one of the same id (TRANSITIONS
schema; self-transitions = a place revisited after a split or an undetected stop
elsewhere). ``matrix`` gives per-id transition counts/probabilities with travel-time
and departure-hour summaries, answering "where does it go after dwelling at X, and
when". ``sequences`` counts length-k place sequences (A->B->C routines).
"""

from __future__ import annotations

import polars as pl

from dwells import schema


def from_dwells(dwells: pl.DataFrame) -> pl.DataFrame:
    d = dwells.sort("id", "start_ts")
    tr = d.select(
        "id",
        pl.col("place_id").alias("from_place"),
        pl.col("place_id").shift(-1).over("id").alias("to_place"),
        pl.col("end_ts").alias("depart_ts"),
        pl.col("start_ts").shift(-1).over("id").alias("arrive_ts"),
    ).drop_nulls("to_place")
    return tr.select([pl.col(c).cast(t) for c, t in schema.TRANSITIONS.items()])


def matrix(tr: pl.DataFrame, utc_offset_h: float = 0.0, drop_self: bool = False) -> pl.DataFrame:
    if drop_self:
        tr = tr.filter(pl.col("from_place") != pl.col("to_place"))
    m = tr.group_by("id", "from_place", "to_place").agg(
        pl.len().alias("count"),
        (pl.col("arrive_ts") - pl.col("depart_ts")).dt.total_seconds().median().alias("travel_s_med"),
        (pl.col("depart_ts") + pl.duration(hours=utc_offset_h)).dt.hour().median().alias("depart_hour_med"),
    )
    return m.with_columns(prob=pl.col("count") / pl.col("count").sum().over("id", "from_place")).sort(
        "id", "from_place", "count", descending=[False, False, True])


def sequences(dwells: pl.DataFrame, k: int = 3) -> pl.DataFrame:
    d = dwells.sort("id", "start_ts").with_columns(
        pl.col("place_id").shift(-i).over("id").alias(f"p{i}") for i in range(k))
    cols = [f"p{i}" for i in range(k)]
    return d.drop_nulls(cols).group_by(["id"] + cols).len("count").sort("count", descending=True)
