"""Colocation output (kept separate on purpose: to be deep-dived later).

Builds the COLOCATIONS table (dwells.pol.schema) from the pipeline's visits table so
rows reference visit and place ids. Matching logic lives in
dwells.downstream.colocation (see docs/algorithms/downstream.md, "Colocation"):

* same_time: two ids' visits overlap >= min_overlap_s and their centroids are within
  radius_m (distance matcher).
* diff_time: the later visit starts within window_s after the earlier one ends, within
  radius_m (one row per id pair: the shortest such gap).

Known limits (main-branch evaluation): adjacent buildings closer than the positional
precision cannot be told apart, so next-door neighbors are routinely flagged; treat
same_time pairs as candidates, not conclusions.
"""

from __future__ import annotations

import polars as pl

from dwells import geo
from dwells.downstream import colocation as C
from dwells.pol import schema as S


def colocations(visits: pl.DataFrame, radius_m: float = 100.0, min_overlap_s: float = 900.0,
                window_s: float = 7 * 86400) -> pl.DataFrame:
    if visits.height < 2:
        return S.conform(pl.DataFrame(), S.COLOCATIONS)
    v = visits.select("id", "visit_id", "place_id", "start_ts", "end_ts", "lat", "lon",
                      "sigma_major", "sigma_minor", "azimuth")
    i, j = C._same_time_pairs(v, radius_m, min_overlap_s)
    same = C._join_pairs(v, i, j)
    if same.height:
        same = same.filter(pl.Series(geo.haversine_m(same["lat"].to_numpy(), same["lon"].to_numpy(),
                                                     same["lat_b"].to_numpy(), same["lon_b"].to_numpy()) <= radius_m))
    same = same.with_columns(kind=pl.lit("same_time"),
                             overlap_start=pl.max_horizontal("start_ts", "start_ts_b"),
                             overlap_end=pl.min_horizontal("end_ts", "end_ts_b"))
    diff = C.diff_time(v.select("id", "start_ts", "end_ts", "lat", "lon", "sigma_major", "sigma_minor", "azimuth"),
                       radius_m=radius_m, window_s=window_s)
    # diff_time rows carry ids and the gap interval; recover the visits by matching end/start times
    if diff.height:
        a = v.select(pl.col("id").alias("id_a"), pl.col("end_ts").alias("overlap_start"),
                     pl.col("visit_id").alias("visit_id_a"), pl.col("place_id").alias("place_id_a"),
                     pl.col("lat").alias("lat_a"), pl.col("lon").alias("lon_a"))
        b = v.select(pl.col("id").alias("id_b"), pl.col("start_ts").alias("overlap_end"),
                     pl.col("visit_id").alias("visit_id_b"), pl.col("place_id").alias("place_id_b"),
                     pl.col("lat").alias("lat_b"), pl.col("lon").alias("lon_b"))
        d1 = diff.join(a, on=["id_a", "overlap_start"], how="left").join(b, on=["id_b", "overlap_end"], how="left")
        # the earlier visit may belong to id_b: try the swapped orientation for unmatched rows
        a2 = a.rename({"id_a": "id_b", "visit_id_a": "visit_id_b", "place_id_a": "place_id_b", "lat_a": "lat_b",
                       "lon_a": "lon_b"})
        b2 = b.rename({"id_b": "id_a", "visit_id_b": "visit_id_a", "place_id_b": "place_id_a", "lat_b": "lat_a",
                       "lon_b": "lon_a"})
        d2 = diff.join(a2, on=["id_b", "overlap_start"], how="left").join(b2, on=["id_a", "overlap_end"], how="left")
        diff = pl.concat([d1.filter(pl.col("visit_id_a").is_not_null() & pl.col("visit_id_b").is_not_null()),
                          d2.filter(pl.col("visit_id_a").is_not_null() & pl.col("visit_id_b").is_not_null())],
                         how="diagonal_relaxed").unique(["id_a", "id_b"])
        diff = diff.with_columns(distance_m=pl.Series(geo.haversine_m(
            diff["lat_a"].to_numpy(), diff["lon_a"].to_numpy(), diff["lat_b"].to_numpy(), diff["lon_b"].to_numpy())))
    swap = pl.col("id") > pl.col("id_b")
    same = same.with_columns(
        id_a=pl.when(swap).then(pl.col("id_b")).otherwise(pl.col("id")),
        id_b=pl.when(swap).then(pl.col("id")).otherwise(pl.col("id_b")),
        visit_id_a=pl.when(swap).then(pl.col("visit_id_b")).otherwise(pl.col("visit_id")),
        visit_id_b=pl.when(swap).then(pl.col("visit_id")).otherwise(pl.col("visit_id_b")),
        place_id_a=pl.when(swap).then(pl.col("place_id_b")).otherwise(pl.col("place_id")),
        place_id_b=pl.when(swap).then(pl.col("place_id")).otherwise(pl.col("place_id_b")),
        distance_m=pl.Series(geo.haversine_m(same["lat"].to_numpy(), same["lon"].to_numpy(),
                                             same["lat_b"].to_numpy(), same["lon_b"].to_numpy()))
        if same.height else pl.lit(None, dtype=pl.Float64),
    ) if same.height else same
    out = pl.concat([S.conform(same.with_columns(
        overlap_s=(pl.col("overlap_end") - pl.col("overlap_start")).dt.total_seconds()) if same.height else same,
        S.COLOCATIONS),
        S.conform(diff.with_columns(overlap_s=pl.lit(0)) if diff.height else pl.DataFrame(), S.COLOCATIONS)])
    return out.sort("kind", "id_a", "id_b", "overlap_start")
