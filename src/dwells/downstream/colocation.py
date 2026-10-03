"""Colocation candidate search between ids (used by dwells.pol.coloc).

* ``_same_time_pairs``: dwell index pairs of different ids that overlap in time by
  >= min_overlap_s within an H3 neighborhood (pol.coloc then applies the distance test).
* ``diff_time``: dwells of different ids within radius_m of each other that do not
  overlap in time, where the later one starts within window_s after the earlier one
  ends (gap >= 0; shared place, different times: shift workers, the same gym). For
  these rows overlap_start/overlap_end hold the gap interval (earlier end, later start).

Candidates come from H3 neighborhoods (each dwell's cell + k=1 ring at a bucket
resolution covering the match scale) scanned with a time-sorted sweep, so memory is
bounded by real temporal overlaps rather than all pairs per cell. In Spark the same
logic is a cell-exploded join with a time-range join condition (or a bucketed
range join). The Mahalanobis and H3 matchers evaluated earlier live on main.
"""

from __future__ import annotations

import h3
import numpy as np
import polars as pl

from dwells import geo, schema

def _res_for(size_m):
    from dwells.h3util import resolution_for

    return resolution_for(size_m)


def _neighborhoods(d: pl.DataFrame, bucket_m: float):
    """Yield (A, B) index arrays: A = dwells in an H3 cell, B = dwells in its k=1 disk
    (including the cell itself), B sorted by start time."""
    r = _res_for(bucket_m)
    cells = [h3.latlng_to_cell(a, b, r) for a, b in zip(d["lat"].to_numpy(), d["lon"].to_numpy())]
    by_cell: dict[str, list[int]] = {}
    for i, c in enumerate(cells):
        by_cell.setdefault(c, []).append(i)
    start = d["start_ts"].dt.epoch("us").to_numpy()
    for c, members in by_cell.items():
        B = np.concatenate([np.asarray(by_cell[n]) for n in h3.grid_disk(c, 1) if n in by_cell])
        yield np.asarray(members), B[np.argsort(start[B], kind="stable")]


def _same_time_pairs(d: pl.DataFrame, bucket_m: float, min_overlap_s: float):
    """Index pairs (i < j, different ids) overlapping in time by >= min_overlap_s.

    Sorted sweep per H3 neighborhood: only genuinely overlapping pairs are produced, so
    memory is bounded by the number of real overlaps (a plain cell join materialized all
    pairs per cell and exhausted memory on dense urban data with fragmented dwells)."""
    ids = d["id"].to_numpy()
    s = d["start_ts"].dt.epoch("us").to_numpy() / 1e6
    e = d["end_ts"].dt.epoch("us").to_numpy() / 1e6
    dmax = float((e - s).max()) if len(s) else 0.0
    out_i, out_j = [], []
    for A, B in _neighborhoods(d, bucket_m):
        sB = s[B]
        for i in A:
            lo = np.searchsorted(sB, s[i] - dmax, "left")
            hi = np.searchsorted(sB, e[i] - min_overlap_s, "right")
            j = B[lo:hi]
            ov = np.minimum(e[i], e[j]) - np.maximum(s[i], s[j])
            j = j[(ov >= min_overlap_s) & (j > i) & (ids[j] != ids[i])]
            out_i.append(np.full(len(j), i))
            out_j.append(j)
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs = np.unique(np.c_[np.concatenate(out_i), np.concatenate(out_j)].astype(np.int64), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _join_pairs(d: pl.DataFrame, i: np.ndarray, j: np.ndarray) -> pl.DataFrame:
    a = d.with_row_index("i").with_columns(pl.col("i").cast(pl.Int64))
    idx = pl.DataFrame({"i": i, "j": j})
    return idx.join(a, on="i").join(a.rename(lambda c: c + "_b"), left_on="j", right_on="i_b")


def _orient(c: pl.DataFrame) -> pl.DataFrame:
    swap = pl.col("id") > pl.col("id_b")
    return c.with_columns(
        id_a=pl.when(swap).then(pl.col("id_b")).otherwise(pl.col("id")),
        id_b=pl.when(swap).then(pl.col("id")).otherwise(pl.col("id_b")),
    )


def diff_time(dwells: pl.DataFrame, radius_m: float = 100, window_s: float = 7 * 86400) -> pl.DataFrame:
    """One row per id pair with at least one diff-time episode (the example with the
    shortest gap found first); enumerating every episode is quadratic at busy places."""
    if dwells.height < 2:
        return schema.empty(schema.COLOCATIONS)
    ids = dwells["id"].to_numpy()
    s = dwells["start_ts"].dt.epoch("us").to_numpy() / 1e6
    e = dwells["end_ts"].dt.epoch("us").to_numpy() / 1e6
    lat, lon = dwells["lat"].to_numpy(), dwells["lon"].to_numpy()
    best: dict[tuple, tuple] = {}
    for A, B in _neighborhoods(dwells, radius_m):
        sB = s[B]
        for i in A:
            # dwells of other ids starting after this one ends, within the window
            lo = np.searchsorted(sB, e[i], "left")
            hi = np.searchsorted(sB, e[i] + window_s, "right")
            j = B[lo:hi]
            j = j[(ids[j] != ids[i]) & (geo.haversine_m(lat[i], lon[i], lat[j], lon[j]) <= radius_m)]
            if len(j) == 0:
                continue
            first = np.unique(ids[j], return_index=True)[1]  # earliest start per other id
            for k in j[first]:
                key = tuple(sorted((ids[i], ids[k])))
                gap = s[k] - e[i]
                if key not in best or gap < best[key][0]:
                    best[key] = (gap, i, k)
    if not best:
        return schema.empty(schema.COLOCATIONS)
    i = np.array([v[1] for v in best.values()], np.int64)
    j = np.array([v[2] for v in best.values()], np.int64)
    c = _join_pairs(dwells, i, j).with_columns(
        overlap_start=pl.col("end_ts"), overlap_end=pl.col("start_ts_b"))  # gap interval
    c = _orient(c).with_columns(
        lat=(pl.col("lat") + pl.col("lat_b")) / 2, lon=(pl.col("lon") + pl.col("lon_b")) / 2,
        overlap_s=pl.lit(0), kind=pl.lit("diff_time"))
    return c.select([pl.col(k).cast(t) for k, t in schema.COLOCATIONS.items()]).sort("id_a", "id_b", "overlap_start")
