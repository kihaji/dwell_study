"""fastmob ``stay_locations`` (Rust reimplementation of scikit-mobility stops).

Each id is split on gaps > ``max_gap_s`` here and the segment number is passed
as ``uid_col`` (fastmob would otherwise treat an ``id`` column as one user), with
``no_data_for_minutes = max_gap_s`` too; one batch call on the polars frame.
Native semantics kept: radius measured from the window's first point (``>=``
tests), ``include_last=True`` (library default), end = first fix outside the
stay (last fix for the trailing stay), centroid = coordinate-wise median.
fastmob returns naive timestamps; UTC is re-attached. It reports no
membership, so ``n_events`` counts the id's events in [start, end), or
[start, end] for the ``include_last`` stay (end = segment's last fix, within
radius of the anchor). Start/end are snapped back to the real fix timestamps
(fastmob truncates to milliseconds).
"""

from __future__ import annotations

import polars as pl
from fastmob.preprocessing import stay_locations

from dwells import schema
from dwells.geo import haversine_m
from dwells.algorithms.common import gap_segments

NAME = "fastmob_stays"
FAMILY = "library"
USES_UNCERTAINTY = False
REFERENCE = ("fastmob 0.3.2 preprocessing.stay_locations (github.com/gefgu/fastmob); "
             "Hariharan & Toyama 2004, Project Lachesis, GIScience LNCS 3234")


def detect(events: pl.DataFrame, *, min_duration_s: float = 300, radius_m: float = 100,
           max_gap_s: float = 1800, **_) -> pl.DataFrame:
    ev = events.select("id", "lat", "lon", "timestamp").with_columns(gap_segments(events, max_gap_s))
    st = stay_locations(ev.select("seg", "lat", "lon", "timestamp"), uid_col="seg", lat_col="lat",
                        lng_col="lon", datetime_col="timestamp", spatial_radius_km=radius_m / 1000,
                        minutes_for_a_stop=min_duration_s / 60, no_data_for_minutes=max_gap_s / 60,
                        include_last=True)
    if st.height == 0:
        return schema.empty(schema.DWELLS)
    st = st.select("seg", "lat", "lon",
                   start_ts=pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
                   end_ts=pl.col("leaving_datetime").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    return schema.validate_dwells(_count_events(ev, st, radius_m))


def _count_events(ev: pl.DataFrame, st: pl.DataFrame, radius_m: float) -> pl.DataFrame:
    """Snap fastmob's ms-truncated times to the real fixes; attach id and n_events.

    n_events = #events in [start, end), plus the end fix when it is the segment's last
    fix and lies within ``radius_m`` of the anchor (the ``include_last`` stay).
    """
    idx = ev.select("seg", "id", "timestamp", i=pl.int_range(pl.len())).sort("timestamp")
    last = idx.group_by("seg").agg(pl.col("id").first(), seg_last=pl.col("i").max())
    st = st.join(last, on="seg")
    for c in ("start_ts", "end_ts"):
        st = st.sort(c).join_asof(idx.select("seg", pl.col("timestamp").alias(c), pl.col("i").alias(c + "_i"),
                                             pl.col("timestamp").alias(c + "_ev")),
                                  on=c, by="seg", strategy="forward", check_sortedness=False)
    i0, i1 = st["start_ts_i"].to_numpy(), st["end_ts_i"].to_numpy()
    lat, lon = ev["lat"].to_numpy(), ev["lon"].to_numpy()
    inside = (i1 == st["seg_last"].to_numpy()) & (haversine_m(lat[i0], lon[i0], lat[i1], lon[i1]) < radius_m)
    return st.select("id", "lat", "lon", start_ts="start_ts_ev", end_ts=pl.max_horizontal("start_ts_ev", "end_ts_ev"),
                     n_events=pl.Series(i1 - i0 + inside))
