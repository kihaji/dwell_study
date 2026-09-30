"""scikit-mobility ``preprocessing.detection.stops`` (Lachesis-style anchor radius).

Each id is split on gaps > ``max_gap_s`` here and the segment number is passed
as skmob's ``uid`` in one ``stops`` call (pandas 3 drops the uid column from
skmob's groupby.apply output, so a copy is carried as an extra column), with
``no_data_for_minutes = max_gap_s`` too. Native semantics kept: radius measured from the window's first point with strict ``>`` tests on
distance and time, end = ``leaving_datetime`` = first fix outside the stay,
centroid = coordinate-wise median. skmob has **no include_last**: the stay still
open at the end of a segment (i.e. before every gap and at the end of the track)
is never emitted. ``n_events`` counts the id's events in [start, end).
"""

from __future__ import annotations

import shapely.ops as so

so.cascaded_union = so.unary_union  # removed in shapely 2, imported by skmob.tessellation
import polars as pl  # noqa: E402
import skmob  # noqa: E402
from skmob.preprocessing import detection  # noqa: E402

from dwells import schema  # noqa: E402
from dwells.algorithms.common import gap_segments  # noqa: E402

NAME = "skmob_stops"
FAMILY = "library"
USES_UNCERTAINTY = False
REFERENCE = ("scikit-mobility 1.2.2 preprocessing.detection.stops; Pappalardo et al. 2022, "
             "doi:10.18637/jss.v103.i04; Hariharan & Toyama 2004, Project Lachesis")


def detect(events: pl.DataFrame, *, min_duration_s: float = 300, radius_m: float = 100,
           max_gap_s: float = 1800, **_) -> pl.DataFrame:
    ev = events.select("id", "lat", "lon", "timestamp").with_columns(gap_segments(events, max_gap_s))
    # skmob groups by uid=seg; under pandas 3 the uid column is dropped from the output,
    # so a copy (seg_) rides along as an extra column (copied from the stop's last row)
    pdf = ev.select("seg", seg_="seg", lat="lat", lon="lon", timestamp="timestamp").to_pandas()
    tdf = skmob.TrajDataFrame(pdf, latitude="lat", longitude="lon", datetime="timestamp", user_id="seg")
    st = detection.stops(tdf, spatial_radius_km=radius_m / 1000, minutes_for_a_stop=min_duration_s / 60,
                         no_data_for_minutes=max_gap_s / 60, leaving_time=True)
    if len(st) == 0:
        return schema.empty(schema.DWELLS)
    st = pl.from_pandas(st[["seg_", "lat", "lng", "datetime", "leaving_datetime"]]).select(
        pl.col("seg_").cast(ev["seg"].dtype).alias("seg"), "lat", lon="lng",
        start_ts=pl.col("datetime").cast(pl.Datetime("us", "UTC")),
        end_ts=pl.col("leaving_datetime").cast(pl.Datetime("us", "UTC")))
    idx = ev.select("seg", "id", "timestamp", i=pl.int_range(pl.len())).sort("timestamp")
    st = st.join(idx.group_by("seg").agg(pl.col("id").first()), on="seg")
    for c in ("start_ts", "end_ts"):  # row index of the start fix and of the first fix outside
        st = st.sort(c).join_asof(idx.select("seg", pl.col("timestamp").alias(c), pl.col("i").alias(c + "_i")),
                                  on=c, by="seg", strategy="forward", check_sortedness=False)
    st = st.with_columns(n_events=pl.col("end_ts_i") - pl.col("start_ts_i"))
    return schema.validate_dwells(st.drop("seg", "start_ts_i", "end_ts_i"))
