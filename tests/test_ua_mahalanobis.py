"""ua_mahalanobis-specific behavior: boundary estimation, windows and departure rules."""

import numpy as np
import polars as pl
import pytest

from dwells import schema
from dwells.algorithms import ua_mahalanobis as ua
from dwells.geo import from_enu

T0 = 1_700_000_000


def events(rows):
    """rows: (minute, x_m, y_m)."""
    m, x, y = (np.array(c, float) for c in zip(*rows))
    lat, lon = from_enu(x, y, 40.0, -75.0)
    df = pl.DataFrame({"id": "a", "lat": lat, "lon": lon, "timestamp": ((T0 + m * 60) * 1e6).astype(np.int64),
                       "horizontalAccuracy": 12.0})
    return schema.validate_events(df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))))


SPARSE = [(0, 3000, 0), (10, 0, 0), (20, 2, 1), (35, 1, -2), (60, 4000, 0), (90, 4200, 0)]
P = dict(min_duration_s=900, radius_m=50, max_gap_s=7200)


def _min(ts):
    return (ts.timestamp() - T0) / 60


def test_midpoint_boundaries_split_the_unobserved_transitions():
    d = ua.detect(events(SPARSE), **P)
    assert _min(d["start_ts"][0]) == pytest.approx(5)    # between 0 (elsewhere) and 10
    assert _min(d["end_ts"][0]) == pytest.approx(47.5)   # between 35 and 60 (elsewhere)


def test_arrival_and_departure_windows_reported():
    d = ua.detect(events(SPARSE), **P)
    r = d.row(0, named=True)
    assert _min(r["arrive_after"]) == pytest.approx(0) and _min(r["arrive_by"]) == pytest.approx(10)
    assert _min(r["depart_after"]) == pytest.approx(35) and _min(r["depart_by"]) == pytest.approx(60)


def test_midpoint_extension_is_capped_at_half_max_gap():
    rows = [(0, 3000, 0), (600, 0, 0), (615, 1, 1), (630, 0, 2), (640, 3000, 0)]
    d = ua.detect(events(rows), **dict(P, max_gap_s=3600))
    assert _min(d["start_ts"][0]) == pytest.approx(600 - 30)  # gap 600 min, capped at 60/2 min


def test_track_start_has_no_earlier_event_so_start_is_observed():
    d = ua.detect(events(SPARSE[1:]), **P)
    assert _min(d["start_ts"][0]) == pytest.approx(10)
    assert d["arrive_after"][0] is None


def test_min_duration_checked_on_estimated_span():
    rows = [(0, 3000, 0), (10, 0, 0), (18, 1, 1), (40, 3000, 0)]  # observed 8 min, estimated ~25 min
    d = ua.detect(events(rows), **P)
    assert d.height == 1
    assert _min(d["first_event_ts"][0]) == pytest.approx(10) and _min(d["last_event_ts"][0]) == pytest.approx(18)


def test_persistent_single_departure_breaks_stay_when_confirm_s_set():
    # sparse: one fix 2 km away, next fix back home 60 min later -> a real errand
    rows = [(0, 0, 0), (20, 1, 1), (40, 0, 2), (60, 2000, 0), (120, 1, 0), (140, 0, 1), (160, 2, 2)]
    p = dict(P, k_break=2)
    assert ua.detect(events(rows), confirm_s=0, **p).height == 1    # count rule only: one miss is an outlier
    assert ua.detect(events(rows), confirm_s=900, **p).height == 2  # persisted 60 min >= 15 min


def test_brief_outlier_does_not_break_stay_with_confirm_s():
    # dense: outlier fix, back in place 1 min later
    rows = [(m, 0, 0) for m in range(0, 30)] + [(30, 300, 0)] + [(m, 1, 1) for m in range(31, 60)]
    assert ua.detect(events(rows), confirm_s=900, **dict(P, k_break=2)).height == 1
