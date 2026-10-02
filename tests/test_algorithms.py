"""Behavioral contract every dwell detector must satisfy on clean, easy tracks."""

import numpy as np
import polars as pl
import pytest

from dwells import algorithms, schema
from dwells.geo import from_enu

LAT0, LON0 = 40.0, -75.0
T0 = 1_700_000_000.0


def track(segments, seed=0, sigma=5.0, shape=2, step_s=60):
    """segments: list of ("stay", x, y, minutes) | ("move", x1, y1, minutes) | ("gap", minutes).
    One fix every step_s seconds; stays jitter with sigma meters."""
    rng = np.random.default_rng(seed)
    t, xs, ys = [], [], []
    now, x, y = T0, 0.0, 0.0
    for seg in segments:
        if seg[0] == "gap":
            now += seg[1] * 60
            continue
        kind, tx, ty, mins = seg
        n = int(mins * 60 / step_s)
        for i in range(n):
            if kind == "stay":
                x, y = tx, ty
                px, py = tx + rng.normal(0, sigma), ty + rng.normal(0, sigma)
            else:
                f = (i + 1) / n
                px, py = x + (tx - x) * f, y + (ty - y) * f
            t.append(now)
            xs.append(px)
            ys.append(py)
            now += step_s
        x, y = tx, ty
    lat, lon = from_enu(np.array(xs), np.array(ys), LAT0, LON0)
    df = pl.DataFrame({"id": "a", "lat": lat, "lon": lon, "timestamp": (np.array(t) * 1e6).astype(np.int64)})
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    if shape == 2:
        df = df.with_columns(horizontalAccuracy=pl.lit(sigma * 1.1774))
    return schema.validate_events(df)


EASY = [("stay", 0, 0, 40), ("move", 2000, 0, 12), ("stay", 2000, 0, 60), ("move", 2000, 3000, 15),
        ("stay", 2000, 3000, 30)]
PARAMS = dict(min_duration_s=900, radius_m=100, max_gap_s=3600)


def _iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / (max(a1, b1) - min(a0, b0))


def _spans(d):
    return list(zip(d["start_ts"].dt.epoch("us").to_numpy() / 1e6, d["end_ts"].dt.epoch("us").to_numpy() / 1e6))


# Faithful ports / native library semantics that knowingly violate a contract test.
# Documented in docs/algorithms/<name>.md; kept visible as xfail rather than hidden.
KNOWN_LIMITATIONS: dict = {}


@pytest.fixture(params=algorithms.names())
def algo(request):
    reason = KNOWN_LIMITATIONS.get((request.param, request.node.originalname))
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    return algorithms.get(request.param)


def test_detects_easy_dwells(algo):
    d = schema.validate_dwells(algo.detect(track(EASY), **PARAMS))
    truth = [(T0, T0 + 40 * 60), (T0 + 52 * 60, T0 + 112 * 60), (T0 + 127 * 60, T0 + 157 * 60)]
    spans = _spans(d)
    assert len(spans) == 3, spans
    for t in truth:
        assert max(_iou(*t, *s) for s in spans) > 0.8


def test_constant_motion_is_not_a_dwell(algo):
    d = algo.detect(track([("move", 20_000, 0, 60)]), **PARAMS)
    assert d.height == 0


def test_long_gap_splits(algo):
    ev = track([("stay", 0, 0, 30), ("gap", 180), ("stay", 0, 0, 30)])
    d = algo.detect(ev, **PARAMS)
    assert d.height == 2


def test_short_stay_below_min_duration_dropped(algo):
    ev = track([("stay", 0, 0, 5), ("move", 3000, 0, 10), ("stay", 3000, 0, 30)])
    d = algo.detect(ev, **PARAMS)
    assert d.height == 1


def test_multiple_ids_kept_separate(algo):
    a = track(EASY)
    b = track(EASY, seed=1).with_columns(id=pl.lit("b"))
    d = algo.detect(schema.validate_events(pl.concat([a, b])), **PARAMS)
    assert sorted(d["id"].unique().to_list()) == ["a", "b"]
    assert d.filter(pl.col("id") == "b").height == 3


def test_dense_sampling_drive_splits_stays(algo):
    # 5 s fixes; 3 km drive at 10 m/s between two 30-min stays
    ev = track([("stay", 0, 0, 30), ("move", 3000, 0, 5), ("stay", 3000, 0, 30)], step_s=5)
    assert algo.detect(ev, **PARAMS).height == 2


def test_slow_walk_splits_stays(algo):
    # 1-min fixes; 30-min walk at ~1.4 m/s (2.5 km) between two 30-min stays
    ev = track([("stay", 0, 0, 30), ("move", 2500, 0, 30), ("stay", 2500, 0, 30)])
    assert algo.detect(ev, **PARAMS).height == 2
