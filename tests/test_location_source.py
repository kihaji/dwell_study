"""Optional location_source field: validation, assumed error by source, forced-vague sources."""

import numpy as np
import polars as pl
import pytest

from dwells import io, schema
from dwells import uncertainty as U
from dwells.algorithms import ua_mahalanobis as ua
from dwells.geo import from_enu

T0 = 1_700_000_000


def _ev(sources, acc=None):
    n = len(sources)
    df = pl.DataFrame({"id": "a", "lat": np.full(n, 40.0), "lon": np.full(n, -75.0),
                       "timestamp": (T0 + np.arange(n) * 60) * 1_000_000, "location_source": sources})
    if acc is not None:
        df = df.with_columns(horizontalAccuracy=pl.lit(acc))
    return df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_validate_accepts_known_sources_and_nulls():
    out = schema.validate_events(_ev(["gps", "network", "fused", "ip", "user_provided", "unknown", None]))
    assert out["location_source"].to_list()[-1] == "unknown"  # null normalized


def test_validate_rejects_unknown_source_value():
    with pytest.raises(ValueError):
        schema.validate_events(_ev(["gps", "satellite"]))


def test_assumed_error_by_source_when_no_accuracy():
    ev = schema.validate_events(_ev(["gps", "network", "unknown"]))
    out = U.with_covariance(ev, default_sigma_m=30.0, source_sigma={"gps": 10.0, "network": 300.0})
    assert out["sigma_m"].to_list() == pytest.approx([10.0, 300.0, 30.0])


def test_stated_accuracy_wins_over_source_label():
    ev = schema.validate_events(_ev(["gps", "network"], acc=11.774))
    out = U.with_covariance(ev, default_sigma_m=30.0, source_sigma={"gps": 10.0, "network": 300.0})
    assert out["sigma_m"].to_list() == pytest.approx([10.0, 10.0], rel=1e-3)


def test_ip_labeled_reports_cannot_form_a_dwell():
    # 4 reports over 45 min at an "IP hub" claiming 50 m accuracy
    lat, lon = from_enu(np.zeros(4), np.zeros(4), 40.0, -75.0)
    df = pl.DataFrame({"id": "a", "lat": lat, "lon": lon, "timestamp": (T0 + np.arange(4) * 900) * 1_000_000,
                       "horizontalAccuracy": 58.87, "location_source": ["ip"] * 4}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    ev = schema.validate_events(df)
    assert ua.detect(ev.drop("location_source"), min_duration_s=900).height == 1  # without the label: a dwell
    assert ua.detect(ev, min_duration_s=900).height == 0                          # labelled ip: vague


def test_generator_sources_map_to_location_source():
    master = pl.DataFrame({"id": ["a"] * 7, "timestamp": np.arange(7) * 1_000_000,
                           "lat": 40.0, "lon": -75.0, "true_lat": 40.0, "true_lon": -75.0,
                           "source": ["gps", "gps_outlier", "wifi", "cell", "ip", "stale", "cdr"],
                           "sig_a": 10.0, "sig_b": 10.0, "az": 0.0, "p_mixed": 0.95,
                           "truth_dwell_id": pl.Series([None] * 7, dtype=pl.Int64)}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    out = io.shape_view(master, "1", with_source=True)
    # stale (cached) fixes keep the label of the report they repeat
    assert out["location_source"].to_list() == ["gps", "gps", "network", "network", "ip", "ip", "network"]


def test_presets_carry_source_sigma():
    from dwells.presets import PRESETS

    for p in PRESETS.values():
        assert p["source_sigma"] == {"gps": 10.0, "network": 300.0, "fused": 30.0}


def _pipeline_events(sources, acc):
    from dwells.geo import from_enu as fe

    n = len(sources)
    lat, lon = fe(np.zeros(n), np.zeros(n), 40.0, -75.0)
    df = pl.DataFrame({"id": "a", "lat": lat, "lon": lon, "timestamp": (T0 + np.arange(n) * 300) * 1_000_000,
                       "horizontalAccuracy": acc, "location_source": sources})
    return schema.validate_events(df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))))


def test_pipeline_carries_location_source_and_quality():
    from dwells.pol import pipeline

    # 12 reports: gps (one claiming 3 km), network, and an ip report claiming 50 m
    src = ["gps"] * 8 + ["network"] * 2 + ["ip", "unknown"]
    acc = [12.0] * 7 + [3532.0] + [118.0] * 2 + [58.9, 12.0]
    t = pipeline.analyze(_pipeline_events(src, acc), profile="mixed")
    assert t["events"]["location_source"].to_list() == src
    q = t["quality"].row(0, named=True)
    assert q["source_coverage"] == pytest.approx(11 / 12)
    # gps claiming > 100 m (1 of 12) and ip claiming < 1 km (1 of 12) contradict their label
    assert q["source_mismatch_frac"] == pytest.approx(2 / 12)


def test_pipeline_without_location_source_leaves_nulls():
    from dwells.pol import pipeline

    ev = _pipeline_events(["gps"] * 6, [12.0] * 6).drop("location_source")
    t = pipeline.analyze(ev, profile="mixed")
    assert t["events"]["location_source"].null_count() == 6
    assert t["quality"]["source_coverage"][0] is None
