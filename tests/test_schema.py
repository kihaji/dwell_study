import polars as pl
import pytest

from dwells import schema


def _ev(**extra):
    d = {"id": ["a"], "lat": [1.0], "lon": [2.0], "timestamp": [0]}
    d |= extra
    return pl.DataFrame(d).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_shape_detection():
    assert schema.event_shape(_ev()) == 1
    assert schema.event_shape(_ev(horizontalAccuracy=[5.0])) == 2
    assert schema.event_shape(
        _ev(semiMajor=[5.0], semiMinor=[3.0], azimuth=[10.0], probability=[0.95])
    ) == 3


def test_validate_events_accepts_and_casts():
    df = _ev().with_columns(pl.col("lat").cast(pl.Float32))
    out = schema.validate_events(df)
    assert out.schema["lat"] == pl.Float64
    assert out.schema["timestamp"] == pl.Datetime("us", "UTC")


@pytest.mark.parametrize(
    "bad",
    [
        {"lat": [91.0]},
        {"lon": [-181.0]},
        {"semiMajor": [1.0], "semiMinor": [3.0], "azimuth": [10.0], "probability": [0.95]},
        {"semiMajor": [5.0], "semiMinor": [3.0], "azimuth": [190.0], "probability": [0.95]},
        {"semiMajor": [5.0], "semiMinor": [3.0], "azimuth": [10.0], "probability": [1.0]},
        {"horizontalAccuracy": [-1.0]},
    ],
)
def test_validate_events_rejects(bad):
    with pytest.raises(ValueError):
        schema.validate_events(_ev(**bad))


def test_validate_events_missing_column():
    with pytest.raises(ValueError):
        schema.validate_events(pl.DataFrame({"id": ["a"], "lat": [1.0]}))


def test_validate_events_sorts_by_id_time():
    df = pl.DataFrame(
        {"id": ["b", "a", "a"], "lat": [0.0] * 3, "lon": [0.0] * 3, "timestamp": [5, 9, 1]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    out = schema.validate_events(df)
    assert out["id"].to_list() == ["a", "a", "b"]
    assert out["timestamp"].dt.epoch("us").to_list() == [1, 9, 5]


def test_empty_dwells_has_schema():
    d = schema.empty(schema.DWELLS)
    assert d.height == 0
    assert set(schema.DWELLS) <= set(d.columns)


def test_validate_dwells_fills_optional_uncertainty():
    d = pl.DataFrame(
        {
            "id": ["a"],
            "start_ts": [0],
            "end_ts": [600_000_000],
            "lat": [1.0],
            "lon": [2.0],
            "n_events": [5],
        }
    ).with_columns(pl.col("start_ts", "end_ts").cast(pl.Datetime("us", "UTC")))
    out = schema.validate_dwells(d)
    assert out["duration_s"][0] == 600
    assert out["dwell_id"][0] == 0
    assert out["sigma_major"][0] is None


@pytest.mark.parametrize(
    "col, extra",
    [
        ("id", {}),
        ("lat", {}),
        ("lon", {}),
        ("timestamp", {}),
        ("horizontalAccuracy", {"horizontalAccuracy": [5.0, 5.0]}),
        ("semiMajor", {"semiMajor": [5.0, 5.0], "semiMinor": [3.0, 3.0], "azimuth": [10.0, 10.0],
                       "probability": [0.95, 0.95]}),
        ("probability", {"semiMajor": [5.0, 5.0], "semiMinor": [3.0, 3.0], "azimuth": [10.0, 10.0],
                         "probability": [0.95, 0.95]}),
    ],
)
def test_validate_events_rejects_nulls(col, extra):
    d = {"id": ["a", "a"], "lat": [1.0, 1.0], "lon": [2.0, 2.0], "timestamp": [0, 1]} | extra
    df = pl.DataFrame(d).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    df = df.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col(col)).alias(col))
    with pytest.raises(ValueError, match=col):
        schema.validate_events(df)


def test_validate_events_rejects_nan_accuracy():
    with pytest.raises(ValueError, match="horizontalAccuracy"):
        schema.validate_events(_ev(horizontalAccuracy=[float("nan")]))


def test_validate_events_rejects_naive_timestamps():
    df = _ev().with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    with pytest.raises(ValueError, match="timezone"):
        schema.validate_events(df)


def test_validate_events_rejects_non_datetime_timestamps():
    df = pl.DataFrame({"id": ["a"], "lat": [1.0], "lon": [2.0], "timestamp": [0]})
    with pytest.raises(ValueError, match="timezone"):
        schema.validate_events(df)


def test_validate_events_converts_other_timezones_to_utc():
    local = pl.DataFrame({"id": ["a"], "lat": [1.0], "lon": [2.0]}).with_columns(
        timestamp=pl.datetime(2026, 1, 5, 8, time_zone="America/Chicago"))
    out = schema.validate_events(local)
    assert out.schema["timestamp"] == pl.Datetime("us", "UTC")
    assert out["timestamp"].dt.hour()[0] == 14  # 08:00 CST = 14:00 UTC


def test_accuracy_probability_validated_and_cast():
    out = schema.validate_events(_ev(horizontalAccuracy=[5.0], accuracyProbability=[0.68]))
    assert out.schema["accuracyProbability"] == pl.Float64


@pytest.mark.parametrize("p", [0.0, 1.0, None, float("nan")])
def test_accuracy_probability_rejected(p):
    df = _ev(horizontalAccuracy=[5.0]).with_columns(accuracyProbability=pl.Series([p], dtype=pl.Float64))
    with pytest.raises(ValueError, match="accuracyProbability"):
        schema.validate_events(df)


def test_accuracy_probability_ignored_without_horizontal_accuracy():
    # only meaningful for shape 2; shape 3 carries its own probability per ellipse
    out = schema.validate_events(_ev(accuracyProbability=[None]))
    assert schema.event_shape(out) == 1
