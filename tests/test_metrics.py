import polars as pl
import pytest

from dwells.metrics import dwell_metrics, match_dwells

M = 60_000_000  # one minute in microseconds


def _d(rows, extra=None):
    """rows: (id, start_min, end_min, lat, lon[, place_id])."""
    cols = ["id", "s", "e", "lat", "lon"] + (["place_id"] if len(rows[0]) > 5 else [])
    df = pl.DataFrame(rows, schema=cols, orient="row")
    return df.with_columns(
        start_ts=(pl.col("s") * M).cast(pl.Datetime("us", "UTC")),
        end_ts=(pl.col("e") * M).cast(pl.Datetime("us", "UTC")),
        duration_s=(pl.col("e") - pl.col("s")) * 60,
        n_events=pl.lit(10),
    ).drop("s", "e")


def test_perfect_match():
    t = _d([("a", 0, 60, 40.0, -75.0), ("a", 120, 300, 40.01, -75.0)])
    m = dwell_metrics(t, t, min_duration_s=0)
    assert m["precision"] == 1 and m["recall"] == 1 and m["f1"] == 1
    assert m["start_err_s_med"] == 0 and m["centroid_err_m_med"] == pytest.approx(0)
    assert m["time_precision"] == 1 and m["time_recall"] == 1


def test_iou_threshold_and_one_to_one():
    t = _d([("a", 0, 100, 40.0, -75.0)])
    # two fragments: only one may match (IoU 0.6), the other (0.3) is a false positive
    p = _d([("a", 0, 60, 40.0, -75.0), ("a", 70, 100, 40.0, -75.0)])
    pairs = match_dwells(p, t, iou_min=0.5)
    assert pairs.height == 1
    m = dwell_metrics(p, t, min_duration_s=0)
    assert m["precision"] == 0.5 and m["recall"] == 1
    assert m["fragmentation"] == 1.0  # the one truth dwell is split by >1 prediction
    assert m["time_recall"] == pytest.approx(0.9)


def test_merged_prediction():
    t = _d([("a", 0, 50, 40.0, -75.0), ("a", 55, 100, 40.0, -75.0)])
    p = _d([("a", 0, 100, 40.0, -75.0)])
    m = dwell_metrics(p, t, min_duration_s=0)
    assert m["merging"] == 1.0
    assert m["recall"] == 0.5


def test_min_duration_filters_truth_only():
    t = _d([("a", 0, 3, 40.0, -75.0), ("a", 10, 70, 40.0, -75.0)])
    p = _d([("a", 10, 70, 40.0, -75.0)])
    m = dwell_metrics(p, t, min_duration_s=300)
    assert m["n_truth"] == 1 and m["recall"] == 1 and m["precision"] == 1


def test_ids_do_not_cross_match():
    t = _d([("a", 0, 60, 40.0, -75.0)])
    p = _d([("b", 0, 60, 40.0, -75.0)])
    m = dwell_metrics(p, t, min_duration_s=0)
    assert m["recall"] == 0 and m["precision"] == 0


def test_observable_recall_excludes_unobservable_truth():
    t = _d([("a", 0, 60, 40.0, -75.0), ("a", 100, 160, 40.0, -75.0)]).with_columns(
        n_events=pl.Series([10, 0]))
    p = _d([("a", 0, 60, 40.0, -75.0)])
    m = dwell_metrics(p, t, min_duration_s=0)
    assert m["recall"] == 0.5 and m["recall_observable"] == 1.0
