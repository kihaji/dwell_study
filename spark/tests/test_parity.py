"""The Spark run must write the same tables as the polars run (pipeline.analyze + write)."""

import json
from pathlib import Path

import polars as pl
import pytest
from conftest import routine, wanderer
from polars.testing import assert_frame_equal

from dwells import io, schema
from dwells.pol import pipeline
from dwells.pol import schema as S
from dwells.pol.geocode import OfflineOSMGeocoder
from dwells.presets import params_for
from dwells_spark import job


def _polars_run(input_path, out, params, geocode_index=None):
    ev = schema.validate_events(pl.read_parquet(input_path))
    geocoder = OfflineOSMGeocoder([geocode_index]) if geocode_index else None
    pipeline.write(pipeline.analyze(ev, params=params, geocoder=geocoder), out)


def _assert_same(a_dir, b_dir):
    """Identical tables. Ids, integers, strings and timestamps must match exactly. Floats
    are allowed last-bits differences (rel 1e-9): polars sums aggregations in an order that
    depends on the frame, so even plain polars per id vs all ids differs by ~1e-16..1e-11."""
    a, b = pipeline.read(a_dir), pipeline.read(b_dir)
    assert set(a) == set(b) == set(S.TABLES)
    for name in S.TABLES:
        x, y = a[name], b[name]
        assert x.schema == y.schema, name
        keys = [c for c, t in x.schema.items() if not isinstance(t, pl.List)]
        try:
            assert_frame_equal(x.sort(keys), y.sort(keys), rel_tol=1e-9, abs_tol=1e-9)
        except AssertionError as e:
            raise AssertionError(f"{name}: {e}") from None


def test_parity_handwritten(spark, tmp_path):
    # ids out of order, one id with no dwells (offsets must skip it), accuracy stated as a
    # 68% radius, location_source labels
    ev = pl.concat([routine("c", (4000, -3000), (-1000, -1000), seed=3), wanderer("b0"),
                    routine("a", (0, 0), (3000, 500), seed=1), routine("b", (-2000, 1500), (3000, 500), seed=2)])
    ev = ev.with_columns(horizontalAccuracy=pl.lit(12.0), accuracyProbability=pl.lit(0.68),
                         location_source=pl.lit("gps"))
    ev.write_parquet(tmp_path / "events.parquet")
    params = params_for("mixed", 900)
    _polars_run(tmp_path / "events.parquet", tmp_path / "polars", params)
    meta = job.run(spark, str(tmp_path / "events.parquet"), str(tmp_path / "spark"), params)
    _assert_same(tmp_path / "polars", tmp_path / "spark")
    assert pipeline.read(tmp_path / "spark")["visits"]["visit_id"].sort().to_list() == list(range(meta["rows"]["visits"]))
    assert meta["rows"]["visits"] > 0 and meta["rows"]["events"] == ev.height


def test_meta_json(spark, tmp_path):
    routine("a", (0, 0), (3000, 500)).write_parquet(tmp_path / "events.parquet")
    params = params_for("gps", 1800)
    job.run(spark, str(tmp_path / "events.parquet"), str(tmp_path / "out"), params, meta={"dataset": "x"})
    m = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert m["dataset"] == "x" and m["engine"] == "spark" and m["params"]["min_duration_s"] == 1800
    assert m["rows"] == {k: v.height for k, v in pipeline.read(tmp_path / "out").items()}


DATASET, INDEX = "eval_urban_adtech", io.DATA_DIR / "osm" / "urban" / "geocode.parquet"


@pytest.mark.skipif(DATASET not in io.dataset_dirs() or not INDEX.exists(), reason="repo dataset not present")
def test_parity_repo_dataset_geocoded(spark, tmp_path):
    d = io.dataset_dirs()[DATASET]
    ids = pl.scan_parquet(Path(d) / "events.parquet").select("id").unique().sort("id").head(40).collect()["id"]
    io.load_events(d, "3_mixed", ids=ids.to_list(), with_source=True).write_parquet(tmp_path / "events.parquet")
    params = params_for("adtech", 900)
    _polars_run(tmp_path / "events.parquet", tmp_path / "polars", params, INDEX)
    job.run(spark, str(tmp_path / "events.parquet"), str(tmp_path / "spark"), params, geocode_index=str(INDEX))
    _assert_same(tmp_path / "polars", tmp_path / "spark")
    assert pipeline.read(tmp_path / "spark")["places"]["address"].is_not_null().any()
