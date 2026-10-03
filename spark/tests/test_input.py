"""Input checks done on the driver before any work is scheduled."""

import polars as pl
import pytest
from conftest import routine

from dwells.presets import params_for
from dwells_spark import job


def test_rejects_naive_timestamps(spark, tmp_path):
    routine("a", (0, 0), (3000, 500)).with_columns(pl.col("timestamp").dt.replace_time_zone(None)).write_parquet(
        tmp_path / "e.parquet")
    with pytest.raises(ValueError, match="timezone"):
        job.run(spark, str(tmp_path / "e.parquet"), str(tmp_path / "out"), params_for("mixed"))


def test_rejects_missing_columns(spark, tmp_path):
    routine("a", (0, 0), (3000, 500)).drop("lon").write_parquet(tmp_path / "e.parquet")
    with pytest.raises(ValueError, match="lon"):
        job.run(spark, str(tmp_path / "e.parquet"), str(tmp_path / "out"), params_for("mixed"))


def test_spark_schemas_cover_all_tables():
    for name, spec in job.TABLES.items():
        assert job.spark_schema(spec).fieldNames() == list(spec), name


def test_id_columns_cover_every_visit_and_place_reference():
    # every column holding a visit or place id must be offset when ids are made global
    for name, spec in job.TABLES.items():
        refs = {c for c in spec if c in ("visit_id", "place_id", "prev_place_id", "next_place_id",
                                        "from_place", "to_place")}
        assert refs == set(job.VISIT_COLS.get(name, ())) | set(job.PLACE_COLS.get(name, ())), name
