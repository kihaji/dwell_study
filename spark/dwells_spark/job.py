"""Pattern-of-life pipeline on Spark.

Every output table is computed per identifier, so Spark partitions the cleaned events by
``id`` and runs the polars pipeline (``dwells.pol.pipeline.analyze``) on each group. The
only cross-id step is numbering: ``visit_id`` and ``place_id`` are made global with
per-id offsets in id order, exactly as a single polars run numbers them. Output:
``<out>/<table>.parquet`` directories (readable by ``pipeline.read`` / polars) plus
``meta.json``, with the same schemas and values as ``pipeline.write``.

    from dwells_spark import job
    spark = job.session("spark://host:7077")
    job.run(spark, "cleaned/events.parquet", "results/upload_1", params_for("mixed"))
"""

from __future__ import annotations

import io
import socket
import time

import polars as pl
import pyarrow as pa
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from dwells import schema
from dwells.pol import pipeline
from dwells.pol import schema as S

TABLES = S.TABLES
# columns holding a visit / place id, offset to make ids global
VISIT_COLS = {"events": ("visit_id",), "visits": ("visit_id",), "anomalies": ("visit_id",)}
PLACE_COLS = {"events": ("place_id",), "visits": ("place_id", "prev_place_id", "next_place_id"),
              "places": ("place_id",), "transitions": ("from_place", "to_place"),
              "next_places": ("from_place", "to_place"), "presence": ("place_id",), "anomalies": ("place_id",)}
# row order of the written tables
ORDER = {"events": ("id", "event_idx"), "visits": ("visit_id",), "places": ("place_id",),
         "transitions": ("id", "from_place", "to_place"), "next_places": ("id", "from_place", "rank"),
         "presence": ("id", "date", "hour", "place_id"), "anomalies": ("visit_id", "kind"), "quality": ("id",)}
# input columns the pipeline reads (others are dropped before the shuffle)
INPUT_COLS = (*schema.EVENTS_BASE, *schema.EVENTS_SHAPE2, schema.ACCURACY_PROBABILITY, *schema.EVENTS_SHAPE3,
              "location_source")

# one row per (id, table): the table's rows for that id as Arrow IPC bytes
PACKED = T.StructType([T.StructField("id", T.StringType()), T.StructField("table", T.StringType()),
                       T.StructField("rows", T.LongType()), T.StructField("payload", T.BinaryType())])
_PACKED_ARROW = pa.schema([("id", pa.string()), ("table", pa.string()), ("rows", pa.int64()),
                           ("payload", pa.binary())])

_SCALARS = {pl.String: (T.StringType(), pa.string()), pl.Float64: (T.DoubleType(), pa.float64()),
            pl.Int64: (T.LongType(), pa.int64()), pl.Int8: (T.ByteType(), pa.int8()),
            pl.Boolean: (T.BooleanType(), pa.bool_()), pl.Date: (T.DateType(), pa.date32())}


def _types(t) -> tuple[T.DataType, pa.DataType]:
    """(Spark, Arrow) type of a polars output dtype."""
    if isinstance(t, pl.List):
        s, a = _types(t.inner)
        return T.ArrayType(s), pa.list_(a)
    if isinstance(t, pl.Datetime):  # all output times are UTC instants
        return T.TimestampType(), pa.timestamp("us", tz="UTC")
    return _SCALARS[t]


def spark_schema(spec: dict) -> T.StructType:
    return T.StructType([T.StructField(c, _types(t)[0]) for c, t in spec.items()])


def session(master: str = "local[*]", app_name: str = "dwells") -> SparkSession:
    b = SparkSession.builder.master(master).appName(app_name)
    if master.startswith("spark://"):  # executors connect back to this driver container
        b = b.config("spark.driver.host", socket.gethostbyname(socket.gethostname())).config(
            "spark.driver.bindAddress", "0.0.0.0")
    return b.getOrCreate()


def run(spark: SparkSession, input_path: str, out_dir: str, params: dict, geocode_index: str | None = None,
        meta: dict | None = None) -> dict:
    """Analyze cleaned events (parquet, docs/data_flow.md §0) and write the output tables.

    ``params`` as from ``presets.params_for``. ``geocode_index``: an offline OSM index
    (``geocode.build_index_*``) readable by every executor at that path. ``out_dir`` must
    be a shared filesystem path (meta.json is written by the driver). Sets the session
    time zone to UTC, parquet timestamps to microseconds, and lets adaptive execution
    coalesce cached partitions."""
    t0 = time.time()
    spark.conf.set("spark.sql.session.timeZone", "UTC")  # Arrow timestamps reach polars as UTC
    spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")  # polars Datetime("us", "UTC")
    # let adaptive execution coalesce the cached per-id results (mostly empty partitions otherwise)
    spark.conf.set("spark.sql.optimizer.canChangeCachedPlanOutputPartitioning", "true")
    events = _checked(spark.read.parquet(input_path))
    packed = (events.select([c for c in INPUT_COLS if c in events.columns])
              .groupBy("id").applyInArrow(_analyze_id(params, geocode_index), PACKED)
              .persist(StorageLevel.MEMORY_AND_DISK))
    rows = {r["table"]: int(r["rows"]) for r in packed.groupBy("table").agg(F.sum("rows").alias("rows")).collect()}
    offsets = _offsets(packed)
    for name, spec in TABLES.items():
        part = packed.filter(F.col("table") == name).join(offsets, "id").select("payload", "visit_off", "place_off")
        (part.mapInArrow(_unpack(name), spark_schema(spec)).orderBy(*ORDER[name])
         .write.mode("overwrite").parquet(f"{out_dir}/{name}.parquet"))
    packed.unpersist()
    meta = (meta or {}) | {"engine": "spark", "input": input_path, "params": params,
                           "rows": {k: rows.get(k, 0) for k in TABLES}, "seconds": round(time.time() - t0, 1)}
    pipeline.write_meta(out_dir, meta)
    return meta


def _checked(events: DataFrame) -> DataFrame:
    """Fail fast on the driver; value checks run per id in schema.validate_events."""
    missing = [c for c in schema.EVENTS_BASE if c not in events.columns]
    if missing:
        raise ValueError(f"events missing columns: {missing}")
    ts = events.schema["timestamp"].dataType
    if not isinstance(ts, T.TimestampType):  # TimestampNTZType = naive
        raise ValueError(f"timestamp must be a timezone-aware datetime (got {ts.simpleString()}); "
                         "convert to UTC upstream")
    return events


_geocoders: dict = {}  # per executor process: the index is loaded once


def _geocoder(path: str | None):
    if path and path not in _geocoders:
        from dwells.pol.geocode import OfflineOSMGeocoder
        _geocoders[path] = OfflineOSMGeocoder([path])
    return _geocoders.get(path)


def _analyze_id(params: dict, geocode_index: str | None):
    def analyze(table: pa.Table) -> pa.Table:
        ev = schema.validate_events(pl.from_arrow(table))
        tables = pipeline.analyze(ev, params=params, geocoder=_geocoder(geocode_index))
        payloads = []
        for df in tables.values():
            buf = io.BytesIO()
            df.write_ipc(buf)
            payloads.append(buf.getvalue())
        return pa.table({"id": [ev["id"][0]] * len(tables), "table": list(tables),
                         "rows": [df.height for df in tables.values()], "payload": payloads}, schema=_PACKED_ARROW)
    return analyze


def _offsets(packed: DataFrame) -> DataFrame:
    """Per id: visits and places of all ids sorting before it (polars numbering order)."""
    counts = (packed.filter(F.col("table").isin("visits", "places")).groupBy("id")
              .pivot("table", ["visits", "places"]).agg(F.first("rows")))
    before = Window.orderBy("id").rowsBetween(Window.unboundedPreceding, -1)
    return counts.select("id", F.coalesce(F.sum("visits").over(before), F.lit(0)).alias("visit_off"),
                         F.coalesce(F.sum("places").over(before), F.lit(0)).alias("place_off"))


def _unpack(name: str):
    target = pa.schema([(c, _types(t)[1]) for c, t in TABLES[name].items()])
    shift = [(c, "visit_off") for c in VISIT_COLS.get(name, ())] + [(c, "place_off") for c in PLACE_COLS.get(name, ())]

    def unpack(batches):
        for batch in batches:
            for r in batch.to_pylist():
                df = pl.read_ipc(io.BytesIO(r["payload"]))
                df = df.with_columns([pl.col(c) + r[off] for c, off in shift])
                yield from df.to_arrow().cast(target).to_batches()
    return unpack
