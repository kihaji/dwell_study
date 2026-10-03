# PySpark runner

Runs the same pattern-of-life analysis as `dwells.pol.pipeline` on Spark and writes the
same output tables (`docs/outputs.md`). The input is the cleaned events parquet described
in [docs/data_flow.md §0](../docs/data_flow.md).

```
cleaned events (parquet, any size)
  │  driver: required columns present, timestamp is timezone-aware (TimestampType)
  ▼
groupBy("id").applyInArrow(analyze_one_id)       per id, on the executors:
  │                                               validate_events → pipeline.analyze
  │                                               → 8 tables, each packed as Arrow IPC bytes
  ▼  persisted: one row per (id, table)
per-id visit / place counts → offsets in id order (window over one row per id)
  ▼
for each table: filter → join offsets → mapInArrow(unpack + shift ids) → orderBy → write
  ▼
<out>/<table>.parquet/   (directories of part files)     <out>/meta.json
```

## Why it wraps the polars pipeline

Every output table is computed per identifier, so the polars pipeline already is the
per-partition function: Spark contributes distribution and I/O, and there is one
implementation of the analysis to maintain. The single cross-id step is numbering.
`visit_id` and `place_id` are unique across the whole run, numbered in id order, so each
id's ids are shifted by the number of visits / places of all ids sorting before it
(`job._offsets`), exactly as one polars run over all ids numbers them.

## Usage

```python
from dwells.presets import params_for
from dwells_spark import job

spark = job.session("spark://master:7077")       # or your application's session
meta = job.run(spark, "cleaned/events.parquet", "results/upload_1", params_for("mixed", 900),
               geocode_index="osm/area/geocode.parquet", meta={"dataset": "upload_1"})
```

CLI:

```
python -m dwells_spark --input cleaned/events.parquet --out results/upload_1 \
    [--profile mixed] [--min-duration 900] [--geocode-index geocode.parquet] [--master spark://…]
```

`job.run` sets three session options: time zone UTC (Arrow timestamps reach polars as
UTC), parquet timestamps in microseconds (read back as polars `Datetime("us", "UTC")`),
and `canChangeCachedPlanOutputPartitioning` (lets adaptive execution coalesce the cached
per-id results; without it a small run takes ~85 s instead of ~3 s).

## Output

The same eight tables and `meta.json` as `pipeline.write`, except that each table is a
**directory** of part files (`visits.parquet/part-*.parquet`), written sorted by its
natural key (`job.ORDER`). `pipeline.read` (and so the app) reads either form;
plain polars reads a directory with `pl.read_parquet("visits.parquet/*.parquet")`.
`meta.json` adds `engine: "spark"` and `input`.

## Parity with the polars run

`tests/test_parity.py` runs both engines on the same input and compares every table:
same schema, same rows, same `visit_id` / `place_id` numbering; integers, strings and
timestamps equal exactly, floats within 1e-9 relative. Floats cannot be bit-identical:
polars sums aggregations in an order that depends on the frame, so even plain polars on
one id versus all ids differs by 1e-16 to 1e-11. Verified on a handwritten fixture (ids
out of order, an id with no dwells, 68% accuracy radii, source labels) and on 40 ids of
`eval_urban_adtech` with geocoding; the full dataset (100 ids, 84k events) also matches.

Building the parity test surfaced four order-dependent spots in the polars pipeline,
fixed there (tests in `tests/test_pol.py`): an empty table conformed to one null row; a
visit without anomalies was `[]` or null depending on other visits; `typical_depart_hour`
and the `next_places` / place ranks broke ties arbitrarily.

## Requirements

* Driver and executors run **the same Python and library versions** (`requirements.txt`
  pins the analysis libraries to `uv.lock`), with `dwells` and `dwells_spark` importable
  (here via `PYTHONPATH=/work/src:/work/spark`; in production, ship a wheel or
  `--py-files`).
* PySpark Arrow UDFs need pandas installed (not used directly; < 3 for PySpark 4.2).
* Input, output and the geocode index must be on storage every executor can reach.
  `meta.json` is written by the driver with plain file I/O, so `out_dir` must be a
  filesystem path (adapt `pipeline.write_meta` for object stores).
* Each id's events are processed in one Python worker. The largest id must fit in an
  executor's memory, and its packed tables in one Spark binary value (< 2 GB).
* The first task in each Python worker compiles the numba kernel (a few seconds); set
  `NUMBA_CACHE_DIR` to a writable directory so it is cached.

## Docker test cluster

One image (`Dockerfile`: Python 3.12, JDK 17, PySpark 4.2) for the master, two workers
(4 cores / 3 GB each) and the driver; the repo is mounted at `/work`, so code changes
need no rebuild. Memory is capped (~11 GB in total) for WSL2.

```
docker compose -f spark/docker-compose.yml build
docker compose -f spark/docker-compose.yml run --rm driver pytest spark/tests                    # local[2]
docker compose -f spark/docker-compose.yml up -d spark-master spark-worker-1 spark-worker-2      # UI: localhost:8080
docker compose -f spark/docker-compose.yml run --rm -e SPARK_MASTER=spark://spark-master:7077 driver pytest spark/tests
docker compose -f spark/docker-compose.yml down
```

Test files go to `spark/.pytest-tmp/` (git-ignored) so the workers can read them. On the
host, `uv run pytest` skips `spark/tests` (no PySpark / Java there).
