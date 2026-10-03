# Data flow: cleaned upload → dwell analysis → UI

Developer reference for wiring `ua_mahalanobis` and the pattern-of-life pipeline into an
application. Starting point: the analyst has uploaded a new dataset, the cleaning pipeline
has run, and nothing has been computed yet. End point: result tables on disk that the UI
reads.

```
cleaned events (polars DataFrame)
  │  1. schema.validate_events          check, cast, sort
  │  2. presets.params_for              detector parameters
  │  3. geocode.OfflineOSMGeocoder      optional, built once per area
  ▼
pipeline.analyze(events, params=…, geocoder=…)   ──►  dict of 8 polars DataFrames
  │  4. pipeline.write(tables, out_dir, meta)
  ▼
<out_dir>/{events,visits,places,transitions,next_places,presence,anomalies,quality}.parquet
<out_dir>/meta.json
  │  5. pipeline.read(out_dir)  (the UI, app/data.py)
  ▼
map layers, tables, charts
```

Minimal end-to-end code:

```python
import polars as pl
from dwells import schema
from dwells.pol import pipeline
from dwells.pol.geocode import OfflineOSMGeocoder
from dwells.presets import params_for

events = schema.validate_events(cleaned)                    # 1
params = params_for("mixed", min_duration_s=900)            # 2
geocoder = OfflineOSMGeocoder(["osm/area/geocode.parquet"]) # 3 (or None)
tables = pipeline.analyze(events, params=params, geocoder=geocoder)
pipeline.write(tables, f"results/{upload_id}",              # 4
               meta={"dataset": upload_id, "profile": "mixed", "params": params,
                     "rows": {k: v.height for k, v in tables.items()}})
```

Everything runs in memory in one process. For large uploads, the PySpark runner in
[`spark/`](../spark/README.md) produces the same output (see "Scaling" below).

---

## 0. What the cleaning pipeline must hand over

One **polars DataFrame** (or a parquet file read into one) per upload, with one row per
location report. The meaning of each field, and how much each defect costs, is in
[data_contract.md](data_contract.md). This section covers only the structure.

### Required columns

| column | polars dtype | notes |
|---|---|---|
| `id` | `String` | device identifier; one id = one device |
| `lat` | `Float64` | WGS84, −90…90 |
| `lon` | `Float64` | WGS84, −180…180 |
| `timestamp` | `Datetime("us", "UTC")` | measurement time; must be **timezone-aware** (any zone is converted to UTC) |

### Error columns: pick exactly one shape for the whole upload

The shape is decided **per DataFrame** by which columns exist, not per row
(`schema.event_shape`).

| shape | extra columns (`Float64`) | meaning |
|---|---|---|
| 1 (centerpoint) | none | the detector assumes an error (`default_sigma_m`, or one per `location_source`) |
| 2 (accuracy) | `horizontalAccuracy`, optional `accuracyProbability` | circular error radius in metres, > 0; `accuracyProbability` (strictly between 0 and 1) states its confidence per report, e.g. 0.68 for Android. Without it the radius is read as **CEP50** |
| 3 (ellipse) | `semiMajor`, `semiMinor`, `azimuth`, `probability` | axes in metres (`semiMajor ≥ semiMinor > 0`), azimuth 0–180° clockwise from north, probability strictly between 0 and 1 |

If both shape-3 and shape-2 columns are present, shape 3 is used.

### Optional column

| column | dtype | values |
|---|---|---|
| `location_source` | `String` | `gps`, `network`, `fused`, `ip`, `user_provided`, `unknown`; null is treated as `unknown` |

Any other columns are carried through validation and ignored by the pipeline.

### Rejected by validation

`validate_events` raises `ValueError` for these, so the cleaning pipeline must fix them
first:

* **Nulls or NaN/inf in required or error columns.** A missing accuracy would otherwise
  become a `NaN` error that breaks gating for its id. If some reports lack accuracy, fill
  them in during cleaning with a per-source nominal value (for example the 1σ values in
  `presets.SOURCE_SIGMA` with `accuracyProbability = 0.3935`), or drop the error columns for the whole upload and
  supply `location_source` instead.
* **Timestamps that are not timezone-aware datetimes.** Naive datetimes and epoch
  numbers are rejected, because the true zone is unknown and a silent guess would shift
  every hour-of-day view. Aware datetimes in any zone are accepted and converted to UTC.

### Obligations that validation cannot check

* **Accuracy semantics.** A `horizontalAccuracy` that is not CEP50 needs its
  `accuracyProbability` (0.68 for Android's `getAccuracy()`, 0.95 for many vendors), and
  an ellipse needs its true `probability`. An unlabelled 68% radius is silently read as
  CEP50 ([data_contract.md §2](data_contract.md)).
* **Measurement time**, not receive time, in `timestamp`.
* **One device per `id`**, and no subsampling (density drives accuracy).

Sorting is **not** an obligation: `validate_events` sorts the data.

---

## 1. Validate: `dwells.schema.validate_events(df) -> DataFrame`

* Checks that the required columns exist and raises `ValueError` if any are missing.
* Casts columns to their canonical dtypes.
* Normalizes `location_source` (null becomes `unknown`) and rejects unknown labels.
* Rejects naive or non-datetime timestamps, and converts aware ones to UTC.
* Rejects nulls and NaN/inf in the required and error columns.
* Range-checks lat/lon and the error fields.
* **Sorts by (`id`, `timestamp`).**

Always call it. `analyze` does not validate, and it relies on the sort order: the per-event
arrays the detector returns are aligned to row order.

Raise its `ValueError` messages to the analyst. They name the failing rule (for example,
`semiMajor < semiMinor`, `horizontalAccuracy has null or non-finite values`).

---

## 2. Parameters: `dwells.presets.params_for(profile, min_duration_s=900, **overrides) -> dict`

| profile | use when | only difference |
|---|---|---|
| `gps` | dense device GPS | `default_sigma_m = 10` |
| `cdr` | telco records | `default_sigma_m = 300` |
| `adtech` | bid-stream locations | `default_sigma_m = 300` |
| `mixed` | unknown or mixed (**recommended default**) | `default_sigma_m = 30` |

All profiles share one tuned base (`presets.BASE`). The profile only matters for
**centerpoint-only reports that have no usable `location_source` label**. With error
columns, or with labels, the choice is nearly irrelevant, so the application does not need
to ask the analyst for the data type. Evidence is in [tuning.md](tuning.md).

* `min_duration_s` is the analyst's definition of a dwell (default 15 min). Expose it in
  the UI.
* `overrides` can be any detector keyword (`radius_m`, `gate_p`, `max_gap_s`, `k_break`,
  `confirm_s`, `coarse_sigma_m`, `coarse_rel`, `min_events`, `default_sigma_m`,
  `source_sigma`, `coarse_sources`). Each one is documented in the
  `ua_mahalanobis.detect_with_members` docstring.

Keep the returned dict: it goes into `meta.json` (step 4) so the run can be reproduced.

---

## 3. Geocoder (optional)

`analyze(..., geocoder=None)` skips geocoding, and the address columns stay null. A
geocoder is any object with `reverse(lat, lon) -> GeoResult(address, name, category,
distance_m, source)`. It is called once per place.

| provider | setup | runtime |
|---|---|---|
| `OfflineOSMGeocoder([index.parquet, …])` | build the index once per study area: `geocode.build_index_for_point((lat, lon), dist_m, out)` (Overpass, needs internet once) or `geocode.build_index_from_xml(osm_xml, out)` (fully offline) | KD-tree lookup, no network; matches within 150 m |
| `NominatimGeocoder(url=…)` | a Nominatim server (self-host for volume; the public server allows ≤ 1 request/s) | HTTP per uncached place, with a JSON disk cache |

For a new upload, the app can choose an index by testing whether the bounding box of the
events falls inside a prepared area, or build one around the events' median position. In
this repo, `pol.build.geocoder_for` selects by dataset name; that is repo-specific and
should not be ported.

---

## 4. Run: `dwells.pol.pipeline.analyze(events, profile="mixed", params=None, min_duration_s=900, place_radius_m=150, geocoder=None) -> dict[str, DataFrame]`

Pass `params` from step 2. If you don't, they are built from `profile` and
`min_duration_s`. Internally:

| step | function | produces |
|---|---|---|
| a. per-event error | `uncertainty.with_covariance` (via `algorithms.common.prepare`) | 2×2 covariance and 1σ (`sigma_m`) per report, from whichever shape is present; `location_source` / `default_sigma_m` for shape 1 |
| b. coarse mask | `ua_mahalanobis.detect_with_members` | `coarse` = 1σ > 1 km **and** > 3 × the id's median 1σ, or the label is `ip` / `user_provided` |
| c. stay detection | `ua_mahalanobis._kernel` (numba, per id in time order) | per-event stay label |
| d. dwell rows | `common.dwells_from_labels`, `ua_mahalanobis._windows` | fused centroid and 1σ ellipse, first/last event, arrival/departure windows, estimated start/end, filtered by `min_duration_s` |
| e. places | `downstream.places.assign(dwells, radius_m=150)` | `place_id`: each id's visits merged by Mahalanobis distance (places never span ids) |
| f. visits | `pipeline._visit_table` | visit rows with extent, hull and ellipse rings, confidence 0–100 |
| g. presence | `stats.hour_overlap` | seconds per (id, date, UTC hour, place) |
| h. places | `stats.place_table`, `pipeline._place_geometry` | per-place profiles, scores, ranks, rings |
| i. geocoding | `pipeline._geocode` | address columns on places |
| j. movement | `stats.transitions` | `transitions` and `next_places` (Wilson 95% lower bound) |
| k. anomalies | `stats.anomalies` | unusual visits, also attached to `visits.anomalies` |
| l. quality | `quality.quality` | per-id input diagnostics |

Every table is then cast to its schema in `dwells.pol.schema` (`conform`). Columns
always exist with fixed dtypes, even when a table is empty, so the UI never has to guess.

**Returned tables** (full column reference: [outputs.md](outputs.md)):

| key | grain | main UI use |
|---|---|---|
| `events` | one per input report (same order as validated input) | member points, coarse/precise styling, source tooltip |
| `visits` | one per detected stay | dwell table, timeline, extent rings, radar overlay |
| `places` | one per (id, place) | map markers, top-10 lists, radar, weekday heatmap, addresses |
| `transitions` | (id, from_place, to_place) | arcs, directed graph |
| `next_places` | ranked next places per from_place | "where next" panel |
| `presence` | (id, date, hour, place) | presence calendar |
| `anomalies` | one per flagged visit | anomaly list |
| `quality` | one per id | data-quality panel, id picker |

`visit_id` and `place_id` are unique **within one `analyze` call**, across all ids.
Join keys are listed in [outputs.md](outputs.md).

---

## 5. Persist: `pipeline.write(tables, out_dir, meta=None)`

Creates `out_dir` and writes one `<table>.parquet` per key, plus `meta.json`. The meta
dict is written as given, with a `written` UTC timestamp added.

Use one directory per upload/run:

```
results/<upload_id>/
  events.parquet  visits.parquet  places.parquet  transitions.parquet
  next_places.parquet  presence.parquet  anomalies.parquet  quality.parquet
  meta.json
```

The current app reads these `meta.json` keys:

| key | used for |
|---|---|
| `profile` | default of the live-tuning profile control |
| `params` | display and reproducibility |
| `shape`, `with_source` | reloading the input for live re-runs (repo-specific, see §6) |

Also record the `min_duration_s`, upload id, row counts and code version, so that a
result can always be traced to the exact run that produced it.

---

## 6. UI reads (`app/data.py`)

Two viewers read the results: the Dash app (`app/app.py`) and the marimo notebook
(`notebooks/viewer.py`). Both are thin layers over shared, framework-independent modules:
`app/data.py` (loading, live re-runs, labels and colors), `app/views.py` (table rows,
place summary, quality stats, selection), `app/map_data.py` (map layer rows, view
and bounds) and `app/figures.py` (Plotly figures). Port those, then bind them to the
target UI.

* `pipeline.read(out_dir) -> dict[str, DataFrame]` reads every `*.parquet` in the
  directory, keyed by file stem: a single file (`pipeline.write`) or a directory of part
  files (the Spark runner). The app wraps it in an LRU cache (`data.load`).
* Most views show one identifier at a time: `data.tables(dataset, id)` filters every
  table to that id.
* The map layers (`app/layers.py`) use the rings (`ellipse_ring`, `hull_ring`: closed
  `[lon, lat]` lists) directly as deck.gl polygon paths. The charts (`app/figures.py`)
  use the fixed-length profile lists on `places` (`hour_presence_s`[24],
  `weekday_hour_presence_s`[168], and others).

**Live tuning** (the parameter panel) re-runs `analyze` for **one id** with the
analyst's parameters, keeps the result in memory (the `_live` cache, 16 entries) and
never overwrites the saved results. It needs the validated input events again. The repo
reloads them from the synthetic dataset folder (`io.load_events`), which is
repo-specific. In the application, **also persist the validated input** (for example
`results/<upload_id>/input.parquet`, written right after step 1) and reload it filtered
to the id.

---

## Scaling and porting notes

* **The work splits by id.** Detection, places, profiles, transitions and quality are
  all computed per id; nothing crosses ids. A large upload can be partitioned by id,
  with each partition calling `analyze`. The PySpark runner in [`spark/`](../spark/README.md)
  does exactly this (`groupBy("id").applyInArrow`) and writes the same tables, verified
  equal to a polars run.
* **Remap ids when partitioning.** `visit_id` and `place_id` start at 0 in each call.
  When merging partitions, offset each id's ids by the visits / places of all ids
  sorting before it (as `spark/dwells_spark/job.py` does) to get the polars numbering.
* **Memory:** keep each partition to a few million events. On WSL2, run batch jobs
  under a memory cap (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`).
* **Startup:** the first call compiles the numba kernel. With `cache=True`, this
  happens once per environment, not once per run.
* **UTC throughout.** Hours and weekdays in the output are UTC. Any local-time display
  must convert at the UI layer.
* `pol/build.py` and `io.py` are batch helpers for the repo's datasets. They are not
  part of the application path: steps 1–5 above are the whole interface.
