# Spark readiness review

Spark was not available locally (project constraint), so this is a design review
backed by the single-node scaling benchmarks (`docs/results/scaling.md`). The target is
millions of events per identifier, spread over many identifiers.

## The common execution pattern

Every detector in this repo works on one identifier's time-ordered events at a time,
with no information shared across identifiers. The natural Spark mapping is therefore:

```python
from pyspark.sql import functions as F

schema = "id string, dwell_id long, start_ts timestamp, end_ts timestamp, duration_s long, " \
         "lat double, lon double, n_events long, sigma_major double, sigma_minor double, azimuth double"

def detect_group(pdf):                      # pandas DataFrame for ONE id
    import polars as pl
    from dwells import schema as S, algorithms
    ev = S.validate_events(pl.from_pandas(pdf))
    return algorithms.get("ua_mahalanobis").detect(ev, min_duration_s=900, radius_m=100,
                                                   max_gap_s=7200).to_pandas()

dwells = (events.repartition("id")          # co-locate each id's events
                .groupBy("id").applyInPandas(detect_group, schema))
```

What this pattern needs:

* **Per-id memory.** `applyInPandas` materializes one id's events in one Python
  worker. At 1M events per id, shape-3 events take about 100 MB in Arrow/pandas,
  which is fine. What differs between algorithms is their own working memory
  (see below).
* **Executor dependencies.** Every executor needs numba, polars and h3, plus the
  library itself for the library wrappers (a geopandas/shapely stack for trackintel
  and movingpandas). Numba compiles once per worker process; `cache=True` helps
  only if the cache directory is writable and shared.
* **Skew.** Identifiers with extreme event counts (adtech bots, stationary IoT)
  dominate task time. Salting is not possible for sequential detectors, because
  one id's events must stay together. Two alternatives:
  * Split an id's events at long gaps (> max_gap_s) before the groupBy, keyed by
    (id, segment). This is exactly what the gap semantics already allow.
  * Cap the event count per id with a pre-filter.
* **Downstream steps.**
  * **Places, transitions, next location:** per id, same groupBy.
  * **Colocation:** the only step that genuinely crosses identifiers. It is
    implemented as an H3 bucket join (`downstream/colocation.py`): explode each
    dwell's cell plus its k=1 ring, equi-join on cell, then filter on time overlap
    and the spatial test. In Spark SQL that is an `explode` + join + filter with
    no UDF, apart from the Mahalanobis test.
  * **Diff-time colocation:** the same join with the time filter inverted.

## Per-algorithm assessment

| algorithm | per-id partitionable | working memory per id | global step | notes / Spark risk |
|---|---|---|---|---|
| li_centroid | yes | O(n) labels | none | trivial; one numba scan |
| ua_mahalanobis | yes | O(n) labels | none | trivial; one numba scan with k_break look-back |
| h3_occupancy | yes | O(n) + unique cells | none | cell ids and neighbor lists can be computed in SQL (`h3-spark`/Sedona H3 functions); only the run scan needs a UDF |
| h3_cell_dwell | yes | O(n) | none | same as above; accuracy weak (fragmentation) |
| hmm_stay_move | yes | O(n) back-pointers | none | linear Viterbi |
| ua_segmentation | yes | O(n) prefix sums + PELT candidate set | none | near-linear with pruning; worst case O(n^2) on a single very long stationary segment without change points (not observed) |
| st_dbscan | yes | O(n) + time-window neighbor scan | none | cost ~ n x events per eps_t window: dense GPS with a large eps_t is expensive |
| st_hdbscan | yes | sklearn HDBSCAN O(n log n) time, O(n) to O(n^2) memory depending on backend | none | heavy per id; not recommended for 1M-event ids |
| kalman_dwell | yes | dense grid over the whole span: span/dt x (4x4 x 2 + ...) floats | none | 60 days at 60 s gives 86k steps, fine; multi-year spans with sparse data allocate large mostly-empty grids |
| changepoint_pelt (ruptures) | yes | O(n) | none | pure-python PELT: impractically slow at 10^6 events per id |
| trackintel_sliding | yes (per user) | O(n) + geopandas | optional dataset-level location clustering | geopandas on executors; ~60k events/s |
| skmob_stops | yes | O(n) | none | slow python loop; must reattach uid; loses the trailing stay |
| fastmob_stays | yes | Arrow buffers | optional H3 place sharing | Rust, the fastest by far; young single-maintainer dependency |
| movingpandas_stops | yes (per trajectory) | window grows with stay length (quadratic time) | none | too slow for 10^6-event ids |
| infostop | **no** (as designed) | stationary events of all users | global Infomap over all users' stops | per-id use loses shared place ids; GPL/AGPL licensing |

## Recommendation for a Spark deployment

1. **Detection:** use `ua_mahalanobis` or `h3_occupancy` inside `groupBy(id,
   segment).applyInPandas`. Both are single-pass, constant-state scans that exploit
   error information, and they have the lowest executor risk. `ua_segmentation` is
   the most accurate option at a small extra cost.
2. **Places:** keep grouping per id (DBSCAN on dwell centroids, or the Mahalanobis
   merge when detectors report centroid uncertainty).
3. **Shared places and colocation:** use the H3 bucket join on dwells, not on raw
   events. Dwells are 2–4 orders of magnitude fewer than events.
4. **Next location:** per-id Markov counts can be computed as a Spark aggregation
   (`groupBy(id, from, to).count()`) and need no UDF.
