# skmob_stops: scikit-mobility `detection.stops` (wrapper)

Module: `src/dwells/algorithms/skmob_stops.py`. Family `library`,
`USES_UNCERTAINTY = False`. The library internals are in
[../libraries/scikit-mobility.md](../libraries/scikit-mobility.md).

## Summary

This wraps `skmob.preprocessing.detection.stops`, a Lachesis-style detector (Hariharan &
Toyama 2004): the radius is measured from the window's first point, with strict `>`
tests on distance and time. Each id is split on gaps longer than `max_gap_s`, and the
segment number is passed as skmob's `uid`.

## Algorithm (`detect`, `skmob_stops.py:32-53`)

1. `seg = gap_segments(events, max_gap_s)`. A copy `seg_` is carried as an extra column,
   because under pandas 3 skmob's `groupby.apply` drops the uid column from the output
   (`:35-37`). The copy is taken from the stop's last row.
2. `detection.stops(tdf, spatial_radius_km=radius_m/1000,
   minutes_for_a_stop=min_duration_s/60, no_data_for_minutes=max_gap_s/60,
   leaving_time=True)` (`:39-40`).
3. `start_ts = datetime` (first fix) and `end_ts = leaving_datetime` (first fix
   outside).
4. `n_events` = the id's events in `[start, end)`. Both times are located by forward
   `join_asof` into the row index, and the count is the index difference (`:47-52`).

The module also patches `shapely.ops.cascaded_union = unary_union` (removed in shapely 2)
so that `skmob.tessellation` imports (`:15`).

## Parameters

| Harness param | Library param |
|---|---|
| `radius_m` | `spatial_radius_km = radius_m/1000` (from the anchor) |
| `min_duration_s` | `minutes_for_a_stop = min_duration_s/60` |
| `max_gap_s` | Pre-split, plus `no_data_for_minutes = max_gap_s/60` |

## Use of positional uncertainty

None.

## Output semantics

- `start_ts`: the first fix of the stay. `end_ts`: `leaving_datetime`, the first fix
  outside.
- Centroid: coordinate-wise **median** of member lat/lng (skmob).
- `sigma_*`: null. `n_events`: events in `[start, end)`.

## Complexity and Spark-readiness

- Pure Python per-point loop in `_stops_array`, so O(n) with a large constant.
  pandas/geopandas overhead. Heavy and partly undeclared dependencies (see the library
  doc).
- Per-id partitionable (`applyInPandas`). There is no global step.

## Known limitations / failure modes

- **No `include_last`**: the stay still open at the end of a segment is never emitted.
  Because segments are cut at gaps, **every stay that ends at a gap or at the end of the
  track is lost**. This behaviour is native to skmob and is kept deliberately. It
  appears as strict xfails in `tests/test_algorithms.py:67-70` for
  `test_detects_easy_dwells` (third, trailing stay lost), `test_long_gap_splits`,
  `test_short_stay_below_min_duration_dropped` and `test_multiple_ids_kept_separate`.
  In sparse data (CDR/adtech), where gaps close most stays, recall drops sharply.
- Anchor-based, with no outlier tolerance.

## References

- Pappalardo, L. et al. (2022). scikit-mobility: a Python library for the analysis, generation, and risk assessment of mobility data. J. Stat. Softw. 103(4). doi:10.18637/jss.v103.i04
- Hariharan, R., Toyama, K. (2004). Project Lachesis. doi:10.1007/978-3-540-30231-5_8
- [../libraries/scikit-mobility.md](../libraries/scikit-mobility.md)
