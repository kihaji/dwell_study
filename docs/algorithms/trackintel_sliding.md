# trackintel_sliding: trackintel sliding-window staypoints (wrapper)

Module: `src/dwells/algorithms/trackintel_sliding.py`. Family `library`,
`USES_UNCERTAINTY = False`. The library internals (the anchor rule, `>=` tests, gap
handling, `include_last`, circular-mean centroid) are detailed in
[../libraries/trackintel.md](../libraries/trackintel.md). This page covers only how the
wrapper maps parameters and which semantics it keeps.

## Summary

The wrapper splits each id on gaps longer than `max_gap_s` and hands every gap-free
segment to trackintel as its own `user_id`. It then calls
`Positionfixes.generate_staypoints(method="sliding")`, which implements the Li et al.
(2008) anchor-radius window: the radius is measured from the window's first point.

## Algorithm (`detect`, `trackintel_sliding.py:29-47`)

1. `seg = common.gap_segments(events, max_gap_s)`. A segment number increments on a new
   id or on a gap longer than `max_gap_s`.
2. Build a GeoDataFrame with `user_id = seg`, `tracked_at = timestamp` and EPSG:4326
   point geometry.
3. Call `generate_staypoints(method="sliding", dist_threshold=radius_m,
   time_threshold=min_duration_s/60, gap_threshold=max_gap_s/60, include_last=True)`
   (`:34-36`).
4. `n_events` = the number of positionfixes whose `staypoint_id` is that staypoint
   (`:39`). Segments are joined back to `id`.

Why pre-split on gaps: trackintel's native `gap_threshold` *discards* a window that
spans a large gap. Pre-splitting makes a gap close the stay instead, which matches the
harness semantics. `gap_threshold = max_gap_s` is also passed, but inside a segment no
gap exceeds it.

## Parameters

| Harness param | Library param |
|---|---|
| `radius_m` | `dist_threshold` (m, haversine, from the anchor = first point of the window) |
| `min_duration_s` | `time_threshold = min_duration_s / 60` (minutes) |
| `max_gap_s` | Pre-split into segments, plus `gap_threshold = max_gap_s / 60` |

`include_last=True` emits the stay still open at the end of each segment.

## Use of positional uncertainty

None.

## Output semantics

- `start_ts`: the first fix of the stay.
- `end_ts`: trackintel `finished_at`, the **first fix outside** the stay (the next
  positionfix). For the trailing `include_last` stay it is the segment's last fix. This
  is the main semantic difference from our event-based detectors, whose end is the last
  fix inside. It lengthens durations by one sampling interval.
- Centroid: trackintel's mean of the **distinct** member positions (see the library
  doc).
- `sigma_*`: null.

## Complexity and Spark-readiness

- trackintel iterates per user in Python/pandas. The window growth makes it O(n·k) per
  segment. It is slower than the numba sequential detectors, and it carries heavy
  dependencies (geopandas, osmnx and others).
- Per-id partitionable: `groupBy("id").applyInPandas` with trackintel installed on the
  executors. There is no global step.

## Known limitations / failure modes

- Anchor-based: a first fix that is an outlier or on the approach anchors the whole
  window, so the stay is shifted or split.
- No outlier tolerance. Coarse fixes break stays.
- `end_ts` is the next fix, so after a long intra-segment gap (up to `max_gap_s`) the
  stay is stretched to the next fix even though the id may have left much earlier.
- No `KNOWN_LIMITATIONS` xfails.

## References

- Martin, H. et al. (2023). Trackintel: an open-source Python library for human mobility analysis. CEUS 101. doi:10.1016/j.compenvurbsys.2023.101938
- Li, Q. et al. (2008). doi:10.1145/1463434.1463477
- [../libraries/trackintel.md](../libraries/trackintel.md)
