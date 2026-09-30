# movingpandas_stops: MovingPandas `TrajectoryStopDetector` (wrapper)

Module: `src/dwells/algorithms/movingpandas_stops.py`. Family `library`,
`USES_UNCERTAINTY = False`. The library internals are in
[../libraries/movingpandas.md](../libraries/movingpandas.md).

## Summary

This is a **diameter-based** detector. A stop is a window whose minimum-rotated-rectangle
diagonal is `< max_diameter` and that spans at least `min_duration`. Each gap-free
segment becomes one `Trajectory`, and the wrapper uses only the stop time ranges,
computing members and centroids itself.

## Algorithm (`detect`, `movingpandas_stops.py:36-56`)

1. `seg = gap_segments(events, max_gap_s)`. MovingPandas has no gap handling of its own.
2. `TrajectoryCollection(pdf, traj_id_col="seg", ...)`, then
   `TrajectoryStopDetector(tc).get_stop_time_ranges(max_diameter=2*radius_m,
   min_duration=timedelta(seconds=min_duration_s))` (`:41-42`).
3. For each range `[t_0, t_n]`, label the events of that segment with
   `t_0 <= t <= t_n` (searchsorted `left`/`right`, `:51-55`).
4. `dwells_from_labels(events, labels, 0, centroid="median")` (`:56`). The library
   already enforced `min_duration`. This is equivalent to `get_stop_points` without its
   per-stop pandas overhead.

## Parameters

| Harness param | Library param |
|---|---|
| `radius_m` | `max_diameter = 2 * radius_m` (geodesic metres for EPSG:4326) |
| `min_duration_s` | `min_duration` (timedelta) |
| `max_gap_s` | Pre-split only |

## Use of positional uncertainty

None.

## Output semantics

- `start_ts`: the first fix of the stop window. `end_ts`: the **last fix inside** (the
  library's `t_n`). The open stop at a segment's end is always emitted.
- Centroid: coordinate-wise median of the fixes in `[start, end]`.
- `sigma_*`: null.
- `n_events` counts all events in the range, including duplicate timestamps that
  MovingPandas itself drops.

## Complexity and Spark-readiness

- Per fix: a geodesic bbox plus a minimum rotated rectangle over the growing window,
  which is O(k²) per stop. The module docstring records about 7k events/s on
  `dev_urban_gps` (54k events in 7.3 s). This is the slowest wrapper.
- Per-id partitionable (`applyInPandas`). `n_processes` exists in the library but is
  unused here.

## Known limitations / failure modes

- A diameter test with `2·radius_m` is not equivalent to a radius test. An elongated
  cloud can fail where a radius detector accepts it, and vice versa.
- A single outlier inflates the rectangle and ends the stop.
- Scalability is the main weakness for GPS-dense ids.
- No `KNOWN_LIMITATIONS` xfails.

## References

- Graser, A. (2019). MovingPandas: efficient structures for movement data in Python. GI_Forum 7(1). doi:10.1553/giscience2019_01_s54
- [../libraries/movingpandas.md](../libraries/movingpandas.md)
