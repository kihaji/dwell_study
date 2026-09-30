# li_centroid: sequential running-centroid stay detection

Module: `src/dwells/algorithms/li_centroid.py`. Family `sequential`,
`USES_UNCERTAINTY = False`. This is a port of `/home/kihaji/projects/new_ellipses/dwell_detection.py`
(`DwellConfig`, `_detect_dwells_for_device`).

## Summary

The detector walks each id's events in time order and keeps one open cluster. The next
event breaks the cluster when it is more than `max_gap_s` after the previous event, or
more than `radius_m` from the cluster's **running mean** centroid. A closed cluster is a
dwell if it has at least `min_events` events and spans at least `min_duration_s`. This is
the "time-based clustering" of Kang et al. (2004). Li et al. (2008) instead anchor the
radius at the cluster's *first* point, and trackintel, skmob and fastmob implement that
variant.

## Algorithm (`_kernel`, `li_centroid.py:26-51`)

For each id (index range `[a, b)` from `common.prepare`), with positions `(x_i, y_i)` in
the id's local ENU frame:

1. Open a cluster at `c0 = a` with sums `sx = x_a, sy = y_a`.
2. For `i = a+1 .. b` (with `i = b` meaning end of track):
   - `cnt = i - c0` and the running centroid is `m = (sx/cnt, sy/cnt)`.
   - Close if `i == b`, or `t_i - t_{i-1} > max_gap_s`, or `|z_i - m| > radius_m`
     (`li_centroid.py:40`).
   - On close, if `i - c0 >= min_events` and `t_{i-1} - t_{c0} >= min_duration_s`,
     label events `c0..i-1` as one dwell. Then re-open at `i`, so the breaking event
     seeds the next cluster.
   - Otherwise add `z_i` to the sums.
3. `dwells_from_labels(events, labels, min_duration_s)` aggregates the labels
   (`li_centroid.py:58`).

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | Max distance from the running mean (strict `>` breaks) |
| `max_gap_s` | Max gap between **consecutive** events (strict `>` breaks) |
| `min_duration_s` | Min `last - first` event time (inclusive `>=`) |

Algorithm-specific: `min_events = 3`. The defaults `300 s / 100 m / 1800 s / 3` equal
`DwellConfig` (`new_ellipses/dwell_detection.py:36-43`).

## Use of positional uncertainty

None. `prepare()` computes covariances, but the kernel ignores them.

## Output semantics

- `start_ts` / `end_ts`: first / last event inside the cluster. The breaking event is not
  included.
- Centroid: arithmetic mean of member lat/lon (`dwells_from_labels` default `mean`).
  This matches `_compute_centroid` in the original.
- `sigma_*`: null.
- `n_events`: member count.

## Comparison with the original

| Aspect | Original (`dwell_detection.py:114-210`) | Port |
|---|---|---|
| Break rule | Gap to previous cluster member > max gap; distance to mean lat/lon > radius | Same |
| Distance | Flat-earth with `METERS_PER_DEGREE_LAT` and a local midpoint latitude (`:82-96`) | Equirectangular in a per-id frame anchored at the id's median (`common.py:31-41`) |
| Breaking point | Starts the next cluster | Same |
| Validation | `len >= min_observations` and `duration >= min_dwell_time_sec` | Same |
| Timestamps | Truncated to whole seconds (`datetime64[s]`) | Microseconds |
| Extras | Per-observation weights (`_compute_weights`), global mode | Not ported (not needed for detection) |

The only behavioural deviation is the projection. For an id whose track spans more than
about 50 km from its median, the east-west scale uses `cos(lat_median)` rather than the
local latitude, so radius tests are slightly distorted far from home. The seconds
truncation in the original can change a `>=` duration test by less than 1 s.

The docstring also names `ellipses/colocation.py _find_dwell_periods_numba` as the same
family, with defaults of 60 min gap and 500 m radius (`ellipses/colocation.py:22-23`).

## Complexity and Spark-readiness

- O(n) time and O(1) state per id (running sums), numba-compiled.
- Per-id partitionable with no global step:
  `df.groupBy("id").applyInPandas(detect_pdf, schema=DWELLS)`, with events sorted by
  timestamp inside the UDF.
- Pure SQL windows cannot express it, because the reset depends on a running mean that
  resets on a data-dependent condition (a stateful scan). Options are `applyInPandas`,
  a Scala `mapGroupsWithState`, or an Arrow UDF over sorted partitions. A single id
  with 1M+ events fits easily, since only O(n) arrays are held.

## Known limitations / failure modes

- **Drift.** The running mean follows slow movement: points within `radius_m` of the
  current mean are accepted even as the mean walks, so slow walking can be absorbed.
- **Outliers break stays.** A single multipath fix more than `radius_m` away closes the
  dwell and starts a new cluster. Typically the stay is split in two, or the second half
  is lost if it is too short. There is no outlier tolerance (compare the `k_break` logic
  in ua_mahalanobis).
- **Coarse points.** With CDR/adtech data every km-scale fix breaks a cluster. Radius
  tuning up to 2000 m (CDR grid) trades this for merging neighbouring places.
- **Sparse sampling.** A stay observed by a single event cannot meet `min_events = 3`.
- No `KNOWN_LIMITATIONS` xfails. It passes the contract tests.

## References

- Kang, J. H. et al. (2004). Extracting places from traces of locations. WMASH '04. doi:10.1145/1024733.1024748 (running-centroid variant)
- Li, Q. et al. (2008). Mining user similarity based on location history. doi:10.1145/1463434.1463477 (anchor variant)
- Original: `/home/kihaji/projects/new_ellipses/dwell_detection.py`
