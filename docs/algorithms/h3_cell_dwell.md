# h3_cell_dwell: sessions + contiguous H3 cell runs

Module: `src/dwells/algorithms/h3_cell_dwell.py`. Family `grid`, `USES_UNCERTAINTY = True`
(uncertainty only sizes the grid). This is a port of `/home/kihaji/projects/ellipses/hexbin.py`
(`select_h3_resolution`, `segment_sessions`, `compute_cell_dwell_times`).

## Summary

The detector first cuts each id's track into coarse **sessions** using a running-mean
centroid with a 500 m radius and a `max_gap_s` gap. It then maps each event to an H3 cell
at one dataset-wide resolution, sized so that an average error ellipse fits in a cell. A
dwell is a maximal run of consecutive events in the same cell within a session that spans
at least `min_duration_s`.

## Algorithm (`detect`, `h3_cell_dwell.py:51-61`)

1. **Sessions** (`:55`): reuse `li_centroid._kernel` with `radius = session_radius_m`,
   `max_gap = max_gap_s`, `min_events = 1`, `min_duration = 0`. With these arguments
   every cluster is labelled, so labels are session ids. The rule is the same as
   `segment_sessions` (`hexbin.py:522-612`): a new session starts when the gap to the
   previous event exceeds the threshold, or when the haversine distance to the running
   mean lat/lon exceeds the radius.
2. **Resolution** (`:56`, `select_resolution` `:34-39`, `_avg_semi_major` `:42-48`):
   `need = 1.16 * avg_semi_major`, and the chosen resolution is the finest `r` in
   15..0 with `average_hexagon_edge_length(r) >= need`. The factor 1.16 is
   `2/sqrt(3)`: a hexagon's inscribed diameter is `sqrt(3) * edge`, and requiring it to
   be at least the ellipse diameter `2a` gives `edge >= 2a/sqrt(3) = 1.1547a`. The
   original rounds this to 1.16 (`hexbin.py:131-156`). The average semi-major is taken
   as follows:
   - shape 3: `mean(semiMajor)`, **at whatever probability the data states** (so p68,
     p95 and mixed data yield different resolutions for identical positions);
   - shape 2: `mean(horizontalAccuracy) / 1.1774 * 2.4477`, the 95% circle radius
     (k(0.95) = 2.4477);
   - shape 1: `radius_m`.
3. **Cells** (`:57-58`): `h3.latlng_to_cell(lat, lon, res)` per event.
4. **Runs** (`:59-60`): a new label starts whenever the session or the cell changes
   between consecutive events. `dwells_from_labels(events, labels, min_duration_s)`
   keeps runs whose `last - first >= min_duration_s`. This is the same span logic as
   `compute_cell_dwell_times` (`hexbin.py:615-681`): span = time of last event in cell -
   time of first event in cell.

Worked sizes (`h3.average_hexagon_edge_length`, m): res 12 = 10.8, 11 = 28.7,
10 = 75.9, 9 = 200.8, 8 = 531.4, 7 = 1406.5, 6 = 3724.5. Examples of the resolution
chosen for shape-2 data with sigma = 5 / 10 / 30 / 100 / 1000 m: res 11 / 11 / 9 / 8 / 6.
For shape-3 p68 data with the same sigmas: res 12 / 11 / 10 / 9 / 6.

## Parameters

| Harness param | Meaning here |
|---|---|
| `max_gap_s` | Session gap (consecutive events, strict `>`) |
| `min_duration_s` | Minimum cell-run span (inclusive) |
| `radius_m` | **Only** the fallback "semi-major" for shape 1 (sets the cell size). It does not bound the dwell extent for shapes 2/3 |

Algorithm-specific: `session_radius_m = 500`. Defaults `min_duration_s = 1800` and
`max_gap_s = 3600` equal the original `TemporalConfig` (`hexbin.py:70-75`: 60 min,
500 m, 30 min).

## Use of positional uncertainty

Uncertainty only chooses the grid resolution, once, from the mean over **all events of
all ids in the frame**. It never affects membership. Coarse fixes in the data enlarge the
cells for everyone. A mixed-source dataset (for example adtech with IP-geolocation fixes)
therefore gets km cells even for its precise ids.

## Output semantics

- `start_ts` / `end_ts`: first / last event of the cell run.
- Centroid: mean lat/lon of the run's events (not the cell centre).
- `sigma_*`: null.
- `n_events`: run length.

## Comparison with the original

| Aspect | Original | Port |
|---|---|---|
| Session distance | haversine to running mean lat/lon | per-id equirectangular ENU (`common.prepare`) |
| Edge-length table | Rounded constants (`hexbin.py:25-42`, e.g. res 10 = 76, res 12 = 11) | `h3.average_hexagon_edge_length` (75.9, 10.8). This can shift the resolution by one exactly at a boundary, for example `need` in (75.9, 76] |
| Output | Per-cell temporal metrics (`max_continuous_dwell_minutes`, `is_dwell_location` if max span >= 30 min, visit counts) | Per-run dwells (a run meeting `min_duration_s` is a dwell) |
| Semi-major input | `obs.semi_major.mean()` (whatever confidence the source uses) | Same for shape 3; shape 2 is converted to a 95% radius; shape 1 uses `radius_m` |

## Complexity and Spark-readiness

- O(n) per id: sessions (numba) plus a Python loop over events for `latlng_to_cell`.
  The per-event Python H3 call dominates, at about 1 µs to several µs per event.
- **One global step**: the resolution depends on a dataset-wide mean
  (`_avg_semi_major`). In Spark, compute `agg(mean(semiMajor))` first and broadcast the
  result. Results for an id therefore depend on which other ids are in the batch. For a
  per-id result, compute the mean per id instead (a deviation from the original).
- Cells are a per-event map (`h3_longlatash3` in Databricks, or the h3 Python UDF). Cell
  runs are a gaps-and-islands window
  (`lag(cell) over (partition by id order by ts)`, then `sum(new_run)` over the window).
  The session step is the stateful running-mean scan, so it needs `applyInPandas`.

## Known limitations / failure modes

- **Boundary flicker** (xfail in `tests/test_algorithms.py:63-64`): with 5 m GPS error
  the grid is about 28 m (res 11). Stay jitter crosses cell edges, fragmenting one stay
  into many short runs, most of which fall below `min_duration_s`. The docstring states
  this known weakness (`h3_cell_dwell.py:12-13`). `h3_occupancy` addresses it with
  anchor-ring hysteresis.
- **Dataset-dependent resolution** (see above). One CDR-like id in a GPS batch makes the
  cells coarse for all.
- **Probability dependence**: p68 and p95 descriptions of the same error give different
  resolutions, with p95 about 1.6x larger `need`.
- The 500 m session radius caps the dwell extent. Places larger than about 500 m of
  wander are split.

## References

- H3: https://h3geo.org ; hexagon edge lengths: https://h3geo.org/docs/core-library/restable
- Original: `/home/kihaji/projects/ellipses/hexbin.py`
