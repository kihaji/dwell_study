# h3_occupancy: H3 anchor-ring occupancy with hysteresis (new)

Module: `src/dwells/algorithms/h3_occupancy.py`. Family `grid`,
`USES_UNCERTAINTY = True` (coarse-event filter). New in this repo. It addresses the
boundary flicker of [h3_cell_dwell](h3_cell_dwell.md).

## Summary

The resolution is the finest H3 resolution whose average edge is at least `radius_m`. A
run is anchored at its first informative event's cell. Later informative events
continue the run while their cell is the anchor or one of its 6 neighbours
(`grid_disk(anchor, 1)`). The run breaks after `k_break` consecutive misses or on a gap.
Coarse events (1-sigma greater than 3 x the edge) are ignored entirely.

## Algorithm

`detect` (`h3_occupancy.py:111-127`):

1. `r = resolution_for(radius_m)`: the finest `r` with `EDGE_M[r] >= radius_m`
   (`:38-40`). Examples: `radius_m` 50 gives res 10 (edge 75.9 m), 100 gives res 9
   (200.8 m), 250 gives res 8 (531.4 m), 1000 gives res 7 (1406.5 m), 2000 gives res 6
   (3724.5 m).
2. `coarse_i ⇔ sqrt((C_ee + C_nn)/2) > coarse_factor · EDGE_M[r]` (`:116`). The
   threshold is 602 m at res 9 and 228 m at res 10. The docstring records that a 1x
   threshold made every CDR event coarse in the dev sweep (F1 0.42 → 0.23), hence the
   default of 3.
3. Cells per event (`:118`). A unique-cell table holds a 7-slot disk per unique cell,
   with slots that are not in the table set to -2 (`:119-124`).
4. `_kernel` (`:43-108`), per id:
   - A gap close (`:57`) happens when an **informative** event arrives more than
     `max_gap_s` after the last accepted event (`t_last`), or at the end of the track.
     The run is accepted if `n_inf >= min_events` and `t_last - t_first >=
     min_duration_s`, otherwise its labels are reset. Pending misses restart at
     `first_miss`.
   - Coarse events are skipped (`:71-73`): they never start, break or extend a run, and
     they are **never labelled**.
   - Seed: the anchor is the event's cell.
   - An informative event whose cell is in the anchor's disk is labelled, with
     `n_inf += 1`, `t_last = t_i` and the miss counter reset.
   - Otherwise it is a miss. After `k_break` consecutive misses the run is closed at its
     last accepted event (accept/reject), and processing restarts at `first_miss`
     (`:92-106`).
5. `dwells_from_labels(events, labels, min_duration_s)`: mean centroid.

### Effective spatial tolerance (important)

The acceptance region is the 7-hex "flower" around the anchor cell's centre. Its
boundary lies between `2·e` (concave notches) and `sqrt(7)·e ≈ 2.65·e` from the anchor
centre, where `e` is the edge length. Because `e >= radius_m`, and at `radius_m = 100` in
fact `e = 200.8` m, the tolerance is several times `radius_m`. Monte Carlo (2000 random
placements per distance, `radius_m = 100`, displacement from the anchor event to a later
event):

| Displacement (m) | 100 | 200 | 300 | 400 | 500 | 600 | 700 |
|---|---|---|---|---|---|---|---|
| Fraction accepted | 1.00 | 1.00 | 0.96 | 0.72 | 0.41 | 0.11 | 0.003 |

So `radius_m` controls the grid, not the dwell extent. At the harness radius of 100 m,
stays are effectively "within about 300-500 m of the first fix's cell". Verified effect:
for a 1.4 m/s walk between two 30-minute stays (60 s sampling), h3_occupancy reports
0-35 and 53-89 min where the truth is 0-30 and 60-90. The stays extend about 5-7 min
into the walk on each side, whereas the sequential detectors report about 0-30 and 59-89.

### Worked intuition for coarse points

At `radius_m = 100` (res 9), a cell-tower fix with σ = 1274 m exceeds 602 m, so it is
skipped. It neither breaks the run (it is not a miss) nor extends it (it never updates
`t_last` and is never labelled). A 600 m-sigma adtech fix just below the threshold is
treated as informative. It is likely to land outside the disk and count as a miss, and
a single miss is tolerated (`k_break = 2`).

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | Selects the resolution (edge >= radius_m), and through it the tolerance (about 2-2.65 edges) and the coarse threshold (3 edges) |
| `max_gap_s` | Max time between accepted informative events |
| `min_duration_s` | Min `t_last - t_first` of accepted events |

Algorithm-specific defaults: `k_break = 2`, `min_events = 2`, `coarse_factor = 3.0`,
`default_sigma_m = 10`. The uncommitted working-tree diff changes `coarse_factor` from
1.0 to 3.0 and adds the rationale to the docstring.

## Use of positional uncertainty

Uncertainty is used only for the per-event coarse filter. The grid is sized from
`radius_m`, not from errors (unlike h3_cell_dwell), and informative fixes are all
treated alike.

## Output semantics

- `start_ts` / `end_ts`: first / last accepted informative event.
- Centroid: arithmetic mean lat/lon of the **accepted informative** events. Coarse and
  missed events are excluded. This is not the cell centre.
- `sigma_*`: null. `n_events` counts accepted informative events only. Unlike
  ua_mahalanobis, coarse events are not attached.

## Complexity and Spark-readiness

- A per-event Python H3 call (`latlng_to_cell`), a per-unique-cell `grid_disk`, then an
  O(n) numba scan (near-linear, with restarts of up to `k_break - 1` events).
- Everything except the run scan is a per-event map: the cell (`h3_longlatash3` or an
  h3 UDF), the coarse flag, and the ring membership, which can be computed as a join
  against `explode(h3_kring(anchor, 1))`. The anchor-hysteresis scan is stateful (the
  anchor changes only on a break), so it needs a per-id sorted scan in `applyInPandas`.
  An SQL gaps-and-islands formulation would only approximate it, for example by testing
  ring membership against the previous event's cell instead of the anchor.
- Per-id partitionable with no global step. The resolution depends only on `radius_m`.

## Known limitations / failure modes

- **Loose spatial tolerance** (see the table). It merges nearby places closer than
  about 2-2.65 edges and extends stays into slow departures and arrivals. Choosing a
  smaller harness `radius_m` refines the grid but also lowers the coarse threshold.
- The anchor is fixed to the first informative event's cell. An arrival fix on the
  approach anchors the run off-centre.
- Grid anisotropy and H3 distortion: the tolerance depends on where the place falls
  relative to cell centres, and on latitude (the average edge is used).
- If every fix is coarse, there are no dwells.
- No `KNOWN_LIMITATIONS` xfails. It passes the coarse-point tests.

## References

- H3: https://h3geo.org ; resolution table: https://h3geo.org/docs/core-library/restable ; `grid_disk`: https://h3geo.org/docs/api/traversal
- Uber Engineering (2018). H3: Uber's hexagonal hierarchical spatial index. https://www.uber.com/blog/h3/
