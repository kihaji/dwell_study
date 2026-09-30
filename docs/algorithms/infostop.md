# infostop: Infostop stationary events + Infomap places (wrapper)

Module: `src/dwells/algorithms/infostop.py`. Family `library`,
`USES_UNCERTAINTY = False`. The library internals are in
[../libraries/infostop.md](../libraries/infostop.md).

## Summary

Infostop has two stages. First it finds sequential stationary groups: a fix joins the
current group if it is within `r1` of the group's **running median** and within
`max_time_between` of the previous fix, and a group needs at least 2 fixes and at least
`min_staying_time`. Second, it links the groups' medians within `r2` into a network and
clusters them with Infomap into stop locations. The output is a per-fix location label.
The wrapper turns runs of consecutive equal labels into visits (dwells).

## Algorithm (`detect`, `infostop.py:34-49`)

1. `seg = gap_segments(events, max_gap_s)`. Each segment becomes one `[lat, lon, t]`
   array, and all segments of all ids go into **one** `fit_predict` call (multi-user
   mode, `:42`), so place labels are shared across ids.
2. `Infostop(r1=radius_m, r2=radius_m, min_staying_time=min_duration_s,
   max_time_between=max_gap_s)` (`:40`).
3. If no trace has a stationary event, Infostop raises "No stop events found", and the
   wrapper returns an empty frame.
4. Visits: a new run starts when the place label or the segment changes. Fixes with
   label `-1` (moving) are unlabelled (`:47-48`). The wrapper builds these runs itself
   because `postprocess.compute_intervals` drops the last stop.
5. `dwells_from_labels(events, labels, min_duration_s, centroid="median")` (`:49`).
   This applies the duration test again at the visit level.

## Parameters

| Harness param | Library param |
|---|---|
| `radius_m` | `r1` (running-median radius) and `r2` (network link distance) |
| `min_duration_s` | `min_staying_time` (s), then re-checked per visit |
| `max_gap_s` | Pre-split, plus `max_time_between` |

**Constraint:** Infostop asserts `max_time_between > min_staying_time`
(`infostop/models.py:_hyperparam_assertions`). A combination with
`max_gap_s <= min_duration_s` raises `AssertionError`, which is not caught (verified with
`min_duration_s=3600, max_gap_s=1800`). In the GPS sweep grid (`max_gap_s ∈ {1800, 7200}`
x `min_duration_s ∈ {300, 900, 1800, 3600}`), the cells `md ∈ {1800, 3600}` at
`gap = 1800` therefore cannot run. `results/sweep.parquet` holds 228 infostop rows versus
240 for the other shape-1 algorithms: 12 missing, which is exactly 2 md x 3 radii x
2 GPS dev datasets.

## Use of positional uncertainty

None.

## Output semantics

- `start_ts` / `end_ts`: first / last fix of the visit (the last stay is kept).
- Centroid: coordinate-wise median of the visit's fixes.
- `sigma_*`: null.
- Place labels are not exported. Only visits are. Two consecutive stationary groups with
  the same Infomap label merge into one visit.

## Complexity and Spark-readiness

- Stage 1 is C++ and sequential per trace. Stage 2 is **global**: one BallTree plus
  Infomap over all stationary-event medians of all ids. The docstring records
  super-linear throughput: about 570k events/s at 54k events and about 200k events/s at
  890k.
- As wrapped, it is **not per-id partitionable** because of the cross-id Infomap step.
  It can be run per id (`applyInPandas`, one `fit_predict` per id). Because visits are
  per-id label runs, the result should change little, but Infomap community structure
  can differ once other ids' medians are removed.
- Infomap is unseeded, so runs are not bit-reproducible.

## Known limitations / failure modes

- The parameter constraint above.
- **Licensing**: GPL/AGPL (infomap 1.0.6 is AGPL-3.0). See the library doc.
- `r2 = r1 = radius_m` ties the place scale to the stay scale.
- No `KNOWN_LIMITATIONS` xfails.

## References

- Aslak, U., Alessandretti, L. (2020). Infostop: scalable stop-location detection in multi-user mobility data. arXiv:2003.14370
- Rosvall, M., Bergstrom, C. T. (2008). Maps of random walks on complex networks reveal community structure. PNAS 105(4). doi:10.1073/pnas.0706851105
- [../libraries/infostop.md](../libraries/infostop.md)
