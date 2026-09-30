# fastmob_stays: fastmob `stay_locations` (wrapper)

Module: `src/dwells/algorithms/fastmob_stays.py`. Family `library`,
`USES_UNCERTAINTY = False`. The library internals are in
[../libraries/fastmob.md](../libraries/fastmob.md).

## Summary

fastmob reimplements the skmob stops algorithm in Rust (anchor radius, `>=` tests), and
by default it adds `include_last=True`. The wrapper makes one batch call on a polars frame
with the gap-segment number as `uid_col`. It then snaps the reported times back to real
fixes and counts member events itself.

## Algorithm (`detect`, `fastmob_stays.py:32-44`; `_count_events` `:47-64`)

1. `seg = gap_segments(events, max_gap_s)`. Passing `uid_col="seg"` matters: without an
   explicit uid, fastmob auto-detects columns and would treat an `id` column as one user.
2. `stay_locations(..., spatial_radius_km=radius_m/1000,
   minutes_for_a_stop=min_duration_s/60, no_data_for_minutes=max_gap_s/60,
   include_last=True)` (`:35-38`).
3. fastmob returns naive timestamps truncated to milliseconds. UTC is re-attached, and
   `start_ts`/`end_ts` are snapped forward to the real fix timestamps with `join_asof`
   (`:56-59`).
4. `n_events = i1 - i0 + inside`: events in `[start, end)`, plus the end fix when it is
   the segment's last fix **and** lies within `radius_m` of the anchor (`:62`). That is
   the `include_last` stay, whose end is the last fix inside rather than the first
   outside. `end_ts = max(start_ev, end_ev)`.

## Parameters

| Harness param | Library param |
|---|---|
| `radius_m` | `spatial_radius_km` (anchor = first point) |
| `min_duration_s` | `minutes_for_a_stop` |
| `max_gap_s` | Pre-split, plus `no_data_for_minutes` |

## Use of positional uncertainty

None.

## Output semantics

- `start_ts`: the first fix. `end_ts`: the first fix outside the stay, or the segment's
  last fix for the trailing stay.
- Centroid: coordinate-wise **median** (fastmob).
- `sigma_*`: null.

## Complexity and Spark-readiness

- The Rust kernel is parallel over users (rayon). It is the fastest library wrapper.
  Wrapper overhead is two polars `join_asof` calls.
- Per-id partitionable. In Spark, call it per partition (`mapInPandas` / `applyInPandas`
  with polars inside). rayon threads inside executors should be capped
  (`RAYON_NUM_THREADS`).

## Known limitations / failure modes

- Anchor-based, with no outlier tolerance.
- **The `inside` heuristic uses the wrapper's own test.** It uses haversine
  `< radius_m`, which may disagree with fastmob's internal test at exact boundary
  distances (fastmob uses `>=` to break). The effect is off-by-one in `n_events` only.
- fastmob drops consecutive exact duplicates (same lat, lon, t) internally (see the
  library doc). The wrapper's `n_events` counts them.
- Young 0.x library whose API is still moving. Pin the version.
- No `KNOWN_LIMITATIONS` xfails.

## References

- fastmob: https://github.com/gefgu/fastmob (0.3.2)
- Hariharan & Toyama (2004). doi:10.1007/978-3-540-30231-5_8
- [../libraries/fastmob.md](../libraries/fastmob.md)
