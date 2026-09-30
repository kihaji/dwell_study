# kalman_dwell: Kalman/RTS-smoothed track, then greedy grid dwells

Module: `src/dwells/algorithms/kalman_dwell.py`. Family `sequential`,
`USES_UNCERTAINTY = True`. Port of new_ellipses branch `cotravel-research`:
`cotravel_cointegration/series.py` (`_kalman_rts`, `kalman_series_for`) and
`cotravel_cointegration/place_follower.py` (`_dwells`, `detect_dwells`), read with
`git -C /home/kihaji/projects/new_ellipses show cotravel-research:<path>`.

## Summary

The detector splits each id on gaps longer than `max_gap_s`. It smooths every segment
with a constant-velocity Kalman filter and Rauch-Tung-Striebel (RTS) smoother, using each
fix's own error covariance as measurement noise, and outputs the smoothed position on a
global regular grid `t_k = k·dt_s`. Dwells are then found on the grid: runs whose smoothed
positions stay within `radius_m` of their running mean for at least
`round(min_duration_s/dt_s)` steps.

## Model

State `s = [e, n, v_e, v_n]^T` in the id's local ENU frame (`common.prepare`).

- Transition over `Δt`: `F = [[I, Δt·I], [0, I]]`.
- Process noise: continuous white-noise acceleration with spectral density
  `q = accel_sigma²` (m²/s³) per axis (Bar-Shalom et al. 2001, §6.2):

  ```
  Q = q · [[Δt³/3 · I, Δt²/2 · I],
           [Δt²/2 · I, Δt    · I]]
  ```
  (`kalman_dwell.py:47-50`).
- Measurement: `z_i = H s + w_i`, `H = [I 0]`, `w_i ~ N(0, R_i)` with `R_i` the event's
  covariance `[[C_ee, C_en], [C_en, C_nn]]` from `with_covariance`. The call passes
  `(a.cee, a.cen, a.cnn)` into parameters `(r_ee, r_en, r_nn)` (`:131`), which is the
  correct order.
- Initialisation (`:38-41`): position = first observation, position covariance = that
  observation's `R`, velocity variance 100 (sd 10 m/s), zero cross terms. The first
  observation is not re-applied as an update (`:55`, `i > 0`).
- Filter: predict `x⁻ = F x`, `P⁻ = F P Fᵀ + Q`. At observation steps,
  `S = P⁻_{pos} + R`, `K = P⁻ Hᵀ S⁻¹`, `x = x⁻ + K (z - H x⁻)`, and the Joseph-form
  update `P = (I - K H) P⁻ (I - K H)ᵀ + K R Kᵀ` (`:55-71`). Grid steps are
  predict-only.
- RTS smoother (`:73-79`): `C_i = P_i Fᵀ (P⁻_{i+1})⁻¹`,
  `x^s_i = x_i + C_i (x^s_{i+1} - x⁻_{i+1})`.

Events and grid instants are merged and sorted by time, with observations placed before
grid steps at equal times (`np.lexsort((ev_obs < 0, ev_t))`, `:129`). The first grid step
is `ceil(t_first/dt)·dt >= t_first`, so the first merged element is always an
observation.

## Algorithm (`detect`, `kalman_dwell.py:109-145`)

1. `min_len = max(round(min_duration_s / dt_s), 1)` (`:115`). Python's `round` is
   banker's rounding.
2. Per id, split on `diff(t) > max_gap_s` (`:121`). Segments with fewer than 2 events
   are skipped.
3. Per segment: grid `k ∈ [ceil(t_p/dt), floor(t_{q-1}/dt)]`, smoother as above, and
   keep the smoothed positions at grid steps.
4. `_dwells` (`:83-106`), a verbatim greedy search: from step `i`, extend `j` while
   `|g_j - mean(g_i..g_{j-1})|² <= radius²`. If `j - i >= min_len`, emit `[i, j)` and
   continue at `j`. Otherwise `i += 1` (a sliding start).
5. For each run: `t0 = grid_t[i]`, `t1 = grid_t[j-1]` (the last step **inside**). Skip
   if `t1 - t0 < min_duration_s - dt_s` (`:136`). The centroid is the mean smoothed
   position. `n_events` is the count of real events with `t0 <= t <= t1` (`:138`).

## Parameters

| Harness param | Meaning here |
|---|---|
| `max_gap_s` | Segment split (no smoothing across gaps) |
| `radius_m` | Max distance of a smoothed grid position from the run's running mean |
| `min_duration_s` | Converted to a step count `min_len`; see the duration caveat below |

Algorithm-specific: `dt_s = 60`, `accel_sigma = 0.2` m/s² per sqrt(s), and
`default_sigma_m = 10` (shape 1). Original defaults (`PlaceConfig`,
`place_follower.py:30-31`; `build_series`, `series.py:230`): `dwell_radius_m = 150`,
`min_dwell_s = 600`, `dt = 60`, `max_gap_s = 900`, `accel_sigma = 0.2`. The port's
detect defaults match (`600 / 150 / 900`).

## Use of positional uncertainty

Uncertainty enters only through `R_i` in the smoother. A precise fix pulls the track
strongly, while a km-scale fix barely moves it (small Kalman gain). A coarse fix inside a
stay therefore does not break the smoothed track, but a coarse fix far away can still
bend it if no precise fixes are nearby. The dwell test itself uses a fixed metric radius
on the smoothed positions. The smoothed covariance is not used for membership and is not
reported.

## Output semantics

- `start_ts` / `end_ts`: **grid instants**, not event times: the first and last grid
  step of the run. They are quantised to `dt_s` and can differ from the true fixes by up
  to `dt_s`.
- Centroid: mean of the smoothed grid positions, converted back with the same per-id
  frame (`:139`).
- `sigma_*`: null. The original computes smoothed marginal variances (`var` in
  `series.py`), but the port drops them.
- `n_events`: real events inside `[t0, t1]`.

## Comparison with the original

| Aspect | Original | Port |
|---|---|---|
| Smoother | `_kalman_rts` returns `(xs, var)` and also runs the covariance smoother | Same filter and state smoother. Covariance smoothing removed (unused) |
| Validity / gaps | Global per-id series with a `valid` mask; `_dwells(e, n, valid, ...)` stops at invalid steps | Runs `_dwells` per gap-free segment, which is equivalent because invalid steps always separate segments when `max_gap_s > dt_s` |
| Dwell end | `t1 = (k0 + en) · dt`, i.e. the **exclusive** end: the first grid step after the run (`place_follower.py:81`) | `t1 = grid_t[en - 1]`, the last step inside: one `dt_s` earlier |
| Duration filter | None beyond `min_len` | Extra `t1 - t0 >= min_duration_s - dt_s` |
| `min_len` | `int(round(min_dwell_s / dt))` | `max(..., 1)` |
| Projection | Tracks' east/north from `tracks.py` | Per-id median-anchored ENU |

## Complexity and Spark-readiness

- Smoother: O(m) per segment, where `m = events + grid steps`. Memory is
  `Pp, Pf ∈ R^{m×4×4}` plus states, about 320 B per step. A 60-day, 1 Hz-ish single id
  with a long gap-free segment needs a few hundred MB.
- `_dwells`: O(m·L) worst case (sliding start, runs of length L).
- There is a Python loop over ids and segments, with numba kernels inside.
- Per-id partitionable. The grid `k·dt` is global and deterministic, so no shuffle is
  needed: `groupBy("id").applyInPandas`. It cannot be expressed as SQL windows (a
  recursive filter).

## Known limitations / failure modes

- **Emitted durations can be below `min_duration_s`** (verified): with
  `min_duration_s = 900`, a 16-minute stay (16 fixes, one per minute) yields
  `duration_s = 840`. Every event-based detector reports 900 for the same stay. The cause
  is the last-step-inside end definition combined with the `min_duration_s - dt_s`
  tolerance. With banker's rounding of `min_len`, the minimum emitted duration is
  `(min_len - 1)·dt_s`. Durations are systematically about `dt_s` shorter than the
  original's.
- Start/end quantisation to `dt_s` (60 s) limits the start/end error floor.
- The smoother lags at arrivals and departures. It rounds corners, so the smoothed
  position enters the radius later and leaves earlier. It also shortens dwells when
  sampling is sparse.
- The constant-velocity prior over long intra-segment gaps (up to `max_gap_s`) lets the
  predicted track wander. The grid positions in a gap are interpolations between fixes,
  which is fine for a stationary id but can create artificial "stays" along straight
  slow segments.
- Coarse fixes are not excluded, only down-weighted. Isolated coarse fixes during travel
  can distort the path.
- The smoothed covariance is not propagated, so the dwell centroid carries no
  uncertainty.

## References

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. J. Basic Eng. 82(1). doi:10.1115/1.3662552
- Rauch, H. E., Tung, F., Striebel, C. T. (1965). Maximum likelihood estimates of linear dynamic systems. AIAA J. 3(8). doi:10.2514/3.3166
- Bar-Shalom, Y., Li, X. R., Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. Wiley. doi:10.1002/0471221279 (white-noise acceleration model, Joseph form)
- Original: new_ellipses `cotravel-research:cotravel_cointegration/series.py`, `place_follower.py`
