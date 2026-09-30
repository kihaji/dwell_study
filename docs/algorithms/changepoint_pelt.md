# changepoint_pelt: PELT change-point baseline (ruptures, L2 cost)

Module: `src/dwells/algorithms/changepoint_pelt.py`. Family `probabilistic`,
`USES_UNCERTAINTY = False`. This is the homoscedastic counterpart of
[ua_segmentation](ua_segmentation.md). Comparing the two isolates what per-event error
weighting contributes.

## Summary

Each gap-free segment's `(east, north)` series is partitioned by
`ruptures.Pelt(model="l2")` into pieces that minimise the sum of squared deviations from
each piece's mean, plus `penalty · #pieces`. A piece becomes a dwell if its RMS spread is
at most `radius_m`, it has at least `min_events` events, and it spans at least
`min_duration_s`.

## Model

For a piece `[a, b)` the L2 cost is

```
c(a, b) = Σ_{i=a}^{b-1} ||z_i - z̄_{a:b}||²
```

(ruptures `CostL2.error`: `signal[a:b].var(axis=0).sum() * (b - a)`). PELT finds the
partition that minimises `Σ c(piece) + β · #changes` exactly, with
`β = penalty_factor · radius_m²` (`changepoint_pelt.py:35`). Read this as a
Gaussian negative log-likelihood with a **common** per-axis variance σ²: `c/(2σ²)` is
the NLL, so `β` corresponds to `2σ²·β'` for a likelihood-scale penalty `β'`. Intuitively,
a split must remove more than about `penalty_factor` points' worth of radius-sized
squared deviation.

## Algorithm (`detect`, `changepoint_pelt.py:27-49`)

1. `seg = gap_segments(events, max_gap_s)`.
2. Per segment with at least `min_events` events:
   `rpt.Pelt(model="l2", min_size=2, jump=jump).fit(X).predict(pen=β)` if the segment
   has more than 3 events, else a single piece (`:40`).
3. For each piece, compute `rms = sqrt(mean ||z_i - z̄||²)` (`:44`). Label the piece if
   `len >= min_events`, `rms <= radius_m` and `t_last - t_first >= min_duration_s`.
4. `dwells_from_labels(events, labels, min_duration_s)`.

Note: for positions uniform in a disk of radius R, the RMS radius is `R/√2`, so the
acceptance test `rms <= radius_m` admits places up to about `√2 · radius_m` in extent.

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | Sets the penalty scale `β = penalty_factor · radius_m²` and the RMS acceptance bound |
| `max_gap_s` | Segment split |
| `min_duration_s` | Minimum piece span |

Algorithm-specific: `penalty_factor = 5.0`, `jump = 1`, `min_events = 2`. `min_size = 2`
is hard-coded.

## Use of positional uncertainty

None. All events have equal weight, so a km-scale fix creates a change point exactly as
a real move does. That is the point of the baseline.

## Output semantics

- `start_ts` / `end_ts`: first / last event of the piece.
- Centroid: mean lat/lon.
- `sigma_*`: null.

## Complexity and Spark-readiness

- ruptures' PELT is pure Python, and `CostL2.error` recomputes the variance over the
  whole slice, so each cost evaluation is O(piece length) instead of O(1) with prefix
  sums. With pruning the typical cost is between O(n) and O(n²) evaluations, each
  O(length), per segment. This is slow for long dense segments (the reason
  ua_segmentation implements its own prefix-sum PELT in numba).
- Per `(id, segment)` partitionable: `groupBy("id")`, with gap segmentation via SQL
  window (`lag`, `sum`), then `applyInPandas`. There is no global step.

## Known limitations / failure modes

- Heteroscedastic data (CDR/adtech) produces spurious change points at every coarse fix.
  Coarse fixes can also create 2-point pieces that fail `min_duration_s`, fragmenting
  stays.
- A constant penalty across segments of very different lengths: long segments get more
  changes.
- Travel produces many short pieces, and some slow, dense travel pieces can pass the RMS
  and duration tests.
- No `KNOWN_LIMITATIONS` xfails.

## References

- Killick, R., Fearnhead, P., Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. JASA 107(500). doi:10.1080/01621459.2012.737745
- Truong, C., Oudre, L., Vayatis, N. (2020). Selective review of offline change point detection methods. Signal Processing 167. doi:10.1016/j.sigpro.2019.107299
- ruptures: https://centre-borelli.github.io/ruptures-docs/
