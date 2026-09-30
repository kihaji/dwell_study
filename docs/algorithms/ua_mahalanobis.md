# ua_mahalanobis: sequential information-filter stay with Mahalanobis gating (new)

Module: `src/dwells/algorithms/ua_mahalanobis.py`. Family `uncertainty`,
`USES_UNCERTAINTY = True`. New in this repo. The gating follows classic Kalman/data-
association practice (Bar-Shalom & Fortmann 1988).

## Summary

This is a running-centroid detector in which "within radius" is replaced by a
chi-square test that uses both the new fix's error covariance and the uncertainty of
the current stay estimate. On top of that it adds three robustness rules:

1. Coarse fixes can never start, break or extend a stay.
2. A stay breaks only after `k_break` consecutive informative misses.
3. The centroid is the information-weighted mean, reported with its 1-sigma ellipse.

## Model

While an id stays at a place with (unknown) centre `μ ∈ R²` (local ENU metres):

```
z_i = μ + u_i + e_i,   u_i ~ N(0, v I)  (movement inside the place),   e_i ~ N(0, C_i)  (reported error)
⇒ z_i ~ N(μ, R_i),     R_i = C_i + v I,     v = radius_m² / 4
```

`v = R²/4` is the per-axis variance of a point uniformly distributed in a disk of radius
`R = radius_m` (`E[x²] = R²/4`). The Gaussian stands in for that uniform distribution.
`C_i` comes from `with_covariance` (see [../uncertainty.md](../uncertainty.md)). For
shape 1, `C_i = default_sigma_m² I = 100 I`.

**Estimate (information form, flat prior on μ).** After accepting events `A`:

```
I = Σ_{i∈A} R_i⁻¹,   b = Σ_{i∈A} R_i⁻¹ z_i,   μ̂ = I⁻¹ b,   P = Cov(μ̂) = I⁻¹
```

(`ua_mahalanobis.py:81-86` seed, `:94-96` estimate, `:107-115` update). 2x2 inverses are
closed-form.

**Gate.** For a candidate `z`, the innovation `z - μ̂` has covariance
`S = R_i + P = C_i + v I + P`, because `μ̂` is independent of the new event. Under the
"same stay" hypothesis, `d² = (z - μ̂)ᵀ S⁻¹ (z - μ̂) ~ χ²₂`. The event is accepted if

```
d² <= χ²₂⁻¹(gate_p) = -2 ln(1 - gate_p)       (= 9.21 for gate_p = 0.99)
```

(`:97-99`, threshold `chi2.ppf(gate_p, 2)` at `:147`).

Effective metric radius: for a precise fix (`σ_i << radius_m`) and a well-established
stay (`P → 0`), accept `|z - μ̂| <= sqrt(9.21 · (σ_i² + r²/4)) = 1.517 · sqrt(r² + 4σ_i²)`.
In words, about **1.52 · radius_m** for GPS. At `r = 100` m and σ = 5 m this is
152 m. For shape 1 (σ = 10 m) it is 155 m. `P` inflates the gate early in a stay (after
one event `P = R_seed`, so `S ≈ 2R` and the gate is `√2` larger), which makes the second
fix easy to accept.

**Coarse events.** `coarse_i ⇔ sqrt((C_ee + C_nn)/2) > coarse_factor · radius_m`
(`:45`, threshold `coarse_factor · radius_m` passed at `:148`). A coarse fix has
`S ≈ C_i`, so almost any displacement passes the gate. Such a fix carries no information
about *which* place, only a little about *whether still here*.

## Algorithm (`_kernel`, `ua_mahalanobis.py:41-139`)

Per id, in time order, with state `is_open, c0, n_inf, n_buf, n_fail, first_fail,
t_first, t_last` and the information sums:

1. **Gap close** (`:59-60`): if a stay is open and an *informative* event arrives more
   than `max_gap_s` after the last **accepted informative** event (`t_last`), close
   the stay. Coarse events never trigger a gap close.
2. **Close handling** (`:61-72`): accept the stay as a dwell if
   `n_inf >= min_events` and `t_last - t_first >= min_duration_s`. Otherwise reset its
   labels to -1. If there were pending failures, restart processing at `first_fail`,
   so those events may seed the next stay.
3. **Seeding** (`:75-92`): only an informative event can seed. Coarse events while no
   stay is open are skipped. Seeding sets `I = R_i⁻¹`, `b = R_i⁻¹ z_i`,
   `n_inf = 1`, `t_first = t_last = t_i`.
4. **Coarse event in an open stay** (`:100-105`): if it passes the gate, append it to a
   pending buffer. Either way, move on. It never counts as a miss.
5. **Informative event passes** (`:106-121`): add its information to `I, b`, label
   it, attach all buffered coarse events (confirmed: the stay continued past them),
   clear the buffer and the miss counter, `n_inf += 1`, `t_last = t_i`.
6. **Informative event fails** (`:122-137`): remember `first_fail` and increment
   `n_fail`. When `n_fail >= k_break`, close the stay at its last confirmed event
   (accept or reject as in step 2) and restart at `first_fail`. A single failing event
   followed by a passing one is skipped as an outlier. It stays unlabelled, but it lies
   inside the dwell span.
7. End of track: close as in step 2.

`detect` (`:142-151`) then calls `dwells_from_labels(..., centroid="fused",
extra_var=v)`, which recomputes the same information-weighted estimate over all
labelled events (informative plus attached coarse) and reports its 1-sigma ellipse.

### Worked intuition for coarse points

Setting: `radius_m = 100` (coarse threshold 300 m), a stay at the origin with 5 m GPS
fixes every minute.

- **Coarse fix in the middle** (`tests/test_uncertainty_algorithms.py:36-41`): at
  minute 30.5 a fix 800 m away with CEP 1500 m, so σ = 1274 m. It is coarse. Its
  `d² ≈ 800² / (1274² + 2500) ≈ 0.39`, so it passes the gate and is buffered. The
  next GPS fix passes and the coarse fix is attached. The stay is not broken, and the
  fix's weight in the centroid is `(1274² + 2500)⁻¹ / (25 + 2500)⁻¹ ≈ 1/640` of a GPS
  fix. Without the coarse rule a plain radius detector would break here. Even as an
  informative event the fix would pass the gate, but it would have been allowed to
  *seed* stays elsewhere.
- **Coarse fix 3 h after the last GPS fix** (`:44-49`): it is buffered but never
  confirmed by a later informative fix, so it is not labelled and the dwell ends at the
  last GPS fix.
- **CDR-only stretches**: if every fix of an id is coarse at the chosen radius, the id
  produces **no dwells at all**. The harness therefore tunes `radius_m` up to 2000 m for
  CDR (coarse threshold 6 km), and then the CDR fixes become informative.

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | Within-place spread `v = radius_m²/4` (so the gate is about 1.52 · radius_m for precise fixes), and the coarse threshold `coarse_factor · radius_m` |
| `max_gap_s` | Max time between accepted informative events |
| `min_duration_s` | Min `t_last - t_first` over informative events |

Algorithm-specific defaults: `gate_p = 0.99`, `coarse_factor = 3.0`, `k_break = 2`,
`min_events = 2` (informative), `default_sigma_m = 10` (shape 1).

## Use of positional uncertainty

It is used in the gate (`C_i` in `S`), the coarse classification, the estimate weights,
and the reported centroid ellipse. For shape 1 all `C_i` are equal, so the method
degenerates to a Gaussian-gated running mean (radius about 155 m at `r = 100`) with
k-miss outlier tolerance.

## Output semantics

- `start_ts` / `end_ts`: first / last **informative** labelled event. Attached coarse
  events always lie strictly inside, because they are attached only when a later
  informative event confirms the stay. Unlabelled outliers inside the span do not
  shorten it.
- Centroid: fused (inverse-covariance-weighted) mean over labelled events, with
  `extra_var = v` added to each event's variance (`common.py:80-99`). This is the same
  estimator as the kernel.
- `sigma_major/minor, azimuth`: 1-sigma ellipse of `P = (Σ (C_i + vI)⁻¹)⁻¹`. For `n`
  GPS fixes, `σ ≈ sqrt((σ_gps² + r²/4)/n)`, for example 7.9 m for 40 fixes at
  `r = 100`. This is the uncertainty of the *mean position of the stay*, treating
  within-place movement as independent per fix. It is optimistic when a device sits
  still (the offset is then persistent, not independent), and it is **not** the place
  extent.
- `n_events`: informative accepted plus attached coarse events. Skipped outliers are
  excluded.

## Complexity and Spark-readiness

- Near-linear: one pass with O(1) state. Each break re-processes up to `k_break - 1`
  events, and rejected stays re-scan their events once to clear labels. numba.
- Per-id partitionable with no global step. It is a stateful scan, so use
  `groupBy("id").applyInPandas(detect)` (or `mapGroupsWithState` in Scala). It cannot
  be expressed as SQL windows.

## Known limitations / failure modes

- Greedy and sequential: events of a rejected (too short) stay are never re-used, and
  the result depends on the seed (an approach fix can seed the stay and be absorbed,
  since the gate is about 1.5 · radius_m).
- Drift: the estimate follows slow movement within the gate, as with any running
  centroid. `P` shrinks, but `S >= C_i + vI` keeps the gate from tightening below
  about 1.5 · r.
- The `v = r²/4` model conflates "place size" with "per-fix jitter". A large place
  sampled at a precise, fixed spot is still given a 1.5 · r gate.
- Calibration: the gate assumes honest `C_i`. The generator draws fixes from
  `Σ_true = Σ_reported / c²` with `c ~ lognormal(0, 0.25)`, so reported errors are only
  roughly honest. Overconfident sources cause spurious breaks, which `k_break` mitigates.
- If every fix is coarse, there are no dwells (see above).
- No `KNOWN_LIMITATIONS` xfails. It passes all four coarse-point tests.

## References

- Bar-Shalom, Y., Fortmann, T. E. (1988). Tracking and Data Association. Academic Press. (validation gates, χ² gating)
- Bar-Shalom, Y., Li, X. R., Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. Wiley. doi:10.1002/0471221279 (information filter)
- Mahalanobis, P. C. (1936). On the generalised distance in statistics. Proc. Natl. Inst. Sci. India 2(1):49–55
- Kang, J. H. et al. (2004). doi:10.1145/1024733.1024748 (running-centroid baseline this generalises)
