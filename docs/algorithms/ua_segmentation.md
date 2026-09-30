# ua_segmentation: heteroscedastic Gaussian PELT segmentation (new)

Module: `src/dwells/algorithms/ua_segmentation.py`. Family `uncertainty`,
`USES_UNCERTAINTY = True`. New in this repo. [changepoint_pelt](changepoint_pelt.md) is
its homoscedastic baseline.

## Summary

Each gap-free segment is optimally partitioned into pieces, where every piece is
modelled as "stationary at an unknown centre" with per-event error covariances. The
exact penalised optimum is found by PELT, using prefix sums that give each piece's cost
in O(1). A piece becomes a dwell if its informative events span at least
`min_duration_s` and its residuals fit the stationary model.

## Model and cost

Same observation model as [ua_mahalanobis](ua_mahalanobis.md): in a stationary piece
with centre `μ`,

```
z_i ~ N(μ, R_i),   R_i = C_i + v I,   W_i = R_i⁻¹,   v = radius_m²/4
```

The negative log-likelihood of a piece `[a, b)` is

```
-log L(μ) = Σ_i [ log 2π + ½ log det R_i ] + ½ Σ_i (z_i - μ)ᵀ W_i (z_i - μ)
```

The first sum does not depend on how the segment is partitioned (every event contributes
once, whatever piece it is in), so it is dropped. Minimising the quadratic over `μ`
(profile likelihood):

```
∂/∂μ = 0  ⇒  μ* = I⁻¹ b,  where  I = Σ W_i,  b = Σ W_i z_i
cost(a, b) = ½ min_μ Σ (z_i - μ)ᵀ W_i (z_i - μ) = ½ [ q - bᵀ I⁻¹ b ],  q = Σ z_iᵀ W_i z_i
```

(Expand `Σ (zᵀWz - 2μᵀWz + μᵀWμ) = q - 2μᵀb + μᵀIμ`, then substitute `μ*`.)

**Prefix sums.** Per event, store the 6-vector

```
c_i = [W_ee, W_nn, W_en, (W z)_e, (W z)_n, zᵀ W z]
    = [w_ee, w_nn, w_en, w_ee x + w_en y, w_en x + w_nn y, w_ee x² + 2 w_en x y + w_nn y²]
```

(`ua_segmentation.py:96-100`). With `S_k = Σ_{i<k} c_i` (`:107`, with a zero row
prepended), every piece's `I, b, q` are differences `S_b - S_a`, and

```
bᵀ I⁻¹ b = (I_nn b_x² - 2 I_en b_x b_y + I_ee b_y²) / (I_ee I_nn - I_en²)
```

(`_cost`, `:38-52`). The result is clamped at 0 for round-off. Numerics: coordinates are
metres in a per-id frame, and the cancellation `q - bᵀI⁻¹b` is benign in float64 at
city scale.

**Objective.** `min over partitions Σ cost(piece) + β · #pieces`, with
`β = penalty` (dimensionless, in NLL units). Because `W` is normalised by `C_i + vI`, a
single `β` works across error scales.

**MAP interpretation.** A constant-hazard prior with change probability `h` per
inter-event boundary has `log P(τ) = m·log(h/(1-h)) + const` for `m` changes, so
`β = log((1-h)/h)`. `β = 10` corresponds to `h ≈ 4.5·10⁻⁵`. Since `μ` is profiled out
(maximised), not integrated, this is MAP under a penalised profile likelihood. A fully
Bayesian version (integrating `μ` under a flat prior) would add a data-dependent
`½ log det I_piece` term per piece, similar to BIC's `log n`. The docstring's reference
to Adams & MacKay (2007) is for context only: this is not online Bayesian change-point
detection.

### What β = 10 means in metres

Merging two stationary groups A and B (n_A and n_B events, equal weights
`w = 1/(σ² + v)`, centre separation D) costs `½ · w · n_A n_B/(n_A+n_B) · D²` more than
keeping them apart. That is the between-group sum of squares. A split therefore pays off
when

```
D > sqrt( 2β (σ² + v) (n_A + n_B) / (n_A n_B) )
```

Example: σ = 5 m, r = 100 m (σ² + v = 2525 m²), n_A = n_B = 10 gives D > 100 m. With
n_A = n_B = 40 it gives D > 50 m. With more events, nearby places separate more easily.

## PELT (`_pelt`, `ua_segmentation.py:55-87`)

```
F(0) = -β
F(t) = min_{s ∈ R_t} [ F(s) + cost(s, t) + β ],   last(t) = argmin
R_{t+1} = { s ∈ R_t : F(s) + cost(s, t) <= F(t) } ∪ {t}
```

The pruning rule (`:72-80`) is Killick et al.'s condition with `K = 0`, which is valid
because this cost satisfies `cost(a,b) + cost(b,c) <= cost(a,c)`: adding a split never
increases the minimised residual. Change points are recovered by backtracking `last`.
Pieces may be a single event (no `min_size`). A single event's cost is 0.

## Algorithm (`detect`, `ua_segmentation.py:90-123`)

1. `prepare` (covariances with `default_sigma_m` for shape 1); compute `W_i` and the
   component matrix.
2. `informative_i ⇔ sqrt((C_ee + C_nn)/2) <= coarse_factor · radius_m` (`:101`).
3. Per gap segment (`gap_segments`): prefix sums, PELT, and for each piece:
   - `inf_idx` = the piece's informative events. It needs `len >= min_events`.
   - `chi = 2·cost / max(n_piece - 1, 1)` (`:114`), where `n_piece` counts all events,
     including coarse ones.
   - `t0, t1` = first / last informative time, and it needs `t1 - t0 >= min_duration_s`
     and `chi <= max_chi2_per_event`.
   - Label all events of the piece with `t0 <= t <= t1` (`:118`). Coarse events inside
     are attached, and coarse events at the piece's edges are excluded.
4. `dwells_from_labels(..., centroid="fused", extra_var=v)`.

**The acceptance statistic.** Under the model, `2·cost ~ χ²` with `2(n-1)` degrees of
freedom (2 coordinates per event, minus 2 for the fitted centre), so
`E[chi] = 2(n-1)/(n-1) = 2`. A Monte Carlo check of 500 stationary 60-event pieces gave
a mean of 1.995. The module docstring calls this the "reduced chi-square", but a reduced
chi-square conventionally divides by the degrees of freedom `2(n-1)` and has expectation
1. The threshold `max_chi2_per_event = 3.0` is therefore **1.5x the expectation**,
equivalent to a conventional reduced χ² of 1.5. In practice, a device sitting still with
5 m error gives `chi ≈ 2·25/2525 ≈ 0.02`, far below the threshold. The test mainly
rejects pieces whose spread is well above `radius_m`, such as slow travel absorbed into
one piece.

### Worked intuition for coarse points

With `r = 100`: a coarse fix with σ = 1274 m that is 800 m from a GPS stay adds about
`½ · 800² / (1274² + 2500) ≈ 0.2` to the piece cost, which is far below `β = 10`, so it
creates no change point. Its weight in `μ*` is about 1/640 of a GPS fix. In the
homoscedastic baseline the same fix adds `800² = 640 000 m²` to the L2 cost, against
`β = 5·100² = 50 000`, so it is isolated into its own (2-event minimum) piece and splits
the stay. Coarse fixes also cannot define `t0/t1`, so a consistent coarse fix hours after
the last GPS fix of a stay, which ends up in the same piece, does not stretch the dwell.

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | Within-place spread `v = r²/4` in `W_i`, and the coarse threshold `coarse_factor · r` |
| `max_gap_s` | Segment split (no piece spans a gap) |
| `min_duration_s` | Min informative span of a piece |

Algorithm-specific defaults: `penalty = 10.0`, `max_chi2_per_event = 3.0`,
`coarse_factor = 3.0`, `min_events = 2` (informative), `default_sigma_m = 10`.

## Use of positional uncertainty

`C_i` weights every residual. Coarse fixes are nearly free, so they neither create
change points nor move centres. It also drives informative/coarse classification and the
reported centroid ellipse.

## Output semantics

- `start_ts` / `end_ts`: first / last informative event of the piece.
- Centroid: fused mean with `extra_var = v`, which is `μ*` of the labelled events
  (edge coarse events excluded).
- `sigma_*`: the 1-sigma ellipse of `I⁻¹` (see the caveats on this sigma in
  [ua_mahalanobis](ua_mahalanobis.md#output-semantics)).

## Complexity and Spark-readiness

- Prefix sums are O(n). PELT is O(n) expected when change points occur regularly
  (Killick et al.), and O(n²) worst case. Within a long stationary stretch of L events,
  most candidates survive pruning (split and unsplit costs are nearly equal), so that
  stretch costs O(L²). This is harmless for GPS stays of a few thousand fixes, but a
  risk for multi-day gap-free stationary stretches at a 1-5 s sampling rate. numba.
- Per `(id, gap segment)` partitionable: segments come from SQL windows (`lag`, then a
  cumulative sum), then `applyInPandas` per `(id, seg)`. There is no global step.

## Known limitations / failure modes

- The `v = r²/4` model again conflates place size with per-fix jitter.
- A single global `β` does not adapt to sampling density. Dense sampling makes small
  displacements significant (the `D` threshold shrinks as `1/sqrt(n)`), which can split
  large places. Sparse sampling can merge adjacent places.
- The acceptance threshold is naming-sensitive (see "The acceptance statistic").
- Pieces are exact optima of the model. Travel is not modelled explicitly: travel fixes
  become many short pieces that fail the duration test, and slow travel close to a place
  can join the stay piece if the chi test tolerates it.
- If every fix is coarse, there are no dwells.
- No `KNOWN_LIMITATIONS` xfails. It passes the coarse-point tests.

## References

- Killick, R., Fearnhead, P., Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. JASA 107(500). doi:10.1080/01621459.2012.737745
- Jackson, B. et al. (2005). An algorithm for optimal partitioning of data on an interval. IEEE Signal Process. Lett. 12(2):105–108 (optimal partitioning recursion; arXiv:math/0309285)
- Adams, R. P., MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742 (context only)
- Truong, C., Oudre, L., Vayatis, N. (2020). doi:10.1016/j.sigpro.2019.107299
