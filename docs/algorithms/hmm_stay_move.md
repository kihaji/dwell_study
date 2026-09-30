# hmm_stay_move: two-state stay/move HMM with heteroscedastic Rayleigh emissions (new)

Module: `src/dwells/algorithms/hmm_stay_move.py`. Family `probabilistic`,
`USES_UNCERTAINTY = True`. New in this repo. It follows the tradition of state-space
movement models (Patterson et al. 2008) and HMM segmentation (Rabiner 1989), with
per-event error in the emissions.

## Summary

A hidden state (stay or move) is attached to every **interval** between consecutive
events of a gap-free segment. The observation is the displacement between the two fixes.
Emissions are Rayleigh densities whose scale includes both fixes' reported errors.
Transitions depend on the elapsed time. Viterbi decoding gives the most probable state
path, and maximal runs of "stay" intervals become dwells.

## Model

For segment events `0..N-1`, interval `j = 1..N-1` covers events `(j-1, j)`, with
`Δt_j = max(t_j - t_{j-1}, 10⁻³)` and displacement `d_j = |z_j - z_{j-1}|`.

**Per-event isotropic variance** (`hmm_stay_move.py:86`): `σ_i² = (C_ee + C_nn)/2`.
The ellipse orientation is ignored.

**Emissions** (`_rayleigh_log`, `:38-40`). If a 2-D displacement is `N(0, s² I)`, then
`d ~ Rayleigh(s)` with `f(d) = (d/s²) exp(-d²/(2s²))`:

```
s_j² = σ_{j-1}² + σ_j² + (radius_m / 3)²           (both fixes' errors + within-place jitter)
stay:  d_j ~ Rayleigh( s_j )
move:  d_j ~ Rayleigh( sqrt(s_j² + (speed_ms · Δt_j)²) )
log f = log(max(d, 10⁻³) / s²) - d² / (2 s²)
```

The jitter `(r/3)²` treats `radius_m` as a 3-sigma bound on consecutive-fix movement
inside a place. The docstring (`:6-9`) explains that independent uniform draws (`r²/4`
per fix) made travel steps look like within-place jumps. The move model is a zero-mean
Gaussian step with per-axis sd `speed_ms·Δt` (mean step `≈ 1.25·speed·Δt`).

**Transitions** (`:62-72`), with state 0 = move and 1 = stay:

```
p_sm(Δt) = 1 - exp(-Δt / mean_stay_s)     (stay → move)
p_ms(Δt) = 1 - exp(-Δt / mean_move_s)     (move → stay)
A(Δt) = [[1 - p_ms, p_ms],
         [p_sm,     1 - p_sm]]           (rows: from move, from stay)
```

These are the probabilities of leaving each state within `Δt` under exponential holding
times. This is the "at least one jump" approximation of a two-state continuous-time
Markov chain. The exact CTMC transition matrix `exp(QΔt)` would give
`p_ms = μ/(λ+μ)·(1 - e^{-(λ+μ)Δt})` with `λ = 1/mean_stay_s` and `μ = 1/mean_move_s`.
The two agree for `Δt << mean_*`, but for long intervals the approximation overstates
switching: at `Δt = 3600` s, `p_ms` is 0.98 versus 0.88 exact. Long gaps (up to
`max_gap_s`) are therefore close to uninformative, as the docstring intends.

**Initial distribution:** `π = (½, ½)`.

## Viterbi recursion (`_viterbi`, `hmm_stay_move.py:43-78`)

In log space, per segment:

```
δ_1(s) = log π_s + log e_1(s)
δ_j(s) = max_{s'} [ δ_{j-1}(s') + log A_{s's}(Δt_j) ] + log e_j(s),     ψ_j(s) = argmax
ŝ_{N-1} = argmax_s δ_{N-1}(s);   ŝ_{j-1} = ψ_j(ŝ_j)
```

`10⁻³⁰⁰` is added inside logs to avoid `log 0`. Ties go to "move" (`>=`). The state of
interval `j` is stored at event index `j` (`state[a + j + 1]`). The first event of a
segment has no interval (`-1`).

## Algorithm (`detect`, `hmm_stay_move.py:81-111`)

1. `prepare`; `var = (C_ee + C_nn)/2`. Segments come from `gap_segments(max_gap_s)`.
2. Viterbi per segment.
3. `informative_i ⇔ σ_i <= coarse_factor · radius_m` (`:90`).
4. For each maximal run of stay intervals `i..j` within a segment, the covered events
   are `i-1..j` (`:104`). Keep the informative subset. If it has at least `min_events`
   and spans at least `min_duration_s`, label **all** events from the first to the last
   informative one (`:107`). Coarse events inside are attached, and coarse edge events
   are excluded.
5. `dwells_from_labels(..., centroid="fused", extra_var=radius_m²/4)`.

### Worked intuition (numbers computed with the model, `r = 100`, `speed_ms = 5`)

Log emission ratio `log e_stay - log e_move` for one interval:

| Fixes σ (m) | d (m) | Δt (s) | log ratio | Reading |
|---|---|---|---|---|
| 10, 10 | 15 | 60 | +4.16 | Clearly stay |
| 10, 10 | 300 | 60 | -29.6 | Clearly move |
| 10, 1000 (coarse) | 800 | 60 | +0.06 | Uninformative |
| 10, 1000 (coarse) | 800 | 1800 | +4.09 | Favours stay: a 30-min move would predict km-scale steps |

Transition log-odds for staying in "stay" (`log(1-p_sm) - log p_sm`) are +4.78 at
`Δt = 60` s, +3.16 at 300 s and +1.26 at 1800 s.

A coarse fix inside a stay enters two intervals (in and out). Both emissions are nearly
flat, so the transition prior keeps the path in "stay", and the coarse fix does not
break the dwell. Because coarse fixes cannot be the first or last event of a dwell, a
lone consistent coarse fix hours later cannot extend it.

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | Jitter `(r/3)²` in the emissions, the coarse threshold `coarse_factor · r`, and the centroid `extra_var = r²/4` |
| `max_gap_s` | Segment split. The chain restarts at `π` after a gap |
| `min_duration_s` | Min informative span of a stay run |

Algorithm-specific defaults: `speed_ms = 5.0`, `mean_stay_s = 7200`,
`mean_move_s = 900`, `coarse_factor = 3.0`, `min_events = 2`, `default_sigma_m = 10`.
These are **fixed**, not estimated: there is no Baum-Welch/EM.

## Use of positional uncertainty

The emission scale includes both fixes' variances, so precise fixes produce sharp
evidence and coarse fixes produce flat evidence. Uncertainty also drives the coarse
classification and the fused centroid ellipse.

## Output semantics

- `start_ts`: the first informative event of the stay run. When the preceding interval
  is "move", this is the arrival fix, i.e. the end of the move interval. `end_ts`: the
  last informative event.
- Centroid: fused mean with `extra_var = r²/4`. Note that this differs from the
  `(r/3)²` jitter used in the emissions.
- `sigma_*`: 1-sigma ellipse of the fused estimate.

## Complexity and Spark-readiness

- O(N) per segment with O(N) back-pointers (int64 x 2 per interval). numba.
- Per `(id, segment)` partitionable: `applyInPandas`. Displacements and `Δt` are SQL
  window features (`lag`). Viterbi is a max-plus linear recurrence. It is sequential
  here, but it could be computed as an associative (parallel-prefix) scan over 2x2
  max-plus matrices if a single id were ever too long for one task.

## Known limitations / failure modes

- **Travel absorbed into stays at dense sampling or slow speeds (verified).** The
  jitter `(r/3)²` is per step and independent of `Δt`, while the move term scales with
  `(speed_ms·Δt)²`. At high sampling rates the move term is small compared with the
  jitter, so per-step evidence for "move" nearly vanishes (at most
  `log(s_move²/s_stay²)` in favour of stay), and the sticky transition prior dominates.
  Measured with `min_duration_s = 900`, `radius_m = 100`, `max_gap_s = 3600` and 5 m
  GPS error:
  - 5 s sampling, 30 min stay, a 3 km drive at 10 m/s, 30 min stay: **one dwell,
    0-64 min**. ua_mahalanobis, ua_segmentation, li_centroid and h3_occupancy give two
    dwells, (0-30, 34-64).
  - 60 s sampling, 30 min stay, a 2.5 km walk at 1.4 m/s over 30 min, 30 min stay:
    **one dwell, 0-89 min**. The others give 0-30 and about 58-89.

  The `scale_gps_single_1m` dataset (4-6 s sampling) is in the affected regime. A fix
  would scale the jitter or the speed prior with `Δt`, or learn the parameters.
- The parameters are fixed guesses (walking-to-driving speed averaged as 5 m/s, a 2 h
  mean stay). There is no per-source adaptation.
- The isotropic `σ_i²` discards ellipse orientation (CDR sectors are elongated).
- The first-jump transition approximation (see Model).
- No `KNOWN_LIMITATIONS` xfails. The contract and coarse-point tests use 1-minute
  sampling with fast moves, so they do not exercise the failure above.

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proc. IEEE 77(2). doi:10.1109/5.18626
- Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Trans. Inf. Theory 13(2). doi:10.1109/TIT.1967.1054010
- Patterson, T. A. et al. (2008). State–space models of individual animal movement. Trends Ecol. Evol. 23(2). doi:10.1016/j.tree.2007.10.009
- Rayleigh distribution: https://en.wikipedia.org/wiki/Rayleigh_distribution
