# Positional uncertainty model

Every event's reported error, whatever form it arrives in, is converted into a 2×2 covariance matrix, in m², in a local east/north frame. All uncertainty-aware code then works with covariances: gating, fusion, centroid ellipses and place merging. This document covers `src/dwells/uncertainty.py`, the related helpers, and how the three dataset shapes are derived from one master event table (`src/dwells/io.py`).

---

## 1. The model

The reported position `z` of an event is modelled as a bivariate normal draw around the true position `x`:

```
z ~ N(x, Σ),   Σ = [[c_ee, c_en],
                    [c_en, c_nn]]      (m², east/north)
```

The columns are `cov_ee`, `cov_nn` and `cov_en`. Everything else follows from the distribution of the squared Mahalanobis distance.

### 1.1 Coverage of a k-sigma ellipse: derivation of k(p)

Let `d² = (z − x)ᵀ Σ⁻¹ (z − x)`. Whiten with `Σ = L Lᵀ` and `u = L⁻¹(z − x) ~ N(0, I₂)`. Then `d² = u₁² + u₂² ~ χ²₂`. The χ² distribution with 2 degrees of freedom is the exponential distribution with mean 2:

```
f(s) = ½ e^{−s/2},      P(d² ≤ s) = 1 − e^{−s/2}.
```

The "k-sigma ellipse" is the contour `d² = k²`. Its axes are k times the 1-sigma axes, i.e. k·√λ₁ and k·√λ₂ along Σ's eigenvectors. So

```
P(inside k-sigma ellipse) = 1 − exp(−k²/2)
⇔  k(p) = sqrt(−2 ln(1 − p))
```

This is `k_factor` (computed with `log1p` for accuracy near p = 0). For p = 0.95 the familiar 1D value 1.96 is **wrong** in 2D; the correct value is 2.4477.

### 1.2 Accuracy radius → sigma

For a circular distribution (Σ = σ²I), the radius containing a share p of true positions
is the k-sigma circle, so a `horizontalAccuracy` radius r stated at probability p gives:

```
σ = r / k(p)
CEP50 (p = 0.5):  k = sqrt(2 ln 2) = 1.17741   ⇒  σ = CEP50 / 1.17741
68% radius:       k = 1.5096                   ⇒  σ = r68 / 1.5096
95% radius:       k = 2.4477                   ⇒  σ = r95 / 2.4477
```

p comes from the optional `accuracyProbability` column; without it p = 0.5 (CEP50, the
`project.md` definition). `CEP50_K` (`uncertainty.py`) is k(0.5). The formula is exact
only for circular errors. Section 5.3 covers elliptical ones.

### 1.3 Numeric table

These values were verified with `uncertainty.k_factor` and `scipy.stats.chi2.ppf(p, 2)`:

| p | k(p) = semi-axis / sigma | χ²₂ threshold = k² |
|---|---|---|
| 0.3935 (= 1 − e^{−1/2}) | 1.0000 | 1.0000 |
| 0.50 | 1.1774 | 1.3863 |
| 0.68 | 1.5096 | 2.2789 |
| 0.90 | 2.1460 | 4.6052 |
| 0.95 | 2.4477 | 5.9915 |
| 0.99 | 3.0349 | 9.2103 |

The χ² column is the Mahalanobis-gate threshold. `ua_mahalanobis` uses `gate_p = 0.99` → 9.21, and `downstream.places` merges places at 9.21. The p = 0.3935 row is `ONE_SIGMA_P` (`algorithms/common.py`, `downstream/places.py`). Passing it to `cov_to_ellipse` or `ellipse_to_cov` makes the axes exactly 1-sigma. That is how dwell centroid ellipses (`sigma_major`, `sigma_minor`) are reported and read back.

---

## 2. Ellipse ↔ covariance

### 2.1 `ellipse_to_cov(a, b, azimuth_deg, p)`

Inputs: semi-major `a`, semi-minor `b` (m) at probability `p`, and `azimuth` in degrees.

**Azimuth convention.** The bearing of the **semi-major axis, clockwise from north**, in [0, 180). Azimuth 0 points the major axis north; azimuth 90 points it east. The major-axis unit vector in (east, north) is therefore

```
u = (sin θ, cos θ),     minor axis v = (cos θ, −sin θ)
```

The 1-sigma axes are `σ_a = a / k(p)` and `σ_b = b / k(p)`, so `Σ = σ_a² u uᵀ + σ_b² v vᵀ`:

```
c_ee = σ_a² sin²θ + σ_b² cos²θ
c_nn = σ_a² cos²θ + σ_b² sin²θ
c_en = (σ_a² − σ_b²) sin θ cos θ
```

`test_ellipse_azimuth_is_clockwise_from_north` checks the convention: azimuth 0 gives `c_nn > c_ee`, azimuth 90 gives `c_ee > c_nn`. The synthetic generator (branch `tools/ua-tuning`) samples errors with the same convention: offsets `u·(sin θ, cos θ) + v·(cos θ, −sin θ)`.

This is a compass bearing, not the mathematical angle counter-clockwise from east. A source that reports the ellipse orientation counter-clockwise from east, or the minor-axis bearing, must be converted before ingestion. Otherwise ellipses come out mirrored or rotated by 90°.

### 2.2 `cov_to_ellipse(c_ee, c_nn, c_en, p)`

Eigenvalues of the symmetric 2×2 matrix:

```
λ₁,₂ = tr/2 ± sqrt((tr/2)² − det),   tr = c_ee + c_nn,  det = c_ee c_nn − c_en²
a = k(p) √λ₁,   b = k(p) √λ₂
```

The discriminant and λ₂ are clamped at 0 against round-off. For the orientation, substitute §2.1 with `Δ = σ_a² − σ_b² ≥ 0`:

```
c_nn − c_ee = Δ (cos²θ − sin²θ) = Δ cos 2θ
2 c_en      = 2Δ sin θ cos θ    = Δ sin 2θ
⇒ θ = ½ atan2(2 c_en, c_nn − c_ee)  (mod 180°)
```

For a circular covariance (Δ = 0), `atan2(0, 0) = 0`, so the azimuth is reported as 0. `test_ellipse_cov_roundtrip` round-trips `(120, 40, 30°, 0.95)` to 1e-9.

### 2.3 `mahalanobis2(dx, dy, c_ee, c_nn, c_en)`

```
d² = (c_nn dx² − 2 c_en dx dy + c_ee dy²) / det
```

This is the closed-form `[dx dy] Σ⁻¹ [dx dy]ᵀ`. For a test between two uncertain positions, pass `Σ = Σ_a + Σ_b`. Compare `d²` against the χ² column of §1.3. `test_monte_carlo_coverage` checks by Monte Carlo (2×10⁵ draws) that `P(d² ≤ k(p)²) ≈ p` for p = 0.5, 0.68 and 0.95, within 0.005.

### 2.4 Information fusion

The dwell and place centroids use the information-form (inverse-covariance-weighted) mean of independent 2D measurements of one fixed point:

```
W_i = Σ_i⁻¹,   I = Σ_i W_i,   b = Σ_i W_i z_i
μ̂ = I⁻¹ b,     Cov(μ̂) = I⁻¹
```

This is the BLUE, and the maximum-likelihood estimate under Gaussian errors. It is the static Kalman update applied to all points at once (for example, points at 0 and 10 with variances 1 and 4 fuse to 2.0 with variance 0.8).

**Where it lives.** There is no shared `fuse` helper; the algebra is inlined where it is used: incrementally in the `ua_mahalanobis` numba kernel, as a polars aggregation in `algorithms/common.py` (`_fused`, the reported dwell centroid), and in `downstream.places`. Their fused centroids add a within-place variance `v` to each event's covariance (`extra_var`). This matters for interpretation. `I⁻¹` is the uncertainty of the *mean position*, and treating events as independent makes it shrink roughly as 1/√n. It is **not** the spatial extent of the place.

---

## 3. `with_covariance(events, default_sigma_m=30.0, source_sigma=None)`

This function appends `cov_ee`, `cov_nn`, `cov_en` and `sigma_m`, choosing by which columns exist:

1. **Shape 3** (`semiMajor`, `semiMinor`, `azimuth` and `probability` all present) → `ellipse_to_cov` per event, each at its own `probability`. This takes precedence even if `horizontalAccuracy` is also present.
2. **Shape 2** (`horizontalAccuracy`) → a circle with `σ = horizontalAccuracy / k(p)`, `c_en = 0`, where p is the event's `accuracyProbability` (0.5 = CEP50 when the column is absent).
3. **Shape 1** → a circle with `σ = source_sigma[location_source]` when the events carry that label and it is mapped, else `σ = default_sigma_m`.

`sigma_m = sqrt((c_ee + c_nn)/2)` is the **RMS 1-sigma radius**, the per-axis average standard deviation. It is the scalar the detector's coarse rule compares against `coarse_sigma_m` and `coarse_rel × the id's median`, and the `sigma_m` column of the events output.

The 30 m default applies only to direct calls. The detector passes its own `default_sigma_m` and `source_sigma` (from `presets.params_for`). The `test_with_covariance_*` tests cover all three branches, including per-event probability normalisation (p = 0.68 and 0.95 in one frame).

Input validity is enforced earlier, in `schema.validate_events`:
- shape 2: `horizontalAccuracy > 0`; `accuracyProbability ∈ (0, 1)` when present
- all shapes: no nulls or NaN/inf in required or error columns; timezone-aware timestamps
- shape 3: `semiMinor > 0`, `semiMajor >= semiMinor`, `azimuth ∈ [0, 180]`, `probability ∈ (0, 1)`

Together these guarantee a positive-definite Σ and a finite k(p).

### 3.1 Local planar frame (`geo.to_enu`)

Covariances are in metres east/north. Positions are projected with an equirectangular ENU approximation, which the docstring says is accurate to better than 0.1% within about 50 km:

```
x = Δλ · R cos φ₀,   y = Δφ · R,   R = 6,371,008.8 m
```

`algorithms/common.py` (`prepare`) anchors one frame per id at that id's **median** lat/lon. Distances inside an id are therefore planar metres. An id whose track spans hundreds of km gets growing east-west scale error far from its median latitude.

---

## 4. Dataset shapes from one master table (`src/dwells/io.py`)

The synthetic generator (on branch `tools/ua-tuning`) writes one master frame per dataset with:

- `lat, lon`: the reported position.
- `sig_a ≥ sig_b`: the **reported** 1-sigma axes (m).
- `az`: the major-axis bearing in [0, 180), same convention as §2.1.
- `p_mixed`: 0.68 or 0.95 per event, with probability ½ each.

The reported point is drawn as `z = x + e`, `e ~ N(0, Σ_reported / c²)`, with `c = exp(N(0, 0.25))` per event (`lognormal(0, calibration_noise)` in the generator). The stated error is therefore honest only up to about ±25% in scale, in either direction. GPS multipath outliers, CDR sector centroids (a quantized position rather than a draw), IP hubs and stale fixes also deviate from the stated distribution by design.

`shape_view(master, shape)` projects this frame onto the five evaluated shapes:

| shape | columns added | formula |
|---|---|---|
| `1` | none | error dropped |
| `2` | `horizontalAccuracy` | `1.17741 · sqrt((sig_a² + sig_b²)/2)`: the CEP50 of the circle with the same RMS radius |
| `3_p68` | `semiMajor, semiMinor, azimuth, probability` | `sig · k(0.68) = sig · 1.5096`, p = 0.68 |
| `3_p95` | same | `sig · k(0.95) = sig · 2.4477`, p = 0.95 |
| `3_mixed` | same | `sig · k(p_mixed)`, per-event p |

Properties:
- **All shapes share identical `lat/lon`.** Differences between shapes isolate the effect of the error information itself.
- **The three shape-3 variants are information-equivalent.** `with_covariance` divides by the same k(p) it was multiplied by, so it recovers exactly `σ_a² u uᵀ + σ_b² v vᵀ` for every p. This was verified numerically: `ellipse_to_cov(50·k, 20·k, 30°, p)` gives `c_ee = 925.0` for both p = 0.68 and 0.95. Any accuracy difference between `3_p68`, `3_p95` and `3_mixed` would therefore indicate a bug; `ua_mahalanobis` gives identical results on all three.
- Shape 2 keeps the trace (`c_ee + c_nn = sig_a² + sig_b²`) but loses the orientation and eccentricity.

---

## 5. Caveats

### 5.1 The accuracy radius must be labelled with its true probability

`project.md` defines `horizontalAccuracy` as CEP50, and that remains the default. Real producers differ. Android documents `Location.getAccuracy()` as the radius of **68%** confidence (https://developer.android.com/reference/android/location/Location#getAccuracy()). Apple's `horizontalAccuracy` is not specified as CEP50 either. Vendors often state 95%. Supply `accuracyProbability` (0.68, 0.95, …) per event instead of rescaling. If a p68 radius arrives unlabelled, it is read as CEP50 and σ is inflated by k(0.68)/k(0.5) = 1.282. The source probability should be confirmed for each feed in production.

### 5.2 Probability range

The schema allows `probability` (shape 3) and `accuracyProbability` (shape 2) anywhere in (0, 1). As p → 0, k(p) → 0 and σ = axis / k blows up. A mislabelled p (e.g. 0.05 meant as 95%) would silently create enormous covariances. A sanity bound on p per source (e.g. ≥ 0.39) would guard against this.

### 5.3 The RMS-circle CEP is not the true CEP50 of an elliptical error

The synthetic shape 2 uses `CEP = 1.17741 · σ_rms`, which is exact only for circles. The same applies to any feed that reduces an ellipse to one radius. Monte Carlo check (10⁶ draws, σ_a = 1):

| σ_b / σ_a | true CEP50 | RMS-based | overstatement |
|---|---|---|---|
| 1.0 | 1.176 | 1.177 | 0% |
| 0.5 | 0.870 | 0.931 | +7% |
| 0.2 | 0.706 | 0.849 | +20% |
| 0.01 | 0.674 | 0.833 | +24% |

In the degenerate limit the true value tends to the 1D median |N(0,1)| = 0.6745. The RMS approach preserves total variance (trace) and is simple, but it overstates the 50% radius for elongated errors. The effect is largest for CDR sectors, which are elongated by design. It does not affect shape 3.

### 5.4 Independence

Fusion and gating assume independent errors between events. Consecutive fixes from the same source (a cached or stale fix, the same serving sector, Wi-Fi positioning off the same access points) are strongly correlated in reality. Fused centroid sigmas are therefore optimistic by up to a factor of √n for correlated runs.

---

## 6. Tests

`tests/test_uncertainty.py`:
- `test_k_factor_values` (k(0.95), k(0.68), k(ONE_SIGMA_P) = 1)
- `test_cep50_is_k_at_half`
- `test_ellipse_cov_roundtrip`
- `test_ellipse_azimuth_is_clockwise_from_north`
- `test_monte_carlo_coverage` (p = 0.5, 0.68, 0.95)
- `test_with_covariance_shape1/2/3`, `test_with_covariance_shape2_accuracy_probability`, `test_with_covariance_shape2_defaults_to_cep50`
- `test_haversine_known_distance`
- `test_enu_roundtrip`

## References

- Hoover, W. E. (1984). *Algorithms for Confidence Circles and Ellipses*. NOAA Technical Report NOS 107 C&GS 3, Rockville, MD.
- van Diggelen, F. (2007). GNSS Accuracy: Lies, Damn Lies, and Statistics. *GPS World* 18(1):26–32.
- Mahalanobis, P. C. (1936). On the generalized distance in statistics. *Proc. National Institute of Sciences of India* 2(1):49–55.
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley. doi:10.1002/0471221279
- Circular error probable. Wikipedia. https://en.wikipedia.org/wiki/Circular_error_probable
- Android `Location.getAccuracy()`. https://developer.android.com/reference/android/location/Location#getAccuracy()
