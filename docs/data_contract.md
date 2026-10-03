# Input data contract for ua_mahalanobis

What the upstream cleaning and transformation pipeline must deliver, and how much each
defect costs. The impact numbers are measured: held-out synthetic data (eval sets,
urban + suburban), tuned presets, shape 3 (ellipses), minimum dwell 15 min. Each defect
is injected on purpose and the resulting change in F1 is reported (F1 is the dwell-
detection score, 0–1). Reproduce with `uv run python -m dwells.harness.tune robustness`
(see `src/dwells/harness/perturb.py`).

Clean baselines: GPS 0.981, CDR 0.622, adtech 0.498.

## 1. Required fields

| field | type | rule |
|---|---|---|
| `id` | string | Stable device identifier. One id = one device. |
| `lat`, `lon` | float64, WGS84 decimal degrees | Must be in range. Keep full precision if you can, although rounding to 3 decimals (about 110 m) cost < 0.002 F1 here (see §4). |
| `timestamp` | UTC instant (microsecond resolution is plenty) | The time the position was **measured**, not received. No timezone-naive local times. |

Optional error fields (strongly recommended for anything but dense GPS):

| shape | fields | rule |
|---|---|---|
| 2 | `horizontalAccuracy` (m) | Must be the **CEP50 radius**: 50% of true positions fall inside. |
| 3 | `semiMajor`, `semiMinor` (m), `azimuth` (deg, 0–180, clockwise from north, of the major axis), `probability` (0–1 exclusive) | `semiMajor >= semiMinor > 0`. `probability` must state the actual confidence of the ellipse (e.g. 0.68 or 0.95). Mixed probabilities within a dataset are fine. |

The 0.68, 0.95 and mixed variants of the same data gave identical results, so the
normalization is correct **as long as each value is labelled with its true meaning**.

Optional provenance field (recommended whenever the feed states it):

| field | values | rule |
|---|---|---|
| `location_source` | `gps`, `network`, `fused`, `ip`, `user_provided`, `unknown` (null = unknown) | How the fix was produced, using the Android provider terms (`gps`, `network`, `fused`). Adtech feeds following OpenRTB map `geo.type` 1 → `gps`, 2 → `ip`, 3 → `user_provided`. CDR and Wi-Fi/cell positioning → `network`. iOS exposes no provider, so use `fused` or leave it unknown. A cached / last-known fix keeps the label of the original fix. |

What the detector does with it (evidence: [tuning.md](tuning.md), "location_source labels"):
* `ip` and `user_provided` reports are always treated as vague: they never start, end,
  extend or move a dwell, whatever accuracy they claim. In held-out adtech data, IP
  reports claiming a 50–200 m accuracy cost 0.08 F1 without the label and nothing with it.
* For reports **without** accuracy fields, the label sets the assumed 1σ: gps 10 m, network
  300 m, fused 30 m (untested: no synthetic fused data). Stated accuracy always wins over
  the label.

## 2. Accuracy semantics: the most consequential pitfall

Different sources mean different things by "accuracy":

* Android `Location.getAccuracy()` is documented as a **68% radius**, not CEP50.
* iOS `horizontalAccuracy` is a 1σ-like radius.
* Vendors and telco geolocation often state 95%.
* Cell-tower data may be the sector radius.

Convert every source to the contract (CEP50 radius, or an ellipse with its true
probability) in the cleaning pipeline. Factors to CEP50: a 68% radius × 0.780; a
1σ circle × 1.177; a 95% radius × 0.481.

Measured impact of a scale mistake on the stated error:

| reported error vs truth | GPS | CDR | adtech | typical cause |
|---|---|---|---|---|
| × 0.5 (stated twice as good) | −0.001 | **−0.125** | −0.046 | trusting vendor accuracy, wrong unit |
| × 0.78 (CEP50 labelled as 68%) | 0.000 | −0.013 | −0.008 | |
| × 1.28 (68% radius labelled CEP50) | 0.000 | −0.044 | +0.003 | Android accuracy passed through as-is |
| × 2.08 (95% radius labelled CEP50) | +0.001 | **−0.153** | +0.004 | vendor 95% passed through as-is |
| × 4 | −0.002 | **−0.368** | −0.002 | unit / semantic bugs |
| random ± (lognormal sd 0.5 / 1.0) | 0.000 / +0.001 | −0.035 / **−0.096** | −0.006 / −0.031 | poorly calibrated estimates |

Dense GPS barely cares because its errors are small compared with the place size.
Coarse, uniform sources (CDR) care most, because their error decides whether a fix
counts as "coarse". The detector protects itself partly: a fix is coarse only if it is
above 1 km **and** 3× worse than that device's typical fix. Without that rule, the 2.08×
case cost −0.33 instead of −0.15.

## 3. Centerpoint-only data

When no error fields exist, the detector assumes a constant error per fix
(`default_sigma_m` in `src/dwells/presets.py`). It must match the source:

| assumed 1σ (m) | 10 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|
| GPS F1 (dev) | **0.979** | 0.970 | 0.952 | 0.838 | 0.358 |
| CDR F1 (dev) | 0.358 | 0.386 | 0.450 | **0.475** | 0.244 |
| adtech F1 (dev) | 0.291 | 0.294 | 0.309 | **0.322** | 0.247 |

Rules:

* Use about 10 m for GPS, about 300 m for CDR and adtech.
* Never use a value at or above the coarse threshold (1 km). Every fix would then be
  ignored and detection collapses.
* Better still, attach at least a per-source nominal `horizontalAccuracy`. Held-out
  F1 with error fields versus centerpoint only: CDR 0.607 vs 0.497, adtech 0.497 vs 0.332.
* If accuracy is unavailable, supply `location_source`. The assumed error is then chosen
  per report, which also handles mixed uploads. Held-out centerpoint-only F1:

  | | GPS | CDR | adtech |
  |---|---|---|---|
  | type unknown (assumed 30 m) | 0.980 | 0.387 | 0.278 |
  | analyst states the type | 0.980 | 0.497 | 0.332 |
  | `location_source` on every report | 0.980 | 0.497 | **0.400** |
  | `location_source` on 50% of reports | 0.981 | 0.436 | 0.329 |

## 4. Defects and how much they matter

| defect (level) | GPS | CDR | adtech | pipeline action |
|---|---|---|---|---|
| **Sampling thinned to 50%** | −0.065 | −0.089 | −0.061 | Do **not** subsample or rate-limit. Keep every fix. |
| **Sampling thinned to 25% / 10%** | −0.094 / −0.142 | −0.202 / −0.371 | −0.161 / −0.296 | Event density is the single biggest accuracy driver. |
| **Outliers with confident accuracy (5% / 10%)** | −0.011 / −0.062 | −0.019 / −0.038 | −0.046 / −0.091 | Drop physically impossible jumps (speed > ~60 m/s and distance > 3× the combined error) and known bad sources. |
| Stale / cached fixes re-reported (15% / 30%) | −0.004 / −0.006 | −0.021 / −0.038 | −0.018 / −0.032 | Drop fixes flagged as cached or last-known. Collapse runs of identical positions with new timestamps when the source is known to cache. |
| Exact duplicate rows (up to 100%) | ≤ −0.013 | ≤ −0.007 | ≈ 0 | Deduplicate exact (id, timestamp, lat, lon) duplicates; low priority. |
| Clock jitter ±10 s / ±60 s / ±300 s | 0 / 0 / −0.018 | 0 / −0.001 / −0.007 | 0 / −0.002 / −0.004 | Use measurement time. Minutes of jitter are tolerable. |
| Coordinates rounded to 4 / 3 decimals | 0 / 0 | +0.001 / +0.001 | 0 / −0.002 | Harmless at this level. |

## 5. Volume and density expectations

* Accuracy rises steeply with reports per stay. A dwell needs at least 2 precise
  (non-coarse) fixes. In the synthetic adtech data only about 60% of true 15-minute
  dwells contain two fixes at all, which caps recall.
* Long silent periods are fine. A stay survives gaps up to `max_gap_s` (12 h in the
  presets). On real phones that stop reporting while stationary, keep it at 12–24 h.
  On the real STAGA dataset, raising it from 2 h to 24 h moved F1 from about 0.5 to
  0.83 (main-branch study).

## 6. Diagnostics to monitor

The pipeline writes a per-id `quality` table (`docs/outputs.md`; shown in the app's
Data quality panel) with the signals above. Flag ids for review when any of these hold:

| metric | warning level | why |
|---|---|---|
| `events_per_active_day` | < 10 | too sparse to delimit most stays |
| `speed_outlier_frac` | > 2% | confident outliers fragment and displace dwells |
| `stale_frac` | > 10% | cached fixes stretch dwells |
| `duplicate_frac` | > 20% | feed merging issue upstream |
| `coarse_frac` | > 50% | mostly cell/IP-level fixes; dwells will be few and vague |
| `sigma_p50_m` | far from the expected source accuracy | semantics or unit problem (§2) |
| `source_coverage` | < 50% (when the feed should carry labels) | most reports fall back to the default assumption |
| `source_mismatch_frac` | > 1% | labels contradict stated accuracy (`gps` claiming > 100 m 1σ, `ip` claiming < 1 km): a labelling or accuracy-semantics problem upstream |
