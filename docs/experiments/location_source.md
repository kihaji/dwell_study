# Experiment: optional `location_source` field (branch `exp/ua-location-source`)

**Question:** does a per-report provenance label help dwell detection when the data
type (GPS / CDR / adtech / mixed) is unknown?

## The field

`location_source` is optional, nullable (null = `unknown`), and validated against:

| value | meaning | typical origin |
|---|---|---|
| `gps` | satellite fix | Android `gps` provider, OpenRTB `geo.type = 1` |
| `network` | Wi-Fi or cell positioning | Android `network` provider; CDR |
| `fused` | platform blend of sources | Android `fused` provider; iOS (no provider exposed) |
| `ip` | IP geolocation | OpenRTB `geo.type = 2` |
| `user_provided` | entered or registered address, not a measurement | OpenRTB `geo.type = 3` |
| `unknown` | not stated | |

How the detector uses it:
1. **Assumed error** (only when a report has no accuracy fields): `source_sigma` maps a label
   to a 1σ (gps 10 m, network 300 m, fused 30 m); otherwise `default_sigma_m`. Stated
   accuracy always wins over the label.
2. **Forced vague:** `ip` and `user_provided` reports are always coarse (`coarse_sources`).
   They can never start, end, extend or move a dwell, whatever accuracy they claim.

## Synthetic labels

The generator records each report's true source, mapped as:
* `gps` and `gps_outlier` → `gps`
* `wifi`, `cell` and `cdr` → `network`
* `ip` → `ip`
* `stale` (a cached fix re-reported) keeps the label of the report it repeats.

The generator produces no `fused` or `user_provided` reports, so their handling is
defined but **untested**.

## Tuning (dev): assumed 1σ for `network`, centerpoint-only data

| network σ | 50 m | 100 m | 300 m | 600 m |
|---|---|---|---|---|
| GPS | 0.979 | 0.979 | 0.979 | 0.979 |
| CDR | 0.409 | 0.450 | **0.475** | 0.418 |
| adtech | 0.355 | 0.359 | 0.366 | 0.375 |

300 m was chosen as the compromise (it is best for CDR and within 0.01 of best for adtech).

## Held-out results (eval sets, min dwell 15 min, F1)

| condition | GPS | CDR | adtech |
|---|---|---|---|
| A. centerpoint only, no label, type unknown (assumed 30 m) | 0.980 | 0.387 | 0.278 |
| B. centerpoint only, no label, analyst states the type | 0.980 | 0.497 | 0.332 |
| **C. centerpoint only + location_source** | 0.980 | **0.497** | **0.400** |
| D. centerpoint only + location_source on 50% of reports | 0.981 | 0.436 | 0.329 |
| E. accuracy fields (ellipses), no label | 0.981 | 0.622 | 0.498 |
| F. accuracy fields + location_source | 0.981 | 0.622 | 0.498 |

For adtech, centerpoint only (A → C):
* precision 0.28 → 0.46
* recall 0.28 → 0.36
* median centroid error 11 m → 6 m

### IP reports claiming a small accuracy (adtech, accuracy fields present)

Real feeds often attach a confident accuracy to IP geolocation. The synthetic IP reports
honestly state about 6 km, which is why E = F above. The `ip_confident` perturbation sets
IP reports' 1σ to a small value:

| IP claims 1σ | without label | with location_source |
|---|---|---|
| 50 m | 0.422 (precision 0.50) | **0.498** (precision 0.68) |
| 200 m | 0.424 (precision 0.50) | **0.498** (precision 0.68) |

## Conclusions

1. **Centerpoint-only data:** labels are at least as good as having the analyst state the
   data type, and better for mixed sources (adtech +0.07 F1 over a declared type, +0.12
   over unknown). They work per report, so mixed uploads are handled.
2. **Partial labels help less:** 50% coverage recovers about half of the CDR gain and
   matches a declared type for adtech. Unlabelled reports fall back to the default
   assumption.
3. **Accuracy fields present:** the label adds nothing when accuracies are honest, but
   fully protects against IP geolocation with an overconfident accuracy (−0.08 F1 avoided).
4. **Not tested:** the `fused` and `user_provided` handling (no synthetic data),
   real-world label reliability, and iOS data (which does not expose a provider).

Reproduce: `src/dwells/harness/tune.py` (`run_one(..., with_source=True, coverage=...)`) and
the `ip_confident` perturbation; results in `results/ua/locsrc.parquet`.
