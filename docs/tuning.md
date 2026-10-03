# Tuning ua_mahalanobis

Protocol:
* Every decision was made on the **dev** datasets (`dev_{urban,suburban}_{gps,cdr,adtech}`)
  and confirmed on the held-out **eval** datasets (different agents and seeds).
* Scores are dwell-detection F1 at a 15-minute minimum dwell, averaged over both areas.
* Raw results are in `results/ua/{baseline,sweep,robustness}.parquet`, produced by
  `src/dwells/harness/tune.py`.

## Result

Held-out F1, before (v0, the main-branch version with its tuned parameters) and after
(v3, final):

| source | centerpoint only (shape 1) | horizontalAccuracy (shape 2) | ellipses (shape 3) |
|---|---|---|---|
| GPS | 0.900 → **0.980** | 0.939 → **0.980** | 0.939 → **0.981** |
| CDR | 0.454 → **0.497** | 0.495 → **0.607** | 0.499 → **0.622** |
| adtech | 0.245 → **0.332** | 0.356 → **0.497** | 0.355 → **0.498** |

Where the gain came from (shape 3, held out):

| | GPS v0 → v3 | CDR v0 → v3 | adtech v0 → v3 |
|---|---|---|---|
| precision | 0.922 → 0.986 | 0.732 → 0.775 | 0.611 → 0.684 |
| recall | 0.957 → 0.975 | 0.379 → 0.520 | 0.250 → 0.391 |
| median duration bias | +4 s → +77 s | **−32 min → +11 min** | **−43 min → +3 min** |

Recall over true dwells that have at least 2 reports ("observable"): GPS 0.986, CDR 0.682,
adtech 0.636. The rest are dwells the data cannot show.

## Algorithm changes (each measured, then kept)

1. **Midpoint boundaries and arrival/departure windows.**
   * The problem: a dwell used to start at its first report and end at its last. On sparse
     data that truncates every stay, by about 43 min on adtech.
   * The change: the start is now estimated halfway between the last precise report
     elsewhere and the first report here (likewise for the end), capped at `max_gap_s / 2`.
     The raw window is output (`arrive_after`/`arrive_by`, `depart_after`/`depart_by`).
   * The filter: the minimum-duration check uses this estimated span.
   * Dev F1: adtech 0.314 → 0.425, CDR 0.442 → 0.489, GPS 0.936 → 0.949.
   * Coarse fixes do not bound the window: a 5 km fix is no evidence of having left.
     Earlier this let a vague fix stretch a stay; it was caught by a test.
2. **Absolute coarse threshold** (`coarse_sigma_m = 1000`).
   * The old rule tied "coarse" to the place radius (3 × radius). That made suburban CDR
     fixes (σ ≈ 700 m) disappear at small radii: dev F1 0.29 vs 0.51.
   * Place size and fix vagueness are separate questions. About 1 km works for every source.
3. **Time-confirmed departures** (`confirm_s = 900`).
   * A stay used to end only after `k_break` consecutive misses.
   * Sparse data shows a real errand as a single fix elsewhere, while dense GPS shows
     outliers as single fixes followed seconds later by a fix back in place.
   * The stay now also ends when the id stayed away for 15 min before being seen back.
   * This lets one robust setting (`k_break = 3`) serve all sources:
     * GPS keeps its outlier tolerance.
     * CDR dev F1 with `k_break` 3: 0.493 → 0.544.
     * adtech dev F1 with `k_break` 3: 0.409 → 0.450.
4. **Relative coarse rule** (`coarse_rel = 3`). A fix is coarse only if it is also 3×
   worse than that id's median fix. Same clean scores; CDR with error stated 2.08× too
   large: −0.333 → −0.153 F1.

## Parameter sensitivity (dev F1)

Radius × max gap, GPS (rows radius m, columns gap h):

| r \ gap | 4 | 6 | 8 | 12 |
|---|---|---|---|---|
| 35 | 0.960 | 0.964 | 0.967 | 0.967 |
| 50 | 0.966 | 0.970 | **0.974** | **0.974** |
| 75 | 0.966 | 0.970 | 0.973 | 0.973 |

Adtech (gap 12 h; columns = coarse threshold in m):

| r \ coarse | 300 | 600 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| 75 | 0.438 | 0.436 | **0.445** | 0.444 | 0.444 |
| 100 | 0.437 | 0.435 | **0.445** | 0.444 | 0.444 |
| 250 | 0.421 | 0.422 | 0.429 | 0.431 | 0.428 |

CDR shows the same pattern with a sharp cliff below 1000 m (0.30 → 0.51), because
all its fixes become coarse.

Gate probability / k_break / confirm_s:
* GPS needs `k_break` ≥ 2 (`k_break` 1 drops F1 to about 0.47 because of outliers) and
  `confirm_s` ≥ 900. At 300 s, GPS throttled 5-minute sampling makes every outlier look
  like a departure (0.67).
* For CDR and adtech, `k_break` 1 was best on dev. The robust `k_break` 3 + `confirm_s`
  900 matched or beat it on held-out data.

## Presets (`src/dwells/presets.py`)

One base works for every source. The per-source entries differ only in the error assumed
for centerpoint-only data:

| parameter | value | meaning |
|---|---|---|
| radius_m | 75 | spread of positions inside a place (sets the gate width) |
| max_gap_s | 43200 (12 h) | longest silence a stay can bridge |
| gate_p | 0.99 | how far (statistically) a fix may sit from the place center and still belong; app label **same-place tolerance** (99% ≈ 3.0σ) |
| k_break | 3 | consecutive misses that end a stay |
| confirm_s | 900 | ...or this long away before being seen back |
| coarse_sigma_m / coarse_rel | 1000 / 3 | fixes worse than 1 km (1σ) and 3× the id's typical fix are not used to locate stays; app label **vague-report cutoff** (sets coarse_sigma_m) |
| default_sigma_m | gps 10, cdr 300, adtech 300, mixed 30 | error assumed when no accuracy fields exist and no usable location_source label |
| source_sigma | gps 10, network 300, fused 30 | error assumed per location_source label for reports without accuracy fields ([experiments/location_source.md](experiments/location_source.md)); ip / user_provided reports are always vague (`coarse_sources`) |

The base matched or beat the per-source tuned values on held-out data, except adtech by
0.011 (0.498 vs 0.509), so it is the recommended default. Use `mixed` when the source is
unknown.

## Places (grouping visits)

Visits are merged into places with the Mahalanobis merge at a 150 m scale. A visit
joins a place when their centers agree given both uncertainties. Held-out adjusted Rand
index against the true places (1 = perfect):

| source | Mahalanobis 150 | DBSCAN 150 | HDBSCAN 150 |
|---|---|---|---|
| GPS | 0.983 | 0.992 | 0.989 |
| CDR | **0.913** | 0.816 | 0.796 |
| adtech | 0.968 | 0.969 | 0.959 |

It is chosen because it adapts to each visit's precision. That matters most for coarse
sources. The other grouping methods were then removed from this branch, along with the
scikit-learn dependency; they remain on `main`.

## Robustness

See `docs/data_contract.md` for the full defect table and pipeline guidance.
