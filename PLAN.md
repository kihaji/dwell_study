# Dwell Analysis Research — Execution Plan

Derived from `project.md` plus clarification Q&A (2026-09-30).

## Decisions captured

| Topic | Decision |
|---|---|
| Accuracy | Synthetic data with ground truth for scored accuracy; public datasets for realism/scaling (qualitative, no labels) |
| Scope | Full pipeline: dwell detection → places → temporal patterns → post-dwell movement → colocation → top-N next location |
| Dwell definition | Parameterized; every algorithm evaluated across a sweep (min duration 5/15/30/60 min × radius 50/100/250 m, adapted per algorithm) |
| Uncertainty | Core focus. All algorithms run on shapes 1/2/3; uncertainty-aware variants built; measure whether error info helps |
| Ellipse probability | Datasets at p=0.68, p=0.95, and mixed. Normalize via bivariate-normal scaling k(p)=sqrt(-2 ln(1-p)); CEP50 → sigma = CEP/1.1774 |
| Error model | Per-source profiles (GPS-like, CDR-like, adtech-like), heterogeneous error within a single identifier's track allowed |
| Generator | Real OSM road networks (osmnx), two contrasting areas: dense urban core + suburban/rural |
| Colocation truth | Shared places (household/coworkers), same place at different times, near-miss decoys (adjacent buildings), occasional meetings |
| Volume | Up to ~10M events total (e.g. 1k ids × 10k events) plus a few single ids at 1M+ events |
| Spark | Design review only (no PySpark runs); assess per-identifier partitionability, state, shuffle needs |
| Stack | Python ≥3.12, uv, Parquet on disk, polars for own code; pandas/geopandas only at library boundaries |
| Prior code | Port distinct algorithms from ellipses / new_ellipses / analytic_app as candidates, with source attribution |
| Viz | Standard output schema (Dash/deckgl-ready parquet) + one small Dash + deckgl-dash comparison viewer |
| Scorecard | Quantitative metrics + qualitative rubric + a weighted single score (weights proposed below) |
| Execution | Fully autonomous; failed experiments stay on branches, documented in final report |

## Proposed score weights (adjustable in `harness/weights.toml`)

- Accuracy (dwell F1, time/centroid error, place/colocation/next-loc metrics): **35%**
- Scalability (runtime + memory scaling slope, 10M-event completion): **20%**
- Uncertainty utilization (accuracy lift from shape 1 → 2 → 3, robustness to coarse points): **15%**
- Spark-readiness (per-id partitionable, bounded state, no global ops): **10%**
- Extensibility (pluggable params, clean API, output richness): **10%**
- Ease of use (install weight, API friction, LOC to integrate, docs): **10%**

Tradeoffs and "best application" remain qualitative per algorithm.

## Standard schemas

- **events**: `id, lat, lon, timestamp` [+ `horizontalAccuracy`] [+ `semiMajor, semiMinor, azimuth, probability`]
- **dwells**: `algorithm, params_hash, id, dwell_id, start_ts, end_ts, duration_s, lat, lon, n_events, [sigma_major, sigma_minor, azimuth]` (uncertainty of the dwell centroid where the algorithm supplies it)
- **places**: `id, place_id, lat, lon, radius_m, n_visits, first_seen, last_seen, total_dwell_s`
- **transitions**: `id, from_place, to_place, depart_ts, arrive_ts`
- **colocations**: `id_a, id_b, place_id/lat/lon, overlap_start, overlap_end, overlap_s, kind (same-time/diff-time)`
- **predictions**: `id, from_place, rank, to_place, prob`
- Ground truth uses the same schemas, plus `truth_*` tables from the generator

## Repository layout

```
dwells/
  pyproject.toml            (uv; optional dependency groups per library)
  src/dwells/
    schema.py  io.py  uncertainty.py  geo.py
    generator/              (OSM, agents, routines, sampling, error profiles)
    algorithms/<name>.py    (one module per implementation, common detect() API)
    downstream/             (places, patterns, transitions, colocation, nextloc)
    metrics/                (matching, scoring)
  scripts/run_<name>.py     (standalone CLI per algorithm: dataset in, parquet out)
  harness/                  (matrix runner, benchmarks, scorecard, weights.toml)
  viewer/                   (Dash + deckgl-dash app)
  data/                     (gitignored; generated + downloaded)
  docs/                     (markdown per algorithm, library, dataset, results)
  tests/
```

## Step-by-step execution

Each step is TDD (tests before implementation), committed to `main` on success. Speculative work goes on `exp/<name>` branches; failures are committed there and left unmerged.

### Phase 0 — Foundation
1. **uv project scaffold**, Python 3.12, ruff, pytest, `.gitignore` for `data/`. → verify: `uv run pytest` runs empty suite.
2. **Schemas + validation** (`schema.py`) for events (3 shapes) and all output tables. → verify: tests for valid/invalid frames.
3. **Uncertainty utilities**: CEP50 ↔ sigma, ellipse(p) ↔ covariance, covariance ↔ ellipse, Mahalanobis, inverse-variance weighted mean, local ENU projection. → verify: property tests (round-trip, known chi-square values, Monte Carlo coverage ≈ p).
4. **Library feasibility check**: install trackintel, scikit-mobility, fastmob (verify repo/API actually exists and works on 3.12), infostop, movingpandas. → verify: smoke import + tiny run each; record install friction in `docs/libraries/*.md`. Anything that fails on 3.12 goes to an `exp/` branch with notes.

### Phase 1 — Data
5. **OSM areas**: download/cached road graphs + POIs for a dense urban core and a suburban/rural area (candidates: e.g. downtown Chicago/Manhattan-like core; a Macon, GA-like suburban area). → verify: graph loads offline from cache.
6. **Agent simulator**: persona routines (home/work/school/gym/shops/leisure), weekday/weekend variation, stochastic durations, road-network travel with speeds; households, coworkers, shift workers sharing places at different times, occasional meetings, near-miss decoys (adjacent buildings). Emits ground-truth dwells, places, transitions, colocations. → verify: tests on routine invariants, truth tables consistent with trajectories.
7. **Sampling + error profiles**: GPS-like (dense, 5–50 m), CDR-like (event-driven, sector/km-scale ellipses), adtech-like (bursty user-driven, heterogeneous 10 m–5 km per id, long gaps). Emit shapes 1/2/3; shape 3 at p=0.68, 0.95, mixed; reported point drawn from the stated error distribution. → verify: empirical coverage matches p; gap distributions match profile.
8. **Dataset catalog**: fixed seeds, small (dev), medium (accuracy matrix), scale ladder (10k → 10M events; single-id 1M+). `docs/datasets.md`. → verify: generation reproducible (hash check).
9. **Public dataset survey + ingest**: GeoLife plus others found (e.g. YJMob100K, Foursquare/Gowalla check-ins, MDC if accessible); normalize to events schema; subsample/add synthetic error to mimic sources. `docs/public_datasets.md` with licenses. → verify: loaders produce valid events.

### Phase 2 — Evaluation core
10. **Metrics**: dwell matching (temporal IoU + centroid distance, Hungarian per id), precision/recall/F1, start/end/centroid error, fragmentation/merging rates; place ARI/matching; colocation P/R; next-loc top-N accuracy + MRR. → verify: hand-built cases with known scores.
11. **Harness**: algorithm registry, param sweeps, algorithm × dataset × shape matrix, subprocess isolation with timeout, runtime + peak-memory capture, results to parquet. → verify: runs a trivial dummy algorithm end to end.

### Phase 3 — Dwell detection implementations (each: module + `scripts/run_*.py` + `docs/algorithms/*.md` with references)
12. **Ports**: Li running-centroid (new_ellipses `dwell_detection.py`), H3 cell dwell (ellipses `hexbin.py`), Kalman/RTS-smoothed dwell (new_ellipses `place_follower.py`).
13. **Libraries**: trackintel staypoints, scikit-mobility `stay_locations`, fastmob, movingpandas stop detector, Infostop.
14. **Density clustering**: ST-DBSCAN, HDBSCAN / ST-HDBSCAN.
15. **Uncertainty-aware (new)**: Mahalanobis/ellipse-overlap membership, inverse-variance centroid + dwell covariance, probabilistic segmentation with per-point likelihoods; coarse-point handling (down-weight vs. drop).
16. **Probabilistic/temporal**: HMM stay/move segmentation, Bayesian change-point detection.
17. **Grid**: H3 occupancy (multi-resolution, accuracy-adaptive resolution).
→ verify each: tiny fixture tests (one clear dwell, gap-split, pass-through non-dwell), then harness run on dev dataset.

### Phase 4 — Downstream analytics
18. **Places**: cluster dwells into places (DBSCAN port, HDBSCAN, GMM, H3, uncertainty-weighted merge); per-algorithm place output where the library provides it (trackintel locations, skmob clustering).
19. **Patterns**: time-of-day/week profiles, return frequency, events per dwell, dwell regularity/entropy.
20. **Post-dwell movement**: transition matrices, departure time distributions, sequence patterns (A→B→C).
21. **Colocation**: same-time (Mahalanobis port, H3 overlap) and different-time co-visitation; evaluated against near-miss decoys.
22. **Next location**: baselines (most-frequent, order-1/2 Markov, time-conditioned Markov); top-N output. → verify all: metrics against generator truth.

### Phase 5 — Evaluation & reporting
23. **Accuracy matrix** across algorithms × sources × shapes × parameter sweep.
24. **Scaling benchmarks** on the ladder; fit scaling curves; note where each fails/OOMs.
25. **Spark-readiness review** per algorithm (per-id `applyInPandas` feasibility, state, global ops).
26. **Scorecard** with weighted score + qualitative tradeoffs/best application. → `docs/results/scorecard.md`.
27. **Viewer**: Dash + deckgl-dash app: pick dataset/id/algorithm(s), map overlay of events (with ellipses), detected vs. true dwells, places, transitions; comparison tables. → verify: launches and renders dev dataset.
28. **Final report** `docs/REPORT.md`: findings, recommendations for the downstream app, index of all docs, and a list of `exp/` branches with what each tried and why it failed.

## Remaining assumptions (proceeding unless corrected)
- Exact OSM areas are chosen in step 5 (dense core + suburban). Macon, GA is preferred for the suburban area, for continuity with the existing repos.
- Sweep grids are trimmed per algorithm to keep the matrix tractable locally.
- Libraries that can't run on Python ≥3.12 are documented as such rather than forced.
