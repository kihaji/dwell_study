# Branch plan: focus/ua-mahalanobis

> Status: complete. To keep this branch a focused porting reference, the evaluation
> tooling (synthetic generator, tuning harness, defect injection, scoring metrics, STAGA
> loader) was moved to branch `tools/ua-tuning`; results and tuned presets remain here.

Goal: a focused, tuned implementation of `ua_mahalanobis` with tight output data
structures, an input-data contract for the upstream cleaning pipeline, and an analyst
app (Plotly Dash + deckgl-dash) for pattern-of-life exploration. Decisions from the
2026-10-02 Q&A are summarized here.

## Decisions

| Topic | Decision |
|---|---|
| Analyst unit | **Places + visits.** Dwells (visits) are grouped into recurring places; rankings, next-location arcs, the directed graph and heatmaps work per place; drill down to visits and their member events. |
| Keep in branch | Synthetic generator, public loaders, metrics, a slim harness for ua_mahalanobis only. Remove the other 14 detectors, library dependencies, the old comparison viewer and the comparison results docs (main keeps them). |
| Time | UTC everywhere (hour-of-day / day-of-week are UTC; labelled as such). |
| Geocoding | Pluggable interface. Offline default = nearest OSM address/building/POI from a local extract; optional online Nominatim (rate-limited, cached). |
| Place score | Composite 0–100 "pattern confidence" from evidence (events, time), precision (centroid uncertainty) and regularity (distinct days, consistent hours, repeat visits); components shown separately. |
| Spatial extent | User-selectable: uncertainty ellipse, convex hull of member events, member events (+ error circles). |
| Radar | Per place (events / presence per UTC hour over all visits), with the selected visit overlaid. |
| Graph | dash-cytoscape directed graph of places and transitions. |
| Compute | Batch CLI writes the output tables for a whole dataset; the app reads them and offers a live tuning panel that re-runs the selected id with other parameters. |
| Colocation | Initially included but isolated; later removed from this branch for a separate focused study (main keeps the evaluated code). |
| Tuning deliverables | Input data contract doc; per-source presets with sensitivity; data-quality diagnostics (CLI + app); robustness experiments (miscalibrated accuracy, duplicates, stale fixes, clock jitter, sparse sampling). |
| Algorithm changes | Allowed when held-out experiments justify them; failed ideas kept on `exp/ua-*` branches. |
| Extra views | Timeline/Gantt, presence calendar (day x hour), data-quality panel, anomaly flags. |
| Inputs | Repo datasets only (synthetic catalog + ingested public). |
| Docs | Analyst guide (markdown, non-mathematician audience) + in-app help drawn from the same text. |
| Memory safety | All heavy runs under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` (WSL2). |

## Steps

1. **Prune.** Remove other detectors, their tests and dependencies; slim the harness and registry; keep tests green.
2. **Baseline + tuning harness.** ua_mahalanobis-only sweep/eval with per-profile results; an experiment runner for robustness perturbations.
3. **Robustness experiments.** Measure the impact of input defects (accuracy semantics 50/68/95%, miscalibration, duplicates, stale fixes, outliers, clock jitter, sampling sparsity, missing error fields).
4. **Algorithm improvements.** Evaluate candidates (start/end estimation, coarse-fix rule, adaptive gate/radius, duplicate handling), adopting only measured wins. Then per-source presets with sensitivity.
5. **Output schemas (tightened).** Event-to-visit assignment, visits, places (ellipse + hull geometry, score components, geocode), transitions / next-place probabilities, hourly and weekday profiles, anomalies, data quality, colocation (separate). Batch CLI to write them.
6. **Geocoding.** Offline OSM provider (address, building, POI) + optional Nominatim provider, cached.
7. **Analyst app.** Map (extent toggles, member events, next-place arcs), top-10 rankings, directed graph, places/visits table, radar, weekday heatmap, timeline, presence calendar, data quality, anomalies, colocation panel, live tuning panel, in-app help.
8. **Docs.** Input data contract; tuning report with presets; output schema reference; analyst guide.
9. **Verify.** Tests, app callback checks, final summary.
