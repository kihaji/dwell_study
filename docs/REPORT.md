# Dwell detection research: final report

Goals are in `project.md`; the agreed plan and decisions are in `PLAN.md`. Every
number below comes from files under `results/` (regenerate the tables with
`uv run python -m dwells.harness.report` and `... .score`). The detailed tables are
in `docs/results/`.

## 1. Bottom line

| Need in the downstream app | Recommendation | Why |
|---|---|---|
| Most accurate dwell detection on any source | **ua_segmentation** (new) | Best held-out F1 on GPS, CDR and adtech among the scalable methods (0.956 / 0.521 / 0.367 at 15 min), best precision (0.78), best observable recall (0.64), best place grouping (ARI 0.94). 1.8M events/s. |
| Spark-scale streaming / single-pass | **ua_mahalanobis** (new) | Within 0.01–0.02 F1 of ua_segmentation. A single O(n) scan per id with O(1) state. 6.9M events/s; 16.5M adtech events in 2.4 s. Easiest to port to `groupBy(id).applyInPandas`. |
| SQL-friendly / grid-native | **h3_occupancy** (new) | F1 0.92 GPS / 0.43 CDR / 0.35 adtech. Everything except the run scan is a per-event H3 map, so it maps onto Spark SQL with H3 functions. |
| Minimal, dependency-free baseline | **li_centroid** (your port) | 14M events/s and trivial, but no error handling: F1 0.76 GPS / 0.37 CDR / 0.15 adtech. |
| Library to adopt as-is | **fastmob** | Same algorithm as trackintel with identical output (verified), ~130x faster (8–13M events/s, Rust). Its first-fix-anchored radius fragments stays (GPS F1 0.69), so use it for its place, next-location and co-presence utilities rather than as the detector. |
| Avoid | infostop, movingpandas, scikit-mobility, h3_cell_dwell (port), st_hdbscan, changepoint_pelt (ruptures) | Licensing (infostop GPL-3.0 + AGPL infomap; global clustering that does not partition and ran out of memory), quadratic or very slow implementations, the lost trailing stay (skmob), or fragmentation (h3_cell_dwell: GPS F1 0.05). |

Weighted scorecard (weights from PLAN.md: accuracy 35, scalability 20, uncertainty
15, Spark 10, extensibility 10, ease of use 10; `src/dwells/harness/weights.toml`):

| rank | algorithm | score | rank | algorithm | score |
|---|---|---|---|---|---|
| 1 | ua_segmentation | 0.905 | 9 | trackintel_sliding | 0.592 |
| 2 | ua_mahalanobis | 0.899 | 10 | h3_cell_dwell | 0.586 |
| 3 | hmm_stay_move | 0.879 | 11 | skmob_stops | 0.536 |
| 4 | h3_occupancy | 0.815 | 12 | changepoint_pelt | 0.534 |
| 5 | li_centroid | 0.767 | 13 | infostop | 0.515 |
| 6 | st_dbscan | 0.728 | 14 | st_hdbscan | 0.514 |
| 7 | fastmob_stays | 0.716 | 15 | movingpandas_stops | 0.512 |
| 8 | kalman_dwell | 0.657 | | | |

`hmm_stay_move` ranks third mainly because it has the largest gain from error
information. Its absolute accuracy (GPS 0.89) is below the top two, so for production
use prefer ua_segmentation or ua_mahalanobis. The full component breakdown and rubric
rationale are in `docs/results/scorecard.md`.

## 2. Key findings

1. **Error information helps exactly where the data is sparse and heterogeneous.**
   Positions are identical across shapes; only the error fields differ. Held-out
   F1 gain from shape 1 (no error) to shape 3 (ellipses):
   * Adtech: +0.16 ua_segmentation, +0.11 ua_mahalanobis, +0.16 hmm, +0.07 h3_occupancy.
   * CDR: +0.07, +0.05, +0.13 and 0.00 for the same four.
   * Dense GPS: −0.01 to +0.02, i.e. no help.

   The mechanism is coarse-fix handling: cell, IP or stale fixes cannot discriminate
   places, so they must not start, break, extend or pull a dwell. Ignoring that is
   why every error-blind method stays at or below F1 0.23 on adtech
   (`docs/results/uncertainty_lift.md`).
2. **Ellipse probability normalization works.** Datasets at p=0.68, p=0.95 and mixed
   produce identical results. Elliptical shape adds little over CEP50 on these data
   (shape 2 is within ±0.01 of shape 3).
3. **The anchor matters more than the library.** Detectors that measure the radius
   from the first fix (trackintel, fastmob, scikit-mobility) fragment long stays:
   GPS precision 0.40 against 0.78 for ua_segmentation. Running-centroid and
   segmentation methods do not have this problem.
4. **Gap tolerance has to follow the device.** On the real labeled STAGA data every
   detector plateaued at F1 ≈ 0.5, because 131 of 683 labeled stops contain silent
   periods of more than 2 h. With a 24 h gap tolerance, F1 rises to 0.83 for
   ua_mahalanobis and 0.79 for ua_segmentation, with the same ranking as on
   synthetic data (`docs/results/public_data.md`).
5. **Sparse data has a hard ceiling.** Only about 60% of true adtech dwells of 15 min
   or more contain two or more events at all (CDR about 75%, GPS 99%), so adtech
   recall cannot exceed about 0.6. Short stops are effectively invisible in adtech:
   fuel 3%, grocery 19% (`docs/datasets.md`).
6. **Colocation is limited by physics, not by the algorithm.** Same-time colocation
   recall is high (GPS 0.97–0.98, adtech about 0.6). But next-door neighbors 8–21 m
   apart are flagged 84–100% of the time even at a 15 m match radius on GPS, because
   dwell centroids are not precise enough to separate adjacent buildings. Precision
   is controlled by the match radius: on GPS with ua_segmentation it is 0.84 at
   15 m, 0.80 at 30 m and 0.41 at 150 m. Different-time co-visitation pair F1:
   0.45 GPS, 0.26–0.28 CDR and adtech.
7. **Place grouping and home/work inference are robust once detection is good.**
   * Place grouping: ARI is about 0.99 on GPS for any grouping method. The grouping
     method matters less than the detector: DBSCAN and HDBSCAN are best, while H3
     cells are consistently worst (0.35–0.64).
   * Home inference is 0.96–0.97 for the top detectors. Work inference is 1.0 on GPS
     and about 0.86–0.91 on sparse sources.
8. **Next-location: detection makes routines look more predictable than they are.**
   Order-1 Markov acc@3 is 0.77 on true visit sequences, but 0.82–0.89 on detected
   sequences, because short visits go undetected. Order-2 and time-conditioned
   Markov models add nothing over order 1 on these synthetic routines.
9. **Scalability spans four orders of magnitude.** On the 16.5M-event adtech rung:
   li_centroid 9M events/s, fastmob 8.7M, ua_mahalanobis 6.9M, st_dbscan 4.1M, hmm 2.9M,
   ua_segmentation 1.8M, h3_occupancy 1.8M. trackintel needs about 6.5k events/s on
   sparse data. movingpandas, trackintel and scikit-mobility exceeded 900 s, and
   infostop was not runnable in memory.
   Per id (one id, ~1M events): st_hdbscan and ruptures PELT time out. HDBSCAN is not
   suitable when one id holds millions of events (`docs/results/scaling.md`).

## 3. What was built

| Component | Where |
|---|---|
| Canonical schemas (3 event shapes, dwells, places, transitions, colocations, predictions) | `src/dwells/schema.py` |
| Uncertainty model (CEP50 / ellipse(p) to covariance, Mahalanobis, fusion) | `src/dwells/uncertainty.py`, `docs/uncertainty.md` |
| Synthetic generator: OSM road networks (Chicago Loop, Macon GA), households/personas/routines, shared workplaces, neighbor decoys, meetings; GPS / CDR / adtech sensing with honest-ish error | `src/dwells/generator/`, `docs/datasets.md` |
| Dataset catalog: dev (tuning), eval (held out), scale ladder to 16.5M events, 1M-event single id | `uv run python -m dwells.generator.build` |
| Public data: STAGA (labeled), PoL benchmark (labeled), GeoLife, Foursquare NYC | `src/dwells/public/`, `docs/public_datasets.md` |
| 15 detectors with a common `detect()` API | `src/dwells/algorithms/`, `docs/algorithms/` |
| Downstream: places (5 methods), patterns + home/work, transitions + sequences, next location (4 Markov models), colocation (same and different time; 3 matchers) | `src/dwells/downstream/`, `docs/algorithms/downstream.md` |
| Metrics (IoU matching, observable recall, fragmentation, colocation scoring) | `src/dwells/metrics.py`, `docs/metrics.md` |
| Harness: dev sweep, held-out eval, downstream eval, scaling benchmark, scorecard, report | `src/dwells/harness/` |
| Spark readiness review | `docs/results/spark.md` |
| Dash + deckgl-dash comparison viewer | `viewer/app.py` (`uv run --extra viewer python viewer/app.py`) |
| Tests: 172 passing plus 11 strict xfails documenting known limitations of ports and libraries | `tests/` (`uv run pytest`) |

**Standalone scripts.** Each detector runs on any dataset through
`dwells.algorithms.get(name).detect(events, ...)`, and the harness drives them all.
I did not add one thin `scripts/run_<name>.py` file per algorithm, because the
registry plus `python -m dwells.harness.run sweep --algos <name> --datasets <glob>`
covers the same need. Say if you still want per-algorithm entry points.

## 4. Detection accuracy (held out, min duration 15 min)

Tuned on dev data (different agents and seeds), scored on eval data. Uncertainty-
aware methods use shape 3_mixed. "gps / cdr / adtech" is the mean of the urban and
suburban eval sets.

| algorithm | gps | cdr | adtech | STAGA* | PoL_50 |
|---|---|---|---|---|---|
| ua_segmentation | 0.956 | **0.521** | **0.367** | 0.517 | 0.996 |
| changepoint_pelt | **0.959** | 0.403 | 0.226 | 0.518 | timeout |
| st_dbscan | 0.942 | 0.305 | 0.229 | 0.292 | 0.965 |
| ua_mahalanobis | 0.939 | 0.499 | 0.355 | 0.500 | 0.992 |
| h3_occupancy | 0.919 | 0.432 | 0.350 | 0.506 | 0.984 |
| hmm_stay_move | 0.891 | 0.466 | 0.312 | 0.352 | 0.901 |
| movingpandas_stops | 0.833 | 0.417 | 0.173 | 0.471 | timeout |
| infostop | 0.777 | 0.354 | 0.149 | 0.495 | 0.969 |
| kalman_dwell | 0.776 | 0.268 | 0.132 | 0.381 | 0.943 |
| li_centroid | 0.758 | 0.367 | 0.146 | 0.501 | 0.972 |
| trackintel_sliding = fastmob_stays | 0.687 | 0.386 | 0.185 | 0.516 | 0.979 |
| skmob_stops | 0.684 | 0.377 | 0.183 | 0.579 | 0.975 |
| st_hdbscan | 0.545 | 0.086 | 0.081 | 0.115 | 0.999 |
| h3_cell_dwell | 0.053 | 0.276 | 0.116 | 0.171 | 0.997 |

\*STAGA at the synthetic-tuned 2 h gap; with a 24 h gap the top methods reach 0.76–0.83
(finding 4). PoL's clean variant has no positional noise, so it validates
correctness but cannot rank methods. Per-dataset tables, other minimum durations,
error anatomy (start/end/centroid errors, fragmentation, merging) and parameter
sensitivity are in `docs/results/accuracy.md` and `docs/results/sensitivity.md`.

## 5. Per-algorithm verdicts

| algorithm | best application | main tradeoff |
|---|---|---|
| ua_segmentation | default detector for mixed-quality data | penalty parameter is less intuitive than a radius; near-linear time but O(n) memory per segment |
| ua_mahalanobis | streaming / Spark, very large ids | slightly below segmentation on CDR; gate and coarse factors fixed at sensible defaults |
| h3_occupancy | SQL/H3 pipelines; joins with other H3 data | hysteresis tolerates up to ~2 cell edges, coarser than radius_m suggests |
| hmm_stay_move | when the stay probability itself is wanted; extensible to transport modes | lower GPS accuracy (0.89); sensitive to its priors |
| changepoint_pelt (ruptures) | offline research on small data | pure-Python PELT: ~1k events/s on GPS, timeouts at 1M events per id |
| st_dbscan | dense GPS with clear stops | chains stays through slow walks or dense driving (known limitation); weak on coarse data |
| li_centroid (port) | baseline; very fast | no error model; a single outlier breaks a stay |
| kalman_dwell (port) | smoothing noisy dense tracks | dense time grid; weak on sparse data (adtech 0.13) |
| h3_cell_dwell (port) | none for detection | resolution set by error size causes fragmentation (GPS 0.05); batch-dependent resolution |
| trackintel / fastmob | ecosystem utilities (trips, tours, locations, next-location, co-presence) | first-fix anchoring fragments; trackintel is slow on sparse data |
| scikit-mobility | none | needs a Shapely 2 shim, drops uid under pandas 3, never emits the trailing stay |
| movingpandas | trajectory analytics, not detection | no gap handling, quadratic stop window, slow |
| infostop | none in this context | GPL/AGPL, non-deterministic, global clustering exceeded memory |
| st_hdbscan | small dense-GPS ids only | relative density needs heavy post-processing; times out per id at 1M events |

## 6. How the analytic questions are answered

| Question (project.md) | Pipeline |
|---|---|
| Where does an identifier dwell? | detector, then `downstream.places.assign` (DBSCAN on dwell centroids, or the Mahalanobis merge when centroid sigma is reported), then the PLACES table |
| Is there a pattern to where they dwell? | `patterns.place_profiles` (hour-of-day and day-of-week histograms, regularity, return interval) and `label_home_work` |
| Pattern of movement after a dwell? | `transitions.matrix` (next-place probabilities, travel time, departure hour) and `transitions.sequences` (A→B→C) |
| Dwell statistics (start, end, return frequency, events) | DWELLS (start, end, duration, n_events, centroid sigma) and the place profiles |
| Colocation (same or different times) | `colocation.same_time` (distance / Mahalanobis / H3) and `colocation.diff_time` |
| Top-N next locations | `nextloc.fit(...).predict_top(n)`; order-1 Markov is as good as the richer models here |

All output tables are parquet in the schemas of `src/dwells/schema.py`, ready for
Dash tables and deckgl-dash layers. The viewer shows them on a map with a timeline.
I verified that the viewer serves and that its callbacks produce the layers, figures
and tables, but I could not open it in a browser here, so please check it visually.

## 7. Experiment branches and approaches that did not work

| Branch / approach | What it tried | Outcome |
|---|---|---|
| `exp/st-hdbscan-spacetime` | classic space-time features for ST-HDBSCAN | **Succeeded on GPS** (dev F1 0.37 → 0.52–0.58) but worse on CDR. Merged as the default; spatial-only mode remains via `time_scale_s=0`. Notes in `docs/experiments/st_hdbscan_spacetime.md`. |
| HMM v1 (constant jitter, single 5 m/s move speed) | stay/move HMM | Merged stays across dense-sampled drives and slow walks (found in code review). Fixed in place: dt-scaled jitter plus a walk/drive mixture. Two new contract tests cover it for every detector. |
| h3_occupancy with a 1x coarse threshold | ignore fixes coarser than one cell edge | Every CDR fix was "coarse", so CDR F1 fell 0.42 → 0.23. Changed to 3x, consistent with the other uncertainty-aware methods. |
| HDBSCAN "leaf" selection and `cluster_selection_epsilon` | finer clusters / merge scale | leaf over-fragmented stays; epsilon crashes in scikit-learn 1.8 under numpy 2.5. Replaced by EOM selection plus an explicit centroid merge. |
| Pinned old-Shapely environment for scikit-mobility | run skmob unmodified | Shapely 1.x has no Python 3.12 wheels. Replaced by a one-line shim. |

These fixes were made directly on main, with the reasoning recorded in docstrings and
commit messages, so only the ST-HDBSCAN experiment has its own branch.

## 8. Threats to validity and known issues

* **Synthetic realism.** Fixed UTC offset (no DST). Drive network also used for
  urban walking. iid per-fix error. Per-event, not per-device, calibration noise
  (nominal 95% coverage ≈ 0.92).
* **CDR error model.** CDR error is a single sector covariance per area, so shapes 2
  and 3 add only the azimuth for CDR; this flattens the CDR uncertainty results.
* **Shared timing.** Urban and suburban datasets share event timing per profile
  (same seed).
* **Held-out scope.** The held-out split uses new agents and seeds but the same two
  cities.
* **Public data.** PoL's clean variant is too easy. STAGA is one person, with
  anonymization jumps. No public per-device CDR or raw adtech data exists.
* **Real-device accuracy semantics.** Android documents `getAccuracy()` as a 68%
  radius, not CEP50. Confirm what the cleaning pipeline emits before trusting
  shape-2 conversions.
* **Code-review findings documented, not fixed:**
  * places DBSCAN runs per id (the original clustered all ids together);
  * place centers ignore the fused estimate;
  * `depart_hour_med` is a linear median of clock hours;
  * f1_observable counts predictions matched to unobservable truth as false positives;
  * episode-level colocation scoring has no spatial check.

  Details in the per-algorithm docs.

## 9. Run incidents (for the record)

The scaling benchmark brought down the WSL2 VM twice:

1. Infostop on 16.5M events grew past the 31 GB the VM has.
2. The downstream evaluation's colocation join materialized all dwell pairs per H3
   cell on dense urban GPS, more than 12 GB.

A results-path variable in the benchmark was also overwritten by child output,
which produced the "file not found" save errors.

Fixes:
* All heavy runs now execute under a kernel-enforced cgroup cap
  (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 ...`). It was
  verified to kill only the capped job.
* Infostop at 16.5M is recorded as skipped for memory.
* Colocation now uses a time-sorted sweep per H3 neighborhood (worst case: under
  0.5 GB in seconds).
* The benchmark variable bug is fixed.

## 10. Document index

* `docs/datasets.md`, `docs/public_datasets.md`: data
* `docs/uncertainty.md`, `docs/metrics.md`: models and scoring
* `docs/algorithms/README.md` (+ one page per detector, `downstream.md`), `docs/libraries/*.md`: methods
* `docs/results/accuracy.md`, `sensitivity.md`, `uncertainty_lift.md`, `public_data.md`,
  `scaling.md`, `downstream.md`, `spark.md`, `scorecard.md`, `scorecard_table.md`: results
* `docs/experiments/`: experiment notes
