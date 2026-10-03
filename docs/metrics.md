# Evaluation metrics

How detected dwells are scored against ground truth (`src/dwells/metrics.py`). These
metrics drive the tuning harness `src/dwells/harness/tune.py` (dev tuning, held-out
evaluation, robustness; see [tuning.md](tuning.md)). Colocation scoring and the
multi-algorithm protocol live on `main`. Code line numbers are approximate.

---

## 1. Dwell scoring (`src/dwells/metrics.py`)

### 1.1 Interval convention

Every dwell is treated as a half-open time interval `[start_ts, end_ts)`, converted to float seconds (`_arr`, `metrics.py:21-26`). Two intervals overlap only if `ts < pe` and `te > ps` (`metrics.py:50`). Intervals that merely touch (one ends exactly when the other starts) do not overlap. Zero-length predictions (`start == end`) can never overlap anything, because the `ts < pe` and `te > ps` tests are strict.

### 1.2 Candidate generation: `overlap_pairs` (`metrics.py:29-61`)

This step finds all (prediction, truth) pairs of the same `id` whose intervals overlap, without an O(n_pred × n_truth) scan:

1. Per id, the truth indices are sorted by start time. The code also builds the **running maximum of the end times**, `tend_max = maximum.accumulate(te)` (`:35-38`). Truths may overlap each other (they do not in the generator), so the plain end times need not be sorted, but their running maximum always is.
2. For each prediction `[ps, pe)` (`:44-52`):
   - `hi = searchsorted(tstart, pe, "left")`: truths at index `>= hi` start at or after `pe`, so they cannot overlap.
   - `lo = searchsorted(tend_max, ps, "right")`: every truth before `lo` has its running-max end `<= ps`, so all of them end at or before `ps`.
   - The exact test `ts < pe & te > ps` is then applied to the slice `[lo, hi)`. This is needed because a truth inside the slice may still end early when truths overlap each other.
3. For each pair the function returns `inter_s` (intersection length) and `iou = inter / union`. It also returns `p_frac = inter / |P|` (the share of the prediction that lies inside the truth) and `t_frac = inter / |T|` (the share of the truth that is covered) (`:54-61`).

The cost is O((n_p + n_t) log n_t + number of pairs) per id.

### 1.3 Matching: temporal IoU only

```
IoU(P, T) = |P ∩ T| / |P ∪ T|      (lengths in seconds)
```

`match_dwells` (`metrics.py:64-74`) keeps the candidate pairs with `iou >= iou_min` (default 0.5). It sorts them by descending IoU and accepts a pair greedily when neither the prediction nor the truth has been used yet. The result is a one-to-one matching.

**Why location is not a match criterion** (module docstring, `metrics.py:3-8`). The centroid error of a correct detection depends on the source profile. For GPS it is metres. For CDR it is often 0.5–2 km, because the reported position is a sector centroid. For adtech it ranges up to km, from IP and cell fixes. A spatial gate tight enough for GPS would make every CDR dwell a false negative. A gate loose enough for CDR would be meaningless for GPS. Temporal identity ("was the device staying somewhere during this time") is what the detector is responsible for. Location quality is reported separately as the centroid error of matched pairs (§1.6). A consequence: a detection at the wrong place but at the right time counts as a true positive. With one agent per id and no concurrent true stays, that can only happen if the detector's centroid is wrong, so read F1 together with `centroid_err_m_*`.

**When greedy equals optimal.** Claim: if predictions are pairwise disjoint within an id, truths are pairwise disjoint within an id, and the threshold is *strictly* greater than 0.5, then each prediction has at most one candidate truth and vice versa. The candidate graph is then already a matching, so greedy, Hungarian (Kuhn 1955) and any other maximum matching coincide.

Proof. Suppose `IoU(P,T) > 1/2`. Since `|P ∪ T| >= max(|P|, |T|)`,
```
|P ∩ T| > |P ∪ T| / 2 >= |P| / 2    and    |P ∩ T| > |T| / 2.
```
If two disjoint truths `T1`, `T2` both had IoU > 1/2 with `P`, then `|P ∩ T1| + |P ∩ T2| > |P|`. That is impossible because `P ∩ T1` and `P ∩ T2` are disjoint subsets of `P`. The symmetric argument, using `|T|/2`, rules out two disjoint predictions sharing one truth. ∎

Caveats:

- The code uses `>=` (`metrics.py:67`), so IoU exactly 0.5 is admitted. Counter-example: `P = [0,2)`, `T1 = [0,1)`, `T2 = [1,2)` gives `IoU(P,T1) = IoU(P,T2) = 0.5`. Uniqueness then fails and the tie is broken by sort order. This is measure-zero with real-valued timestamps, but it is possible with synthetic round-number data.
- The argument needs disjoint predictions. Every detector in `src/dwells/algorithms/` emits non-overlapping dwells per id, but this is not enforced by `schema.validate_dwells`.
- The **lenient** matching at IoU >= 0.2 (`metrics.py:95`) is *not* guaranteed to be maximum-cardinality. A prediction can have several candidates, and greedy-by-IoU can block a pairing that a maximum matching would find. The resulting `recall_lenient` is a lower bound on the maximum-matching recall at 0.2.

### 1.4 Precision / recall / F1 (`metrics.py:77-80, 83-103`)

`dwell_metrics(pred, truth, min_duration_s, iou_min=0.5, min_obs_events=2)`:

- The truth is filtered to `duration_s >= min_duration_s` (`:85`). **Predictions are not filtered.** The detector is responsible for honouring `min_duration_s`. A detector that emits dwells shorter than `min_duration_s` (e.g. `kalman_dwell`, whose grid-based end time can be one step below the threshold) is charged a false positive for each one, unless it happens to match a slightly longer truth.
- `precision = matched / n_pred`, `recall = matched / n_truth`, `F1 = 2PR / (P + R)`. Each is 0 when its denominator is 0.

### 1.5 Observable recall and lenient scores

A truth dwell is **observable** when the sampled event stream contains at least `min_obs_events = 2` events inside it (`truth.n_events`, computed in `generator/build.py` from `truth_dwell_id`). A stay with 0–1 events cannot be detected by any method, so it is excluded from the denominator:

- `recall_observable = (# matched truths that are observable) / n_truth_observable` (`:91-94`).
- `f1_observable = F1(matched_observable, n_pred, n_obs)` (`:101`). Its precision numerator is also *matched observable truths*. A prediction matched to an unobservable truth (possible only if the detector bridges a truth with a single event, using neighbouring events) therefore counts as a false positive in `f1_observable`, even though it is a true positive in `f1`. In practice such matches are rare. Strictly, this makes `f1_observable` slightly pessimistic.
- `recall_lenient`, `f1_lenient`: the same scores at IoU >= 0.2 (see §1.3). They separate "found the stay but with poor boundaries" from "missed the stay".

### 1.6 Boundary and location errors (matched pairs at IoU >= 0.5; `metrics.py:104-119`)

| key | definition |
|---|---|
| `start_err_s_med` | median of \|pred.start − truth.start\| |
| `end_err_s_med` | median of \|pred.end − truth.end\| |
| `duration_bias_s_med` | median of (pred duration − truth duration). Negative means detections are too short. This is expected for "first/last event inside" semantics under sparse sampling |
| `centroid_err_m_med`, `centroid_err_m_p90` | haversine distance from the prediction centroid to the truth `lat/lon` |

The truth `lat/lon` is the **true place position** (`generator/build.py:75-82` joins `area.places`). The true in-place position of the agent varies within `radius_m` of the place, so even a perfect centroid has a non-zero error of up to the place radius. All error values are `None` when nothing matched.

Start/end semantics differ by detector: first/last event inside the stay, versus first fix outside it (trackintel, skmob, fastmob), versus grid times (kalman). This shows up directly in these medians. See each `docs/algorithms/<name>.md`.

### 1.7 Fragmentation, merging, time-based scores (`metrics.py:120-129`)

These use **all overlapping pairs**, not only matched ones:

- `fragmentation` = fraction of truths that have at least 2 predictions, each with at least 50% of its *own* duration inside that truth (`p_frac >= 0.5`). This means one stay was split into several.
- `merging` = fraction of predictions that cover at least 50% of each of at least 2 truths (`t_frac >= 0.5`). This means several stays were merged into one.
- `time_precision = Σ inter_s / Σ pred durations` and `time_recall = Σ inter_s / Σ truth durations`. These are duration-weighted and do not need a matching. If predictions (or truths) overlapped each other, the numerator would double-count. With disjoint sets it does not.

---

## 2. Caveats and threats to validity

1. **Same geography in dev and eval.** Parameters tuned on dev may overfit the place density and sizes of the two OSM areas. The public labeled set `staga` is the only out-of-distribution check on this branch.
2. **Simulator-defined truth.** A "dwell" is whatever the simulator calls a stay: arrive/depart at a catalogued place. Short stops that the simulator does not model (traffic lights, drive-throughs) appear as false positives if detected.
3. **Temporal-only matching** rewards a detection at the right time even at the wrong place. Always read F1 alongside `centroid_err_m_*`.
4. **Duration estimates.** Truths are filtered by `min_duration_s`; the detector applies the same threshold to its *estimated* span (halfway into the arrival/departure windows). On sparse data the window midpoints can push near-threshold dwells across `min_duration_s` in either direction.
5. **Tuning on F1 only.** Presets were chosen on dev F1 at shape 3 and a 15-min dwell definition; other shapes and definitions use the same parameters.
6. **Greedy matching** is exactly optimal only under the conditions in §1.3. The lenient (0.2) numbers are lower bounds.
7. **Observable recall** uses `n_events >= 2`. That is the minimum for a *span*, but many detectors also require `min_events` of 2–3 and a duration of at least `min_duration_s` between the first and last event. A stay with 2 events 1 minute apart is "observable" but undetectable at `min_duration_s = 900`.
8. **Runtime** reported by the harness includes JIT warm-up on first use and is measured under concurrent workers; it is not a benchmark.

## References

- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly* 2(1–2):83–97. doi:10.1002/nav.3800020109
