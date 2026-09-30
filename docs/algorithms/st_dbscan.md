# st_dbscan: spatio-temporal DBSCAN per id

Module: `src/dwells/algorithms/st_dbscan.py`. Family `density`, `USES_UNCERTAINTY = False`.
This is our own numba implementation inspired by Birant & Kut (2007). No library is used.

## Summary

DBSCAN in which two events are neighbours only if they are within `radius_m` in space
**and** within `eps_t_s` in time (default `max_gap_s`). Clusters grow through core events
as in DBSCAN. Each cluster is then cut into time-contiguous runs, so that separate visits
to the same spot become separate dwells, and runs shorter than `min_duration_s` are
dropped.

## Algorithm

`_st_dbscan` (`st_dbscan.py:25-67`), per id, on events sorted by time:

1. **Neighbour counts** (`:34-41`): a symmetric forward scan. For each `i`, the scan
   visits `j > i` while `t_j - t_i <= eps_t`. If `|z_j - z_i|² <= eps_s²`, both counts
   are incremented. Counts start at 1 (self), so `core_i ⇔ cnt_i >= min_samples`.
2. **Expansion** (`:43-66`): for each unlabelled core `i`, start cluster `nxt`. A stack
   pops `p`. If `p` is core, the code scans its time window
   `[t_p - eps_t, t_p + eps_t]`, and every unlabelled `q` within `eps_s` gets label
   `nxt` and is pushed. Non-core points are labelled but not expanded (border points).
   A border point takes the first cluster that reaches it, and unreached points stay
   `-1` (noise).
3. **Runs** (`detect`, `:74`): `common.contiguous_runs(starts, t, lab, max_gap_s,
   max_noise_run=min_samples)` (`common.py:108-133`). A run of cluster `c` continues
   across at most `min_samples` consecutive noise events and while
   `t_i - t_last_c <= max_gap_s`. Any event of another cluster ends it.
4. `dwells_from_labels(events, runs, min_duration_s)`.

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | `eps_s`, the spatial neighbourhood radius |
| `max_gap_s` | Default `eps_t` (temporal neighbourhood half-width) and the run-splitting gap |
| `min_duration_s` | Minimum run span |

Algorithm-specific: `eps_t_s = None` (which means `max_gap_s`), `min_samples = 3`.

## Use of positional uncertainty

None.

## Output semantics

- `start_ts` / `end_ts`: first / last event of the run.
- Centroid: mean lat/lon.
- `sigma_*`: null.

## Comparison with Birant & Kut (2007)

The original ST-DBSCAN uses `Eps1` (spatial), `Eps2` (a non-spatial attribute), `MinPts`,
and `Δε`, a cluster-average check that prevents drift between adjacent clusters. Its
temporal neighbourhood is defined by the application, for example consecutive days. This
implementation:

- uses time itself as the second distance (`|Δt| <= eps_t`);
- has no non-spatial attribute and no `Δε` check;
- adds the contiguous-run split, which is not in the paper, to separate revisits.

## Complexity and Spark-readiness

- O(n·w) per id, where `w` is the number of events in a `±eps_t` window. Because
  `eps_t = max_gap_s`, this can be large: dense GPS (5 s sampling) with
  `max_gap_s = 7200` gives `w ≈ 1440`, so about 1.4·10⁹ distance checks for a 1M-event
  id. For CDR grid values (up to 43 200 s) `w` also grows. Memory is O(n).
- Per-id partitionable (`applyInPandas`). There is no global step and no SQL-window
  formulation, because cluster expansion is a graph traversal.

## Known limitations / failure modes

- **Slow pass-through at dense sampling**: a slowly moving id sampled densely has
  `>= min_samples` neighbours within `radius_m` at every point. Chains of core points
  then turn the trajectory into one long cluster. The contiguous-run split does not
  help, because the run is contiguous. Only the duration gate limits it, and there is no
  extent check (compare st_hdbscan's `trim_runs`).
- Coarse fixes are simply noise or border points. With km-scale error, `radius_m` must
  be km-scale, which merges places.
- `max_noise_run = min_samples` couples two unrelated knobs.
- Border-point assignment depends on scan order (standard DBSCAN behaviour).
- No `KNOWN_LIMITATIONS` xfails.

## References

- Birant, D., Kut, A. (2007). ST-DBSCAN: an algorithm for clustering spatial–temporal data. Data Knowl. Eng. 60(1). doi:10.1016/j.datak.2006.01.013
- Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD-96. https://cdn.aaai.org/KDD/1996/KDD96-037.pdf
