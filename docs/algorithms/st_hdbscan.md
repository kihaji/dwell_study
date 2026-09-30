# st_hdbscan: HDBSCAN + temporal run splitting per id

Module: `src/dwells/algorithms/st_hdbscan.py`. Family `density`,
`USES_UNCERTAINTY = False`. It uses `sklearn.cluster.HDBSCAN`.

## Summary

The detector runs HDBSCAN on each id's positions (space only, by default) with
excess-of-mass cluster selection. It cuts clusters into time-contiguous runs, and then,
because HDBSCAN's density is relative, trims every run to the events within `radius_m`
of its median centre. A run that loses most of its events is rejected.

## Algorithm (`detect`, `st_hdbscan.py:30-54`)

1. Per id, skip ids with fewer than `min_cluster_size` events. Otherwise
   `X = [x, y]`. If `time_scale_s` is set, append a third feature
   `(t - t_0) · radius_m / time_scale_s` (`:43-44`), so that `time_scale_s` seconds
   weigh like `radius_m` metres (classic ST-HDBSCAN).
2. `HDBSCAN(min_cluster_size=k, min_samples=k, cluster_selection_method="eom",
   allow_single_cluster=True)` (`:45-47`). Labels are offset to be unique across ids.
3. `contiguous_runs(..., max_gap_s, max_noise_run=min_cluster_size)` (`:50`), with the
   same run semantics as st_dbscan.
4. `trim_runs(x, y, runs, radius_m)` (`:53`, `common.py:137-160`). Events farther than
   `radius_m` from the run's coordinate-wise median centre are unlabelled. If fewer than
   `min_keep = 50%` of the events remain, the run is dropped.
5. `dwells_from_labels(events, runs, min_duration_s)`.

Design notes recorded in the docstring (`:1-14`): `leaf` selection over-fragmented stays;
`cluster_selection_epsilon` is not used because scikit-learn 1.8's `epsilon_search`
raises `TypeError` under numpy 2.5; and the time feature, when enabled, stretched stays
along the time axis and merged them with approach/departure points, so it is off by
default.

## Parameters

| Harness param | Meaning here |
|---|---|
| `radius_m` | **Only** the post-hoc trim radius (absolute stay extent). HDBSCAN itself has no scale |
| `max_gap_s` | Run-splitting gap |
| `min_duration_s` | Minimum run span |

Algorithm-specific: `min_cluster_size = 3` (also used as `min_samples` and as the
maximum noise run), `selection = "eom"`, `time_scale_s = None`.

## Use of positional uncertainty

None.

## Output semantics

- `start_ts` / `end_ts`: first / last **retained** event of the trimmed run. Trimmed
  approach/departure events do not count.
- Centroid: mean lat/lon of the retained events.
- `sigma_*`: null.

## Complexity and Spark-readiness

- HDBSCAN costs roughly O(n log n) to O(n²) per id, depending on data and backend, with
  O(n) or more memory. It is run on **all** of an id's positions at once, which is the
  worst case for the 1M-event single id.
- Per-id partitionable (`applyInPandas`). There is no global step.

## Known limitations / failure modes

- **Relative density**: without the trim, evenly sampled travel forms clusters and
  approach points attach to stays. The trim restores an absolute scale but can reject
  genuine large places (more than 50% of events beyond `radius_m`).
- EOM selection is data-dependent: two nearby places can be merged or split depending
  on the id's other data (the global density hierarchy of that id). Results for a given
  day can change when more days are added.
- Clusters are spatial only, so a place revisited many times forms one cluster, which is
  good. But two places that alternate quickly within `max_noise_run` cannot be separated
  by time.
- No `KNOWN_LIMITATIONS` xfails.

## References

- Campello, R. J. G. B., Moulavi, D., Sander, J. (2013). Density-based clustering based on hierarchical density estimates. PAKDD 2013, LNCS 7819. doi:10.1007/978-3-642-37456-2_14
- McInnes, L., Healy, J., Astels, S. (2017). hdbscan: hierarchical density based clustering. JOSS 2(11). doi:10.21105/joss.00205
- scikit-learn `HDBSCAN`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
