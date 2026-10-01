# Experiment: space-time ST-HDBSCAN (branch exp/st-hdbscan-spacetime)

```
Experiment: classic space-time ST-HDBSCAN (features x, y, t * radius_m / (max_gap_s / 2))
vs the spatial-only HDBSCAN + temporal run splitting used on main.

History: the space-time variant was first rejected on main because it failed the
contract tests (stays elongated along t merged with approach points). After main
added common.trim_runs (robust spatial trim) and allow_single_cluster=True, the
space-time variant passes all 5 contract tests on this branch.

Dev-set F1 (md=900, shape 1):
  dataset             spatial(main)  spacetime(this branch)
  dev_urban_gps       0.367          0.517
  dev_suburban_gps    0.382          0.580
  dev_urban_adtech    0.057          0.059   (radius 250, gap 6 h)
  dev_suburban_cdr    0.128          0.074   (radius 1000, gap 6 h)

Outcome: space-time is better on dense GPS, worse on CDR (sector-quantized positions
repeat exactly, so time is the only thing separating revisits). Adopted as the default
on main (merged); the spatial mode stays available via time_scale_s=0.
```
