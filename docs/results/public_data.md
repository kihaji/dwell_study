# Results on labeled public data

Two ingested datasets have stop labels (see `docs/public_datasets.md`):
* **STAGA:** one person, 126 days, dense GPS, diary-labeled stops.
* **PoL benchmark:** a 50-agent subset, `pol_50`, of an external simulator's
  clean variant.

Both use a nominal, constant 8 m error, so shapes 2 and 3 carry no information here.
Held-out F1 at the dev-tuned GPS parameters is in `accuracy.md`.

## PoL (`pol_50`): easy, as expected

Positions in the clean PoL variant do not move at all during a stay. Every detector
reaches F1 0.90–1.00, apart from the two that timed out: changepoint_pelt
(ruptures) and movingpandas, which each needed more than 30 min on 2.2M events.
PoL confirms that the implementations are correct, but it cannot rank them. The
published noisy variants, 245–400 MB each, were not ingested (see
public_datasets.md).

## STAGA: the ~0.5 F1 plateau comes from the data

All detectors score F1 ≈ 0.5 at the dev-tuned GPS parameters, and all show about
23% fragmentation. Inspecting the labeled stops explains both:

* 131 of 683 labeled stops contain a recording gap of more than 2 h inside the
  stop. The median largest gap is 6 min; the 80th percentile is 1.9 h and the
  95th is 9.8 h (the phone is silent overnight or indoors). Any `max_gap_s` of
  2 h or less, which is what the synthetic GPS tuning selected, splits these stops.
* 51 stops contain a jump of more than 200 m between consecutive fixes, caused by
  the dataset's non-uniform coordinate anonymization.

Re-running with a longer gap tolerance (min duration 15 min, radius 100 m):

| algorithm | max_gap 2 h | 12 h | 24 h |
|---|---|---|---|
| li_centroid | 0.494 | 0.710 | 0.755 |
| fastmob_stays | 0.510 | 0.719 | 0.765 |
| ua_segmentation | 0.517 | 0.741 | 0.792 |
| ua_mahalanobis | 0.522 | 0.771 | **0.825** |
| st_dbscan | 0.305 | 0.332 | 0.290 |

Takeaways:

1. On real GPS, the gap tolerance has to follow the device's reporting behavior,
   not the synthetic profile. Real phones go quiet for hours while stationary, and
   synthetic GPS throttles but rarely goes silent.
2. With a realistic gap, the ranking matches the synthetic results: uncertainty-
   aware sequential methods lead and first-point-anchored libraries trail.
3. ST-DBSCAN does not recover as the gap grows (0.29–0.33): its time-window
   neighborhood chains stays through travel fixes (see its known limitations).
