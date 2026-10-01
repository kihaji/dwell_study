# Scaling benchmarks

Single process per run, sequential, warm (JIT compiled on a 500-event slice first). Timeout 900 s; an algorithm stops climbing a ladder after a failure.

## Ladder: gps

| dataset | n_events | n_ids |
|---|---|---|
| scale_gps_1 | 9849 | 1 |
| scale_gps_8 | 97283 | 8 |
| scale_gps_80 | 907807 | 80 |
| scale_gps_800 | 9367454 | 800 |

Runtime (s):

| algo | scale_gps_1 | scale_gps_8 | scale_gps_80 | scale_gps_800 |
|---|---|---|---|---|
| changepoint_pelt | 11.74 | 97.57 | timeout>900s |  |
| fastmob_stays | 0.02 | 0.03 | 0.1 | 0.74 |
| h3_cell_dwell | 0.02 | 0.07 | 0.51 | 4.98 |
| h3_occupancy | 0.02 | 0.07 | 0.54 | 5.45 |
| hmm_stay_move | 0.02 | 0.05 | 0.26 | 2.37 |
| infostop | 0.03 | 0.26 | 7.76 | 87.99 |
| kalman_dwell | 0.08 | 0.6 | 5.63 | 57.49 |
| li_centroid | 0.01 | 0.02 | 0.09 | 0.66 |
| movingpandas_stops | 1.21 | 10.21 | 99.37 | timeout>900s |
| skmob_stops | 0.06 | 0.45 | 3.95 | 40.57 |
| st_dbscan | 0.01 | 0.04 | 0.22 | 2.22 |
| st_hdbscan | 0.17 | 2.02 | 17.26 | 181.48 |
| trackintel_sliding | 0.21 | 1.48 | 13.49 | 141.47 |
| ua_mahalanobis | 0.02 | 0.03 | 0.13 | 1.03 |
| ua_segmentation | 0.02 | 0.07 | 0.45 | 4.66 |

Peak RSS (MB, includes the loaded frame):

| algo | scale_gps_1 | scale_gps_8 | scale_gps_80 | scale_gps_800 |
|---|---|---|---|---|
| changepoint_pelt | 409.0 | 409.0 |  |  |
| fastmob_stays | 399.0 | 399.0 | 588.0 | 2211.0 |
| h3_cell_dwell | 228.0 | 273.0 | 678.0 | 2893.0 |
| h3_occupancy | 409.0 | 409.0 | 597.0 | 2711.0 |
| hmm_stay_move | 404.0 | 404.0 | 665.0 | 3027.0 |
| infostop | 404.0 | 410.0 | 2272.0 | 17958.0 |
| kalman_dwell | 234.0 | 269.0 | 562.0 | 2730.0 |
| li_centroid | 220.0 | 258.0 | 537.0 | 2857.0 |
| movingpandas_stops | 400.0 | 400.0 | 684.0 |  |
| skmob_stops | 391.0 | 391.0 | 700.0 | 2769.0 |
| st_dbscan | 404.0 | 404.0 | 558.0 | 2450.0 |
| st_hdbscan | 404.0 | 404.0 | 621.0 | 2407.0 |
| trackintel_sliding | 397.0 | 472.0 | 1011.0 | 6627.0 |
| ua_mahalanobis | 404.0 | 404.0 | 701.0 | 2841.0 |
| ua_segmentation | 404.0 | 404.0 | 754.0 | 3844.0 |

## Ladder: adtech

| dataset | n_events | n_ids |
|---|---|---|
| scale_adtech_20 | 19719 | 20 |
| scale_adtech_200 | 197517 | 200 |
| scale_adtech_2000 | 1969661 | 2000 |
| scale_adtech_18000 | 16494494 | 18000 |

Runtime (s):

| algo | scale_adtech_20 | scale_adtech_200 | scale_adtech_2000 | scale_adtech_18000 |
|---|---|---|---|---|
| changepoint_pelt | 0.47 | 8.95 | 161.2 | 827.11 |
| fastmob_stays | 0.03 | 0.06 | 0.21 | 1.9 |
| h3_cell_dwell | 0.03 | 0.13 | 1.11 | 9.54 |
| h3_occupancy | 0.02 | 0.12 | 1.06 | 9.27 |
| hmm_stay_move | 0.03 | 0.23 | 0.66 | 5.71 |
| infostop | 0.2 | 2.08 | 107.4 | skipped: memory (8.6 GB at 2M events, global Infomap) |
| kalman_dwell | 0.63 | 5.52 | 54.24 | 476.88 |
| li_centroid | 0.01 | 0.04 | 0.2 | 1.82 |
| movingpandas_stops | 2.69 | 26.87 | 267.29 | timeout>900s |
| skmob_stops | 1.57 | 16.53 | 161.53 | timeout>900s |
| st_dbscan | 0.02 | 0.07 | 0.54 | 4.01 |
| st_hdbscan | 0.19 | 2.11 | 22.62 | 171.09 |
| trackintel_sliding | 3.25 | 31.4 | 302.36 | timeout>900s |
| ua_mahalanobis | 0.03 | 0.06 | 0.25 | 2.39 |
| ua_segmentation | 0.04 | 0.14 | 1.08 | 9.25 |

Peak RSS (MB, includes the loaded frame):

| algo | scale_adtech_20 | scale_adtech_200 | scale_adtech_2000 | scale_adtech_18000 |
|---|---|---|---|---|
| changepoint_pelt | 416.0 | 480.0 | 9470.0 | 10091.0 |
| fastmob_stays | 410.0 | 410.0 | 1117.0 | 3555.0 |
| h3_cell_dwell | 409.0 | 409.0 | 937.0 | 4735.0 |
| h3_occupancy | 416.0 | 416.0 | 865.0 | 4286.0 |
| hmm_stay_move | 416.0 | 416.0 | 1010.0 | 5094.0 |
| infostop | 410.0 | 704.0 | 8581.0 |  |
| kalman_dwell | 409.0 | 409.0 | 1061.0 | 6933.0 |
| li_centroid | 409.0 | 409.0 | 868.0 | 3332.0 |
| movingpandas_stops | 410.0 | 464.0 | 1936.0 |  |
| skmob_stops | 410.0 | 490.0 | 1856.0 |  |
| st_dbscan | 411.0 | 412.0 | 946.0 | 3567.0 |
| st_hdbscan | 414.0 | 414.0 | 888.0 | 3862.0 |
| trackintel_sliding | 409.0 | 571.0 | 2128.0 |  |
| ua_mahalanobis | 415.0 | 415.0 | 1092.0 | 4777.0 |
| ua_segmentation | 415.0 | 415.0 | 1138.0 | 6546.0 |

## Ladder: single

| dataset | n_events | n_ids |
|---|---|---|
| scale_gps_single_1m | 957479 | 1 |

Runtime (s):

| algo | scale_gps_single_1m |
|---|---|
| changepoint_pelt | timeout>900s |
| fastmob_stays | 0.17 |
| h3_cell_dwell | 0.5 |
| h3_occupancy | 0.5 |
| hmm_stay_move | 0.23 |
| infostop | 0.18 |
| kalman_dwell | 2.15 |
| li_centroid | 0.09 |
| movingpandas_stops | 69.75 |
| skmob_stops | 2.58 |
| st_dbscan | 3.79 |
| st_hdbscan | timeout>900s |
| trackintel_sliding | 3.36 |
| ua_mahalanobis | 0.15 |
| ua_segmentation | 0.86 |

Peak RSS (MB, includes the loaded frame):

| algo | scale_gps_single_1m |
|---|---|
| fastmob_stays | 569.0 |
| h3_cell_dwell | 584.0 |
| h3_occupancy | 631.0 |
| hmm_stay_move | 578.0 |
| infostop | 591.0 |
| kalman_dwell | 644.0 |
| li_centroid | 523.0 |
| movingpandas_stops | 645.0 |
| skmob_stops | 775.0 |
| st_dbscan | 509.0 |
| trackintel_sliding | 1047.0 |
| ua_mahalanobis | 745.0 |
| ua_segmentation | 776.0 |

## Throughput at the largest completed size

| algo | single | gps | adtech |
|---|---|---|---|
| changepoint_pelt |  | 997.0 | 19942.0 |
| fastmob_stays | 5504105.0 | 12701849.0 | 8670185.0 |
| h3_cell_dwell | 1908568.0 | 1882874.0 | 1729524.0 |
| h3_occupancy | 1916734.0 | 1717685.0 | 1779472.0 |
| hmm_stay_move | 4156964.0 | 3944889.0 | 2890704.0 |
| infostop | 5359958.0 | 106462.0 | 18340.0 |
| kalman_dwell | 445463.0 | 162927.0 | 34588.0 |
| li_centroid | 10545234.0 | 14100126.0 | 9063802.0 |
| movingpandas_stops | 13727.0 | 9136.0 | 7369.0 |
| skmob_stops | 371077.0 | 230880.0 | 12194.0 |
| st_dbscan | 252754.0 | 4223529.0 | 4111727.0 |
| st_hdbscan |  | 51617.0 | 96407.0 |
| trackintel_sliding | 284916.0 | 66213.0 | 6514.0 |
| ua_mahalanobis | 6489619.0 | 9131297.0 | 6898067.0 |
| ua_segmentation | 1115284.0 | 2010803.0 | 1782375.0 |
