# Downstream analytics (synthetic eval datasets, min duration 15 min)

## Place grouping: ARI vs true places (mean over datasets)

| algo | gmm | h3 | hdbscan | dbscan | mahalanobis |
|---|---|---|---|---|---|
| changepoint_pelt | 0.86 | 0.65 | 0.87 | 0.87 | 0.8 |
| fastmob_stays | 0.76 | 0.45 | 0.79 | 0.78 | 0.71 |
| h3_cell_dwell | 0.8 | 0.62 | 0.85 | 0.87 | 0.79 |
| h3_occupancy | 0.91 | 0.67 | 0.93 | 0.93 | 0.83 |
| hmm_stay_move | 0.8 | 0.58 | 0.79 | 0.78 | 0.8 |
| infostop | 0.86 | 0.65 | 0.89 | 0.9 | 0.81 |
| kalman_dwell | 0.69 | 0.34 | 0.69 | 0.66 | 0.72 |
| li_centroid | 0.87 | 0.65 | 0.9 | 0.89 | 0.81 |
| movingpandas_stops | 0.85 | 0.61 | 0.87 | 0.87 | 0.79 |
| skmob_stops | 0.76 | 0.45 | 0.79 | 0.78 | 0.7 |
| st_dbscan | 0.9 | 0.67 | 0.92 | 0.92 | 0.82 |
| st_hdbscan | 0.77 | 0.59 | 0.81 | 0.82 | 0.74 |
| trackintel_sliding | 0.76 | 0.43 | 0.75 | 0.74 | 0.69 |
| ua_mahalanobis | 0.9 | 0.66 | 0.92 | 0.91 | 0.91 |
| ua_segmentation | 0.92 | 0.68 | 0.94 | 0.94 | 0.91 |

## Place grouping by profile (dbscan)

| algo | cdr | adtech | gps |
|---|---|---|---|
| changepoint_pelt | 0.82 | 0.8 | 0.99 |
| fastmob_stays | 0.84 | 0.51 | 0.99 |
| h3_cell_dwell | 0.78 | 0.87 | 0.97 |
| h3_occupancy | 0.84 | 0.95 | 0.99 |
| hmm_stay_move | 0.77 | 0.6 | 0.97 |
| infostop | 0.87 | 0.82 | 1.0 |
| kalman_dwell | 0.45 | 0.53 | 0.99 |
| li_centroid | 0.85 | 0.83 | 0.99 |
| movingpandas_stops | 0.87 | 0.75 | 0.99 |
| skmob_stops | 0.84 | 0.52 | 0.99 |
| st_dbscan | 0.82 | 0.94 | 0.99 |
| st_hdbscan | 0.59 | 0.87 | 0.99 |
| trackintel_sliding | 0.71 | 0.52 | 0.99 |
| ua_mahalanobis | 0.8 | 0.95 | 0.99 |
| ua_segmentation | 0.85 | 0.97 | 0.99 |

## Home / work inference accuracy (dbscan places)

| algo | {"cdr","home"} | {"gps","home"} | {"gps","work"} | {"adtech","home"} | {"cdr","work"} | {"adtech","work"} |
|---|---|---|---|---|---|---|
| changepoint_pelt | 0.88 | 0.98 | 1.0 | 0.9 | 0.76 | 0.82 |
| fastmob_stays | 0.88 | 0.98 | 1.0 | 0.94 | 0.76 | 0.92 |
| h3_cell_dwell | 0.88 | 0.98 | 1.0 | 0.95 | 0.79 | 0.89 |
| h3_occupancy | 0.87 | 0.98 | 1.0 | 0.96 | 0.8 | 0.91 |
| hmm_stay_move | 0.92 | 0.96 | 0.99 | 0.89 | 0.75 | 0.84 |
| infostop | 0.87 | 0.98 | 1.0 | 0.96 | 0.8 | 0.89 |
| kalman_dwell | 0.68 | 0.98 | 1.0 | 0.7 | 0.38 | 0.69 |
| li_centroid | 0.88 | 0.98 | 1.0 | 0.92 | 0.82 | 0.81 |
| movingpandas_stops | 0.88 | 0.98 | 1.0 | 0.94 | 0.77 | 0.89 |
| skmob_stops | 0.85 | 0.98 | 1.0 | 0.94 | 0.76 | 0.91 |
| st_dbscan | 0.86 | 0.98 | 1.0 | 0.96 | 0.74 | 0.89 |
| st_hdbscan | 0.86 | 0.98 | 1.0 | 0.92 | 0.79 | 0.91 |
| trackintel_sliding | 0.8 | 0.98 | 1.0 | 0.92 | 0.69 | 0.9 |
| ua_mahalanobis | 0.93 | 0.98 | 1.0 | 0.96 | 0.79 | 0.91 |
| ua_segmentation | 0.97 | 0.98 | 1.0 | 0.96 | 0.86 | 0.91 |

## Next location: acc@3 by model (truth = model ceiling on true sequences)

| stage | markov1_time | markov1 | markov2 | most_frequent |
|---|---|---|---|---|
| nextloc_detected | 0.81 | 0.815 | 0.812 | 0.809 |
| nextloc_mapped | 0.805 | 0.805 | 0.803 | 0.798 |
| nextloc_truth | 0.765 | 0.766 | 0.764 | 0.743 |

### acc@3, markov1_time, per detector

| algo | nextloc_mapped | nextloc_detected | nextloc_truth |
|---|---|---|---|
| _truth |  |  | 0.765 |
| changepoint_pelt | 0.822 | 0.822 |  |
| fastmob_stays | 0.814 | 0.749 |  |
| h3_cell_dwell | 0.741 | 0.873 |  |
| h3_occupancy | 0.847 | 0.87 |  |
| hmm_stay_move | 0.857 | 0.827 |  |
| infostop | 0.812 | 0.84 |  |
| kalman_dwell | 0.727 | 0.532 |  |
| li_centroid | 0.809 | 0.857 |  |
| movingpandas_stops | 0.822 | 0.816 |  |
| skmob_stops | 0.805 | 0.741 |  |
| st_dbscan | 0.838 | 0.869 |  |
| st_hdbscan | 0.675 | 0.862 |  |
| trackintel_sliding | 0.814 | 0.761 |  |
| ua_mahalanobis | 0.839 | 0.864 |  |
| ua_segmentation | 0.846 | 0.87 |  |

## Colocation same-time: pair_f1

| algo | h3 | distance | mahalanobis |
|---|---|---|---|
| changepoint_pelt | 0.32 | 0.369 | 0.228 |
| fastmob_stays | 0.255 | 0.29 | 0.181 |
| h3_cell_dwell | 0.349 | 0.4 | 0.259 |
| h3_occupancy | 0.337 | 0.399 | 0.243 |
| hmm_stay_move | 0.304 | 0.365 | 0.262 |
| infostop | 0.329 | 0.391 | 0.233 |
| kalman_dwell | 0.25 | 0.286 | 0.179 |
| li_centroid | 0.333 | 0.392 | 0.241 |
| movingpandas_stops | 0.311 | 0.376 | 0.221 |
| skmob_stops | 0.258 | 0.293 | 0.182 |
| st_dbscan | 0.337 | 0.397 | 0.243 |
| st_hdbscan | 0.346 | 0.405 | 0.248 |
| trackintel_sliding | 0.251 | 0.283 | 0.18 |
| ua_mahalanobis | 0.333 | 0.4 | 0.29 |
| ua_segmentation | 0.341 | 0.406 | 0.302 |

## Colocation same-time: episode_f1

| algo | h3 | distance | mahalanobis |
|---|---|---|---|
| changepoint_pelt | 0.363 | 0.372 | 0.27 |
| fastmob_stays | 0.373 | 0.41 | 0.278 |
| h3_cell_dwell | 0.364 | 0.389 | 0.282 |
| h3_occupancy | 0.396 | 0.435 | 0.291 |
| hmm_stay_move | 0.364 | 0.392 | 0.32 |
| infostop | 0.386 | 0.416 | 0.285 |
| kalman_dwell | 0.322 | 0.332 | 0.247 |
| li_centroid | 0.381 | 0.411 | 0.282 |
| movingpandas_stops | 0.388 | 0.421 | 0.285 |
| skmob_stops | 0.364 | 0.399 | 0.273 |
| st_dbscan | 0.396 | 0.431 | 0.289 |
| st_hdbscan | 0.359 | 0.382 | 0.276 |
| trackintel_sliding | 0.368 | 0.406 | 0.278 |
| ua_mahalanobis | 0.394 | 0.435 | 0.347 |
| ua_segmentation | 0.392 | 0.43 | 0.347 |

## Colocation same-time: decoy_fp_rate

| algo | h3 | distance | mahalanobis |
|---|---|---|---|
| changepoint_pelt | 0.996 | 0.975 | 0.996 |
| fastmob_stays | 1.0 | 1.0 | 1.0 |
| h3_cell_dwell | 0.993 | 0.993 | 0.993 |
| h3_occupancy | 0.996 | 0.996 | 0.996 |
| hmm_stay_move | 0.988 | 0.988 | 0.988 |
| infostop | 0.996 | 0.996 | 0.996 |
| kalman_dwell | 0.976 | 0.949 | 0.993 |
| li_centroid | 0.996 | 0.993 | 0.996 |
| movingpandas_stops | 0.996 | 0.996 | 0.996 |
| skmob_stops | 0.971 | 0.966 | 0.971 |
| st_dbscan | 0.996 | 0.996 | 0.996 |
| st_hdbscan | 0.989 | 0.989 | 0.989 |
| trackintel_sliding | 1.0 | 1.0 | 1.0 |
| ua_mahalanobis | 0.996 | 0.996 | 0.996 |
| ua_segmentation | 0.996 | 0.996 | 0.996 |

## Colocation radius sweep: recall / precision / decoy false-positive rate (by profile)

| profile | algo | method | pair_precision | pair_recall | decoy_fp_rate |
|---|---|---|---|---|---|
| adtech | fastmob_stays | distance@150 | 0.113 | 0.753 | 1.0 |
| adtech | fastmob_stays | distance@250 | 0.095 | 0.782 | 1.0 |
| adtech | fastmob_stays | distance@30 | 0.133 | 0.728 | 0.956 |
| adtech | fastmob_stays | distance@60 | 0.13 | 0.74 | 0.989 |
| adtech | fastmob_stays | mahalanobis@150 | 0.084 | 0.801 | 1.0 |
| adtech | fastmob_stays | mahalanobis@250 | 0.069 | 0.857 | 1.0 |
| adtech | fastmob_stays | mahalanobis@30 | 0.127 | 0.74 | 0.989 |
| adtech | fastmob_stays | mahalanobis@60 | 0.114 | 0.753 | 1.0 |
| adtech | li_centroid | distance@150 | 0.437 | 0.572 | 0.944 |
| adtech | li_centroid | distance@250 | 0.363 | 0.589 | 0.978 |
| adtech | li_centroid | distance@30 | 0.619 | 0.422 | 0.725 |
| adtech | li_centroid | distance@60 | 0.527 | 0.484 | 0.848 |
| adtech | li_centroid | mahalanobis@150 | 0.292 | 0.598 | 0.989 |
| adtech | li_centroid | mahalanobis@250 | 0.208 | 0.622 | 0.989 |
| adtech | li_centroid | mahalanobis@30 | 0.515 | 0.513 | 0.871 |
| adtech | li_centroid | mahalanobis@60 | 0.441 | 0.572 | 0.944 |
| adtech | st_dbscan | distance@150 | 0.478 | 0.604 | 0.989 |
| adtech | st_dbscan | distance@250 | 0.381 | 0.612 | 0.989 |
| adtech | st_dbscan | distance@30 | 0.689 | 0.579 | 0.92 |
| adtech | st_dbscan | distance@60 | 0.629 | 0.6 | 0.989 |
| adtech | st_dbscan | mahalanobis@150 | 0.313 | 0.619 | 0.989 |
| adtech | st_dbscan | mahalanobis@250 | 0.214 | 0.642 | 0.989 |
| adtech | st_dbscan | mahalanobis@30 | 0.603 | 0.602 | 0.989 |
| adtech | st_dbscan | mahalanobis@60 | 0.487 | 0.604 | 0.989 |
| adtech | ua_mahalanobis | distance@150 | 0.496 | 0.61 | 0.989 |
| adtech | ua_mahalanobis | distance@250 | 0.386 | 0.618 | 0.989 |
| adtech | ua_mahalanobis | distance@30 | 0.713 | 0.592 | 0.956 |
| adtech | ua_mahalanobis | distance@60 | 0.65 | 0.606 | 0.989 |
| adtech | ua_mahalanobis | mahalanobis@150 | 0.33 | 0.621 | 0.989 |
| adtech | ua_mahalanobis | mahalanobis@250 | 0.255 | 0.631 | 0.989 |
| adtech | ua_mahalanobis | mahalanobis@30 | 0.446 | 0.616 | 0.989 |
| adtech | ua_mahalanobis | mahalanobis@60 | 0.42 | 0.616 | 0.989 |
| adtech | ua_segmentation | distance@150 | 0.533 | 0.608 | 0.989 |
| adtech | ua_segmentation | distance@250 | 0.415 | 0.614 | 0.989 |
| adtech | ua_segmentation | distance@30 | 0.73 | 0.595 | 0.931 |
| adtech | ua_segmentation | distance@60 | 0.672 | 0.604 | 0.989 |
| adtech | ua_segmentation | mahalanobis@150 | 0.395 | 0.614 | 0.989 |
| adtech | ua_segmentation | mahalanobis@250 | 0.295 | 0.621 | 0.989 |
| adtech | ua_segmentation | mahalanobis@30 | 0.621 | 0.605 | 0.989 |
| adtech | ua_segmentation | mahalanobis@60 | 0.554 | 0.605 | 0.989 |
| cdr | fastmob_stays | distance@1000 | 0.069 | 0.94 | 1.0 |
| cdr | fastmob_stays | distance@250 | 0.133 | 0.811 | 1.0 |
| cdr | fastmob_stays | distance@500 | 0.095 | 0.873 | 1.0 |
| cdr | fastmob_stays | mahalanobis@1000 | 0.055 | 0.997 | 1.0 |
| cdr | fastmob_stays | mahalanobis@250 | 0.082 | 0.892 | 1.0 |
| cdr | fastmob_stays | mahalanobis@500 | 0.066 | 0.962 | 1.0 |
| cdr | li_centroid | distance@1000 | 0.106 | 0.801 | 1.0 |
| cdr | li_centroid | distance@250 | 0.197 | 0.704 | 1.0 |
| cdr | li_centroid | distance@500 | 0.137 | 0.753 | 1.0 |
| cdr | li_centroid | mahalanobis@1000 | 0.065 | 0.92 | 1.0 |
| cdr | li_centroid | mahalanobis@250 | 0.123 | 0.767 | 1.0 |
| cdr | li_centroid | mahalanobis@500 | 0.098 | 0.813 | 1.0 |
| cdr | st_dbscan | distance@1000 | 0.098 | 0.822 | 1.0 |
| cdr | st_dbscan | distance@250 | 0.185 | 0.697 | 1.0 |
| cdr | st_dbscan | distance@500 | 0.127 | 0.763 | 1.0 |
| cdr | st_dbscan | mahalanobis@1000 | 0.061 | 0.937 | 1.0 |
| cdr | st_dbscan | mahalanobis@250 | 0.114 | 0.78 | 1.0 |
| cdr | st_dbscan | mahalanobis@500 | 0.091 | 0.833 | 1.0 |
| cdr | ua_mahalanobis | distance@1000 | 0.091 | 0.839 | 1.0 |
| cdr | ua_mahalanobis | distance@250 | 0.215 | 0.719 | 0.989 |
| cdr | ua_mahalanobis | distance@500 | 0.139 | 0.789 | 1.0 |
| cdr | ua_mahalanobis | mahalanobis@1000 | 0.066 | 0.947 | 1.0 |
| cdr | ua_mahalanobis | mahalanobis@250 | 0.121 | 0.81 | 1.0 |
| cdr | ua_mahalanobis | mahalanobis@500 | 0.088 | 0.851 | 1.0 |
| cdr | ua_segmentation | distance@1000 | 0.094 | 0.819 | 1.0 |
| cdr | ua_segmentation | distance@250 | 0.221 | 0.73 | 0.989 |
| cdr | ua_segmentation | distance@500 | 0.142 | 0.788 | 1.0 |
| cdr | ua_segmentation | mahalanobis@1000 | 0.068 | 0.923 | 1.0 |
| cdr | ua_segmentation | mahalanobis@250 | 0.125 | 0.794 | 1.0 |
| cdr | ua_segmentation | mahalanobis@500 | 0.091 | 0.823 | 1.0 |
| gps | fastmob_stays | distance@15 | 0.839 | 0.936 | 0.829 |
| gps | fastmob_stays | distance@150 | 0.414 | 0.981 | 1.0 |
| gps | fastmob_stays | distance@30 | 0.78 | 0.963 | 1.0 |
| gps | fastmob_stays | distance@60 | 0.671 | 0.974 | 1.0 |
| gps | fastmob_stays | mahalanobis@15 | 0.756 | 0.972 | 1.0 |
| gps | fastmob_stays | mahalanobis@150 | 0.193 | 0.983 | 1.0 |
| gps | fastmob_stays | mahalanobis@30 | 0.626 | 0.979 | 1.0 |
| gps | fastmob_stays | mahalanobis@60 | 0.423 | 0.981 | 1.0 |
| gps | li_centroid | distance@15 | 0.809 | 0.842 | 0.851 |
| gps | li_centroid | distance@150 | 0.406 | 0.984 | 1.0 |
| gps | li_centroid | distance@30 | 0.776 | 0.958 | 0.909 |
| gps | li_centroid | distance@60 | 0.671 | 0.978 | 1.0 |
| gps | li_centroid | mahalanobis@15 | 0.744 | 0.965 | 1.0 |
| gps | li_centroid | mahalanobis@150 | 0.193 | 0.988 | 1.0 |
| gps | li_centroid | mahalanobis@30 | 0.614 | 0.982 | 1.0 |
| gps | li_centroid | mahalanobis@60 | 0.414 | 0.984 | 1.0 |
| gps | st_dbscan | distance@15 | 0.866 | 0.934 | 0.762 |
| gps | st_dbscan | distance@150 | 0.416 | 0.966 | 1.0 |
| gps | st_dbscan | distance@30 | 0.795 | 0.953 | 0.884 |
| gps | st_dbscan | distance@60 | 0.693 | 0.959 | 1.0 |
| gps | st_dbscan | mahalanobis@15 | 0.77 | 0.955 | 0.94 |
| gps | st_dbscan | mahalanobis@150 | 0.197 | 0.972 | 1.0 |
| gps | st_dbscan | mahalanobis@30 | 0.646 | 0.962 | 1.0 |
| gps | st_dbscan | mahalanobis@60 | 0.421 | 0.966 | 1.0 |
| gps | ua_mahalanobis | distance@15 | 0.867 | 0.944 | 0.751 |
| gps | ua_mahalanobis | distance@150 | 0.42 | 0.981 | 1.0 |
| gps | ua_mahalanobis | distance@30 | 0.804 | 0.966 | 0.884 |
| gps | ua_mahalanobis | distance@60 | 0.698 | 0.974 | 1.0 |
| gps | ua_mahalanobis | mahalanobis@15 | 0.798 | 0.97 | 0.884 |
| gps | ua_mahalanobis | mahalanobis@150 | 0.269 | 0.983 | 1.0 |
| gps | ua_mahalanobis | mahalanobis@30 | 0.711 | 0.976 | 1.0 |
| gps | ua_mahalanobis | mahalanobis@60 | 0.514 | 0.977 | 1.0 |
| gps | ua_segmentation | distance@15 | 0.835 | 0.909 | 0.84 |
| gps | ua_segmentation | distance@150 | 0.41 | 0.979 | 1.0 |
| gps | ua_segmentation | distance@30 | 0.795 | 0.974 | 0.884 |
| gps | ua_segmentation | distance@60 | 0.686 | 0.975 | 1.0 |
| gps | ua_segmentation | mahalanobis@15 | 0.757 | 0.975 | 0.917 |
| gps | ua_segmentation | mahalanobis@150 | 0.267 | 0.981 | 1.0 |
| gps | ua_segmentation | mahalanobis@30 | 0.689 | 0.975 | 1.0 |
| gps | ua_segmentation | mahalanobis@60 | 0.503 | 0.977 | 1.0 |

## Colocation different-time: pair F1

| algo | distance |
|---|---|
| changepoint_pelt | 0.332 |
| fastmob_stays | 0.319 |
| h3_cell_dwell | 0.337 |
| h3_occupancy | 0.345 |
| hmm_stay_move | 0.333 |
| infostop | 0.34 |
| kalman_dwell | 0.314 |
| li_centroid | 0.334 |
| movingpandas_stops | 0.333 |
| skmob_stops | 0.319 |
| st_dbscan | 0.341 |
| st_hdbscan | 0.336 |
| trackintel_sliding | 0.315 |
| ua_mahalanobis | 0.344 |
| ua_segmentation | 0.347 |
