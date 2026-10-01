# Parameter sensitivity (dev sweep)

F1 averaged over both areas' dev datasets; for each cell the other tuning parameter is at its best value.

## gps: F1 vs radius (min duration 15 min)

| algo | r=50 | r=100 | r=250 |
|---|---|---|---|
| changepoint_pelt | 0.665 | 0.943 | 0.93 |
| fastmob_stays | 0.049 | 0.277 | 0.695 |
| h3_cell_dwell | 0.05 | 0.05 | 0.05 |
| h3_occupancy | 0.895 | 0.894 | 0.774 |
| hmm_stay_move | 0.491 | 0.581 | 0.868 |
| infostop | 0.079 | 0.374 | 0.784 |
| kalman_dwell | 0.08 | 0.37 | 0.783 |
| li_centroid | 0.072 | 0.347 | 0.772 |
| movingpandas_stops | 0.072 | 0.363 | 0.819 |
| skmob_stops | 0.049 | 0.277 | 0.684 |
| st_dbscan | 0.927 | 0.887 | 0.508 |
| st_hdbscan | 0.635 | 0.548 | 0.344 |
| trackintel_sliding | 0.049 | 0.277 | 0.695 |
| ua_mahalanobis | 0.936 | 0.929 | 0.902 |
| ua_segmentation | 0.659 | 0.941 | 0.93 |

## gps: F1 vs max gap

| algo | gap=30m | gap=2h |
|---|---|---|
| changepoint_pelt | 0.938 | 0.943 |
| fastmob_stays | 0.691 | 0.695 |
| h3_cell_dwell | 0.05 | 0.05 |
| h3_occupancy | 0.89 | 0.895 |
| hmm_stay_move | 0.862 | 0.868 |
| infostop | 0.778 | 0.784 |
| kalman_dwell | 0.78 | 0.783 |
| li_centroid | 0.766 | 0.772 |
| movingpandas_stops | 0.814 | 0.819 |
| skmob_stops | 0.68 | 0.684 |
| st_dbscan | 0.927 | 0.912 |
| st_hdbscan | 0.412 | 0.635 |
| trackintel_sliding | 0.691 | 0.695 |
| ua_mahalanobis | 0.931 | 0.936 |
| ua_segmentation | 0.936 | 0.941 |

## cdr: F1 vs radius (min duration 15 min)

| algo | r=250 | r=500 | r=1000 | r=2000 |
|---|---|---|---|---|
| changepoint_pelt | 0.356 | 0.46 | 0.443 | 0.313 |
| fastmob_stays | 0.291 | 0.334 | 0.391 | 0.417 |
| h3_cell_dwell | 0.309 | 0.309 | 0.309 | 0.309 |
| h3_occupancy | 0.419 | 0.419 | 0.209 | 0.132 |
| hmm_stay_move | 0.421 | 0.407 | 0.362 | 0.289 |
| infostop | 0.266 | 0.316 | 0.386 | 0.353 |
| kalman_dwell | 0.081 | 0.134 | 0.2 | 0.271 |
| li_centroid | 0.277 | 0.324 | 0.39 | 0.381 |
| movingpandas_stops | 0.31 | 0.409 | 0.434 | 0.234 |
| skmob_stops | 0.286 | 0.326 | 0.374 | 0.379 |
| st_dbscan | 0.284 | 0.312 | 0.332 | 0.257 |
| st_hdbscan | 0.086 | 0.084 | 0.091 | 0.08 |
| trackintel_sliding | 0.291 | 0.334 | 0.391 | 0.417 |
| ua_mahalanobis | 0.442 | 0.433 | 0.36 | 0.217 |
| ua_segmentation | 0.465 | 0.451 | 0.421 | 0.292 |

## cdr: F1 vs max gap

| algo | gap=2h | gap=6h | gap=12h |
|---|---|---|---|
| changepoint_pelt | 0.191 | 0.369 | 0.46 |
| fastmob_stays | 0.17 | 0.319 | 0.417 |
| h3_cell_dwell | 0.119 | 0.242 | 0.309 |
| h3_occupancy | 0.175 | 0.339 | 0.419 |
| hmm_stay_move | 0.172 | 0.342 | 0.421 |
| infostop | 0.159 | 0.308 | 0.386 |
| kalman_dwell | 0.18 | 0.26 | 0.271 |
| li_centroid | 0.17 | 0.321 | 0.39 |
| movingpandas_stops | 0.177 | 0.347 | 0.434 |
| skmob_stops | 0.162 | 0.276 | 0.379 |
| st_dbscan | 0.164 | 0.288 | 0.332 |
| st_hdbscan | 0.058 | 0.076 | 0.091 |
| trackintel_sliding | 0.17 | 0.319 | 0.417 |
| ua_mahalanobis | 0.177 | 0.358 | 0.442 |
| ua_segmentation | 0.173 | 0.361 | 0.465 |

## adtech: F1 vs radius (min duration 15 min)

| algo | r=100 | r=250 | r=500 | r=1000 |
|---|---|---|---|---|
| changepoint_pelt | 0.1 | 0.131 | 0.158 | 0.206 |
| fastmob_stays | 0.163 | 0.175 | 0.182 | 0.213 |
| h3_cell_dwell | 0.131 | 0.131 | 0.131 | 0.131 |
| h3_occupancy | 0.308 | 0.279 | 0.279 | 0.165 |
| hmm_stay_move | 0.281 | 0.285 | 0.281 | 0.257 |
| infostop | 0.095 | 0.111 | 0.133 | 0.169 |
| kalman_dwell | 0.026 | 0.044 | 0.069 | 0.114 |
| li_centroid | 0.093 | 0.11 | 0.131 | 0.17 |
| movingpandas_stops | 0.103 | 0.129 | 0.16 | 0.187 |
| skmob_stops | 0.16 | 0.17 | 0.178 | 0.2 |
| st_dbscan | 0.215 | 0.22 | 0.208 | 0.195 |
| st_hdbscan | 0.092 | 0.09 | 0.091 | 0.078 |
| trackintel_sliding | 0.163 | 0.175 | 0.182 | 0.213 |
| ua_mahalanobis | 0.304 | 0.314 | 0.291 | 0.253 |
| ua_segmentation | 0.317 | 0.316 | 0.293 | 0.275 |

## adtech: F1 vs max gap

| algo | gap=2h | gap=6h | gap=12h |
|---|---|---|---|
| changepoint_pelt | 0.056 | 0.141 | 0.206 |
| fastmob_stays | 0.06 | 0.129 | 0.213 |
| h3_cell_dwell | 0.048 | 0.073 | 0.131 |
| h3_occupancy | 0.079 | 0.202 | 0.308 |
| hmm_stay_move | 0.087 | 0.182 | 0.285 |
| infostop | 0.059 | 0.108 | 0.169 |
| kalman_dwell | 0.086 | 0.114 | 0.104 |
| li_centroid | 0.06 | 0.109 | 0.17 |
| movingpandas_stops | 0.054 | 0.12 | 0.187 |
| skmob_stops | 0.06 | 0.121 | 0.2 |
| st_dbscan | 0.081 | 0.162 | 0.22 |
| st_hdbscan | 0.031 | 0.066 | 0.092 |
| trackintel_sliding | 0.06 | 0.129 | 0.213 |
| ua_mahalanobis | 0.086 | 0.208 | 0.314 |
| ua_segmentation | 0.099 | 0.213 | 0.317 |

## Best F1 vs min duration (dwell definition)

| algo | profile | md=5m | md=15m | md=30m | md=60m |
|---|---|---|---|---|---|
| changepoint_pelt | adtech | 0.187 | 0.206 | 0.22 | 0.253 |
| fastmob_stays | adtech | 0.185 | 0.213 | 0.232 | 0.264 |
| h3_cell_dwell | adtech | 0.114 | 0.131 | 0.137 | 0.155 |
| h3_occupancy | adtech | 0.279 | 0.308 | 0.323 | 0.364 |
| hmm_stay_move | adtech | 0.26 | 0.285 | 0.3 | 0.345 |
| infostop | adtech | 0.153 | 0.169 | 0.179 | 0.204 |
| kalman_dwell | adtech | 0.097 | 0.114 | 0.133 | 0.151 |
| li_centroid | adtech | 0.152 | 0.17 | 0.176 | 0.201 |
| movingpandas_stops | adtech | 0.167 | 0.187 | 0.198 | 0.226 |
| skmob_stops | adtech | 0.175 | 0.2 | 0.218 | 0.248 |
| st_dbscan | adtech | 0.197 | 0.22 | 0.224 | 0.251 |
| st_hdbscan | adtech | 0.079 | 0.092 | 0.097 | 0.111 |
| trackintel_sliding | adtech | 0.185 | 0.213 | 0.232 | 0.264 |
| ua_mahalanobis | adtech | 0.283 | 0.314 | 0.329 | 0.372 |
| ua_segmentation | adtech | 0.287 | 0.317 | 0.33 | 0.372 |
| changepoint_pelt | cdr | 0.435 | 0.46 | 0.476 | 0.522 |
| fastmob_stays | cdr | 0.399 | 0.417 | 0.431 | 0.486 |
| h3_cell_dwell | cdr | 0.288 | 0.309 | 0.326 | 0.364 |
| h3_occupancy | cdr | 0.395 | 0.419 | 0.434 | 0.476 |
| hmm_stay_move | cdr | 0.399 | 0.421 | 0.439 | 0.484 |
| infostop | cdr | 0.365 | 0.386 | 0.403 | 0.443 |
| kalman_dwell | cdr | 0.258 | 0.271 | 0.291 | 0.327 |
| li_centroid | cdr | 0.368 | 0.39 | 0.403 | 0.44 |
| movingpandas_stops | cdr | 0.408 | 0.434 | 0.445 | 0.495 |
| skmob_stops | cdr | 0.362 | 0.379 | 0.395 | 0.447 |
| st_dbscan | cdr | 0.311 | 0.332 | 0.345 | 0.385 |
| st_hdbscan | cdr | 0.079 | 0.091 | 0.096 | 0.1 |
| trackintel_sliding | cdr | 0.399 | 0.417 | 0.431 | 0.486 |
| ua_mahalanobis | cdr | 0.418 | 0.442 | 0.458 | 0.501 |
| ua_segmentation | cdr | 0.44 | 0.465 | 0.482 | 0.531 |
| changepoint_pelt | gps | 0.939 | 0.943 | 0.946 | 0.933 |
| fastmob_stays | gps | 0.638 | 0.695 | 0.704 | 0.683 |
| h3_cell_dwell | gps | 0.033 | 0.05 | 0.066 | 0.049 |
| h3_occupancy | gps | 0.881 | 0.895 | 0.896 | 0.892 |
| hmm_stay_move | gps | 0.839 | 0.868 | 0.867 | 0.859 |
| infostop | gps | 0.771 | 0.784 | 0.788 | 0.76 |
| kalman_dwell | gps | 0.767 | 0.783 | 0.784 | 0.755 |
| li_centroid | gps | 0.759 | 0.772 | 0.773 | 0.741 |
| movingpandas_stops | gps | 0.722 | 0.819 | 0.821 | 0.801 |
| skmob_stops | gps | 0.627 | 0.684 | 0.689 | 0.665 |
| st_dbscan | gps | 0.925 | 0.927 | 0.929 | 0.918 |
| st_hdbscan | gps | 0.606 | 0.635 | 0.687 | 0.749 |
| trackintel_sliding | gps | 0.638 | 0.695 | 0.704 | 0.683 |
| ua_mahalanobis | gps | 0.934 | 0.936 | 0.936 | 0.925 |
| ua_segmentation | gps | 0.937 | 0.941 | 0.945 | 0.933 |
