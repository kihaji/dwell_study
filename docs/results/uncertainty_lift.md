# Does positional-error information help? (held-out, min duration 15 min)

Same positions in every shape; only the error fields differ. Algorithms that ignore error are shape-invariant and omitted.

| algo | profile | 1 | 2 | 3_mixed | 3_p68 | 3_p95 | gain_1_to_3mixed |
|---|---|---|---|---|---|---|---|
| h3_cell_dwell | adtech | 0.113 | 0.116 | 0.116 | 0.116 | 0.116 | 0.004 |
| h3_occupancy | adtech | 0.284 | 0.35 | 0.35 | 0.35 | 0.35 | 0.066 |
| hmm_stay_move | adtech | 0.148 | 0.312 | 0.312 | 0.312 | 0.312 | 0.164 |
| kalman_dwell | adtech | 0.04 | 0.134 | 0.132 | 0.132 | 0.132 | 0.092 |
| ua_mahalanobis | adtech | 0.245 | 0.356 | 0.355 | 0.355 | 0.355 | 0.11 |
| ua_segmentation | adtech | 0.208 | 0.366 | 0.367 | 0.367 | 0.367 | 0.159 |
| h3_cell_dwell | cdr | 0.354 | 0.276 | 0.276 | 0.276 | 0.334 | -0.078 |
| h3_occupancy | cdr | 0.432 | 0.432 | 0.432 | 0.432 | 0.432 | 0.0 |
| hmm_stay_move | cdr | 0.333 | 0.465 | 0.465 | 0.465 | 0.465 | 0.132 |
| kalman_dwell | cdr | 0.141 | 0.265 | 0.267 | 0.267 | 0.267 | 0.127 |
| ua_mahalanobis | cdr | 0.448 | 0.495 | 0.499 | 0.499 | 0.499 | 0.051 |
| ua_segmentation | cdr | 0.449 | 0.511 | 0.521 | 0.521 | 0.521 | 0.072 |
| h3_cell_dwell | gps | 0.515 | 0.053 | 0.053 | 0.053 | 0.053 | -0.462 |
| h3_occupancy | gps | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 | 0.0 |
| hmm_stay_move | gps | 0.869 | 0.891 | 0.891 | 0.891 | 0.891 | 0.022 |
| kalman_dwell | gps | 0.773 | 0.776 | 0.776 | 0.776 | 0.776 | 0.003 |
| ua_mahalanobis | gps | 0.953 | 0.939 | 0.939 | 0.939 | 0.939 | -0.014 |
| ua_segmentation | gps | 0.958 | 0.955 | 0.956 | 0.956 | 0.956 | -0.002 |
