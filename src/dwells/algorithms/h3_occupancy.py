"""H3 occupancy with hysteresis and accuracy-adaptive resolution (new; grid family).

* Resolution r: the finest H3 resolution whose average edge length >= radius_m, so
  a place fits in about one cell.
* Each event gets its cell at r. A run's anchor is its first informative event's
  cell; a later informative event continues the run if its cell is the anchor or
  one of the anchor's 6 neighbors (grid_disk k=1). This hysteresis absorbs the
  boundary flicker that fragments plain cell runs (see h3_cell_dwell).
* Accuracy-adaptive: an event whose 1-sigma exceeds ``coarse_factor`` (3) x the edge
  length cannot tell neighboring cells apart, so it is ignored: it never starts,
  breaks or extends a run (like ua_mahalanobis coarse handling). A 1x threshold made
  every CDR event coarse in the dev sweep (F1 0.42 -> 0.23).
* A run closes when an informative event falls outside the anchor's disk (after
  ``k_break`` consecutive misses, so single outliers are skipped) or on a gap >
  max_gap_s; runs spanning >= min_duration_s with >= min_events informative events
  are dwells (centroid = mean of informative member positions).
Everything except the run scan is a per-event map (cell ids, neighbor lists), so
it translates directly to Spark SQL + a per-id sorted scan.
"""

from __future__ import annotations

import h3
import numba
import numpy as np
import polars as pl

from dwells.algorithms.common import dwells_from_labels, prepare

NAME = "h3_occupancy"
FAMILY = "grid"
USES_UNCERTAINTY = True
REFERENCE = "New (this repo). H3: Uber hierarchical hexagonal grid, https://h3geo.org"

EDGE_M = np.array([h3.average_hexagon_edge_length(r, "m") for r in range(16)])


def resolution_for(size_m: float) -> int:
    ok = np.flatnonzero(EDGE_M >= size_m)
    return int(ok.max()) if len(ok) else 0


@numba.njit(cache=True)
def _kernel(starts, t, cell_idx, disk, coarse, max_gap, k_break, min_events, min_duration):
    """cell_idx[i]: event i's cell as an index into the unique-cell table;
    disk[u]: the 7 cells (unique-table indices, -2 if absent) of u's k=1 disk."""
    n = len(t)
    labels = np.full(n, -1, np.int64)
    nxt = 0
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        i = a
        is_open = False
        anchor, c0, n_inf, misses, first_miss = -1, a, 0, 0, a
        t_first, t_last = 0.0, 0.0
        while i <= b:
            if is_open and (i == b or (not coarse[i] and t[i] - t_last > max_gap)):
                if n_inf >= min_events and t_last - t_first >= min_duration:
                    nxt += 1
                else:
                    for j in range(c0, i):
                        if labels[j] == nxt:
                            labels[j] = -1
                is_open = False
                if misses > 0:
                    i = first_miss
                    misses = 0
                    continue
            if i == b:
                break
            if coarse[i]:
                i += 1
                continue
            u = cell_idx[i]
            if not is_open:
                is_open = True
                anchor, c0, n_inf, misses = u, i, 1, 0
                t_first = t[i]
                t_last = t[i]
                labels[i] = nxt
                i += 1
                continue
            inside = False
            for c in disk[anchor]:
                if c == u:
                    inside = True
            if inside:
                labels[i] = nxt
                n_inf += 1
                t_last = t[i]
                misses = 0
            else:
                if misses == 0:
                    first_miss = i
                misses += 1
                if misses >= k_break:
                    if n_inf >= min_events and t_last - t_first >= min_duration:
                        nxt += 1
                    else:
                        for j in range(c0, i + 1):
                            if labels[j] == nxt:
                                labels[j] = -1
                    is_open = False
                    misses = 0
                    i = first_miss
                    continue
            i += 1
    return labels


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, k_break: int = 2, min_events: int = 2, coarse_factor: float = 3.0,
           default_sigma_m: float = 10.0, **_) -> pl.DataFrame:
    a = prepare(events, default_sigma_m)
    r = resolution_for(radius_m)
    coarse = np.sqrt((a.cee + a.cnn) / 2) > coarse_factor * EDGE_M[r]
    lat, lon = events["lat"].to_numpy(), events["lon"].to_numpy()
    cells = np.array([h3.str_to_int(h3.latlng_to_cell(la, lo, r)) for la, lo in zip(lat, lon)], np.int64)
    uniq, cell_idx = np.unique(cells, return_inverse=True)
    pos = {c: i for i, c in enumerate(uniq.tolist())}
    disk = np.full((len(uniq), 7), -2, np.int64)
    for u, c in enumerate(uniq.tolist()):
        for j, nb in enumerate(h3.grid_disk(h3.int_to_str(c), 1)[:7]):
            disk[u, j] = pos.get(h3.str_to_int(nb), -2)
    labels = _kernel(a.starts, a.t, cell_idx.astype(np.int64), disk, coarse, max_gap_s, k_break,
                     min_events, min_duration_s)
    return dwells_from_labels(events, labels, min_duration_s)
