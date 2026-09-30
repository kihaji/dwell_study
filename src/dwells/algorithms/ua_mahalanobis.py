"""Uncertainty-aware sequential stay detection with Mahalanobis gating (new).

Model: while staying at a place with center mu, an event is z_i ~ N(mu, C_i + v I),
where C_i is the event's reported error covariance (uncertainty.with_covariance)
and v = radius_m^2 / 4 is the per-axis variance of a uniform position inside a
disk of radius ``radius_m`` (movement within the place).

Per id, in time order:
* The open stay keeps the information-weighted (inverse-covariance) estimate of
  mu and its covariance P. A new event joins when
  (z - mu)^T (C_i + v I + P)^-1 (z - mu) <= chi2_2(gate_p).
* Coarse events (1-sigma > coarse_factor * radius_m) cannot discriminate places:
  they never start, break or extend a stay. A consistent coarse event is attached
  only if a later informative event confirms the stay continued past it.
* Robust breaking: the stay closes only after ``k_break`` consecutive informative
  events fail the gate (single multipath outliers are skipped), or when no
  informative event arrives within ``max_gap_s``. The next stay is re-seeded at the
  first failing event.
* A closed stay is a dwell if it has >= ``min_events`` informative events and
  (last - first informative time) >= ``min_duration_s``.
The dwell centroid is the fused estimate, reported with its 1-sigma ellipse.
Shape 1 data uses a constant ``default_sigma_m`` so the method degrades to a
Gaussian-gated running-centroid detector with outlier tolerance.
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl
from scipy.stats import chi2

from dwells.algorithms.common import dwells_from_labels, prepare

NAME = "ua_mahalanobis"
FAMILY = "uncertainty"
USES_UNCERTAINTY = True
REFERENCE = "New (this repo). Gating as in Kalman/Mahalanobis data association (Bar-Shalom & Fortmann 1988)."


@numba.njit(cache=True)
def _kernel(starts, t, x, y, cee, cnn, cen, v, gate, coarse_sigma, k_break, max_gap, min_events, min_duration):
    n = len(t)
    labels = np.full(n, -1, np.int64)
    coarse = np.sqrt((cee + cnn) / 2) > coarse_sigma
    buf = np.empty(n, np.int64)  # pending coarse candidates of the open stay
    nxt = 0
    for k in range(len(starts) - 1):
        a, b = starts[k], starts[k + 1]
        i = a
        is_open = False
        c0, n_inf, n_buf, n_fail, first_fail = a, 0, 0, 0, a
        t_first, t_last = 0.0, 0.0
        Iee, Inn, Ien, bx, by = 1.0, 1.0, 0.0, 0.0, 0.0
        while i <= b:
            close = False
            if i == b:
                close = is_open
            elif is_open and not coarse[i] and t[i] - t_last > max_gap:
                close = True
            if close:
                if n_inf >= min_events and t_last - t_first >= min_duration:
                    nxt += 1
                else:
                    for j in range(c0, i):
                        if labels[j] == nxt:
                            labels[j] = -1
                is_open = False
                if n_fail > 0:
                    i = first_fail  # pending failures may start the next stay
                    n_fail = 0
                    continue
            if i == b:
                break
            if not is_open:
                if coarse[i]:
                    i += 1
                    continue
                # seed a new stay
                is_open = True
                c0 = i
                de = cee[i] + v
                dn = cnn[i] + v
                dd = de * dn - cen[i] ** 2
                Iee, Inn, Ien = dn / dd, de / dd, -cen[i] / dd
                bx, by = Iee * x[i] + Ien * y[i], Ien * x[i] + Inn * y[i]
                labels[i] = nxt
                n_inf, n_buf, n_fail = 1, 0, 0
                t_first = t[i]
                t_last = t[i]
                i += 1
                continue
            # current estimate
            D = Iee * Inn - Ien * Ien
            pee, pnn, pen = Inn / D, Iee / D, -Ien / D
            mx, my = pee * bx + pen * by, pen * bx + pnn * by
            see, snn, sen = cee[i] + v + pee, cnn[i] + v + pnn, cen[i] + pen
            dx, dy = x[i] - mx, y[i] - my
            d2 = (snn * dx * dx - 2 * sen * dx * dy + see * dy * dy) / (see * snn - sen * sen)
            if coarse[i]:
                if d2 <= gate:
                    buf[n_buf] = i
                    n_buf += 1
                i += 1
                continue
            if d2 <= gate:
                de = cee[i] + v
                dn = cnn[i] + v
                dd = de * dn - cen[i] ** 2
                wee, wnn, wen = dn / dd, de / dd, -cen[i] / dd
                Iee += wee
                Inn += wnn
                Ien += wen
                bx += wee * x[i] + wen * y[i]
                by += wen * x[i] + wnn * y[i]
                labels[i] = nxt
                for j in range(n_buf):
                    labels[buf[j]] = nxt
                n_buf, n_fail = 0, 0
                n_inf += 1
                t_last = t[i]
            else:
                if n_fail == 0:
                    first_fail = i
                n_fail += 1
                if n_fail >= k_break:
                    # close the stay at its last confirmed event; re-seed at first_fail
                    if n_inf >= min_events and t_last - t_first >= min_duration:
                        nxt += 1
                    else:
                        for j in range(c0, i + 1):
                            if labels[j] == nxt:
                                labels[j] = -1
                    is_open = False
                    n_fail = 0
                    i = first_fail
                    continue
            i += 1
    return labels


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, gate_p: float = 0.99, coarse_factor: float = 3.0,
           k_break: int = 2, min_events: int = 2, default_sigma_m: float = 10.0, **_) -> pl.DataFrame:
    a = prepare(events, default_sigma_m)
    v = radius_m**2 / 4
    labels = _kernel(a.starts, a.t, a.x, a.y, a.cee, a.cnn, a.cen, v, chi2.ppf(gate_p, 2),
                     coarse_factor * radius_m, k_break, max_gap_s, min_events, min_duration_s)
    # a dwell's start/end come from informative events only (coarse ones lie inside the span)
    return dwells_from_labels(events, labels, min_duration_s, centroid="fused",
                              default_sigma_m=default_sigma_m, extra_var=v)
