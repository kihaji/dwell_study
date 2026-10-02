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
* Robust breaking: the stay closes after ``k_break`` consecutive informative events
  fail the gate (single multipath outliers are skipped), when the misses persisted
  for ``confirm_s`` seconds before the id was seen back in place (a real short trip),
  or when no informative event arrives within ``max_gap_s``. The next stay is
  re-seeded at the first failing event.
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
def _kernel(starts, t, x, y, cee, cnn, cen, v, gate, coarse, k_break, max_gap, min_events, min_duration,
            confirm):
    n = len(t)
    labels = np.full(n, -1, np.int64)
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
            if d2 <= gate and n_fail > 0 and confirm > 0 and t[i] - t[first_fail] >= confirm:
                # the id was away for >= confirm seconds: the misses were a real departure
                if n_inf >= min_events and t_last - t_first >= min_duration:
                    nxt += 1
                else:
                    for j in range(c0, i):
                        if labels[j] == nxt:
                            labels[j] = -1
                is_open = False
                n_fail = 0
                i = first_fail
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
           k_break: int = 2, min_events: int = 2, default_sigma_m: float = 10.0,
           boundary: str = "midpoint", duration_basis: str = "estimated",
           coarse_sigma_m: float | None = None, confirm_s: float | None = None,
           coarse_rel: float | None = None, **_) -> pl.DataFrame:
    """Dwells only; see detect_with_members for the event-to-dwell membership.

    See module docstring. ``boundary``: "observed" (first/last member event) or
    "midpoint" (half-way to the previous/next event of the id, capped at max_gap_s / 2).
    ``duration_basis``: which span ("observed" or "estimated") must reach min_duration_s.
    ``coarse_sigma_m``: absolute 1-sigma (m) above which a fix is coarse; overrides
    coarse_factor * radius_m (place size and fix vagueness are separate questions).
    ``confirm_s``: fewer than k_break misses still end the stay when the id stayed away
    that long (the next in-place fix came >= confirm_s after the first miss); separates
    brief GPS outliers from real short trips seen as a single fix in sparse data.
    Extra output columns: first_event_ts, last_event_ts, arrive_after/arrive_by,
    depart_after/depart_by: the unobserved arrival/departure windows, bounded by the
    nearest informative (non-coarse) fixes outside the dwell; null at track ends."""
    return detect_with_members(events, min_duration_s=min_duration_s, radius_m=radius_m, max_gap_s=max_gap_s,
                               gate_p=gate_p, coarse_factor=coarse_factor, k_break=k_break,
                               min_events=min_events, default_sigma_m=default_sigma_m, boundary=boundary,
                               duration_basis=duration_basis, coarse_sigma_m=coarse_sigma_m,
                               confirm_s=confirm_s, coarse_rel=coarse_rel)[0].drop("dwell_label")


def detect_with_members(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
                        max_gap_s: float = 3600, gate_p: float = 0.99, coarse_factor: float = 3.0,
                        k_break: int = 2, min_events: int = 2, default_sigma_m: float = 10.0,
                        boundary: str = "midpoint", duration_basis: str = "estimated",
                        coarse_sigma_m: float | None = None, confirm_s: float | None = None,
                        coarse_rel: float | None = None, **_):
    """Returns (dwells with a ``dwell_label`` column, per-event dwell_label array (-1 =
    not in a kept dwell), per-event coarse mask, per-event 1-sigma)."""
    a = prepare(events, default_sigma_m)
    v = radius_m**2 / 4
    sigma = np.sqrt((a.cee + a.cnn) / 2)
    coarse = sigma > (coarse_sigma_m if coarse_sigma_m is not None else coarse_factor * radius_m)
    if coarse_rel is not None:
        # only fixes much vaguer than this id's typical fix are coarse (robust to a
        # uniformly coarse source or a mis-scaled accuracy field)
        med = np.repeat([np.median(sigma[s:e]) for s, e in zip(a.starts[:-1], a.starts[1:])], np.diff(a.starts))
        coarse &= sigma > coarse_rel * med
    md_kernel = min_duration_s if duration_basis == "observed" else 0.0
    labels = _kernel(a.starts, a.t, a.x, a.y, a.cee, a.cnn, a.cen, v, chi2.ppf(gate_p, 2),
                     coarse, k_break, max_gap_s, min_events, md_kernel, confirm_s or 0.0)
    # a dwell's start/end come from informative events only (coarse ones lie inside the span)
    d = dwells_from_labels(events, labels, md_kernel, centroid="fused",
                           default_sigma_m=default_sigma_m, extra_var=v, keep_label=True)
    if d.height == 0:
        d = d.drop("label", strict=False).with_columns(dwell_label=pl.lit(None, dtype=pl.Int64))
        return d, np.full(events.height, -1), coarse, sigma
    informative = ~coarse
    d = d.join(_windows(events, labels, informative), on="label").rename({"label": "dwell_label"})
    if boundary == "midpoint":
        cap = pl.duration(seconds=max_gap_s / 2)
        d = d.with_columns(
            start_ts=pl.when(pl.col("arrive_after").is_null()).then(pl.col("first_event_ts")).otherwise(
                pl.max_horizontal(pl.col("first_event_ts") - (pl.col("first_event_ts") - pl.col("arrive_after")) / 2,
                                  pl.col("first_event_ts") - cap)),
            end_ts=pl.when(pl.col("depart_by").is_null()).then(pl.col("last_event_ts")).otherwise(
                pl.min_horizontal(pl.col("last_event_ts") + (pl.col("depart_by") - pl.col("last_event_ts")) / 2,
                                  pl.col("last_event_ts") + cap)),
        )
    d = d.with_columns(duration_s=(pl.col("end_ts") - pl.col("start_ts")).dt.total_seconds())
    span = (pl.col("last_event_ts") - pl.col("first_event_ts")).dt.total_seconds()
    d = d.filter((span if duration_basis == "observed" else pl.col("duration_s")) >= min_duration_s)
    kept = np.isin(labels, d["dwell_label"].to_numpy())
    return d, np.where(kept, labels, -1), coarse, sigma


def _windows(events: pl.DataFrame, labels: np.ndarray, informative: np.ndarray) -> pl.DataFrame:
    """Per label: first/last member event time and the neighboring informative event times.

    Coarse fixes are skipped when looking for the previous/next event: a fix too vague to
    tell places apart is no evidence that the id had arrived or left."""
    idx = np.flatnonzero(labels >= 0)
    lab = labels[idx]
    order = np.lexsort((idx, lab))
    idx, lab = idx[order], lab[order]
    starts = np.r_[0, np.flatnonzero(np.diff(lab)) + 1]
    ends = np.r_[starts[1:], len(lab)] - 1
    first, last = idx[starts], idx[ends]
    ts, ids = events["timestamp"], events["id"].to_numpy()
    inf_idx = np.flatnonzero(informative)
    p = np.searchsorted(inf_idx, first, "left") - 1
    q = np.searchsorted(inf_idx, last, "right")
    prev = inf_idx[np.clip(p, 0, max(len(inf_idx) - 1, 0))] if len(inf_idx) else first
    nxt = inf_idx[np.clip(q, 0, max(len(inf_idx) - 1, 0))] if len(inf_idx) else last
    has_prev = (p >= 0) & (ids[prev] == ids[first])
    has_next = (q < len(inf_idx)) & (ids[nxt] == ids[last])
    return pl.DataFrame({
        "label": lab[starts], "first_event_ts": ts.gather(first), "last_event_ts": ts.gather(last),
        "arrive_after": ts.gather(prev), "arrive_by": ts.gather(first),
        "depart_after": ts.gather(last), "depart_by": ts.gather(nxt),
    }).with_columns(
        arrive_after=pl.when(pl.Series(has_prev)).then(pl.col("arrive_after")),
        depart_by=pl.when(pl.Series(has_next)).then(pl.col("depart_by")),
    )
