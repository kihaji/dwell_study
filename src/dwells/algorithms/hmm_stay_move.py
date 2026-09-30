"""Two-state (stay / move) hidden Markov model over consecutive-event displacements (new).

Hidden state is attached to each interval between consecutive events i-1 -> i.
Emissions use the displacement d_i with combined per-axis noise s_i^2 =
sigma_{i-1}^2 + sigma_i^2 + (radius_m / 3)^2: both fixes' reported error plus
movement between consecutive fixes inside the place (radius_m ~ a 3-sigma bound;
consecutive positions within a place are correlated, so the jitter is much smaller
than two independent uniform draws over the place, which made travel steps look
like within-place jumps in testing):

* stay: d ~ Rayleigh(s_i)
* move: d ~ Rayleigh(sqrt(s_i^2 + (speed_ms * dt_i)^2))

Transitions are time-inhomogeneous (continuous-time Markov chain): P(stay->move) =
1 - exp(-dt / mean_stay_s), P(move->stay) = 1 - exp(-dt / mean_move_s), so long
gaps are naturally uncertain. Viterbi decoding (numba) gives the state path; maximal
runs of 'stay' intervals become stays spanning their endpoint events. Coarse events
(sigma > coarse_factor * radius_m) are not allowed to define a stay's first/last
event. Stays need >= ``min_events`` informative events and >= ``min_duration_s``.
Relation to literature: HMM stop/move segmentation (e.g. Patterson et al. 2008 for
animal movement; hmmlearn-style models) but with per-event heteroscedastic emissions.
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl

from dwells.algorithms.common import dwells_from_labels, gap_segments, prepare

NAME = "hmm_stay_move"
FAMILY = "probabilistic"
USES_UNCERTAINTY = True
REFERENCE = "New (this repo); cf. Patterson et al. 2008 state-space movement models, doi:10.1016/j.tree.2007.10.009"


@numba.njit(cache=True)
def _rayleigh_log(d, s2):
    return np.log(max(d, 1e-3) / s2) - d * d / (2 * s2)


@numba.njit(cache=True)
def _viterbi(seg_starts, t, x, y, var, jitter, speed, tau_stay, tau_move):
    """Returns per-interval state (index i = interval (i-1, i)); 1 = stay, 0 = move, -1 = none."""
    n = len(t)
    state = np.full(n, -1, np.int64)
    for k in range(len(seg_starts) - 1):
        a, b = seg_starts[k], seg_starts[k + 1]
        m = b - a - 1  # intervals
        if m < 1:
            continue
        back = np.zeros((m, 2), np.int64)
        prev = np.array([np.log(0.5), np.log(0.5)])  # [move, stay]
        for j in range(m):
            i = a + j + 1
            dt = max(t[i] - t[i - 1], 1e-3)
            d = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            s2 = var[i] + var[i - 1] + jitter
            e_stay = _rayleigh_log(d, s2)
            e_move = _rayleigh_log(d, s2 + (speed * dt) ** 2)
            p_sm = 1 - np.exp(-dt / tau_stay)
            p_ms = 1 - np.exp(-dt / tau_move)
            if j == 0:
                cur = np.array([prev[0] + e_move, prev[1] + e_stay])
            else:
                # to move: from move (1 - p_ms) or from stay (p_sm)
                mm, sm = prev[0] + np.log(1 - p_ms + 1e-300), prev[1] + np.log(p_sm + 1e-300)
                ms, ss = prev[0] + np.log(p_ms + 1e-300), prev[1] + np.log(1 - p_sm + 1e-300)
                back[j, 0] = 0 if mm >= sm else 1
                back[j, 1] = 0 if ms >= ss else 1
                cur = np.array([max(mm, sm) + e_move, max(ms, ss) + e_stay])
            prev = cur
        s = 0 if prev[0] >= prev[1] else 1
        for j in range(m - 1, -1, -1):
            state[a + j + 1] = s
            s = back[j, s]
    return state


def detect(events: pl.DataFrame, *, min_duration_s: float = 900, radius_m: float = 100,
           max_gap_s: float = 3600, speed_ms: float = 5.0, mean_stay_s: float = 7200,
           mean_move_s: float = 900, coarse_factor: float = 3.0, min_events: int = 2,
           default_sigma_m: float = 10.0, **_) -> pl.DataFrame:
    a = prepare(events, default_sigma_m)
    var = (a.cee + a.cnn) / 2
    seg = gap_segments(events, max_gap_s).to_numpy()
    seg_starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(seg)]
    st = _viterbi(seg_starts, a.t, a.x, a.y, var, (radius_m / 3) ** 2, speed_ms, mean_stay_s, mean_move_s)
    informative = np.sqrt(var) <= coarse_factor * radius_m
    labels = np.full(len(st), -1, np.int64)
    nxt = 0
    new_seg = np.zeros(len(st), bool)
    new_seg[seg_starts[:-1]] = True
    i = 0
    n = len(st)
    while i < n:
        if st[i] != 1 or new_seg[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and st[j + 1] == 1 and not new_seg[j + 1]:
            j += 1
        idx = np.arange(i - 1, j + 1)  # interval run i..j covers events i-1..j
        inf = idx[informative[idx]]
        if len(inf) >= min_events and a.t[inf[-1]] - a.t[inf[0]] >= min_duration_s:
            labels[inf[0]:inf[-1] + 1] = nxt
            nxt += 1
        i = j + 1
    return dwells_from_labels(events, labels, min_duration_s, centroid="fused",
                              default_sigma_m=default_sigma_m, extra_var=radius_m**2 / 4)
