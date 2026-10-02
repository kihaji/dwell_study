"""Tuned ua_mahalanobis parameter presets per source type.

Chosen on the dev datasets (tuning) and validated on held-out eval datasets; see
docs/tuning.md for the sweeps, sensitivity and the rationale for each value.
``min_duration_s`` is the analyst's dwell definition, not a tuning knob, so it is
supplied separately (default 15 min).
"""

from __future__ import annotations

# One robust base for every source (held-out F1 at least as good as per-source tuning,
# except adtech by ~0.01): outlier-tolerant (3 consecutive misses) yet a stay still ends
# after 15 min away; fixes vaguer than 1 km (1-sigma) never define a dwell.
BASE = {"radius_m": 75, "max_gap_s": 43200, "gate_p": 0.99, "k_break": 3, "confirm_s": 900,
        "coarse_sigma_m": 1000.0, "min_events": 2, "boundary": "midpoint", "duration_basis": "estimated"}
# default_sigma_m only matters for centerpoint-only data (no accuracy fields): it should be
# the source's typical 1-sigma error (keep it below coarse_sigma_m).
PRESETS: dict[str, dict] = {
    "gps": BASE | {"default_sigma_m": 10.0},
    "cdr": BASE | {"default_sigma_m": 300.0},
    "adtech": BASE | {"default_sigma_m": 300.0},
    "mixed": BASE | {"default_sigma_m": 30.0},
}
DEFAULT_MIN_DURATION_S = 900


def params_for(profile: str, min_duration_s: float = DEFAULT_MIN_DURATION_S, **overrides) -> dict:
    return PRESETS[profile] | {"min_duration_s": min_duration_s} | overrides
