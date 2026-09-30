"""Dwell-detection algorithm registry.

Every module exposes::

    NAME: str
    FAMILY: str            # sequential | library | density | uncertainty | probabilistic | grid
    USES_UNCERTAINTY: bool # whether horizontalAccuracy / ellipses change the result
    REFERENCE: str         # origin (paper / library / ported repo)
    def detect(events, *, min_duration_s, radius_m, max_gap_s, **kw) -> pl.DataFrame  # DWELLS-like

``events`` is a validated event frame (any of the three shapes) sorted by id, timestamp.
"""

from __future__ import annotations

import importlib
import importlib.util

_PLANNED = [
    # ports of prior work
    "li_centroid", "h3_cell_dwell", "kalman_dwell",
    # libraries
    "trackintel_sliding", "skmob_stops", "fastmob_stays", "movingpandas_stops", "infostop",
    # density clustering
    "st_dbscan", "st_hdbscan",
    # uncertainty-aware (new)
    "ua_mahalanobis", "ua_segmentation",
    # probabilistic / temporal
    "hmm_stay_move", "changepoint_pelt",
    # grid
    "h3_occupancy",
]
_MODULES = {n: f"dwells.algorithms.{n}" for n in _PLANNED}


def names(include_missing: bool = False) -> list[str]:
    """Registered algorithms (only those whose module exists unless include_missing)."""
    return [n for n, m in _MODULES.items() if include_missing or importlib.util.find_spec(m)]


def get(name: str):
    return importlib.import_module(_MODULES[name])
