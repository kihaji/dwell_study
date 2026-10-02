"""H3 helpers shared by downstream modules."""

from __future__ import annotations

import h3
import numpy as np

EDGE_M = np.array([h3.average_hexagon_edge_length(r, "m") for r in range(16)])


def resolution_for(size_m: float) -> int:
    """Finest H3 resolution whose average edge length is >= size_m."""
    ok = np.flatnonzero(EDGE_M >= size_m)
    return int(ok.max()) if len(ok) else 0
