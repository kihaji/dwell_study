import os

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pyspark")  # Spark tests run in the spark/ Docker image

from dwells.geo import from_enu  # noqa: E402
from dwells_spark import job  # noqa: E402

T0 = 1_709_510_400  # 2024-03-04 00:00 UTC, a Monday
H = 3600


@pytest.fixture(scope="session")
def spark():
    s = job.session(os.environ.get("SPARK_MASTER", "local[2]"), app_name="dwells-tests")
    yield s
    s.stop()


def routine(uid, home, work, days=7, seed=0):
    """Home overnight, work 08-17 on weekdays; a fix every 5 min (8 m jitter), every
    minute while traveling (straight line)."""
    rng = np.random.default_rng(seed)
    plan = []
    for d in range(days):
        b = d * 24
        plan += [(b, b + 7.5, home), (b + 8, b + 17, work), (b + 17.5, b + 24, home)] if d % 7 < 5 \
            else [(b, b + 24, home)]
    t, x, y = [], [], []
    for k, (s0, s1, (px, py)) in enumerate(plan):
        ts = np.arange(s0 * H, s1 * H, 300.0)
        t += list(ts); x += list(px + rng.normal(0, 8, len(ts))); y += list(py + rng.normal(0, 8, len(ts)))
        if k + 1 < len(plan) and plan[k + 1][2] != (px, py):
            nx, ny = plan[k + 1][2]
            tt = np.arange(s1 * H, plan[k + 1][0] * H, 60.0)
            f = (tt - s1 * H) / ((plan[k + 1][0] - s1) * H)
            t += list(tt); x += list(px + (nx - px) * f); y += list(py + (ny - py) * f)
    return _frame(uid, t, x, y)


def wanderer(uid, n=300):
    """Always moving (500 m between fixes): no dwells at all."""
    t = np.arange(n) * 60.0
    return _frame(uid, list(t), list(np.arange(n) * 500.0), [0.0] * n)


def _frame(uid, t, x, y):
    lat, lon = from_enu(np.array(x), np.array(y), 40.0, -75.0)
    return pl.DataFrame({"id": uid, "lat": lat, "lon": lon,
                         "timestamp": ((T0 + np.array(t)) * 1e6).astype(np.int64)}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
