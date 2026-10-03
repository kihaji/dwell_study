"""Sensing models: turn true trajectories into observed events.

Profiles (see docs/datasets.md for rationale):

* ``gps``    dense app/OS location logging. ~30 s while moving or shortly after
             arrival, throttled to ~5 min when stationary; 1-sigma 3-20 m outdoors,
             x2.5 indoors (during stays); rare multipath outliers; outages.
* ``cdr``    network events (calls/SMS/data) as a diurnal, bursty Poisson process.
             Position is the serving *sector centroid* (quantized, not a noisy draw);
             the ellipse is the sector's spatial covariance. Occasional handover to
             the second-nearest tower.
* ``adtech`` sparse, user-driven app sessions (bursts of pings, long gaps) with a
             per-ping location-source mixture: GPS / Wi-Fi / cell / IP-geolocation,
             plus stale cached fixes. Very heterogeneous error within one id.

For gps/wifi/cell pings the reported point is drawn from N(true, Σ_true), where
Σ_true = Σ_reported / c² and c ~ lognormal(0, calibration_noise): the stated
error is roughly, not exactly, honest.

Output: one master frame per dataset with columns
``id, timestamp, lat, lon, true_lat, true_lon, source, sig_a, sig_b, az, p_mixed,
truth_dwell_id``; ``sig_a >= sig_b`` are reported 1-sigma axes (m), ``az`` the
major-axis bearing in [0, 180). ``io.shape_view`` derives the three dataset shapes.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from dwells import geo
from dwells.generator.area import Area
from dwells.generator.simulate import SimResult

H = 3600.0
PROFILES = ("gps", "cdr", "adtech")


def _diurnal(t, utc_offset_h):
    """Relative activity by local hour: low at night, peak in the day."""
    h = ((t / H) + utc_offset_h) % 24
    return np.select([h < 6, h < 8, h < 22], [0.08, 0.6, 1.0], 0.5)


def _poisson_times(rng, t0, t1, rate_per_h, utc_offset_h):
    """Inhomogeneous Poisson process by thinning."""
    n = rng.poisson(rate_per_h * (t1 - t0) / H)
    t = np.sort(rng.uniform(t0, t1, n))
    return t[rng.random(n) < _diurnal(t, utc_offset_h)]


def _bursts(rng, t, mean_extra, max_span_s):
    """Each seed event spawns a geometric number of follow-ups within max_span_s."""
    k = rng.geometric(1.0 / (1.0 + mean_extra), len(t)) - 1
    extra = np.repeat(t, k) + rng.uniform(0, max_span_s, k.sum())
    return np.sort(np.r_[t, extra])


def _outages(rng, times, t0, t1, rate_per_day, hours):
    """Drop events inside random outage windows (phone off, no signal, app killed)."""
    n = rng.poisson(rate_per_day * (t1 - t0) / (24 * H))
    s = rng.uniform(t0, t1, n)
    e = s + rng.uniform(*hours, n) * H
    keep = np.ones(len(times), bool)
    for a, b in zip(s, e):
        keep &= ~((times >= a) & (times < b))
    return times[keep]


def _gps_times(rng, stays, t0, t1):
    base = rng.uniform(10, 60)
    out = []
    arrive, depart = stays[:, 0], stays[:, 1]
    # moving segments: between stays
    for a, b in zip(np.r_[t0, depart[:-1]], arrive):
        if b > a:
            out.append(np.arange(a, b, base) + rng.uniform(0, base))
    for a, d in zip(arrive, depart):
        warm = min(d, a + 600)
        out.append(np.arange(a, warm, base) + rng.uniform(0, base))
        if d > warm:
            step = rng.uniform(240, 360)
            out.append(np.arange(warm, d, step) + rng.uniform(0, step))
    t = np.sort(np.concatenate(out)) if out else np.array([])
    t = t[(t >= t0) & (t < t1)]
    return _outages(rng, t, t0, t1, 0.15, (1, 8))


def _positions(knots, t):
    T, X, Y = knots
    return np.interp(t, T, X), np.interp(t, T, Y)


def _stay_index(stays, t):
    i = np.searchsorted(stays[:, 0], t, side="right") - 1
    ok = (i >= 0) & (t < stays[np.clip(i, 0, None), 1])
    return np.where(ok, i, -1)


def _rand_axes(rng, sigma, ratio_hi):
    ratio = rng.uniform(1.0, ratio_hi, len(sigma))
    return sigma * np.sqrt(ratio), sigma / np.sqrt(ratio), rng.uniform(0, 180, len(sigma))


def _draw(rng, sa, sb, az):
    """Sample offsets (east, north) from N(0, Σ(sa, sb, az))."""
    u, v = rng.standard_normal(len(sa)) * sa, rng.standard_normal(len(sa)) * sb
    th = np.radians(az)
    return u * np.sin(th) + v * np.cos(th), u * np.cos(th) - v * np.sin(th)


class _Towers:
    def __init__(self, area: Area, seed):
        rng = np.random.default_rng(seed + 7_777)
        spacing = 600.0 if area.name == "urban" else 2500.0 if area.name == "suburban" else 500.0
        px, py = geo.to_enu(area.places["lat"].to_numpy(), area.places["lon"].to_numpy(), *area.center)
        ext = max(np.abs(px).max(), np.abs(py).max()) + spacing
        g = np.arange(-ext, ext + spacing, spacing)
        gx, gy = np.meshgrid(g, g)
        self.x = gx.ravel() + rng.normal(0, spacing * 0.2, gx.size)
        self.y = gy.ravel() + rng.normal(0, spacing * 0.2, gx.size)
        self.rot = rng.uniform(0, 120, gx.size)
        self.R = spacing * 0.9
        from scipy.spatial import cKDTree

        self.tree = cKDTree(np.c_[self.x, self.y])
        # sector centroid/covariance of a uniform 120-degree wedge, radius R, centered at bearing 0
        s = np.random.default_rng(0)
        r = self.R * np.sqrt(s.random(20_000))
        b = np.radians(s.uniform(-60, 60, 20_000))
        wx, wy = r * np.sin(b), r * np.cos(b)  # bearing 0 = north
        self.c_n = wy.mean()
        self.s_rad = wy.std()  # along bearing
        self.s_tan = wx.std()  # across bearing

    def observe(self, rng, x, y):
        _, idx = self.tree.query(np.c_[x, y], k=2)
        pick = np.where(rng.random(len(x)) < 0.2, idx[:, 1], idx[:, 0])
        tx, ty = self.x[pick], self.y[pick]
        bearing = (np.degrees(np.arctan2(x - tx, y - ty)) - self.rot[pick]) % 360
        sector_bearing = (np.floor(bearing / 120) * 120 + 60 + self.rot[pick]) % 360
        th = np.radians(sector_bearing)
        ox, oy = tx + self.c_n * np.sin(th), ty + self.c_n * np.cos(th)
        sa = np.full(len(x), max(self.s_rad, self.s_tan))
        sb = np.full(len(x), min(self.s_rad, self.s_tan))
        az_major = sector_bearing if self.s_rad >= self.s_tan else sector_bearing + 90
        return ox, oy, sa, sb, az_major % 180


def _sample_agent(rng, profile, knots, stays, t0, t1, utc, towers, hubs, calibration_noise, outlier_rate):
    if profile == "gps":
        t = _gps_times(rng, stays, t0, t1)
    elif profile == "cdr":
        t = _bursts(rng, _poisson_times(rng, t0, t1, rng.lognormal(np.log(1.2), 0.6), utc), 0.8, 300)
        t = _outages(rng, t, t0, t1, 0.05, (2, 12))
    else:
        heavy = rng.lognormal(np.log(0.4), 1.0)  # sessions/h; broad light-to-heavy user spread
        t = _bursts(rng, _poisson_times(rng, t0, t1, heavy, utc), 2.0, 480)
    t = t[(t >= t0) & (t < t1)]
    n = len(t)
    x, y = _positions(knots, t)
    si = _stay_index(stays, t)
    in_stay = si >= 0
    # wander inside the place footprint while staying
    rad = np.where(in_stay, stays[np.clip(si, 0, None), 2], 0.0)
    rr, ang = rad * np.sqrt(rng.random(n)), rng.uniform(0, 2 * np.pi, n)
    x, y = x + rr * np.cos(ang), y + rr * np.sin(ang)

    src = np.full(n, profile, dtype=object)
    if profile == "gps":
        sig = rng.lognormal(np.log(6.0), 0.5, n) * np.where(in_stay, 2.5, 1.0)
        sa, sb, az = _rand_axes(rng, sig, 1.6)
        ox, oy = None, None
    elif profile == "cdr":
        ox, oy, sa, sb, az = towers.observe(rng, x, y)
    else:
        kind = rng.choice(["gps", "wifi", "cell", "ip"], size=n, p=[0.45, 0.30, 0.17, 0.08])
        src = kind.astype(object)
        sig = np.select([kind == "gps", kind == "wifi", kind == "cell"],
                        [rng.lognormal(np.log(8), 0.4, n), rng.lognormal(np.log(35), 0.4, n),
                         rng.lognormal(np.log(900), 0.5, n)],
                        rng.lognormal(np.log(6000), 0.4, n))
        sa, sb, az = _rand_axes(rng, sig, np.select([kind == "cell"], [3.0], 1.5))
        ox, oy = None, None

    c = np.exp(rng.normal(0, calibration_noise, n)) if calibration_noise > 0 else np.ones(n)
    if ox is None:
        ex, ey = _draw(rng, sa / c, sb / c, az)
        if profile == "gps" and outlier_rate > 0:
            out = rng.random(n) < outlier_rate
            big = rng.uniform(50, 300, out.sum())
            ex[out] += big * np.cos(ang[out])
            ey[out] += big * np.sin(ang[out])
            src[out] = "gps_outlier"
        ox, oy = x + ex, y + ey
    if profile == "adtech":
        ip = src == "ip"
        h = rng.integers(len(hubs), size=ip.sum())
        ox[ip], oy[ip] = hubs[h, 0], hubs[h, 1]
        stale = np.r_[False, rng.random(n - 1) < 0.04] if n else np.zeros(0, bool)
        for i in np.flatnonzero(stale):  # cached fix re-reported with a new timestamp
            ox[i], oy[i], sa[i], sb[i], az[i], src[i] = ox[i - 1], oy[i - 1], sa[i - 1], sb[i - 1], az[i - 1], "stale"
    return t, x, y, ox, oy, sa, sb, az, src, si


def sample(sim: SimResult, area: Area, profile: str, seed: int = 0,
           calibration_noise: float = 0.25, outlier_rate: float = 0.02) -> pl.DataFrame:
    assert profile in PROFILES
    rng = np.random.default_rng([seed, PROFILES.index(profile)])
    towers = _Towers(area, seed) if profile == "cdr" else None
    hubs = np.array([[0.0, 0.0], [3000.0, -2000.0], [-2500.0, 1500.0]])
    rad = area.places["radius_m"].to_numpy()
    utc = sim.config.utc_offset_h
    frames = []
    stays_by_id = {k[0]: g for k, g in sim.stays.group_by("id")}
    for aid, knots in sim.knots.items():
        st = stays_by_id[aid].sort("dwell_id")
        stays = np.c_[st["arrive"].to_numpy(), st["depart"].to_numpy(), rad[st["place_id"].to_numpy()]]
        t, x, y, ox, oy, sa, sb, az, src, si = _sample_agent(
            rng, profile, knots, stays, sim.t0, sim.t1, utc, towers, hubs, calibration_noise, outlier_rate)
        if len(t) == 0:
            continue
        lat, lon = geo.from_enu(ox, oy, *area.center)
        tlat, tlon = geo.from_enu(x, y, *area.center)
        dwell = st["dwell_id"].to_numpy()[np.clip(si, 0, None)]
        frames.append(pl.DataFrame({
            "id": aid, "timestamp": (t * 1e6).astype(np.int64), "lat": lat, "lon": lon,
            "true_lat": tlat, "true_lon": tlon, "source": src.astype(str),
            "sig_a": np.maximum(sa, sb), "sig_b": np.minimum(sa, sb), "az": az % 180,
            "p_mixed": np.where(rng.random(len(t)) < 0.5, 0.68, 0.95),
            "truth_dwell_id": pl.Series(np.where(si >= 0, dwell, -1)).replace(-1, None),
        }))
    ev = pl.concat(frames).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    return ev.sort("id", "timestamp")


def describe(ev: pl.DataFrame) -> pl.DataFrame:
    """Per-dataset summary used in docs: events/id/day, gap and error quantiles."""
    days = (ev["timestamp"].max() - ev["timestamp"].min()).total_seconds() / 86400
    gaps = ev.select(pl.col("timestamp").diff().over("id").dt.total_seconds().alias("g")).drop_nulls()["g"]
    return pl.DataFrame({
        "events": [ev.height], "ids": [ev["id"].n_unique()],
        "events_per_id_day": [ev.height / ev["id"].n_unique() / max(days, 1e-9)],
        "gap_p50_s": [gaps.quantile(0.5)], "gap_p95_s": [gaps.quantile(0.95)],
        "sigma_p50_m": [ev["sig_a"].median()], "sigma_p95_m": [ev["sig_a"].quantile(0.95)],
    })

