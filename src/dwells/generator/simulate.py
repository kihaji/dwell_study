"""Agent-based ground-truth simulator.

Agents live in households, follow persona routines (commuter, shift worker,
student, at-home, remote), visit favorite places chosen with distance decay,
share workplaces (coworkers / shift workers at different times), live next door
to other households (near-miss colocation decoys) and occasionally meet friends.
Travel follows the area's road graph. The output is the *true* continuous
trajectory (as time-knots) plus the true stays; sensing is done in sampling.py.

Times are float seconds since the Unix epoch (UTC). Routines are planned in
local time using a fixed ``utc_offset_h`` (DST ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import polars as pl

from dwells import geo
from dwells.generator.area import Area

H = 3600.0
M = 60.0
PERSONAS = {"worker": 0.55, "shift": 0.08, "student": 0.10, "home": 0.15, "remote": 0.12}
DURATION_MIN = {  # stay duration ranges (minutes) for discretionary activities
    "food": (25, 90), "grocery": (12, 45), "shop": (10, 60), "gym": (40, 100),
    "park": (20, 120), "worship": (60, 120), "fuel": (3, 9), "friend": (60, 180),
}
MIN_STAY_S = 60.0


@dataclass
class SimConfig:
    n_agents: int = 50
    n_days: int = 14
    start_date: str = "2024-03-04"  # a Monday
    utc_offset_h: float = -6.0
    seed: int = 0
    workplace_pool_frac: float = 0.2
    neighbor_frac: float = 0.25
    friend_pairs_frac: float = 0.3
    meet_rate_per_week: float = 1.0
    travel_noise: float = 0.15


@dataclass
class SimResult:
    config: SimConfig
    agents: pl.DataFrame
    stays: pl.DataFrame  # id, dwell_id, place_id, arrive, depart (epoch s)
    meetings: pl.DataFrame
    knots: dict = field(repr=False, default_factory=dict)  # id -> (T, X, Y) ENU meters
    t0: float = 0.0
    t1: float = 0.0


# ------------------------------------------------------------------ agents

class _PlacePicker:
    def __init__(self, area: Area):
        p = area.places
        self.x, self.y = geo.to_enu(p["lat"].to_numpy(), p["lon"].to_numpy(), *area.center)
        self.ids = p["place_id"].to_numpy()
        cats = p["category"].to_numpy()
        self.by_cat = {c: np.flatnonzero(cats == c) for c in np.unique(cats)}

    def near(self, rng, cat, anchor, scale_m, k=1, exclude=()):
        idx = self.by_cat.get(cat, np.array([], int))
        idx = idx[~np.isin(self.ids[idx], list(exclude))] if len(exclude) else idx
        if len(idx) == 0:
            return []
        d = np.hypot(self.x[idx] - anchor[0], self.y[idx] - anchor[1])
        w = np.exp(-d / scale_m) + 1e-12
        k = min(k, len(idx))
        return list(self.ids[rng.choice(idx, size=k, replace=False, p=w / w.sum())])

    def xy(self, pid):
        return self.x[pid], self.y[pid]


def _build_agents(area: Area, cfg: SimConfig, rng) -> list[dict]:
    pk = _PlacePicker(area)
    homes = pk.by_cat["home"]
    agents: list[dict] = []
    used_homes: set[int] = set()
    hh = 0
    household_homes: list[int] = []
    n_work = max(3, int(cfg.workplace_pool_frac * cfg.n_agents))
    work_pool = list(rng.choice(pk.by_cat["work"], size=min(n_work, len(pk.by_cat["work"])), replace=False))
    school_pool = list(rng.choice(pk.by_cat["school"], size=min(max(2, cfg.n_agents // 20), len(pk.by_cat["school"])), replace=False))
    while len(agents) < cfg.n_agents:
        size = int(rng.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.2, 0.15]))
        neighbor_of = -1
        if household_homes and rng.random() < cfg.neighbor_frac:
            # live next door to an existing household: nearest unused home
            j = int(rng.integers(len(household_homes)))
            ax, ay = pk.xy(household_homes[j])
            cand = homes[~np.isin(homes, list(used_homes))]
            d = np.hypot(pk.x[cand] - ax, pk.y[cand] - ay)
            home = int(cand[np.argmin(d)])
            neighbor_of = j
        else:
            home = int(rng.choice(homes[~np.isin(homes, list(used_homes))]))
        used_homes.add(home)
        household_homes.append(home)
        hx = pk.xy(home)
        shared_grocery = pk.near(rng, "grocery", hx, 1500)
        shared_worship = pk.near(rng, "worship", hx, 2000) if rng.random() < 0.4 else []
        for _ in range(size):
            if len(agents) >= cfg.n_agents:
                break
            persona = str(rng.choice(list(PERSONAS), p=list(PERSONAS.values())))
            work = None
            if persona in ("worker", "shift"):
                work = int(rng.choice(work_pool))
            elif persona == "student":
                work = int(rng.choice(school_pool))
            anchor = pk.xy(work) if work is not None else hx
            favs = {
                "grocery": shared_grocery or pk.near(rng, "grocery", hx, 1500),
                "food": pk.near(rng, "food", hx, 1500, k=2),
                "lunch": pk.near(rng, "food", anchor, 400, k=2),
                "shop": pk.near(rng, "shop", hx, 2000, k=2),
                "park": pk.near(rng, "park", hx, 1500),
                "fuel": pk.near(rng, "fuel", hx, 2000),
                "gym": pk.near(rng, "gym", anchor, 2000) if rng.random() < 0.35 else [],
                "worship": shared_worship,
            }
            agents.append({
                "id": f"u{len(agents):05d}", "household": hh, "persona": persona,
                "home": home, "work": work, "neighbor_of": neighbor_of,
                "shift": str(rng.choice(["night", "swing"])) if persona == "shift" else None,
                "favs": {k: [int(x) for x in v] for k, v in favs.items() if v},
            })
        hh += 1
    return agents


def _friend_meetings(agents, area, cfg, rng, day0s) -> list[dict]:
    pk = _PlacePicker(area)
    n_pairs = int(cfg.friend_pairs_frac * len(agents))
    meetings = []
    if len(agents) < 2:
        return meetings
    for _ in range(n_pairs):
        a, b = rng.choice(len(agents), size=2, replace=False)
        if agents[a]["household"] == agents[b]["household"]:
            continue
        mid = np.add(pk.xy(agents[a]["home"]), pk.xy(agents[b]["home"])) / 2
        places = pk.near(rng, "food", mid, 1500, k=2) + pk.near(rng, "park", mid, 1500)
        n = rng.poisson(cfg.meet_rate_per_week * len(day0s) / 7)
        for d in rng.choice(len(day0s), size=min(n, len(day0s)), replace=False):
            weekend = datetime.fromtimestamp(day0s[d], timezone.utc).weekday() >= 5
            start = day0s[d] + (rng.uniform(12, 16) if weekend else rng.uniform(18.5, 20)) * H
            dur = rng.uniform(*DURATION_MIN["friend"]) * M
            meetings.append({"id_a": agents[a]["id"], "id_b": agents[b]["id"], "day": int(d),
                             "place_id": int(rng.choice(places)), "start": start, "end": start + dur})
    return meetings


# ------------------------------------------------------------------ routing

class _Router:
    """Travel times and road polylines between an agent's places (ENU meters)."""

    def __init__(self, area: Area, place_ids, rng, noise):
        self.area, self.rng, self.noise = area, rng, noise
        pl_ = area.places
        self.px, self.py = geo.to_enu(pl_["lat"].to_numpy(), pl_["lon"].to_numpy(), *area.center)
        self.pnode = pl_["node"].to_numpy()
        self.nx, self.ny = geo.to_enu(area.node_lat, area.node_lon, *area.center)
        nodes = np.unique(self.pnode[list(place_ids)])
        self.src = {int(n): i for i, n in enumerate(nodes)}
        self.dist, self.pred = area.paths_from(nodes)

    def _access(self, pid):
        n = self.pnode[pid]
        return np.hypot(self.px[pid] - self.nx[n], self.py[pid] - self.ny[n])

    def expected_tt(self, a, b):
        d = self.dist[self.src[int(self.pnode[a])], self.pnode[b]]
        if not np.isfinite(d):  # disconnected: straight line at 10 m/s
            d = np.hypot(self.px[a] - self.px[b], self.py[a] - self.py[b]) / 10.0
        return d + (self._access(a) + self._access(b)) / 1.4 + 90.0  # walk to/from car, parking

    def trip(self, a, b):
        """Returns (travel seconds, xs, ys, fraction-of-time at each knot)."""
        tt = self.expected_tt(a, b) * float(np.exp(self.rng.normal(0, self.noise)))
        s, tgt = self.src[int(self.pnode[a])], int(self.pnode[b])
        path = []
        if np.isfinite(self.dist[s, tgt]):
            n = tgt
            while n >= 0:
                path.append(n)
                n = self.pred[s, n]
            path.reverse()
        xs = np.r_[self.px[a], self.nx[path], self.px[b]]
        ys = np.r_[self.py[a], self.ny[path], self.py[b]]
        seg = np.hypot(np.diff(xs), np.diff(ys))
        cum = np.r_[0, np.cumsum(seg)]
        frac = cum / cum[-1] if cum[-1] > 0 else np.linspace(0, 1, len(xs))
        return tt, xs, ys, frac


class _Timeline:
    def __init__(self, router: _Router, place, t0):
        self.r = router
        self.place, self.arrive = place, t0
        self.stays: list[tuple] = []
        self.knots: list[tuple] = []

    def move(self, place, depart):
        """Leave the current place at ``depart`` (or ASAP) and travel to ``place``."""
        if place == self.place:
            return
        depart = max(depart, self.arrive + MIN_STAY_S)
        self.stays.append((self.place, self.arrive, depart))
        tt, xs, ys, frac = self.r.trip(self.place, place)
        self.knots.append((depart + frac * tt, xs, ys))
        self.place, self.arrive = place, depart + tt

    def stay_for(self, place, minutes_range, rng, depart_hint=None):
        """Travel to place (leaving at depart_hint or now) and stay a random duration."""
        self.move(place, depart_hint if depart_hint is not None else self.arrive)
        return self.arrive + rng.uniform(*minutes_range) * M

    @property
    def now(self):
        return self.arrive


# ------------------------------------------------------------------ routines

def _errand(ag, rng, cats):
    cats = [c for c in cats if c in ag["favs"]]
    if not cats:
        return None, None
    c = str(rng.choice(cats))
    return int(rng.choice(ag["favs"][c])), c


def _plan_day(tl: _Timeline, ag, d0, weekday, rng, meetings):
    home, work, p = ag["home"], ag["work"], ag["persona"]
    N = lambda mu, sd: d0 + rng.normal(mu, sd) * H  # noqa: E731

    def errands(n, cats, t_start):
        t = t_start
        for _ in range(n):
            pid, c = _errand(ag, rng, cats)
            if pid is None:
                continue
            end = tl.stay_for(pid, DURATION_MIN[c], rng, depart_hint=t)
            t = end if rng.random() < 0.5 else None  # chain directly or return home
            if t is None:
                tl.move(home, end)
                t = tl.now + rng.uniform(0.5, 2.0) * H
        tl.move(home, t if t is not None else tl.now)

    def meet(before=None):
        for m in meetings:
            if m["start"] < tl.now + 15 * M or (before is not None and m["start"] >= before):
                continue  # agent busy: meeting does not happen for this agent
            tl.move(m["place_id"], m["start"] - tl.r.expected_tt(tl.place, m["place_id"]))
            tl.move(home, m["end"])

    if p == "shift" and ag["shift"] == "night":  # works 22:00-06:00 on weekday nights
        if tl.place == work:
            tl.move(home, N(6.0, 0.2))
        if rng.random() < 0.4:
            errands(1, ["grocery", "shop", "food", "gym"], N(14, 1))
        meet(before=d0 + 20.5 * H)
        if weekday < 5:
            tl.move(work, N(21.5, 0.2))
        return

    if p == "shift" and weekday < 5:  # swing shift 14:00-22:30
        if rng.random() < 0.4:
            errands(1, ["grocery", "gym", "shop"], N(10, 1))
        tl.move(work, N(13.6, 0.15))
        tl.move(home, N(22.5, 0.2))
        return

    if weekday < 5 and p in ("worker", "student"):
        if rng.random() < 0.12 and "fuel" in ag["favs"]:
            end = tl.stay_for(ag["favs"]["fuel"][0], DURATION_MIN["fuel"], rng, depart_hint=N(7.5, 0.4))
            tl.move(work, end)
        else:
            tl.move(work, N(7.75 if p == "worker" else 7.4, 0.4))
        if p == "worker" and rng.random() < 0.3 and "lunch" in ag["favs"]:
            end = tl.stay_for(int(rng.choice(ag["favs"]["lunch"])), (25, 60), rng, depart_hint=N(12, 0.25))
            tl.move(work, end)
        leave = N(17.0 if p == "worker" else 15.0, 0.6)
        if rng.random() < 0.35:
            errands(1, ["grocery", "gym", "shop", "fuel"], leave)
        else:
            tl.move(home, leave)
    else:
        n = int(rng.integers(0, 3)) if weekday < 5 else int(rng.integers(1, 4))
        if weekday == 6 and "worship" in ag["favs"]:
            end = tl.stay_for(ag["favs"]["worship"][0], DURATION_MIN["worship"], rng, depart_hint=N(9.3, 0.3))
            tl.move(home, end)
        if p == "remote" and weekday < 5 and rng.random() < 0.5:
            errands(1, ["gym", "food", "park"], N(12, 1.5))
        errands(n, ["grocery", "shop", "park", "food", "gym", "fuel"], N(10.5, 1.5))
    if meetings:
        meet()
    elif rng.random() < 0.15:
        errands(1, ["food", "park", "shop"], N(19.5, 0.8))


def simulate(area: Area, cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    agents = _build_agents(area, cfg, rng)
    start = datetime.fromisoformat(cfg.start_date).replace(tzinfo=timezone.utc).timestamp()
    # local midnight of each day expressed in UTC epoch seconds
    day0s = start - cfg.utc_offset_h * H + np.arange(cfg.n_days) * 24 * H
    meetings = _friend_meetings(agents, area, cfg, rng, day0s)
    t_end = day0s[-1] + 24 * H

    stays, knots = [], {}
    for ag in agents:
        mine = [m for m in meetings if ag["id"] in (m["id_a"], m["id_b"])]
        places = {ag["home"], *( [ag["work"]] if ag["work"] is not None else [] ),
                  *(p for v in ag["favs"].values() for p in v), *(m["place_id"] for m in mine)}
        tl = _Timeline(_Router(area, places, rng, cfg.travel_noise), ag["home"], day0s[0])
        for d, d0 in enumerate(day0s):
            wd = datetime.fromtimestamp(d0 + 12 * H, timezone.utc).weekday()
            _plan_day(tl, ag, d0, wd, rng, sorted([m for m in mine if m["day"] == d], key=lambda m: m["start"]))
        tl.stays.append((tl.place, tl.arrive, max(t_end, tl.arrive + MIN_STAY_S)))
        # stays -> knots: each stay contributes its two endpoints at the place center
        T, X, Y = [], [], []
        r = tl.r
        for pid, a, d in tl.stays:
            T += [a, d]
            X += [r.px[pid]] * 2
            Y += [r.py[pid]] * 2
        for t, xs, ys in tl.knots:
            T += list(t[1:-1])
            X += list(xs[1:-1])
            Y += list(ys[1:-1])
        o = np.argsort(T, kind="stable")
        knots[ag["id"]] = (np.asarray(T)[o], np.asarray(X)[o], np.asarray(Y)[o])
        for i, (pid, a, d) in enumerate(tl.stays):
            stays.append((ag["id"], i, int(pid), float(a), float(d)))

    ag_df = pl.DataFrame([{k: v for k, v in a.items() if k != "favs"} for a in agents],
                         schema_overrides={"work": pl.Int64, "shift": pl.String})
    st_df = pl.DataFrame(stays, schema=["id", "dwell_id", "place_id", "arrive", "depart"], orient="row")
    mt_df = pl.DataFrame(meetings, schema={"id_a": pl.String, "id_b": pl.String, "day": pl.Int64,
                                           "place_id": pl.Int64, "start": pl.Float64, "end": pl.Float64})
    return SimResult(cfg, ag_df, st_df, mt_df, knots, float(day0s[0]), float(t_end))
