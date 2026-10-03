"""Pattern-of-life statistics built from visits (all times UTC).

Definitions (plain-language versions are in docs/analyst_guide.md):

* presence: seconds of visit time falling in each UTC hour (visits are spread over the
  clock hours they span, using estimated start/end).
* regularity (0-1): mean of
    - day coverage   = distinct days with a visit / days the id was observed at all
    - hour focus     = 1 - (entropy of the place's hour-of-day presence / ln 24)
                       (1 = always the same hours, 0 = spread evenly over the day)
    - repeat         = 1 - exp(-(visits - 1) / 3)  (0 for one visit, ~0.5 at 3, ~0.95 at 10)
* evidence (0-1): 0.5 x (1 - exp(-informative events / 30)) + 0.5 x (1 - exp(-hours / 24))
* precision (0-1): 1 / (1 + center 1-sigma / 50 m)
* score (0-100): 100 x (0.35 evidence + 0.15 precision + 0.50 regularity)
* next-place probability: share of departures from a place that went to each other
  place; prob_low is the Wilson 95% lower bound (a cautious estimate that is low when
  based on few departures).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from dwells import geo
from dwells.downstream.places import ONE_SIGMA_P
from dwells.uncertainty import cov_to_ellipse, ellipse_to_cov

Z95 = norm.ppf(0.975)


def wilson_low(k: np.ndarray, n: np.ndarray, z: float = Z95) -> np.ndarray:
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z**2 / np.maximum(n, 1)
    center = p + z**2 / (2 * np.maximum(n, 1))
    half = z * np.sqrt(p * (1 - p) / np.maximum(n, 1) + z**2 / (4 * np.maximum(n, 1) ** 2))
    return np.where(n > 0, np.maximum((center - half) / den, 0.0), 0.0)


def hour_overlap(visits: pl.DataFrame) -> pl.DataFrame:
    """One row per (visit, clock hour) with the seconds of the visit inside that hour."""
    x = visits.select("id", "visit_id", "place_id", "start_ts", "end_ts")
    x = x.with_columns(h=pl.datetime_ranges(pl.col("start_ts").dt.truncate("1h"), pl.col("end_ts"), "1h")).explode(
        "h", empty_as_null=True)  # ranges are never empty (start <= end); explicit for Polars 2.0
    x = x.with_columns(
        seconds=(pl.min_horizontal(pl.col("end_ts"), pl.col("h") + pl.duration(hours=1))
                 - pl.max_horizontal(pl.col("start_ts"), pl.col("h"))).dt.total_seconds().clip(0).cast(pl.Float64),
        date=pl.col("h").dt.date(), hour=pl.col("h").dt.hour().cast(pl.Int8),
        weekday=(pl.col("h").dt.weekday() - 1).cast(pl.Int8),
    )
    return x.filter(pl.col("seconds") > 0)


def _fixed_list(df: pl.DataFrame, key: str, n: int, value: str, name: str, dtype) -> pl.DataFrame:
    """(id, place_id, key, value) -> (id, place_id, name: list of length n)."""
    full = df.group_by("id", "place_id", key).agg(pl.col(value).sum())
    rows = {}
    for r in full.iter_rows(named=True):
        rows.setdefault((r["id"], r["place_id"]), [0] * n)[int(r[key])] = r[value]
    return pl.DataFrame({"id": [k[0] for k in rows], "place_id": [k[1] for k in rows],
                         name: [v for v in rows.values()]}, schema={"id": pl.String, "place_id": pl.Int64,
                                                                    name: pl.List(dtype)})


def place_table(visits: pl.DataFrame, events: pl.DataFrame, presence: pl.DataFrame,
                active_days: pl.DataFrame) -> pl.DataFrame:
    """Aggregate visits into places with profiles, score components and ranks."""
    base = visits.group_by("id", "place_id").agg(
        pl.len().cast(pl.Int64).alias("n_visits"),
        pl.col("start_ts").dt.date().n_unique().cast(pl.Int64).alias("n_days"),
        pl.col("n_events").sum(), pl.col("n_informative").sum().alias("n_informative"),
        pl.col("duration_s").sum().alias("total_dwell_s"), pl.col("duration_s").median().alias("median_dwell_s"),
        pl.col("start_ts").min().alias("first_seen"), pl.col("end_ts").max().alias("last_seen"),
    )
    base = base.join(_place_centers(visits), on=["id", "place_id"])
    base = base.join(_place_extent(events, base), on=["id", "place_id"], how="left")
    hp = _fixed_list(presence, "hour", 24, "seconds", "hour_presence_s", pl.Float64)
    wh = _fixed_list(presence.with_columns(wh=pl.col("weekday").cast(pl.Int32) * 24 + pl.col("hour")),
                     "wh", 168, "seconds", "weekday_hour_presence_s", pl.Float64)
    he = _fixed_list(events.filter(pl.col("place_id").is_not_null()).with_columns(
        hour=pl.col("timestamp").dt.hour(), one=pl.lit(1)), "hour", 24, "one", "hour_events", pl.Int64)
    wv = _fixed_list(visits.with_columns(one=pl.lit(1)), "weekday", 7, "one", "weekday_visits", pl.Int64)
    p = base
    for t in (hp, wh, he, wv):
        p = p.join(t, on=["id", "place_id"], how="left")
    p = p.join(active_days, on="id", how="left")

    hours = p["hour_presence_s"].to_list()
    focus = []
    for h in hours:
        h = np.asarray(h, float)
        tot = h.sum()
        q = h[h > 0] / tot if tot > 0 else np.array([1.0])
        focus.append(float(1 + (q * np.log(q)).sum() / np.log(24)))
    p = p.with_columns(
        hour_focus=pl.Series(focus),
        day_coverage=(pl.col("n_days") / pl.col("active_days").clip(1)).clip(0, 1),
        repeat=1 - (-(pl.col("n_visits") - 1) / 3).exp(),
    ).with_columns(regularity=(pl.col("hour_focus") + pl.col("day_coverage") + pl.col("repeat")) / 3)
    p = p.with_columns(
        score_evidence=0.5 * (1 - (-pl.col("n_informative") / 30).exp())
        + 0.5 * (1 - (-pl.col("total_dwell_s") / 3600 / 24).exp()),
        score_precision=1 / (1 + pl.col("sigma_major") / 50),
        score_regularity=pl.col("regularity"),
        time_share=pl.col("total_dwell_s") / pl.col("total_dwell_s").sum().over("id"),
    ).with_columns(score=100 * (0.35 * pl.col("score_evidence") + 0.15 * pl.col("score_precision")
                                + 0.5 * pl.col("score_regularity")))
    p = p.sort("id", "total_dwell_s", descending=[False, True]).with_columns(
        rank_duration=pl.int_range(1, pl.len() + 1).over("id"))
    p = p.sort("id", "score", descending=[False, True]).with_columns(rank_score=pl.int_range(1, pl.len() + 1).over("id"))
    p = p.sort(["id", "n_visits", "total_dwell_s"], descending=[False, True, True]).with_columns(
        rank_visits=pl.int_range(1, pl.len() + 1).over("id"))
    # anchors: most time (primary) and second most time on >= 3 distinct days (secondary);
    # usually home and work, but no local-time inference is made (all times are UTC)
    p = p.with_columns(
        label=pl.when(pl.col("rank_duration") == 1).then(pl.lit("primary"))
        .when((pl.col("rank_duration") == 2) & (pl.col("n_days") >= 3)).then(pl.lit("secondary"))
        .otherwise(pl.lit(None, dtype=pl.String)))
    return p.sort("id", "rank_duration")


def _place_centers(visits: pl.DataFrame) -> pl.DataFrame:
    """Place center = information-weighted mean of visit centroids; 1-sigma combines the
    fused visit uncertainty with the scatter of visit centroids around the center."""
    rows = []
    for (uid, pid), g in visits.group_by("id", "place_id"):
        lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
        lat0, lon0 = float(lat.mean()), float(lon.mean())
        x, y = geo.to_enu(lat, lon, lat0, lon0)
        a = g["sigma_major"].fill_null(25.0).to_numpy()
        b = g["sigma_minor"].fill_null(25.0).to_numpy()
        az = g["azimuth"].fill_null(0.0).to_numpy()
        cee, cnn, cen = ellipse_to_cov(np.maximum(a, 1.0), np.maximum(b, 1.0), az, ONE_SIGMA_P)
        det = cee * cnn - cen**2
        iee, inn, ien = cnn / det, cee / det, -cen / det
        Iee, Inn, Ien = iee.sum(), inn.sum(), ien.sum()
        D = Iee * Inn - Ien**2
        fee, fnn, fen = Inn / D, Iee / D, -Ien / D
        bx, by = (iee * x + ien * y).sum(), (ien * x + inn * y).sum()
        mx, my = fee * bx + fen * by, fen * bx + fnn * by
        n = len(x)
        sx, sy = (np.var(x) / n, np.var(y) / n) if n > 1 else (0.0, 0.0)
        sa, sb, saz = cov_to_ellipse(fee + sx, fnn + sy, fen, ONE_SIGMA_P)
        clat, clon = geo.from_enu(mx, my, lat0, lon0)
        rows.append((uid, pid, float(clat), float(clon), float(sa), float(sb), float(saz)))
    return pl.DataFrame(rows, schema={"id": pl.String, "place_id": pl.Int64, "lat": pl.Float64, "lon": pl.Float64,
                                      "sigma_major": pl.Float64, "sigma_minor": pl.Float64, "azimuth": pl.Float64},
                        orient="row")


def _place_extent(events: pl.DataFrame, places: pl.DataFrame) -> pl.DataFrame:
    """RMS distance of informative member events to the place center (activity extent)."""
    e = events.filter(pl.col("place_id").is_not_null() & ~pl.col("coarse")).join(
        places.select("id", "place_id", pl.col("lat").alias("clat"), pl.col("lon").alias("clon")), on=["id", "place_id"])
    d = geo.haversine_m(e["lat"].to_numpy(), e["lon"].to_numpy(), e["clat"].to_numpy(), e["clon"].to_numpy())
    return e.with_columns(d2=pl.Series(d**2)).group_by("id", "place_id").agg(
        pl.col("d2").mean().sqrt().alias("extent_m"))


def transitions(visits: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Place-to-place moves (consecutive visits at different places) and top-5 next places."""
    v = visits.sort("id", "start_ts").with_columns(
        to_place=pl.col("place_id").shift(-1).over("id"),
        arrive=pl.col("start_ts").shift(-1).over("id"),
    ).filter(pl.col("to_place").is_not_null() & (pl.col("to_place") != pl.col("place_id")))
    t = v.group_by("id", pl.col("place_id").alias("from_place"), "to_place").agg(
        pl.len().alias("n"),
        (pl.col("arrive") - pl.col("end_ts")).dt.total_seconds().median().cast(pl.Float64).alias("median_travel_s"),
        pl.col("end_ts").dt.hour().mode().first().cast(pl.Int8).alias("typical_depart_hour"),
        pl.col("end_ts").max().alias("last_ts"),
    ).with_columns(n_from=pl.col("n").sum().over("id", "from_place"))
    t = t.with_columns(prob=pl.col("n") / pl.col("n_from"),
                       prob_low=pl.Series(wilson_low(t["n"].to_numpy(), t["n_from"].to_numpy())))
    nxt = t.sort(["id", "from_place", "prob", "n"], descending=[False, False, True, True]).with_columns(
        rank=pl.int_range(1, pl.len() + 1).over("id", "from_place")).filter(pl.col("rank") <= 5)
    return t.sort("id", "from_place", "prob", descending=[False, False, True]), nxt


def anomalies(visits: pl.DataFrame, places: pl.DataFrame, active_days: pl.DataFrame) -> pl.DataFrame:
    """Visits that break the routine of their place / identifier."""
    out = []
    pinfo = {(r["id"], r["place_id"]): r for r in places.select(
        "id", "place_id", "n_visits", "weekday_visits").iter_rows(named=True)}
    days = dict(active_days.iter_rows())
    logdur = visits.with_columns(ld=pl.col("duration_s").clip(60).log()).group_by("id", "place_id").agg(
        pl.col("ld").median().alias("ld_med"), (pl.col("ld") - pl.col("ld").median()).abs().median().alias("ld_mad"))
    stats = {(r["id"], r["place_id"]): r for r in logdur.iter_rows(named=True)}
    starts: dict[tuple, list] = {}
    for r in visits.select("id", "place_id", "visit_id", "start_hour").iter_rows(named=True):
        starts.setdefault((r["id"], r["place_id"]), []).append((r["visit_id"], r["start_hour"]))
    for v in visits.iter_rows(named=True):
        key = (v["id"], v["place_id"])
        p, st = pinfo[key], stats[key]
        if p["n_visits"] == 1 and days.get(v["id"], 0) >= 7:
            out.append((v["id"], v["visit_id"], v["place_id"], "rare_place",
                        "only visit to this place in an established routine", 0.5))
        if p["n_visits"] >= 5:
            # arrival hour compared with the other visits' arrival hours (circular, +-1 h)
            others = [h for vid, h in starts[key] if vid != v["visit_id"]]
            near = sum(min(abs(h - v["start_hour"]), 24 - abs(h - v["start_hour"])) <= 1 for h in others)
            frac = near / len(others)
            if frac < 0.10:
                out.append((v["id"], v["visit_id"], v["place_id"], "unusual_hour",
                            f"arrived {v['start_hour']:02d}:00 UTC; {frac:.0%} of other visits arrived within 1 h of that",
                            float(1 - frac / 0.10)))
            mad = (st["ld_mad"] or 0) * 1.4826
            if mad > 0 and p["n_visits"] >= 6:
                z = abs(np.log(max(v["duration_s"], 60)) - st["ld_med"]) / mad
                if z > 3.5:
                    kind = "unusually_long" if np.log(max(v["duration_s"], 60)) > st["ld_med"] else "unusually_short"
                    out.append((v["id"], v["visit_id"], v["place_id"], kind,
                                f"{v['duration_s'] / 3600:.1f} h vs typical {np.exp(st['ld_med']) / 3600:.1f} h",
                                float(min(1.0, z / 6))))
        if p["n_visits"] >= 8 and p["weekday_visits"][v["weekday"]] == 1:
            out.append((v["id"], v["visit_id"], v["place_id"], "unusual_weekday",
                        "only visit to this place on this weekday", 0.5))
    return pl.DataFrame(out, schema={"id": pl.String, "visit_id": pl.Int64, "place_id": pl.Int64, "kind": pl.String,
                                     "detail": pl.String, "severity": pl.Float64}, orient="row")
