"""Framework-independent view logic shared by the Dash app and the marimo notebook:
table rows, place summary, quality stats and selection handling (all times UTC)."""

from __future__ import annotations

import polars as pl

from data import place_label

RANK_COLS = {"total time": "rank_duration", "pattern confidence": "rank_score", "repeat visits": "rank_visits",
             "distinct days": None}
NO_SELECTION = {"place_id": None, "visit_id": None}


def fmt_dur(s):
    if s is None:
        return ""
    h, m = divmod(int(s) // 60, 60)
    return f"{h}h{m:02d}" if h else f"{m}m"


def fmt_ts(t):
    return t.strftime("%Y-%m-%d %H:%M") if t is not None else ""


def labels(t) -> dict[int, str]:
    return {r["place_id"]: place_label(r) for r in t["places"].iter_rows(named=True)}


# ------------------------------------------------------------------ selection

def default_selection(t) -> dict:
    """The identifier's top place by total time (nothing when it has no places)."""
    p = t["places"]
    return {"place_id": int(p.sort("rank_duration")["place_id"][0]), "visit_id": None} if p.height else dict(NO_SELECTION)


def selection_from_pick(obj: dict | None) -> dict | None:
    """Selection from a picked map object (an event, hull, ellipse, place or arc); None when
    the object belongs to no place."""
    if not obj or obj.get("place_id") is None:
        return None
    return {"place_id": int(obj["place_id"]), "visit_id": obj.get("visit_id")}


# ------------------------------------------------------------------ tables

def rank_rows(t, by: str) -> list[dict]:
    p = t["places"]
    col = RANK_COLS[by]
    p = p.sort(col) if col else p.sort(["n_days", "total_dwell_s"], descending=True)
    return [{"#": i + 1, "place": place_label(r), "total time": fmt_dur(r["total_dwell_s"]),
             "visits": r["n_visits"], "days": r["n_days"], "confidence": round(r["score"]), "place_id": r["place_id"]}
            for i, r in enumerate(p.head(10).iter_rows(named=True))]


def places_rows(t) -> list[dict]:
    return [{"place": place_label(r), "label": r["label"] or "", "address": r["address"] or "",
             "visits": r["n_visits"], "days": r["n_days"], "total time": fmt_dur(r["total_dwell_s"]),
             "median visit": fmt_dur(r["median_dwell_s"]), "time share %": round(100 * r["time_share"], 1),
             "reports": r["n_events"], "confidence": round(r["score"]), "evidence": round(r["score_evidence"], 2),
             "precision": round(r["score_precision"], 2), "regularity": round(r["score_regularity"], 2),
             "center ±m (1σ)": round(r["sigma_major"], 1), "extent m": round(r["extent_m"] or 0, 1),
             "first seen": fmt_ts(r["first_seen"]), "last seen": fmt_ts(r["last_seen"]), "place_id": r["place_id"]}
            for r in t["places"].sort("rank_duration").iter_rows(named=True)]


def visits_rows(t, place_id, all_places: bool = False) -> list[dict]:
    lab = labels(t)
    v = t["visits"] if all_places else t["visits"].filter(pl.col("place_id").eq_missing(place_id))
    return [{"start": fmt_ts(r["start_ts"]), "end": fmt_ts(r["end_ts"]), "duration": fmt_dur(r["duration_s"]),
             "arrived between": f"{fmt_ts(r['arrive_after'])} – {fmt_ts(r['arrive_by'])}",
             "left between": f"{fmt_ts(r['depart_after'])} – {fmt_ts(r['depart_by'])}",
             "reports": r["n_events"], "precise": r["n_informative"], "confidence": round(r["confidence"]),
             "center ±m": round(r["sigma_major"] or 0, 1), "from": lab.get(r["prev_place_id"], ""),
             "to": lab.get(r["next_place_id"], ""), "flags": ", ".join(r["anomalies"] or []),
             "place": lab.get(r["place_id"], ""), "place_id": r["place_id"], "visit_id": r["visit_id"]}
            for r in v.sort("start_ts", descending=True).iter_rows(named=True)]


def next_rows(t, place_id) -> list[dict]:
    lab = labels(t)
    nx = t["next_places"].filter(pl.col("from_place").eq_missing(place_id))
    return [{"rank": r["rank"], "next place": lab.get(r["to_place"], str(r["to_place"])),
             "probability": f"{r['prob']:.0%}", "at least": f"{r['prob_low']:.0%}", "departures seen": r["n_from"],
             "place_id": r["to_place"]}
            for r in nx.iter_rows(named=True)]


def anomaly_rows(t, min_severity: float) -> list[dict]:
    lab = labels(t)
    a = t["anomalies"].filter(pl.col("severity") >= min_severity).join(
        t["visits"].select("visit_id", "start_ts", "duration_s"), on="visit_id").sort("severity", descending=True)
    return [{"when": fmt_ts(r["start_ts"]), "place": lab.get(r["place_id"], ""), "kind": r["kind"].replace("_", " "),
             "detail": r["detail"], "severity": round(r["severity"], 2), "place_id": r["place_id"], "visit_id": r["visit_id"]}
            for r in a.iter_rows(named=True)]


def quality_items(t) -> list[tuple[str, str]]:
    q = t["quality"].row(0, named=True) if t["quality"].height else {}
    return [("reports", f"{q.get('n_events', 0):,}"), ("active days", str(q.get("active_days"))),
            ("reports / active day", f"{q.get('events_per_active_day', 0):.0f}"),
            ("hours with data", f"{100 * (q.get('hour_coverage') or 0):.0f}%"),
            ("typical gap", fmt_dur(q.get("gap_p50_s"))), ("95th pct gap", fmt_dur(q.get("gap_p95_s"))),
            ("typical accuracy", f"{q.get('sigma_p50_m', 0):.0f} m"), ("coarse reports", f"{100 * q.get('coarse_frac', 0):.0f}%"),
            ("duplicates", f"{100 * q.get('duplicate_frac', 0):.1f}%"), ("repeated positions", f"{100 * q.get('stale_frac', 0):.1f}%"),
            ("implausible jumps", f"{100 * q.get('speed_outlier_frac', 0):.1f}%"),
            ("reports in visits", f"{100 * q.get('events_in_visits_frac', 0):.0f}%"),
            ("source labelled", "n/a" if q.get("source_coverage") is None else f"{100 * q['source_coverage']:.0f}%"),
            ("label vs accuracy conflicts", "n/a" if q.get("source_mismatch_frac") is None
             else f"{100 * q['source_mismatch_frac']:.1f}%")]


def place_summary(t, place_id) -> dict | None:
    """Text of the selected-place card (None when nothing is selected)."""
    p = t["places"].filter(pl.col("place_id").eq_missing(place_id))
    if p.height == 0:
        return None
    r = p.row(0, named=True)
    return {
        "title": place_label(r),
        "address": r["address"] or "no address found nearby",
        "match": f"{r['place_category'] or ''}  (matched {r['geocode_distance_m']:.0f} m away)" if r["geocode_distance_m"] else "",
        "score": f"Pattern confidence {r['score']:.0f}/100",
        "components": (f"evidence {r['score_evidence']:.2f}", f"precision {r['score_precision']:.2f}",
                       f"regularity {r['score_regularity']:.2f}"),
        "stats": (f"{r['n_visits']} visits on {r['n_days']} days · {fmt_dur(r['total_dwell_s'])} total · "
                  f"median visit {fmt_dur(r['median_dwell_s'])} · {100 * r['time_share']:.0f}% of dwell time"),
        "extent": (f"center ±{r['sigma_major']:.0f} m (1σ) · activity extent {r['extent_m'] or 0:.0f} m · "
                   f"first {fmt_ts(r['first_seen'])} · last {fmt_ts(r['last_seen'])}"),
    }


def selection_from_points(points: list[dict] | None, fig) -> dict | None:
    """Selection from clicked Plotly points whose trace customdata is [place_id(, visit_id)]
    (movement graph nodes, timeline bars)."""
    for p in points or []:
        cd = p.get("customdata")
        if cd is None and p.get("curveNumber") is not None and p.get("pointIndex") is not None:
            trace_cd = fig.data[p["curveNumber"]].customdata
            cd = trace_cd[p["pointIndex"]] if trace_cd is not None else None
        if cd is not None and len(cd) and cd[0] is not None:
            return {"place_id": int(cd[0]), "visit_id": int(cd[1]) if len(cd) > 1 and cd[1] is not None else None}
    return None
