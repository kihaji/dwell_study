"""Plotly figures for the app (all hour/day breakdowns in UTC)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from data import OTHER, place_colors, place_label
from dwells.pol.stats import hour_overlap

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOURS = [f"{h:02d}" for h in range(24)]
LAYOUT = dict(margin=dict(l=40, r=10, t=36, b=30), template="plotly_white", font=dict(size=11))


def _empty(msg: str) -> go.Figure:
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    f.update_layout(**LAYOUT, xaxis_visible=False, yaxis_visible=False)
    return f


def radar(t, place_id, visit_id) -> go.Figure:
    p = t["places"].filter(pl.col("place_id") == place_id)
    if p.height == 0:
        return _empty("Select a place")
    r = p.row(0, named=True)
    h = np.asarray(r["hour_presence_s"]) / 3600
    ev = np.asarray(r["hour_events"], float)
    f = go.Figure()
    f.add_trace(go.Scatterpolar(r=np.r_[h, h[0]], theta=HOURS + [HOURS[0]], fill="toself", name="hours present",
                                line_color=place_colors(t["places"]).get(place_id, OTHER)))
    f.add_trace(go.Scatterpolar(r=np.r_[ev, ev[0]] * (h.max() / max(ev.max(), 1)), theta=HOURS + [HOURS[0]],
                                name="reports (scaled)", line=dict(color="#555", width=1)))
    sel = t["visits"].filter(pl.col("visit_id") == visit_id) if visit_id is not None else None
    if sel is not None and sel.height and sel["place_id"][0] == place_id:
        vis = hour_overlap(sel)
        if vis.height:
            vh = np.zeros(24)
            for row in vis.iter_rows(named=True):
                vh[row["hour"]] += row["seconds"] / 3600
            f.add_trace(go.Scatterpolar(r=np.r_[vh, vh[0]] * (h.max() / max(vh.max(), 1e-9)),
                                        theta=HOURS + [HOURS[0]], name="selected visit (scaled)",
                                        line=dict(dash="dash", color="black")))
    f.update_layout(**LAYOUT, title=f"When at {place_label(r)} (UTC hour)", legend=dict(orientation="h", y=-0.1),
                    polar=dict(angularaxis=dict(direction="clockwise", rotation=90)))
    return f


def weekday_heatmap(t, top: int = 10) -> go.Figure:
    p = t["places"].sort("rank_duration").head(top)
    if p.height == 0:
        return _empty("No places")
    z = np.array([r for r in p["weekday_visits"].to_list()])
    f = go.Figure(go.Heatmap(z=z, x=DAYS, y=[place_label(r) for r in p.iter_rows(named=True)], colorscale="Blues",
                             hovertemplate="%{y}<br>%{x}: %{z} visits<extra></extra>"))
    f.update_layout(**LAYOUT, title="Visits per day of week (UTC)", yaxis=dict(autorange="reversed"))
    return f


def weekday_hour_heatmap(t, place_id) -> go.Figure:
    p = t["places"].filter(pl.col("place_id") == place_id)
    if p.height == 0:
        return _empty("Select a place")
    z = np.asarray(p["weekday_hour_presence_s"][0].to_list()).reshape(7, 24) / 3600
    f = go.Figure(go.Heatmap(z=z, x=HOURS, y=DAYS, colorscale="YlOrRd",
                             hovertemplate="%{y} %{x}:00 UTC<br>%{z:.1f} h total<extra></extra>"))
    f.update_layout(**LAYOUT, title="Hours present by weekday and hour (UTC)", yaxis=dict(autorange="reversed"))
    return f


def timeline(t, day_range=None) -> go.Figure:
    v = t["visits"]
    if v.height == 0:
        return _empty("No visits")
    colors = place_colors(t["places"])
    labels = {r["place_id"]: place_label(r) for r in t["places"].iter_rows(named=True)}
    rows = []
    for r in v.iter_rows(named=True):  # split visits at midnight so each day is one row
        s, e = r["start_ts"], r["end_ts"]
        while s < e:
            day_end = (s + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            seg_end = min(e, day_end)
            rows.append((s.date(), s.hour + s.minute / 60, (seg_end - s).total_seconds() / 3600, r["place_id"], r["visit_id"]))
            s = seg_end
    df = pl.DataFrame(rows, schema=["day", "h0", "len", "place_id", "visit_id"], orient="row")
    if day_range:
        df = df.filter(pl.col("day").is_between(*day_range))
    f = go.Figure()
    for pid, g in df.group_by("place_id"):
        pid = pid[0]
        f.add_trace(go.Bar(x=g["len"], base=g["h0"], y=g["day"].cast(pl.String), orientation="h",
                           marker_color=colors.get(pid, OTHER), name=labels.get(pid, str(pid)),
                           customdata=g.select("place_id", "visit_id").to_numpy(),
                           hovertemplate="%{y} %{base:.1f}h +%{x:.1f}h<extra>" + labels.get(pid, "") + "</extra>"))
    f.update_layout(**LAYOUT, barmode="overlay", title="Daily timeline (UTC)", xaxis=dict(range=[0, 24], dtick=3, title="hour"),
                    yaxis=dict(autorange="reversed", type="category"), legend=dict(orientation="h", y=-0.15),
                    height=max(300, 18 * df["day"].n_unique() + 120))
    return f


def presence_calendar(t, day_range=None) -> go.Figure:
    pr = t["presence"]
    if pr.height == 0:
        return _empty("No presence")
    if day_range:
        pr = pr.filter(pl.col("date").is_between(*day_range))
    dom = pr.sort("seconds", descending=True).unique(["date", "hour"], keep="first")
    ranks = {r["place_id"]: r["rank_duration"] for r in t["places"].iter_rows(named=True)}
    days = sorted(dom["date"].unique().to_list())
    z = np.full((len(days), 24), np.nan)
    txt = np.full((len(days), 24), "", dtype=object)
    labels = {r["place_id"]: place_label(r) for r in t["places"].iter_rows(named=True)}
    di = {d: i for i, d in enumerate(days)}
    for r in dom.iter_rows(named=True):
        rk = ranks.get(r["place_id"], 11)
        z[di[r["date"]], r["hour"]] = min(rk, 11)
        txt[di[r["date"]], r["hour"]] = labels.get(r["place_id"], "")
    colors = [c for c in place_colors(t["places"].sort("rank_duration").head(10)).values()] + [OTHER]
    scale = [[i / 10, c] for i, c in enumerate(colors)]
    f = go.Figure(go.Heatmap(z=z, x=HOURS, y=[str(d) for d in days], colorscale=scale, zmin=1, zmax=11, showscale=False,
                             text=txt, hovertemplate="%{y} %{x}:00 UTC<br>%{text}<extra></extra>"))
    f.update_layout(**LAYOUT, title="Presence calendar (dominant place per hour, UTC)", yaxis=dict(autorange="reversed"),
                    height=max(300, 14 * len(days) + 120))
    return f


def quality_figs(t, events_all) -> tuple[go.Figure, go.Figure]:
    ev = events_all
    if ev.height == 0:
        return _empty("No events"), _empty("No events")
    per_day = ev.group_by(pl.col("timestamp").dt.date().alias("day")).agg(pl.len(), pl.col("visit_id").is_not_null().sum().alias("in_visits")).sort("day")
    f1 = go.Figure([go.Bar(x=per_day["day"], y=per_day["len"], name="all reports", marker_color="#90a4ae"),
                    go.Bar(x=per_day["day"], y=per_day["in_visits"], name="in visits", marker_color="#1565c0")])
    f1.update_layout(**LAYOUT, barmode="overlay", title="Reports per day (UTC)", legend=dict(orientation="h", y=-0.2))
    s = ev["sigma_m"].to_numpy()
    f2 = go.Figure(go.Histogram(x=np.log10(np.maximum(s, 0.1)), nbinsx=40, marker_color="#6a1b9a"))
    f2.update_layout(**LAYOUT, title="Reported accuracy (1-sigma)", xaxis=dict(
        title="meters", tickvals=[0, 1, 2, 3, 4], ticktext=["1", "10", "100", "1 km", "10 km"]))
    return f1, f2
