"""Pattern-of-life analyst app (Plotly Dash + deckgl-dash + dash-cytoscape).

    uv run python -m dwells.pol.build          # precompute outputs (once per dataset)
    uv run --extra app python app/app.py       # http://127.0.0.1:8050

Select a dataset and identifier; click a place on the map, in the rankings, the graph,
the places table or the timeline to focus every view on it. Times are UTC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import dash_cytoscape as cyto
import polars as pl
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from deckgl_dash import DeckGL

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
import figures  # noqa: E402
import layers  # noqa: E402

from dwells.pol.help import HELP  # noqa: E402
from dwells.presets import params_for  # noqa: E402

DS = data.datasets()
CARD = {"border": "1px solid #ddd", "borderRadius": "6px", "padding": "8px", "background": "white"}
TABLE_STYLE = dict(style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "padding": "3px", "textAlign": "left"},
                   style_header={"fontWeight": "bold", "background": "#f5f5f5"}, page_size=10)
RANK_COLS = {"total time": "rank_duration", "pattern confidence": "rank_score", "repeat visits": "rank_visits",
             "distinct days": None}


def info(key: str):
    return html.Abbr(" ⓘ", title=HELP[key], style={"cursor": "help", "textDecoration": "none", "color": "#1565c0"})


def title(text: str, key: str):
    return html.H4([text, info(key)], style={"margin": "4px 0"})


def card(*children, style=None):
    return html.Div(children, style=CARD | (style or {}))


def fmt_dur(s):
    if s is None:
        return ""
    h, m = divmod(int(s) // 60, 60)
    return f"{h}h{m:02d}" if h else f"{m}m"


def fmt_ts(t):
    return t.strftime("%Y-%m-%d %H:%M") if t is not None else ""


app = Dash(__name__, title="Pattern of life")
app.layout = html.Div(style={"fontFamily": "sans-serif", "background": "#fafafa", "padding": "10px"}, children=[
    dcc.Store(id="sel", data={"place_id": None, "visit_id": None}),
    dcc.Store(id="live", data=None),
    html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "end"}, children=[
        html.H3(["Pattern of life", info("utc")], style={"margin": "0 12px 0 0"}),
        html.Div([html.Label("Dataset"), dcc.Dropdown(DS, DS[0] if DS else None, id="ds", clearable=False)],
                 style={"width": "220px"}),
        html.Div([html.Label("Identifier"), dcc.Dropdown(id="uid", clearable=False)], style={"width": "300px"}),
        html.Div([html.Label("Dates (UTC)"), dcc.DatePickerRange(id="days", display_format="YYYY-MM-DD")]),
        html.Div([html.Label(["Map overlays", info("map")]),
                  dcc.Checklist(id="extent", value=["ellipse", "hull"], inline=True, options=[
                      {"label": html.Span([" uncertainty ellipse", info("uncertainty_ellipse")]), "value": "ellipse"},
                      {"label": html.Span([" convex hull", info("convex_hull")]), "value": "hull"},
                      {"label": html.Span([" member events + links", info("member_events")]), "value": "member"}])]),
        html.Div(id="live-banner", style={"color": "#b71c1c", "fontWeight": "bold"}),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "3fr 2fr", "gap": "10px", "marginTop": "8px"}, children=[
        card(title("Map", "map"), html.Div(id="map-wrap", style={"height": "560px", "position": "relative"}),
             html.Div([info("next_place_arcs"), " arcs show likely next places from the selected place · map © ",
                       html.A("OpenStreetMap contributors", href="https://www.openstreetmap.org/copyright", target="_blank")],
                      style={"fontSize": 11, "color": "#555"})),
        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "10px"}, children=[
            card(title("Selected place", "place"), html.Div(id="place-card")),
            card(title("When there (radar)", "radar"), dcc.Graph(id="radar", style={"height": "330px"})),
        ]),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[
        card(title("Top 10 places", "rankings"),
             dcc.RadioItems(id="rank-by", options=list(RANK_COLS), value="total time", inline=True),
             dash_table.DataTable(id="rank-table", **TABLE_STYLE)),
        card(title("Movement graph", "graph"),
             cyto.Cytoscape(id="graph", layout={"name": "cose", "animate": False}, style={"height": "360px"},
                            stylesheet=[
                                {"selector": "node", "style": {"label": "data(label)", "background-color": "data(color)",
                                                               "width": "data(size)", "height": "data(size)",
                                                               "font-size": 9, "text-wrap": "wrap", "text-max-width": 90}},
                                {"selector": "edge", "style": {"curve-style": "bezier", "target-arrow-shape": "triangle",
                                                               "width": "data(w)", "label": "data(label)", "font-size": 8,
                                                               "line-color": "#999", "target-arrow-color": "#999"}},
                                {"selector": ".selected", "style": {"border-width": 4, "border-color": "#000"}}])),
    ]),
    card(title("Places", "places_table"), dash_table.DataTable(id="places-table", sort_action="native",
                                                                 filter_action="native", **TABLE_STYLE),
         style={"marginTop": "10px"}),
    html.Div(style={"display": "grid", "gridTemplateColumns": "3fr 2fr", "gap": "10px", "marginTop": "10px"}, children=[
        card(title("Visits", "visits_table"), dcc.Checklist(id="visits-all", options=[{"label": " all places", "value": "all"}],
                                                            value=[]),
             dash_table.DataTable(id="visits-table", sort_action="native", **TABLE_STYLE)),
        card(title("Next places from selected", "next_places"), dash_table.DataTable(id="next-table", **TABLE_STYLE),
             html.Div([info("prob_low"), " 'at least' = cautious lower bound"], style={"fontSize": 11, "color": "#555"})),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[
        card(title("Visits per day of week", "weekday_heatmap"), dcc.Graph(id="weekday-heat")),
        card(title("Selected place: weekday x hour", "weekday_hour_heatmap"), dcc.Graph(id="weekday-hour-heat")),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[
        card(title("Daily timeline", "timeline"), dcc.Graph(id="timeline")),
        card(title("Presence calendar", "presence_calendar"), dcc.Graph(id="calendar")),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[
        card(title("Anomalies", "anomalies"),
             html.Div(["minimum severity ", dcc.Slider(id="sev", min=0, max=1, step=0.1, value=0.5)]),
             dash_table.DataTable(id="anom-table", sort_action="native", **TABLE_STYLE)),
        card(title("Data quality", "data_quality"), html.Div(id="quality-stats"),
             html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr"}, children=[
                 dcc.Graph(id="q-per-day", style={"height": "240px"}), dcc.Graph(id="q-acc", style={"height": "240px"})])),
    ]),
    card(html.Details([html.Summary(html.B(["Tuning: re-run detection for this identifier", info("tuning")])),
                       html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}, children=[
                           html.Div(["place radius (m)", dcc.Slider(id="t-radius", min=25, max=300, step=25, value=75)]),
                           html.Div(["max gap between reports (h)", dcc.Slider(id="t-gap", min=1, max=24, step=1, value=12)]),
                           html.Div(["min dwell (min)", dcc.Slider(id="t-md", min=5, max=120, step=5, value=15)]),
                           html.Div(["same-place tolerance", info("same_place_tolerance"),
                                     dcc.Slider(id="t-gate", min=0.9, max=0.999, step=None, value=0.99,
                                                marks={0.9: "90% (2.1σ)", 0.95: "95% (2.4σ)", 0.99: "99% (3.0σ)",
                                                       0.999: "99.9% (3.7σ)"})]),
                           html.Div(["misses to end a stay", dcc.Slider(id="t-kb", min=1, max=5, step=1, value=3)]),
                           html.Div(["away time to end a stay (min)", dcc.Slider(id="t-confirm", min=0, max=60, step=5, value=15)]),
                           html.Div(["vague-report cutoff (1σ, m)", info("vague_report_cutoff"),
                                     dcc.Slider(id="t-coarse", min=250, max=5000, step=250, value=1000,
                                                marks={250: "250", 1000: "1k", 5000: "5k"})]),
                       ]),
                       html.Button("Re-run for this identifier", id="t-run"), html.Button("Back to saved results", id="t-reset"),
                       html.Div(id="t-summary")]), style={"marginTop": "10px"}),
])


# ------------------------------------------------------------------ identifier / dates / tuning

@app.callback(Output("uid", "options"), Output("uid", "value"), Input("ds", "value"))
def _ids(ds):
    opts = data.ids(ds) if ds else []
    return opts, opts[0]["value"] if opts else None


@app.callback(Output("days", "min_date_allowed"), Output("days", "max_date_allowed"), Output("days", "start_date"),
              Output("days", "end_date"), Input("ds", "value"), Input("uid", "value"))
def _days(ds, uid):
    ev = data.tables(ds, uid)["events"]
    lo, hi = ev["timestamp"].min().date(), ev["timestamp"].max().date()
    return lo, hi, lo, hi


@app.callback(Output("live", "data"), Output("t-summary", "children"), Input("t-run", "n_clicks"), Input("t-reset", "n_clicks"),
              Input("uid", "value"), State("ds", "value"), State("t-radius", "value"), State("t-gap", "value"),
              State("t-md", "value"), State("t-gate", "value"), State("t-kb", "value"), State("t-confirm", "value"),
              State("t-coarse", "value"), prevent_initial_call=True)
def _tune(_, __, uid, ds, radius, gap, md, gate, kb, confirm, coarse):
    if ctx.triggered_id != "t-run":
        return None, ""
    prof = data.meta(ds)["profile"]
    params = params_for(prof, md * 60, radius_m=radius, max_gap_s=gap * 3600, gate_p=gate, k_break=kb,
                        confirm_s=confirm * 60 or None, coarse_sigma_m=float(coarse))
    key = data.run_live(ds, uid, params)
    saved, live = data.tables(ds, uid), data.tables(ds, uid, key)
    return key, (f"Re-run: {live['visits'].height} visits / {live['places'].height} places "
                 f"(saved: {saved['visits'].height} / {saved['places'].height}).")


@app.callback(Output("live-banner", "children"), Input("live", "data"))
def _banner(live):
    return "Showing a live re-run with custom settings (not saved)" if live else ""


# ------------------------------------------------------------------ selection

@app.callback(Output("sel", "data"),
              Input("uid", "value"), Input("live", "data"), Input("map", "clickInfo"),
              Input("rank-table", "active_cell"), Input("places-table", "active_cell"), Input("graph", "tapNodeData"),
              Input("visits-table", "active_cell"), Input("timeline", "clickData"), Input("anom-table", "active_cell"),
              State("rank-table", "derived_viewport_data"), State("places-table", "derived_viewport_data"),
              State("visits-table", "derived_viewport_data"), State("anom-table", "derived_viewport_data"),
              State("ds", "value"), State("sel", "data"))
def _select(uid, live, click, rank_cell, place_cell, node, visit_cell, tl_click, anom_cell,
            rank_rows, place_rows, visit_rows, anom_rows, ds, sel):
    trig = ctx.triggered_id
    if trig in ("uid", "live", None):
        p = data.tables(ds, uid, live)["places"]
        return {"place_id": int(p.sort("rank_duration")["place_id"][0]) if p.height else None, "visit_id": None}
    if trig == "map" and click and click.get("object"):
        o = click["object"]
        if o.get("place_id") is not None:
            return {"place_id": int(o["place_id"]), "visit_id": o.get("visit_id")}
    if trig == "rank-table" and rank_cell:
        return {"place_id": rank_rows[rank_cell["row"]]["place_id"], "visit_id": None}
    if trig == "places-table" and place_cell:
        return {"place_id": place_rows[place_cell["row"]]["place_id"], "visit_id": None}
    if trig == "graph" and node:
        return {"place_id": int(node["id"]), "visit_id": None}
    if trig == "visits-table" and visit_cell:
        r = visit_rows[visit_cell["row"]]
        return {"place_id": r["place_id"], "visit_id": r["visit_id"]}
    if trig == "anom-table" and anom_cell:
        r = anom_rows[anom_cell["row"]]
        return {"place_id": r["place_id"], "visit_id": r["visit_id"]}
    if trig == "timeline" and tl_click:
        cd = tl_click["points"][0].get("customdata")
        if cd:
            return {"place_id": int(cd[0]), "visit_id": int(cd[1])}
    return no_update


# ------------------------------------------------------------------ identifier-level views

@app.callback(Output("rank-table", "data"), Output("rank-table", "columns"), Input("ds", "value"), Input("uid", "value"),
              Input("live", "data"), Input("rank-by", "value"))
def _rankings(ds, uid, live, by):
    p = data.tables(ds, uid, live)["places"]
    col = RANK_COLS[by]
    p = p.sort(col) if col else p.sort(["n_days", "total_dwell_s"], descending=True)
    rows = [{"#": i + 1, "place": data.place_label(r), "total time": fmt_dur(r["total_dwell_s"]),
             "visits": r["n_visits"], "days": r["n_days"], "confidence": round(r["score"]), "place_id": r["place_id"]}
            for i, r in enumerate(p.head(10).iter_rows(named=True))]
    return rows, [{"name": c, "id": c} for c in ("#", "place", "total time", "visits", "days", "confidence")]


@app.callback(Output("graph", "elements"), Input("ds", "value"), Input("uid", "value"), Input("live", "data"),
              Input("sel", "data"))
def _graph(ds, uid, live, sel):
    t = data.tables(ds, uid, live)
    p, tr = t["places"], t["transitions"]
    colors = data.place_colors(p)
    keep = set(p.filter(pl.col("rank_duration") <= 15)["place_id"].to_list()) | ({sel["place_id"]} if sel["place_id"] else set())
    nodes = [{"data": {"id": str(r["place_id"]), "label": data.place_label(r), "color": colors[r["place_id"]],
                       "size": 18 + 50 * r["time_share"] ** 0.5},
              "classes": "selected" if r["place_id"] == sel["place_id"] else ""}
             for r in p.iter_rows(named=True) if r["place_id"] in keep]
    edges = [{"data": {"source": str(r["from_place"]), "target": str(r["to_place"]), "w": 1 + 6 * r["prob"],
                       "label": f"{r['prob']:.0%}" if r["n"] >= 2 else ""}}
             for r in tr.iter_rows(named=True) if r["from_place"] in keep and r["to_place"] in keep]
    return nodes + edges


@app.callback(Output("places-table", "data"), Output("places-table", "columns"), Input("ds", "value"), Input("uid", "value"),
              Input("live", "data"))
def _places_table(ds, uid, live):
    p = data.tables(ds, uid, live)["places"].sort("rank_duration")
    rows = [{"place": data.place_label(r), "label": r["label"] or "", "address": r["address"] or "",
             "visits": r["n_visits"], "days": r["n_days"], "total time": fmt_dur(r["total_dwell_s"]),
             "median visit": fmt_dur(r["median_dwell_s"]), "time share %": round(100 * r["time_share"], 1),
             "reports": r["n_events"], "confidence": round(r["score"]), "evidence": round(r["score_evidence"], 2),
             "precision": round(r["score_precision"], 2), "regularity": round(r["score_regularity"], 2),
             "center ±m (1σ)": round(r["sigma_major"], 1), "extent m": round(r["extent_m"] or 0, 1),
             "first seen": fmt_ts(r["first_seen"]), "last seen": fmt_ts(r["last_seen"]), "place_id": r["place_id"]}
            for r in p.iter_rows(named=True)]
    cols = [c for c in rows[0] if c != "place_id"] if rows else []
    return rows, [{"name": c, "id": c} for c in cols]


@app.callback(Output("weekday-heat", "figure"), Output("timeline", "figure"), Output("calendar", "figure"),
              Input("ds", "value"), Input("uid", "value"), Input("live", "data"), Input("days", "start_date"),
              Input("days", "end_date"))
def _id_figs(ds, uid, live, d0, d1):
    t = data.tables(ds, uid, live)
    rng = _range(d0, d1)
    return figures.weekday_heatmap(t), figures.timeline(t, rng), figures.presence_calendar(t, rng)


@app.callback(Output("anom-table", "data"), Output("anom-table", "columns"), Input("ds", "value"), Input("uid", "value"),
              Input("live", "data"), Input("sev", "value"))
def _anomalies(ds, uid, live, sev):
    t = data.tables(ds, uid, live)
    a = t["anomalies"].filter(pl.col("severity") >= sev).join(
        t["visits"].select("visit_id", "start_ts", "duration_s"), on="visit_id").sort("severity", descending=True)
    labels = {r["place_id"]: data.place_label(r) for r in t["places"].iter_rows(named=True)}
    rows = [{"when": fmt_ts(r["start_ts"]), "place": labels.get(r["place_id"], ""), "kind": r["kind"].replace("_", " "),
             "detail": r["detail"], "severity": round(r["severity"], 2), "place_id": r["place_id"], "visit_id": r["visit_id"]}
            for r in a.iter_rows(named=True)]
    return rows, [{"name": c, "id": c} for c in ("when", "place", "kind", "detail", "severity")]


@app.callback(Output("quality-stats", "children"), Output("q-per-day", "figure"), Output("q-acc", "figure"),
              Input("ds", "value"), Input("uid", "value"), Input("live", "data"))
def _quality(ds, uid, live):
    t = data.tables(ds, uid, live)
    q = t["quality"].row(0, named=True) if t["quality"].height else {}
    items = [("reports", f"{q.get('n_events', 0):,}"), ("active days", q.get("active_days")),
             ("reports / active day", f"{q.get('events_per_active_day', 0):.0f}"),
             ("hours with data", f"{100 * (q.get('hour_coverage') or 0):.0f}%"),
             ("typical gap", fmt_dur(q.get("gap_p50_s"))), ("95th pct gap", fmt_dur(q.get("gap_p95_s"))),
             ("typical accuracy", f"{q.get('sigma_p50_m', 0):.0f} m"), ("coarse reports", f"{100 * q.get('coarse_frac', 0):.0f}%"),
             ("duplicates", f"{100 * q.get('duplicate_frac', 0):.1f}%"), ("repeated positions", f"{100 * q.get('stale_frac', 0):.1f}%"),
             ("implausible jumps", f"{100 * q.get('speed_outlier_frac', 0):.1f}%"),
             ("reports in visits", f"{100 * q.get('events_in_visits_frac', 0):.0f}%")]
    stats = html.Div([html.Div([html.B(f"{k}: "), str(v)]) for k, v in items],
                     style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "fontSize": 12})
    f1, f2 = figures.quality_figs(t, t["events"])
    return stats, f1, f2


# ------------------------------------------------------------------ selection-level views

@app.callback(Output("map-wrap", "children"), Input("ds", "value"), Input("uid", "value"), Input("live", "data"),
              Input("sel", "data"), Input("extent", "value"), Input("days", "start_date"), Input("days", "end_date"))
def _map(ds, uid, live, sel, extent, d0, d1):
    t = data.tables(ds, uid, live)
    lay = layers.build(t, extent or [], sel["place_id"], sel["visit_id"], _range(d0, d1))
    p = t["places"]
    focus = p.filter(pl.col("place_id") == sel["place_id"]) if sel["place_id"] is not None else p
    view = {"longitude": float(focus["lon"].mean()), "latitude": float(focus["lat"].mean()), "zoom": 14} if focus.height else \
        {"longitude": -83.63, "latitude": 32.84, "zoom": 12}
    return DeckGL(id="map", layers=lay, initial_view_state=view, controller=True, tooltip=True,
                  enable_events=["click"], style={"width": "100%", "height": "560px"})


@app.callback(Output("place-card", "children"), Output("radar", "figure"), Output("weekday-hour-heat", "figure"),
              Output("visits-table", "data"), Output("visits-table", "columns"), Output("next-table", "data"),
              Output("next-table", "columns"),
              Input("ds", "value"), Input("uid", "value"), Input("live", "data"), Input("sel", "data"),
              Input("visits-all", "value"))
def _selection_views(ds, uid, live, sel, vall):
    t = data.tables(ds, uid, live)
    pid, vid = sel["place_id"], sel["visit_id"]
    p = t["places"].filter(pl.col("place_id").eq_missing(pid))  # pid may be None (no selection)
    labels = {r["place_id"]: data.place_label(r) for r in t["places"].iter_rows(named=True)}
    if p.height:
        r = p.row(0, named=True)
        place_card = html.Div([
            html.Div(html.B(data.place_label(r))),
            html.Div(r["address"] or "no address found nearby", style={"color": "#555"}),
            html.Div(f"{r['place_category'] or ''}  (matched {r['geocode_distance_m']:.0f} m away)" if r["geocode_distance_m"] else ""),
            html.Div([html.B(f"Pattern confidence {r['score']:.0f}/100"), info("score")]),
            html.Div([f"evidence {r['score_evidence']:.2f}", info("score_evidence"), " · ",
                      f"precision {r['score_precision']:.2f}", info("score_precision"), " · ",
                      f"regularity {r['score_regularity']:.2f}", info("score_regularity")]),
            html.Div(f"{r['n_visits']} visits on {r['n_days']} days · {fmt_dur(r['total_dwell_s'])} total · "
                     f"median visit {fmt_dur(r['median_dwell_s'])} · {100 * r['time_share']:.0f}% of dwell time"),
            html.Div(f"center ±{r['sigma_major']:.0f} m (1σ) · activity extent {r['extent_m'] or 0:.0f} m · "
                     f"first {fmt_ts(r['first_seen'])} · last {fmt_ts(r['last_seen'])}"),
        ], style={"fontSize": 12, "lineHeight": "1.6"})
    else:
        place_card = "Select a place"
    v = t["visits"] if "all" in (vall or []) else t["visits"].filter(pl.col("place_id").eq_missing(pid))
    vrows = [{"start": fmt_ts(r["start_ts"]), "end": fmt_ts(r["end_ts"]), "duration": fmt_dur(r["duration_s"]),
              "arrived between": f"{fmt_ts(r['arrive_after'])} – {fmt_ts(r['arrive_by'])}",
              "left between": f"{fmt_ts(r['depart_after'])} – {fmt_ts(r['depart_by'])}",
              "reports": r["n_events"], "precise": r["n_informative"], "confidence": round(r["confidence"]),
              "center ±m": round(r["sigma_major"] or 0, 1), "from": labels.get(r["prev_place_id"], ""),
              "to": labels.get(r["next_place_id"], ""), "flags": ", ".join(r["anomalies"] or []),
              "place": labels.get(r["place_id"], ""), "place_id": r["place_id"], "visit_id": r["visit_id"]}
             for r in v.sort("start_ts", descending=True).iter_rows(named=True)]
    vcols = [{"name": c, "id": c} for c in ("start", "end", "duration", "arrived between", "left between", "reports",
                                            "precise", "confidence", "center ±m", "from", "to", "flags", "place")]
    nx = t["next_places"].filter(pl.col("from_place").eq_missing(pid))
    nrows = [{"rank": r["rank"], "next place": labels.get(r["to_place"], str(r["to_place"])),
              "probability": f"{r['prob']:.0%}", "at least": f"{r['prob_low']:.0%}", "departures seen": r["n_from"]}
             for r in nx.iter_rows(named=True)]
    ncols = [{"name": c, "id": c} for c in ("rank", "next place", "probability", "at least", "departures seen")]
    return (place_card, figures.radar(t, pid, vid), figures.weekday_hour_heatmap(t, pid), vrows, vcols, nrows, ncols)


def _range(d0, d1):
    from datetime import date

    return (date.fromisoformat(d0[:10]), date.fromisoformat(d1[:10])) if d0 and d1 else None



if __name__ == "__main__":
    app.run(debug=False, port=8050)
