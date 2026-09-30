"""Dwell comparison viewer (Dash + deckgl-dash).

    uv run --extra viewer python viewer/app.py   # then open http://127.0.0.1:8050

Pick a dataset, an id, a day and up to four algorithms (tuned eval-stage detections from
results/dwells/). The map shows raw events (radius = reported 1-sigma error), the
track, true dwells (green rings, radius = place footprint) and detected dwells (one
color per algorithm, radius = centroid 1-sigma when reported). Below: a timeline of
true vs. detected dwells and the held-out metrics table for the dataset.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, dash_table, dcc, html
from deckgl_dash import DeckGL
from deckgl_dash.layers import PathLayer, ScatterplotLayer, TileLayer

from dwells import algorithms, io
from dwells.harness.run import RESULTS, dataset_dirs

COLORS = [[230, 97, 1], [94, 60, 153], [0, 114, 178], [204, 121, 167]]
TRUTH_COLOR = [0, 158, 115]
MD = 900


def _datasets():
    d = RESULTS / "dwells"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def _det_file(ds: str, algo: str) -> Path:
    shape = "3_mixed" if algorithms.get(algo).USES_UNCERTAINTY else "1"
    return RESULTS / "dwells" / ds / f"{algo}__{shape}__{MD}.parquet"


app = Dash(__name__, title="Dwell viewer")
DS = _datasets()
app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "12px"}, children=[
    html.H3("Dwell detection comparison"),
    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
        html.Div([html.Label("Dataset"), dcc.Dropdown(DS, DS[0] if DS else None, id="ds", clearable=False)], style={"width": "260px"}),
        html.Div([html.Label("Id"), dcc.Dropdown(id="uid", clearable=False)], style={"width": "160px"}),
        html.Div([html.Label("Day (UTC)"), dcc.Dropdown(id="day", clearable=False)], style={"width": "160px"}),
        html.Div([html.Label("Algorithms (max 4)"),
                  dcc.Dropdown(algorithms.names(), ["ua_segmentation", "li_centroid"], id="algos", multi=True)],
                 style={"minWidth": "420px", "flex": "1"}),
    ]),
    html.Div(id="map-wrap", style={"height": "560px", "marginTop": "10px", "position": "relative"}),
    dcc.Graph(id="timeline", style={"height": "300px"}),
    html.H4("Held-out metrics (min duration 15 min)"),
    dash_table.DataTable(id="metrics", sort_action="native", page_size=20,
                         style_cell={"fontSize": 12, "padding": "3px"}),
])


@app.callback(Output("uid", "options"), Output("uid", "value"), Input("ds", "value"))
def _ids(ds):
    ids = pl.scan_parquet(dataset_dirs()[ds] / "events.parquet").select("id").unique().sort("id").collect()["id"].to_list()
    return ids, ids[0]


@app.callback(Output("day", "options"), Output("day", "value"), Input("ds", "value"), Input("uid", "value"))
def _days(ds, uid):
    ev = pl.scan_parquet(dataset_dirs()[ds] / "events.parquet").filter(pl.col("id") == uid)
    days = ev.select(pl.col("timestamp").dt.date().unique().sort()).collect().to_series().cast(pl.String).to_list()
    return days, days[min(1, len(days) - 1)] if days else None


def _window(day):
    t0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return t0, t0 + timedelta(days=1)


def _in_window(df, t0, t1, s="start_ts", e="end_ts"):
    return df.filter((pl.col(s) < t1) & (pl.col(e) > t0))


@app.callback(Output("map-wrap", "children"), Output("timeline", "figure"), Output("metrics", "data"),
              Output("metrics", "columns"), Input("ds", "value"), Input("uid", "value"), Input("day", "value"),
              Input("algos", "value"))
def _render(ds, uid, day, algos):
    algos = (algos or [])[:4]
    t0, t1 = _window(day)
    ev = io.load_events(dataset_dirs()[ds], "2", keep_truth=True, ids=[uid]).filter(
        pl.col("timestamp").is_between(t0, t1))
    truth_p = dataset_dirs()[ds] / "truth_dwells.parquet"
    truth = _in_window(pl.read_parquet(truth_p).filter(pl.col("id") == uid), t0, t1) if truth_p.exists() else None
    dets = {}
    for a in algos:
        f = _det_file(ds, a)
        if f.exists():
            dets[a] = _in_window(pl.read_parquet(f).filter(pl.col("id") == uid), t0, t1)

    sigma = ev["horizontalAccuracy"] / 1.1774
    pts = [{"p": [lo, la], "r": float(s), "t": str(t)} for lo, la, s, t in
           zip(ev["lon"], ev["lat"], sigma, ev["timestamp"])]
    layers = [
        TileLayer(id="osm", data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", min_zoom=0, max_zoom=19, tile_size=256),
        PathLayer(id="track", data=[{"path": [p["p"] for p in pts]}], get_path="@@=path",
                  get_color=[120, 120, 120, 120], width_min_pixels=1),
        ScatterplotLayer(id="events", data=pts, get_position="@@=p", get_radius="@@=r", radius_min_pixels=2,
                         get_fill_color=[80, 80, 80, 40], stroked=True, get_line_color=[60, 60, 60, 160],
                         line_width_min_pixels=1, pickable=True),
    ]
    if truth is not None and truth.height:
        layers.append(ScatterplotLayer(
            id="truth", data=[{"p": [r["lon"], r["lat"]], "r": r["radius_m"] or 30, "cat": r["category"]}
                              for r in truth.iter_rows(named=True)],
            get_position="@@=p", get_radius="@@=r", radius_min_pixels=6, filled=False, stroked=True,
            get_line_color=TRUTH_COLOR, line_width_min_pixels=3, pickable=True))
    for i, (a, d) in enumerate(dets.items()):
        layers.append(ScatterplotLayer(
            id=f"det-{a}", data=[{"p": [r["lon"], r["lat"]], "r": r["sigma_major"] or 15, "algo": a}
                                 for r in d.iter_rows(named=True)],
            get_position="@@=p", get_radius="@@=r", radius_min_pixels=4, get_fill_color=COLORS[i] + [170],
            pickable=True))
    view = {"longitude": float(ev["lon"].median()) if ev.height else -83.63,
            "latitude": float(ev["lat"].median()) if ev.height else 32.84, "zoom": 13}
    deck = DeckGL(id="map", layers=layers, initial_view_state=view, controller=True, tooltip=True,
                  style={"width": "100%", "height": "560px"})

    fig = go.Figure()
    rows = ([("truth", truth, TRUTH_COLOR)] if truth is not None else []) + [
        (a, d, COLORS[i]) for i, (a, d) in enumerate(dets.items())]
    for name, d, col in rows:
        for r in d.iter_rows(named=True):
            fig.add_trace(go.Scatter(x=[r["start_ts"], r["end_ts"]], y=[name, name], mode="lines",
                                     line={"width": 14, "color": f"rgb({col[0]},{col[1]},{col[2]})"},
                                     showlegend=False, hovertext=f"{r['start_ts']:%H:%M}-{r['end_ts']:%H:%M}"))
    fig.add_trace(go.Scatter(x=ev["timestamp"].to_list(), y=["events"] * ev.height, mode="markers",
                             marker={"size": 4, "color": "gray"}, showlegend=False))
    fig.update_layout(margin={"l": 140, "r": 10, "t": 10, "b": 30}, xaxis={"range": [t0, t1]})

    data, cols = [], []
    mp = RESULTS / "eval.parquet"
    if mp.exists():
        m = pl.read_parquet(mp).filter((pl.col("dataset") == ds) & pl.col("error").is_null()
                                       & pl.col("params").str.contains(f'"min_duration_s": {MD}'))
        keep = ["algo", "shape", "f1", "precision", "recall", "recall_observable", "start_err_s_med",
                "end_err_s_med", "centroid_err_m_med", "fragmentation", "merging", "runtime_s"]
        m = m.select([c for c in keep if c in m.columns]).sort("f1", descending=True)
        m = m.with_columns(pl.col(pl.Float64).round(3))
        data, cols = m.to_dicts(), [{"name": c, "id": c} for c in m.columns]
    return deck, fig, data, cols


if __name__ == "__main__":
    app.run(debug=False, port=8050)
