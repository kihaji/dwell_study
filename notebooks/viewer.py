"""Pattern-of-life results viewer (marimo + deckgl-marimo): the Dash app's views in a notebook.

    uv run python -m dwells.pol.build                                     # precompute outputs
    uv run --extra notebook marimo run notebooks/viewer.py                # app view
    uv run --extra notebook marimo edit notebooks/viewer.py               # edit

Select a dataset and identifier; click a place on the map, a row in any table, a node in
the movement graph or a bar in the timeline to focus every view on it. Times are UTC.
Shares its data, rows and figures with the Dash app (app/data.py, views.py, map_data.py,
figures.py).
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Pattern of life")


@app.cell
def _():
    import html
    import sys

    import deckgl_marimo as dgl
    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir().parent / "app"))
    import data
    import figures
    import map_data
    import views

    from dwells.pol.help import HELP
    from dwells.presets import params_for

    def info(key):
        return (f'<abbr title="{html.escape(HELP[key])}" '
                'style="cursor:help;text-decoration:none;color:#1565c0"> ⓘ</abbr>')

    def title(text, key):
        return mo.md(f"#### {text}{info(key)}")

    return data, dgl, figures, info, map_data, mo, params_for, title, views


@app.cell
def _(data, info, mo):
    _ds = data.datasets()
    ds = mo.ui.dropdown(_ds, value=_ds[0] if _ds else None, label="Dataset")
    mo.hstack([mo.md(f"### Pattern of life{info('utc')}"), ds], justify="start", align="end", gap=2)
    return (ds,)


@app.cell
def _(data, ds, mo):
    mo.stop(ds.value is None, mo.md("No results found: run `uv run python -m dwells.pol.build` first."))
    _ids = {o["label"]: o["value"] for o in data.ids(ds.value)}
    uid = mo.ui.dropdown(_ids, value=next(iter(_ids)), label="Identifier", searchable=True)
    return (uid,)


@app.cell
def _(data, ds, info, mo, uid):
    _ev = data.tables(ds.value, uid.value)["events"]
    _lo, _hi = _ev["timestamp"].min().date(), _ev["timestamp"].max().date()
    dates = mo.ui.date_range(start=_lo, stop=_hi, value=(_lo, _hi), label="Dates (UTC)")
    overlays = mo.ui.multiselect(
        {"uncertainty ellipse": "ellipse", "convex hull": "hull", "member events + links": "member"},
        value=["uncertainty ellipse", "convex hull"], label="Map overlays")
    mo.hstack([uid, dates, overlays, mo.md(f"overlays{info('map')}")], justify="start", align="end", gap=2)
    return dates, overlays


@app.cell
def _(mo):
    # live re-run for one identifier: (dataset, identifier, cache key) or None
    get_live, set_live = mo.state(None)
    # selected place / visit, tagged with the (dataset, identifier, live key) it belongs to
    get_sel, set_sel = mo.state(None)
    return get_live, get_sel, set_live, set_sel


@app.cell
def _(data, dates, ds, get_live, get_sel, set_sel, uid, views):
    _live = get_live()
    live_key = _live[2] if _live and _live[:2] == (ds.value, uid.value) else None
    t = data.tables(ds.value, uid.value, live_key)
    day_range = tuple(dates.value) if dates.value else None
    ctx = (ds.value, uid.value, live_key)
    _s = get_sel()
    sel = {k: _s[k] for k in ("place_id", "visit_id")} if _s and _s["ctx"] == ctx else views.default_selection(t)

    def select(s):
        if s:
            set_sel({**s, "ctx": ctx})

    return day_range, live_key, select, sel, t


@app.cell
def _(live_key, mo):
    mo.callout(mo.md("**Showing a live re-run with custom settings (not saved)**"), kind="danger") if live_key else None
    return


@app.cell
def _(dgl, map_data, mo):
    # created once (no reactive inputs) so the map is never rebuilt; layers and the camera
    # are updated from the cells below. OSM raster tiles as in the Dash app.
    _osm = {"version": 8,
            "sources": {"osm": {"type": "raster", "tiles": [map_data.OSM_TILES], "tileSize": 256, "maxzoom": 19,
                                "attribution": map_data.OSM_ATTRIBUTION}},
            "layers": [{"id": "osm", "type": "raster", "source": "osm"}]}
    deck = dgl.Map(basemap=_osm, height="560px", center=(-83.63, 32.84), zoom=12)
    map_view = mo.ui.anywidget(deck)
    return deck, map_view


@app.cell
def _(day_range, deck, dgl, map_data, overlays, sel, t):
    def _layers(d, extent, sel_place):
        L = [dgl.ScatterplotLayer(id="events", data=[e | {"tooltip": f"{e['time']} UTC · ±{e['acc_m']} m · {e['source']}"}
                                                    for e in d["events"]],
                                  get_position="p", get_radius=3, radius_min_pixels=2, radius_max_pixels=6,
                                  get_fill_color="c", pickable=True)]
        if "member" in extent:
            L.append(dgl.ScatterplotLayer(id="event-error", data=d["event_error"], get_position="p", get_radius="s",
                                          filled=False, stroked=True, get_line_color=[90, 90, 90, 120],
                                          line_width_min_pixels=1, pickable=False))
            L.append(dgl.LineLayer(id="member-links", data=d["member_links"], get_source_position="s",
                                   get_target_position="t", get_color=[60, 60, 60, 110], get_width=1,
                                   width_min_pixels=1, pickable=False))
        if "hull" in extent:
            L.append(dgl.PolygonLayer(id="hulls", data=d["hulls"], get_polygon="poly", get_fill_color="c",
                                      get_line_color="l", line_width_min_pixels=1.5, pickable=True))
        if "ellipse" in extent:
            L.append(dgl.PolygonLayer(id="ellipses", data=d["ellipses"], get_polygon="poly", filled=False, stroked=True,
                                      get_line_color="l", line_width_min_pixels=2, pickable=True))
        L.append(dgl.ScatterplotLayer(id="places", data=[p | {"tooltip": f"{p['name']} · confidence {p['score']} · "
                                                                         f"{p['visits']} visits"} for p in d["places"]],
                                      get_position="p", get_radius="r", radius_units="pixels", get_fill_color="c",
                                      stroked=True, get_line_color=[255, 255, 255, 255], line_width_min_pixels=2,
                                      pickable=True))
        L.append(dgl.TextLayer(id="place-labels", data=d["place_labels"], get_position="p", get_text="name", get_size=12,
                               get_color=[20, 20, 20, 255], get_pixel_offset=[0, -18], background=True, pickable=False))
        if d["arcs"]:
            L.append(dgl.ArcLayer(id="next-arcs", data=[a | {"tooltip": f"next: {a['label']}"} for a in d["arcs"]],
                                  get_source_position="s", get_target_position="t", get_source_color=[30, 30, 30, 220],
                                  get_target_color=[214, 39, 40, 230], get_width="w", pickable=True))
            L.append(dgl.TextLayer(id="arc-labels", data=d["arc_labels"], get_position="p", get_text="label",
                                   get_size=12, get_color=[180, 0, 0, 255], get_pixel_offset=[0, 16], pickable=False))
        return L

    deck.set_layers(_layers(map_data.rows(t, overlays.value, sel["place_id"], sel["visit_id"], day_range),
                            overlays.value, sel["place_id"]))
    return


@app.cell
def _(deck, map_data, sel, t):
    deck.fit_bounds(map_data.bounds(t, sel["place_id"]))  # recenter on the selection
    return


@app.cell
def _():
    seen_click = {}  # last handled map click
    return (seen_click,)


@app.cell
def _(map_view, seen_click, select, views):
    # a click on the map selects the picked place / visit (each click handled once)
    _click = map_view.value.get("click_info") or {}
    if _click and _click != seen_click.get("click"):
        seen_click["click"] = _click
        select(views.selection_from_pick(_click.get("object")))
    return


@app.cell
def _(info, map_view, mo, t, title, views, sel, figures):
    _s = views.place_summary(t, sel["place_id"])
    if _s:
        _ev, _pr, _rg = _s["components"]
        _card = mo.md(
            f"**{_s['title']}**  \n<span style='color:#555'>{_s['address']}</span>  \n{_s['match']}  \n"
            f"**{_s['score']}**{info('score')}  \n{_ev}{info('score_evidence')} · {_pr}{info('score_precision')} · "
            f"{_rg}{info('score_regularity')}  \n{_s['stats']}  \n{_s['extent']}")
    else:
        _card = mo.md("Select a place")
    mo.hstack([
        mo.vstack([title("Map", "map"), map_view,
                   mo.md(f"<small>{info('next_place_arcs')} arcs show likely next places from the selected place · "
                         "map © <a href='https://www.openstreetmap.org/copyright' target='_blank'>OpenStreetMap "
                         "contributors</a></small>")]),
        mo.vstack([title("Selected place", "place"), _card, title("When there (radar)", "radar"),
                   figures.radar(t, sel["place_id"], sel["visit_id"])]),
    ], widths=[3, 2], align="start")
    return


@app.cell
def _(mo, views):
    rank_by = mo.ui.radio(list(views.RANK_COLS), value="total time", inline=True)
    return (rank_by,)


@app.cell
def _(figures, mo, rank_by, sel, select, t, title, views):
    _rank = mo.ui.table(views.rank_rows(t, rank_by.value), selection="single", hidden_columns=["place_id"],
                        page_size=10, show_download=False, show_column_summaries=False,
                        on_change=lambda rows: rows and select({"place_id": rows[0]["place_id"], "visit_id": None}))
    _fig = figures.movement_graph(t, sel["place_id"])
    _graph = mo.ui.plotly(_fig, on_change=lambda pts: select(views.selection_from_points(pts, _fig)))
    mo.hstack([mo.vstack([title("Top 10 places", "rankings"), rank_by, _rank]),
               mo.vstack([title("Movement graph", "graph"), _graph])], widths=[1, 1], align="start")
    return


@app.cell
def _(mo, select, t, title, views):
    _rows = views.places_rows(t)
    mo.vstack([title("Places", "places_table"),
               mo.ui.table(_rows, selection="single", hidden_columns=["place_id"], page_size=10,
                           on_change=lambda rows: rows and select({"place_id": rows[0]["place_id"], "visit_id": None}))])
    return


@app.cell
def _(mo):
    visits_all = mo.ui.checkbox(label="all places")
    return (visits_all,)


@app.cell
def _(info, mo, sel, select, t, title, views, visits_all):
    _visits = mo.ui.table(views.visits_rows(t, sel["place_id"], all_places=visits_all.value), selection="single",
                          hidden_columns=["place_id", "visit_id"], page_size=10,
                          on_change=lambda rows: rows and select({"place_id": rows[0]["place_id"],
                                                                  "visit_id": rows[0]["visit_id"]}))
    _next = mo.ui.table(views.next_rows(t, sel["place_id"]), selection="single", hidden_columns=["place_id"],
                        show_download=False, show_column_summaries=False,
                        on_change=lambda rows: rows and select({"place_id": rows[0]["place_id"], "visit_id": None}))
    mo.hstack([mo.vstack([title("Visits", "visits_table"), visits_all, _visits]),
               mo.vstack([title("Next places from selected", "next_places"), _next,
                          mo.md(f"<small>{info('prob_low')} 'at least' = cautious lower bound</small>")])],
              widths=[3, 2], align="start")
    return


@app.cell
def _(figures, mo, sel, t, title):
    mo.hstack([mo.vstack([title("Visits per day of week", "weekday_heatmap"), figures.weekday_heatmap(t)]),
               mo.vstack([title("Selected place: weekday x hour", "weekday_hour_heatmap"),
                          figures.weekday_hour_heatmap(t, sel["place_id"])])], widths=[1, 1], align="start")
    return


@app.cell
def _(day_range, figures, mo, select, t, title, views):
    _tl = figures.timeline(t, day_range)
    _tl.update_layout(clickmode="event+select")
    mo.hstack([mo.vstack([title("Daily timeline", "timeline"),
                          mo.ui.plotly(_tl, on_change=lambda pts: select(views.selection_from_points(pts, _tl)))]),
               mo.vstack([title("Presence calendar", "presence_calendar"), figures.presence_calendar(t, day_range)])],
              widths=[1, 1], align="start")
    return


@app.cell
def _(mo):
    severity = mo.ui.slider(0, 1, step=0.1, value=0.5, label="minimum severity", show_value=True)
    return (severity,)


@app.cell
def _(figures, mo, select, severity, t, title, views):
    _anom = mo.ui.table(views.anomaly_rows(t, severity.value), selection="single",
                        hidden_columns=["place_id", "visit_id"], page_size=10,
                        on_change=lambda rows: rows and select({"place_id": rows[0]["place_id"],
                                                                "visit_id": rows[0]["visit_id"]}))
    _stats = mo.Html('<div style="display:grid;grid-template-columns:repeat(3,1fr);font-size:12px">' + "".join(
        f"<div><b>{k}:</b> {v}</div>" for k, v in views.quality_items(t)) + "</div>")
    _f1, _f2 = figures.quality_figs(t, t["events"])
    _f1.update_layout(height=240)
    _f2.update_layout(height=240)
    mo.hstack([mo.vstack([title("Anomalies", "anomalies"), severity, _anom]),
               mo.vstack([title("Data quality", "data_quality"), _stats, mo.hstack([_f1, _f2], widths=[1, 1])])],
              widths=[1, 1], align="start")
    return


@app.cell
def _(info, mo, set_live):
    t_radius = mo.ui.slider(25, 300, step=25, value=75, label="place radius (m)", show_value=True)
    t_gap = mo.ui.slider(1, 24, step=1, value=12, label="max gap between reports (h)", show_value=True)
    t_md = mo.ui.slider(5, 120, step=5, value=15, label="min dwell (min)", show_value=True)
    t_gate = mo.ui.dropdown({"90% (2.1σ)": 0.9, "95% (2.4σ)": 0.95, "99% (3.0σ)": 0.99, "99.9% (3.7σ)": 0.999},
                            value="99% (3.0σ)", label="same-place tolerance")
    t_kb = mo.ui.slider(1, 5, step=1, value=3, label="misses to end a stay", show_value=True)
    t_confirm = mo.ui.slider(0, 60, step=5, value=15, label="away time to end a stay (min)", show_value=True)
    t_coarse = mo.ui.slider(250, 5000, step=250, value=1000, label="vague-report cutoff (1σ, m)", show_value=True)
    t_run = mo.ui.run_button(label="Re-run for this identifier")
    t_reset = mo.ui.button(label="Back to saved results", on_click=lambda _: set_live(None))
    mo.accordion({"Tuning: re-run detection for this identifier": mo.vstack([
        mo.md(f"Re-runs detection with these settings for the selected identifier only; saved results are "
              f"not changed.{info('tuning')}"),
        mo.hstack([t_radius, t_gap, t_md], justify="start", gap=2),
        mo.hstack([mo.vstack([t_gate, mo.md(f"<small>{info('same_place_tolerance')}</small>")]), t_kb, t_confirm],
                  justify="start", gap=2),
        mo.hstack([t_coarse, mo.md(f"<small>{info('vague_report_cutoff')}</small>")], justify="start"),
        mo.hstack([t_run, t_reset], justify="start")])})
    return t_coarse, t_confirm, t_gap, t_gate, t_kb, t_md, t_radius, t_run


@app.cell
def _(data, ds, mo, params_for, set_live, t_coarse, t_confirm, t_gap, t_gate, t_kb, t_md, t_radius, t_run, uid):
    mo.stop(not t_run.value)
    _params = params_for(data.meta(ds.value)["profile"], t_md.value * 60, radius_m=t_radius.value,
                         max_gap_s=t_gap.value * 3600, gate_p=t_gate.value, k_break=t_kb.value,
                         confirm_s=t_confirm.value * 60 or None, coarse_sigma_m=float(t_coarse.value))
    _key = data.run_live(ds.value, uid.value, _params)
    set_live((ds.value, uid.value, _key))
    _saved, _new = data.tables(ds.value, uid.value), data.tables(ds.value, uid.value, _key)
    mo.md(f"Re-run: {_new['visits'].height} visits / {_new['places'].height} places "
          f"(saved: {_saved['visits'].height} / {_saved['places'].height}).")
    return


if __name__ == "__main__":
    app.run()
