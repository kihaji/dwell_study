"""Render results parquet files into markdown tables under docs/results/.

    uv run python -m dwells.harness.report

Writes accuracy.md (held-out detection metrics), sensitivity.md (dev sweep: F1 vs
radius / max gap / min duration), uncertainty.md (shape comparison), scaling.md
(benchmarks) and downstream.md (places, home/work, next location, colocation).
"""

from __future__ import annotations

import json

import polars as pl

from dwells import algorithms
from dwells.harness.run import RESULTS, profile_of
from dwells.harness.score import DOCS

UA = {a for a in algorithms.names() if algorithms.get(a).USES_UNCERTAINTY}


def md_table(df: pl.DataFrame, digits: int = 3) -> str:
    df = df.with_columns(pl.col(pl.Float64).round(digits))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def _eval() -> pl.DataFrame:
    e = pl.read_parquet(RESULTS / "eval.parquet")
    return e.with_columns(
        md=pl.col("params").map_elements(lambda p: json.loads(p)["min_duration_s"], return_dtype=pl.Int64),
        profile=pl.col("dataset").map_elements(profile_of, return_dtype=pl.String))


def _pref_shape(df):
    return df.filter(pl.when(pl.col("algo").is_in(list(UA))).then(pl.col("shape") == "3_mixed")
                     .otherwise(pl.col("shape") == "1"))


def accuracy() -> str:
    e = _eval()
    ok = _pref_shape(e.filter(pl.col("error").is_null()))
    out = ["# Held-out detection accuracy\n",
           "Tuned on dev datasets (radius, max gap per profile and min duration), evaluated on eval "
           "datasets with different agents and seeds, plus the labeled public sets STAGA and PoL (50-agent subset).",
           "Uncertainty-aware algorithms are shown at shape 3_mixed, the others at shape 1 (same positions).\n"]
    for md in (900, 1800, 3600, 300):
        t = ok.filter(pl.col("md") == md).pivot(on="dataset", index="algo", values="f1", aggregate_function="first")
        cols = sorted(c for c in t.columns if c != "algo")
        t = t.select(["algo"] + cols)
        syn = [c for c in cols if c.startswith("eval_")]
        t = t.with_columns(mean_synthetic=pl.mean_horizontal(syn)).sort("mean_synthetic", descending=True)
        out += [f"## F1, min duration {md // 60} min\n", md_table(t), ""]
    det = ok.filter(pl.col("md") == 900).group_by("algo", "profile").agg(
        pl.col("precision").mean(), pl.col("recall").mean(), pl.col("recall_observable").mean(),
        pl.col("f1_lenient").mean(), pl.col("start_err_s_med").mean(), pl.col("end_err_s_med").mean(),
        pl.col("duration_bias_s_med").mean(), pl.col("centroid_err_m_med").mean(),
        pl.col("fragmentation").mean(), pl.col("merging").mean(), pl.col("time_recall").mean(),
        pl.col("runtime_s").mean()).sort("profile", "algo")
    out += ["## Error anatomy at 15 min (mean over the eval datasets of each profile)\n",
            "recall_observable = recall over true dwells with >= 2 events; f1_lenient uses IoU >= 0.2; "
            "errors are medians over matched dwells; duration_bias > 0 = detected longer than true.\n",
            md_table(det, 2), ""]
    errs = e.filter(pl.col("error").is_not_null()).group_by("algo", "dataset").agg(pl.col("error").first(), pl.len())
    if errs.height:
        out += ["## Failed runs\n", md_table(errs), ""]
    return "\n".join(out)


def sensitivity() -> str:
    s = pl.read_parquet(RESULTS / "sweep.parquet").filter(pl.col("error").is_null())
    s = _pref_shape(s).with_columns(
        md=pl.col("params").str.json_path_match("$.min_duration_s").cast(pl.Int64),
        radius=pl.col("params").str.json_path_match("$.radius_m").cast(pl.Int64),
        gap=pl.col("params").str.json_path_match("$.max_gap_s").cast(pl.Int64),
        profile=pl.col("dataset").map_elements(profile_of, return_dtype=pl.String))
    out = ["# Parameter sensitivity (dev sweep)\n",
           "F1 averaged over both areas' dev datasets; for each cell the other tuning parameter is at its best value.\n"]
    for prof in ("gps", "cdr", "adtech"):
        sp = s.filter((pl.col("profile") == prof) & (pl.col("md") == 900))
        r = sp.group_by("algo", "radius", "gap").agg(pl.col("f1").mean()).group_by("algo", "radius").agg(pl.col("f1").max())
        t = r.pivot(on="radius", index="algo", values="f1", sort_columns=True).sort("algo")
        t = t.rename({c: f"r={c}" for c in t.columns if c != "algo"})
        g = sp.group_by("algo", "radius", "gap").agg(pl.col("f1").mean()).group_by("algo", "gap").agg(pl.col("f1").max())
        tg = g.pivot(on="gap", index="algo", values="f1", sort_columns=True).sort("algo")
        tg = tg.rename({c: f"gap={int(c) // 3600}h" if int(c) >= 3600 else f"gap={int(c) // 60}m" for c in tg.columns if c != "algo"})
        out += [f"## {prof}: F1 vs radius (min duration 15 min)\n", md_table(t), "",
                f"## {prof}: F1 vs max gap\n", md_table(tg), ""]
    m = s.group_by("algo", "profile", "md", "radius", "gap").agg(pl.col("f1").mean()).group_by("algo", "profile", "md").agg(pl.col("f1").max())
    out += ["## Best F1 vs min duration (dwell definition)\n",
            md_table(m.pivot(on="md", index=["algo", "profile"], values="f1", sort_columns=True).sort("profile", "algo")
                     .rename({str(x): f"md={x // 60}m" for x in (300, 900, 1800, 3600)})), ""]
    return "\n".join(out)


def uncertainty() -> str:
    e = _eval().filter(pl.col("error").is_null() & pl.col("algo").is_in(list(UA)) & (pl.col("md") == 900)
                       & pl.col("dataset").str.starts_with("eval_"))
    t = e.group_by("algo", "profile", "shape").agg(pl.col("f1").mean()).pivot(
        on="shape", index=["algo", "profile"], values="f1", sort_columns=True).sort("profile", "algo")
    t = t.with_columns(gain_1_to_3mixed=pl.col("3_mixed") - pl.col("1"))
    return "\n".join(["# Does positional-error information help? (held-out, min duration 15 min)\n",
                      "Same positions in every shape; only the error fields differ. Algorithms that ignore error "
                      "are shape-invariant and omitted.\n", md_table(t), ""])


def scaling() -> str:
    p = RESULTS / "bench.parquet"
    if not p.exists():
        return "# Scaling\n\n(benchmark not run)\n"
    b = pl.read_parquet(p).with_columns(eps=pl.col("n_events") / pl.col("runtime_s"))
    out = ["# Scaling benchmarks\n",
           "Single process per run, sequential, warm (JIT compiled on a 500-event slice first). "
           "Timeout 900 s; an algorithm stops climbing a ladder after a failure.\n"]
    for lad in ("gps", "adtech", "single"):
        x = b.filter(pl.col("ladder") == lad)
        if x.height == 0:
            continue
        rt = x.with_columns(cell=pl.when(pl.col("error").is_null()).then(pl.col("runtime_s").round(2).cast(pl.String))
                            .otherwise(pl.col("error"))).pivot(on="dataset", index="algo", values="cell")
        mem = x.filter(pl.col("error").is_null()).pivot(on="dataset", index="algo", values="peak_rss_mb")
        sizes = x.filter(pl.col("error").is_null()).group_by("dataset").agg(pl.col("n_events").first(), pl.col("n_ids").first()).sort("n_events")
        out += [f"## Ladder: {lad}\n", md_table(sizes), "", "Runtime (s):\n", md_table(rt.sort("algo"), 2), "",
                "Peak RSS (MB, includes the loaded frame):\n", md_table(mem.sort("algo"), 0), ""]
    tp = b.filter(pl.col("error").is_null()).sort("n_events").group_by("algo", "ladder").agg(
        pl.col("eps").last().round(0).alias("events_per_s_at_largest"), pl.col("n_events").max().alias("largest_ok"))
    out += ["## Throughput at the largest completed size\n",
            md_table(tp.pivot(on="ladder", index="algo", values="events_per_s_at_largest").sort("algo"), 0), ""]
    return "\n".join(out)


def downstream() -> str:
    p = RESULTS / "downstream.parquet"
    if not p.exists():
        return "# Downstream\n\n(not run)\n"
    d = pl.read_parquet(p).with_columns(profile=pl.col("dataset").map_elements(profile_of, return_dtype=pl.String))
    out = ["# Downstream analytics (synthetic eval datasets, min duration 15 min)\n"]

    def piv(stage, metric, on="method", index=("algo",)):
        x = d.filter((pl.col("stage") == stage) & (pl.col("metric") == metric))
        return x.group_by(list(index) + [on]).agg(pl.col("value").mean()).pivot(on=on, index=list(index), values="value").sort(list(index))

    out += ["## Place grouping: ARI vs true places (mean over datasets)\n", md_table(piv("places", "ari"), 2), ""]
    out += ["## Place grouping by profile (dbscan)\n",
            md_table(d.filter((pl.col("stage") == "places") & (pl.col("metric") == "ari") & (pl.col("method") == "dbscan"))
                     .group_by("algo", "profile").agg(pl.col("value").mean()).pivot(on="profile", index="algo", values="value").sort("algo"), 2), ""]
    out += ["## Home / work inference accuracy (dbscan places)\n",
            md_table(d.filter(pl.col("stage") == "home_work").group_by("algo", "profile", "method").agg(pl.col("value").mean())
                     .pivot(on=["profile", "method"], index="algo", values="value").sort("algo"), 2), ""]
    nl = d.filter(pl.col("stage").str.starts_with("nextloc") & (pl.col("metric") == "acc@3"))
    out += ["## Next location: acc@3 by model (truth = model ceiling on true sequences)\n",
            md_table(nl.group_by("stage", "method").agg(pl.col("value").mean()).pivot(on="method", index="stage", values="value").sort("stage"), 3), "",
            "### acc@3, markov1_time, per detector\n",
            md_table(nl.filter(pl.col("method") == "markov1_time").group_by("algo", "stage").agg(pl.col("value").mean())
                     .pivot(on="stage", index="algo", values="value").sort("algo"), 3), ""]
    for metric in ("pair_f1", "episode_f1", "decoy_fp_rate"):
        out += [f"## Colocation same-time: {metric}\n", md_table(piv("colocation_same_time", metric), 3), ""]
    cr = d.filter(pl.col("stage") == "colocation_radius", pl.col("metric").is_in(["pair_recall", "pair_precision", "decoy_fp_rate"]),
                  pl.col("algo").is_in(["ua_segmentation", "ua_mahalanobis", "st_dbscan", "li_centroid", "fastmob_stays"]))
    out += ["## Colocation radius sweep: recall / precision / decoy false-positive rate (by profile)\n",
            md_table(cr.group_by("profile", "algo", "method", "metric").agg(pl.col("value").mean())
                     .pivot(on="metric", index=["profile", "algo", "method"], values="value").sort("profile", "algo", "method"), 3), ""]
    out += ["## Colocation different-time: pair F1\n", md_table(piv("colocation_diff_time", "pair_f1"), 3), ""]
    return "\n".join(out)


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("accuracy", accuracy), ("sensitivity", sensitivity), ("uncertainty_lift", uncertainty),
                     ("scaling", scaling), ("downstream", downstream)):
        (DOCS / f"{name}.md").write_text(fn())
        print("wrote", DOCS / f"{name}.md")


if __name__ == "__main__":
    main()
