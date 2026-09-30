"""Weighted scorecard from eval, downstream, benchmark results and the qualitative rubric.

    uv run python -m dwells.harness.score

Components (each scaled to [0, 1] across algorithms, then weighted by weights.toml):
* accuracy     0.6 x mean F1 on synthetic eval datasets (md 900 and 1800; uncertainty-aware
               algorithms at shape 3_mixed) + 0.2 x mean F1 on labeled public datasets
               (staga, pol_50) + 0.2 x mean place ARI (dbscan grouping); divided by the best.
* scalability  0.5 x log10 throughput (events/s at the largest completed rung, min-max over
               algorithms) + 0.5 x fraction of benchmark rungs completed (all ladders).
* uncertainty  mean F1 gain from shape 1 to 3_mixed on cdr + adtech eval datasets (md 900,
               1800), clipped at 0, divided by the best gain (0 for error-blind methods).
* spark / extensibility / ease_of_use  rubric.toml score / 5.
Writes results/scorecard.parquet and docs/results/scorecard_table.md.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from dwells import algorithms
from dwells.harness.run import RESULTS, profile_of

HERE = Path(__file__).parent
DOCS = HERE.parents[2] / "docs" / "results"


def _eval():
    e = pl.read_parquet(RESULTS / "eval.parquet").filter(pl.col("error").is_null())
    return e.with_columns(md=pl.col("params").map_elements(lambda p: json.loads(p)["min_duration_s"], return_dtype=pl.Int64),
                          profile=pl.col("dataset").map_elements(profile_of, return_dtype=pl.String))


def accuracy(e: pl.DataFrame) -> pl.DataFrame:
    ua = {a for a in algorithms.names() if algorithms.get(a).USES_UNCERTAINTY}
    use = e.filter(pl.col("md").is_in([900, 1800])).filter(
        pl.when(pl.col("algo").is_in(list(ua))).then(pl.col("shape") == "3_mixed").otherwise(pl.col("shape") == "1"))
    syn = use.filter(pl.col("dataset").str.starts_with("eval_")).group_by("algo").agg(pl.col("f1").mean().alias("f1_synthetic"))
    pub = use.filter(pl.col("dataset").is_in(["staga", "pol_50"])).group_by("algo").agg(pl.col("f1").mean().alias("f1_public"))
    out = syn.join(pub, on="algo", how="left")
    ds = RESULTS / "downstream.parquet"
    if ds.exists():
        d = pl.read_parquet(ds).filter((pl.col("stage") == "places") & (pl.col("method") == "dbscan") & (pl.col("metric") == "ari"))
        out = out.join(d.group_by("algo").agg(pl.col("value").mean().alias("place_ari")), on="algo", how="left")
    else:
        out = out.with_columns(place_ari=pl.lit(None, dtype=pl.Float64))
    raw = 0.6 * pl.col("f1_synthetic") + 0.2 * pl.col("f1_public").fill_null(0) + 0.2 * pl.col("place_ari").fill_null(0)
    return out.with_columns(accuracy_raw=raw)


def uncertainty(e: pl.DataFrame) -> pl.DataFrame:
    s = e.filter(pl.col("dataset").str.starts_with("eval_") & pl.col("profile").is_in(["cdr", "adtech"])
                 & pl.col("md").is_in([900, 1800]) & pl.col("shape").is_in(["1", "3_mixed"]))
    p = s.pivot(on="shape", index=["algo", "dataset", "md"], values="f1", aggregate_function="mean")
    if "3_mixed" not in p.columns:
        p = p.with_columns(pl.lit(None, dtype=pl.Float64).alias("3_mixed"))
    p = p.with_columns(gain=(pl.col("3_mixed") - pl.col("1")).fill_null(0.0))
    return p.group_by("algo").agg(pl.col("gain").mean().alias("f1_gain_from_error"))


def scalability() -> pl.DataFrame:
    b = pl.read_parquet(RESULTS / "bench.parquet")
    from dwells.harness.bench import LADDERS

    n_rungs = sum(len(v) for v in LADDERS.values())
    ok = b.filter(pl.col("error").is_null())
    thr = ok.with_columns(tp=pl.col("n_events") / pl.col("runtime_s")).sort("n_events").group_by("algo").agg(
        pl.col("tp").last().alias("throughput_eps"), pl.col("n_events").max().alias("max_events"),
        pl.len().alias("rungs_ok"))
    return thr.with_columns(rung_frac=pl.col("rungs_ok") / n_rungs)


def _minmax(s: pl.Series) -> pl.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi and hi > lo else s * 0 + 1


def main():
    w = tomllib.loads((HERE / "weights.toml").read_text())
    rub = tomllib.loads((HERE / "rubric.toml").read_text())
    e = _eval()
    df = pl.DataFrame({"algo": algorithms.names()})
    df = df.join(accuracy(e), on="algo", how="left").join(uncertainty(e), on="algo", how="left")
    if (RESULTS / "bench.parquet").exists():
        df = df.join(scalability(), on="algo", how="left")
    df = df.with_columns(pl.col("rungs_ok").fill_null(0), pl.col("rung_frac").fill_null(0.0))
    df = df.with_columns(
        s_accuracy=pl.col("accuracy_raw") / pl.col("accuracy_raw").max(),
        s_uncertainty=(pl.col("f1_gain_from_error").clip(0) / pl.col("f1_gain_from_error").clip(0).max()).fill_null(0),
        s_spark=pl.Series([rub[a]["spark"][0] / 5 for a in df["algo"]]),
        s_extensibility=pl.Series([rub[a]["extensibility"][0] / 5 for a in df["algo"]]),
        s_ease_of_use=pl.Series([rub[a]["ease_of_use"][0] / 5 for a in df["algo"]]),
    )
    lt = pl.Series(np.log10(df["throughput_eps"].fill_null(1.0).to_numpy()))
    df = df.with_columns(s_scalability=0.5 * _minmax(lt) + 0.5 * pl.col("rung_frac"))
    df = df.with_columns(score=sum(w[k] * pl.col(f"s_{k}") for k in w)).sort("score", descending=True)
    df.write_parquet(RESULTS / "scorecard.parquet")
    DOCS.mkdir(parents=True, exist_ok=True)
    cols = ["algo", "score", "s_accuracy", "s_scalability", "s_uncertainty", "s_spark", "s_extensibility",
            "s_ease_of_use", "f1_synthetic", "f1_public", "place_ari", "f1_gain_from_error", "throughput_eps", "max_events"]
    t = df.select([c for c in cols if c in df.columns]).with_columns(pl.col(pl.Float64).round(3))
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    for r in t.iter_rows():
        lines.append("| " + " | ".join("" if v is None else (f"{v:,.0f}" if isinstance(v, float) and v > 100 else str(v)) for v in r) + " |")
    (DOCS / "scorecard_table.md").write_text(
        f"Weights: {w}\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
