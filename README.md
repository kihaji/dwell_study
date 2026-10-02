# dwells: focus/ua-mahalanobis

The focused branch for the uncertainty-aware dwell detector `ua_mahalanobis`. It covers:
* tuning the detector and its input data contract;
* tightened output tables for downstream analysis;
* a pattern-of-life analyst app (Plotly Dash + deckgl-dash).

The full multi-algorithm comparison lives on `main` (`docs/REPORT.md` there). The branch
plan is in `PLAN_UA.md`.

```bash
uv sync --extra app
uv run pytest

# data (synthetic catalog + OSM geocode index; once)
uv run python -m dwells.generator.build --group dev
uv run python -m dwells.generator.build --group eval
uv run python -c "from dwells.pol.geocode import build_index_for_area as b; b('urban'); b('suburban')"

# tuning / robustness (WSL2: wrap heavy runs in a memory cap)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 uv run python -m dwells.harness.tune baseline
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 uv run python -m dwells.harness.tune robustness

# pattern-of-life outputs, then the app
uv run python -m dwells.pol.build --datasets "eval_*,dev_*,staga"
uv run --extra app python app/app.py        # http://127.0.0.1:8050
```

Docs:
* `docs/analyst_guide.md`: every view and metric, for analysts.
* `docs/data_contract.md`: what the cleaning pipeline must deliver, with measured impact.
* `docs/tuning.md`: algorithm changes, sweeps, presets, held-out results.
* `docs/outputs.md`: output table reference.
* `docs/algorithms/ua_mahalanobis.md`, `docs/algorithms/downstream.md`, `docs/uncertainty.md`,
  `docs/metrics.md`, `docs/datasets.md`: methods and data.
