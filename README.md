# dwells: focus/ua-mahalanobis

A focused reference for porting the uncertainty-aware dwell detector `ua_mahalanobis` and
its pattern-of-life outputs into the application. It contains:

* the detector (`src/dwells/algorithms/`) with tuned presets (`src/dwells/presets.py`);
* place grouping and the pattern-of-life output tables (`src/dwells/downstream/places.py`,
  `src/dwells/pol/`);
* the input / output schemas and the uncertainty model (`schema.py`, `uncertainty.py`);
* an analyst app (Plotly Dash + deckgl-dash) in `app/`.

Elsewhere:
* the multi-algorithm comparison lives on `main`;
* the evaluation tooling (synthetic data generator, tuning harness, defect injection,
  scoring, public-data loader) lives on `tools/ua-tuning`.

```bash
uv sync --extra app
uv run pytest

# pattern-of-life outputs for the existing repo datasets (data/, git-ignored), then the app
uv run python -m dwells.pol.build --datasets "eval_*,dev_*,staga"
uv run --extra app python app/app.py        # http://127.0.0.1:8050
```

Docs:
* `docs/analyst_guide.md`: every view and metric, for analysts.
* `docs/data_contract.md`: what the cleaning pipeline must deliver, with measured impact.
* `docs/outputs.md`: output table reference.
* `docs/tuning.md`: algorithm changes, presets and the evidence behind them.
* `docs/algorithms/ua_mahalanobis.md`, `docs/algorithms/downstream.md`, `docs/uncertainty.md`:
  methods.
