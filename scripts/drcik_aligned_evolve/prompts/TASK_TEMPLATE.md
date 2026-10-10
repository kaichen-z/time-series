# Task: evolve one module of a document-aware forecaster on {DATASET_NAME} ({STAGE})

{DATASET_DESC}

## The pipeline (three modules, one agent role each)
- **Numerical** module `forecast(view) -> list[float]`: the base forecast (H floats).
- **Retrieval** module `retrieve(view) -> list[dict]`: turns the task's documents and their event cards into future
  corrections `{"start": int, "end": int, "multiplier": float, ...}` with `0 <= start < end <= H` (steps of the
  forecast window, end exclusive). Invalid corrections are dropped.
- **Decision** module `adjust(view) -> list[float]`: decides whether / how much to apply `view["corrections"]` to
  `view["base_forecast"]`; must return H floats. `view["toto_forecast"]` is also given.

{ROLE_TEXT}
The other modules are fixed to the current shared best (`$RUN_DIR/shared/best_forecast.py`, `best_retrieve.py`,
`best_adjust.py`); read them and their notes: your change must work WITH them.

`view` (no labels): `tid, dataset, H, freq, history, history_timestamps, future_timestamps, target_description,
method_forecasts` (frozen forecasts: toto_2_0 always; timesfm_2_5, moirai_2_0, chronos_bolt, granite_ttm_r2 where
cached; five frozen combined candidates `combined_*`), `documents` (list of `{document_id, content, events}`; `events`
are audited Haiku event cards `{time_start, time_end, direction (up/down/mixed/unknown), confidence, evidence}`).

## Evaluation (you cannot run it yourself)
`python3 {SUBMIT} my_module.py --role {ROLE_ARG} --note "what/why"`
Per task: relative gain = (Toto error - final error) / (mean Toto error), error = sMAE + sRMSE (each capped at 5).
Fitness: mean over the two VISIBLE folds of (mean gain + 0.5 x mean negative gain) - 0.25 x std. Synchronous rounds:
every agent of this round starts from the same frozen champion; a submission is *eligible* if its visible fitness beats
the round's frozen champion AND the HIDDEN fold does not get worse; at round close the best eligible submission becomes
the next champion. Pure Python + math/statistics/numpy; no files, no network, no other data; keep it fast (all Train
tasks are run per submission).

## Data you may analyse
- `$RUN_DIR/shared/views_train.json` (no labels; all tasks of this run).
- `$RUN_DIR/shared/traces/visible.jsonl`: VISIBLE-fold tasks only, with truth, per-method errors, the seed's base and
  final forecasts and gains. Never look for other labels (hidden fold, official Dev or Test).
{NOTES_TEXT}
