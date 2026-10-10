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

`view` (no labels): `dataset, H, freq, history, history_timestamps, future_timestamps, target_description,
method_forecasts` (frozen forecasts: toto_2_0 always; timesfm_2_5, moirai_2_0, chronos_bolt, granite_ttm_r2 where
cached; five frozen combined candidates `combined_*`), `documents` (list of `{document_id, content, events}`; `events`
are audited Haiku event cards `{time_start, time_end, direction (up/down/mixed/unknown), confidence, evidence}`).
Every fact and event card is dated strictly before the forecast origin (later or undated ones were removed); documents
marked `calendar: true` (TimesX holiday information) are deterministic calendar covariates known in advance.

## Evaluation (you cannot run it yourself)
`python3 {SUBMIT} my_module.py --role {ROLE_ARG} --note "what/why"`
Per task: relative gain = (Toto error - final error) / (mean Toto error), error = sMAE + sRMSE (each capped at 5).
Feedback fitness: (mean gain + 0.5 x mean negative gain) over the FEEDBACK tasks of this run (the only tasks you can see).
Synchronous rounds: every agent of a round starts from the same frozen champion; a submission is *eligible* if its
feedback fitness beats the round's frozen champion; at round close the best eligible submission becomes the next
champion. The final program of the whole evolution is chosen later on data groups you never see, so changes that only
fit the feedback tasks (rules for particular series/groups, memorised values) do not help. Pure Python +
math/statistics/numpy; no files, no network, no other data; keep it fast.

## Data you may analyse
- `$RUN_DIR/shared/store/` (no labels; the feedback tasks of this run) - a compact store: frozen forecasts are float32 in
  `forecasts.f32`, the rest in `views_meta.json` / `documents.json`. Read it with the bundled reader:
  `import sys; sys.path.insert(0, "$RUN_DIR/shared/store"); import viewstore; s = viewstore.Store("$RUN_DIR/shared/store")`,
  then `for k in s.keys(): view = s.view(k)` gives exactly the view dict your module receives (keys are opaque handles).
  The set can be large; sample keys for exploratory analysis.
- `$RUN_DIR/shared/traces/visible_summary.json`: aggregate diagnostics of the feedback tasks only (per (freq,H) cell with
  enough tasks, and overall): mean Toto error, mean seed gain, mean error of every frozen forecast. No per-task values.
- Submissions must be general programs: they are rejected if they exceed 20 KB, contain more than 300 numeric constants or
  2,000 characters of string data, use banned modules/builtins (encoders, compression, file/OS access, exec/eval, hashing),
  or reproduce many feedback tasks almost exactly (treated as memorisation).
{NOTES_TEXT}
