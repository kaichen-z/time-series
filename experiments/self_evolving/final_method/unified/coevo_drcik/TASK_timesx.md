# Task: co-evolution of a document-aware forecaster on ONE dataset (two agent roles)

The same pipeline code is used on three datasets, but THIS run evolves it on ONE dataset only: **timesx** (all views
have `dataset == "timesx"`). The three datasets are:
- **drcik** (Dr-CiK): event-type documents with distractors; corrections from documents matter a lot. 80 Train tasks.
- **tmmd** (Time-MMD): background reports; monthly/weekly/daily. 477 Train tasks.
- **timesx** (TimesX): commodity prices, FX rates, Google-search trends; retrospective news. 2173 Train tasks.

The pipeline has two evolvable modules, owned by two agent ROLES that take turns (co-evolution):
- **Numerical role** writes `forecast(view) -> list[float]` (the base forecast, H floats).
- **Decision role** writes `adjust(view) -> list[float]` (uses document corrections `view["corrections"]` — windows
  [start, end) and multipliers — to adjust `view["base_forecast"]`; must return H floats).
In each phase only one role may submit; the other role's module is fixed to the current shared best. Your role and
the current phase are given below. Read the other role's current code (`shared/best_forecast.py`,
`shared/best_adjust.py`) and its notes: your change must work WITH it.

`view` (no labels): `tid, dataset, part, domain, variable, seen_variable, H, freq, history, method_forecasts`
(toto_2_0, timesfm_2_5, and where available moirai_2_0, chronos_bolt, arima_auto and ~30 more for drcik),
`toto_forecast_repaired_history` (drcik only; Toto on a document-repaired history, or None), `backtests` (the task's
own history cut 1/2/3 horizons before the end: Toto and TimesFM forecasts and the realised `target`), and the
correction inputs (`corrections, documents, doc_confidence, docbase, cell, cell_toto_backtest_error,
task_toto_backtest_error, sigma_main_calib`). In the Decision phase `view` also has `base_forecast` (from the
current best forecast) and `toto_forecast`.

## Start point (seed, the current unified framework)
- `forecast`: per-dataset choices made on each Train set (drcik: 0.854 Toto[repaired] + 0.046 ARIMA + 0.1 TimesFM,
  shrink 0.2; tmmd: 0.4 Toto + 0.6 TimesFM; timesx: per frequency Toto/TimesFM/seasonal blends) + a cold start for
  variables unseen in Train (choose the blend by the task's own `backtests`).
- `adjust`: the Dr-CiK-evolved correction function (routes by task cell; rejects nearly all corrections on tmmd/timesx).

## Evaluation (you cannot run it yourself)
`python3 $RUN_DIR/../../submit_x.py my_code.py --role <numerical|decision> --note "what/why"`.
Per task: relative gain = (Toto error − final error) / (mean Toto error of that dataset), error = sMAE + sRMSE (cap 5).
Fitness (this dataset only): mean over VISIBLE folds 0,1 of (mean gain + 0.5 × mean negative gain) − 0.25 × std.
Accepted as new shared best only if visible fitness improves AND the HIDDEN fold does not get worse.
Pure Python + math/statistics/numpy; no files, no network; all Train tasks are run per submission (keep it fast).

## Data you may analyse
- `shared/views_train_timesx.json` (no labels); `shared/traces/visible_timesx.jsonl`: visible tasks with truth,
  per-method errors, the seed's base and final forecasts and gains. Never look for other labels (dev / test / hidden).
- `shared/numerical/`, `shared/decision/`, `shared/attempts/`: all submissions with results.

## Lessons from earlier rounds (single-dataset code evolution overfitted)
- Task-specific branches (exact cells, thresholds that fire on 1–3 tasks) pass visible folds and then fail on dev.
  Prefer general mechanisms; check how many tasks a change touches.
- Text corrections help on drcik and are unreliable on tmmd/timesx (direction accuracy ~55%).
- Own-history backtests are a label-free signal; they helped a lot for unseen variables.

## Shared memory (please use it)
`shared/notes/` (write what you tried, what worked, what NEVER worked; say which role you are) and `shared/skills/`.
