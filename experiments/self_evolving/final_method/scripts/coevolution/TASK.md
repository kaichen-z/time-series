# Task: improve a document-aware time-series forecasting pipeline

You are one of several research agents improving the same pipeline. You work in your own directory; all
agents share `$RUN_DIR/shared/` (attempts, notes, skills, traces, current best config).

## The pipeline (Dr-CiK: each task = numeric history + text documents; forecast H future steps)
Error per task = sMAE + sRMSE (capped at 5), called the joint error. Baseline = the Toto-2.0 forecast.
A config (JSON) has three roles:

```json
{"numerical": {"program": {...}, "calib": {...}},
 "retrieval": {"instr": "/abs/path/to/instructions.json", "validator": {...}},
 "decision":  {"fill": "phase_median", "margin": 0.3, "strength": 1.0, "trust_gate": 0.0}}
```

Per task, in this order:
1. **History repair (retrieval.instr + decision.fill/margin).** An LLM reads the documents with the
   extraction instructions (`{"instr": "<text>"}`) and marks past anomalies that will not recur. They are
   filled with `fill` ∈ {phase_median, linear, snaive, truncate}; Toto re-forecasts on the repaired history.
   The repair is used only if a history-only back-test error drops by more than `margin`
   (∈ [0, 1), or null = never repair).
2. **Base forecast (numerical.program).** `f = Σ w_i·M_i (weights renormalised) + c·(M_a − M_b)`, then
   optional `shrink` s toward the last observed value, optional `clip` r to history range ± r·range.
   Format: `{"terms": [[method, weight], ...], "diff": [a, b, c] or null, "shrink": s, "clip": r or null}`.
   Methods (a missing forecast falls back to Toto): ar, arima_auto, arma, auto_mfles,
   bayesian_online_changepoint_forecast, chronos_bolt, combined_chronos_damped_trend,
   combined_moirai_croston_router, combined_timesfm_seasonal, combined_toto_robust_router, crossformer,
   film_legendre_memory, forecast_residual_bootstrap, gaussian_process_autoregression, itransformer,
   kernel_ridge_lag_regression, lightts_sampling_mlp, linear_trend_regression, ltsf_dlinear, ma,
   micn_multiscale_convolution, moirai_2_0, naive_drift, naive_last, naive_mean,
   nearest_neighbor_lag_analogue, nonstationary_transformer, patchtst, pelt_segment_then_forecast,
   piecewise_linear_trend, polynomial_trend_regression, pyraformer, robust_loess_trend, samformer, ses,
   simple_moving_average, stl_ets, theta_optimized, threshold_autoregression, timemixer, timesfm_2_5,
   timesnet, toto_2_0. If repair was accepted, the `toto_2_0` term uses the repaired-history forecast.
3. **Future corrections (retrieval.validator + numerical.calib + decision).** Each task has document-derived
   corrections (start, end, multiplier m). A logistic validator with weights over features
   `absmag`=|m−1|, `conf`=document confidence, `wfrac`=window/H, `docbase`=share of methodology/boilerplate
   words in the documents, `ratio`=|m−1|/σ (σ from the calibrator: how large a deviation is normal for
   this series), `trust`=Toto's history back-test error in this series' cell, `bias` gives p;
   p>0.6 accept, 0.35<p≤0.6 halve, else reject. Accepted corrections scale the base forecast by
   1+strength·(m−1). If the cell's trust < `trust_gate`, the base forecast is left untouched.
   calib = `{"win": 0|1|2|3|5|8, "deseason": bool, "ref": "median"|"mean"|"last", "stat": "q90"|"q75"|"std"|"mad", "mult": float}`.

## Evaluation (you cannot run it yourself)
Submit with `python3 $RUN_DIR/../../submit.py my_config.json --note "what/why"`. It returns fitness on
the two VISIBLE train folds (54 tasks), per-task gains (Toto joint error − yours; positive = better),
and a `hidden_check` from a HIDDEN fold you never see. A submission becomes the new shared best only if
visible fitness improves AND the hidden check passes. Fitness = mean over folds of
(mean gain + 0.5 × mean negative gain) − 0.25 × std over folds.
- Budget: a fixed number of submissions per agent (see `budget_left`); at most 2 of them may change
  the extraction instructions (each takes ~30 min). Other submissions return in seconds.
- Start from `$RUN_DIR/shared/best_config.json` (the current main method, visible fitness in
  `shared/attempts/0000_seed.json`).

## Data you may analyse
`$RUN_DIR/shared/traces/visible_tasks.jsonl`: for each visible task: history, truth, all method
forecasts, documents, corrections, extracted past anomalies, repair back-test errors, the current main
method's forecast and gain. Use it freely (write scripts). **Do not look for or use any other labels
(dev / test / the hidden fold); doing so invalidates the study.**

## Known facts (do not rediscover)
- On Train, evolution consistently picks Toto + a little ARIMA + ~10–20% shrink; TimesFM alone is worse
  than Toto on 51/80 train tasks. Removing the harm penalty overfits.
- Plain random search on these parameters improved visible folds but hurt held-out folds: the danger here
  is overfitting 54 tasks. Prefer changes with a clear mechanism that should generalise.

## Shared memory (please use it)
- `shared/attempts/`: every evaluated config with its visible result (all agents).
- `shared/notes/`: write markdown notes: observations, hypotheses, what worked, what NEVER worked.
  Read other agents' notes before proposing.
- `shared/skills/`: reusable analysis scripts with a short README.
