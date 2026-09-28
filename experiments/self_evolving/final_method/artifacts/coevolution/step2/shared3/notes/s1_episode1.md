# s1 episode 1: numerical probes

The visible joint error is `(MAE + RMSE) / mean(abs(truth))`, capped at 5. I wrote `ws_s1/numerical_probe.py` to replay numerical changes on the visible traces while holding the main method's correction multiplier fixed. It excludes the 13 tasks with repair variants because their repaired Toto forecast is unavailable in the trace. The replay exactly reconstructs the main forecast on uncorrected, unrepaired tasks.

The seed has effective weights 94.9% Toto and 5.1% ARIMA, with 20.15% shrink. On the 41 unrepaired tasks, shrinking less (10% or 15%) improved the local objective slightly. Both submissions improved visible fitness but **failed the hidden check**: 10% gave 0.1961; 15% gave 0.1944. A 5% Chronos Bolt blend also improved visible fitness to 0.1936 and **failed the hidden check**. These are examples of visible-only numerical tuning that should not replace the seed.

Clipping forecasts near the history range worsened the visible replay, especially task_61. Several rejected document corrections also look harmful under a direct full-application counterfactual (task_77, task_83, task_234, task_161); the existing conservative validator is useful.

Repair observations for others: phase_median back-test reductions are very large for task_43 and task_193, while most other repair-variant tasks have reductions below the current 30% margin. Task_193's linear variant had a lower back-test error than phase_median, but this single case is insufficient to select fill globally.
