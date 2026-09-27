# Evolution log: per-generation results and changes

## 0. Numerical part 1: combination-program dictionary (num_part1.py, PEN=3)

- Dictionary: the 31 original methods plus the statistical portfolio run by `stat_full.py`. Of the 93 statistical methods, 48 produce forecasts on Dr-CiK; the rest are intermittent-demand methods, methods that need long histories, or methods that do not support the task frequency. Together this gives 50 methods, and the **37 that cover at least 90% of Train tasks** are used (a missing forecast falls back to Toto).
- Generation 0 is the 37 single methods; the best of them at generation 0 is Toto, with fitness 0 relative to Toto.
- 12 parents and 24 children per generation, 30 generations.
- Fitness = mean gain vs Toto + 3 × mean negative gain on Train. Diversity is kept by leading method.

Best fitness per generation: 0.000 → 0.021 → 0.021 → 0.033 → 0.033 → 0.033 → 0.033 → 0.033 → 0.033 → 0.033 → 0.033 → 0.033 → 0.034 → 0.034 → 0.034 → 0.034 → 0.049 → 0.049 → 0.049 → 0.049 → 0.049 → 0.088 → 0.088 → 0.088 → 0.088 → 0.088 → 0.088 → 0.088 → 0.088 → 0.099 → 0.099

Best program after weight normalisation: **0.95·Toto + 0.05·arima_auto, shrunk 20% toward the last observation**, no difference term, no clipping.

| Rank | Fitness | Program |
|---|---|---|
| 1 | +0.0987 | `{"terms": [["toto_2_0", 1.1005169383119369], ["arima_auto", 0.059]], "diff": null, "shrink": 0.20153008320833654, "clip": null}` |
| 2 | -0.1426 | `{"terms": [["simple_moving_average", 0.29779519107337626], ["toto_2_0", 0.2236982709733526]], "diff": null, "shrink": 0.0, "clip": null}` |
| 3 | -0.2564 | `{"terms": [["toto_2_0", 0.21629149933713063], ["bayesian_online_changepoint_forecast", 0.2751553601453943]], "diff": ["forecast_residual_bootstrap", "kernel_ridge_lag_regression", 0.002], "shrink": 0.18083088582877407, "clip": 0.25}` |
| 4 | -0.4294 | `{"terms": [["stl_ets", 0.6289872537795519], ["samformer", 0.2323071530078904], ["bayesian_online_changepoint_forecast", 0.5313907251836608], ["pelt_segment_then_forecast", 0.4710073533139805]], "diff": null, "shrink": 0.08827201885195486, "clip": null}` |
| 5 | -0.4324 | `{"terms": [["ses", 0.9434314586984685], ["patchtst", 0.095], ["chronos_bolt", 0.42135598167679916], ["robust_loess_trend", 0.09087707803452005]], "diff": ["polynomial_trend_regression", "ar", 0.037], "shrink": 0.0, "clip": 0.1}` |
| 6 | -0.4534 | `{"terms": [["theta_optimized", 0.6289872537795519], ["samformer", 0.4682409345581937], ["bayesian_online_changepoint_forecast", 0.5157630283412165], ["stl_ets", 0.32207456259094736]], "diff": ["theta_optimized", "pelt_segment_then_forecast", -0.069], "shrink": 0.0, "clip": null}` |
| 7 | -0.4644 | `{"terms": [["theta_optimized", 0.6289872537795519], ["samformer", 0.6894763706304862], ["bayesian_online_changepoint_forecast", 0.5157630283412165], ["stl_ets", 0.35808040328839585]], "diff": ["theta_optimized", "pelt_segment_then_forecast", -0.069], "shrink": 0.02019483131291253, "clip": null}` |
| 8 | -0.5353 | `{"terms": [["naive_last", 0.9498975490719375], ["bayesian_online_changepoint_forecast", 0.14466993445659374], ["samformer", 0.432]], "diff": ["stl_ets", "simple_moving_average", 0.206], "shrink": 0.0, "clip": null}` |

Nested stratified CV (evolve on 2 train folds, score the 3rd), held-out sMAE / sRMSE vs Toto:
- fold 0: 0.3663 → 0.3147 / 0.5576 → 0.5006 (W/R 12/15);
- fold 1: 0.2951 → 0.2633 / 0.4757 → 0.4334 (W/R 8/19);
- fold 2: 0.5082 → 0.4893 / 0.7785 → 0.7634 (W/R 12/14).

The previous 31-method dictionary reached fitness 0.093 with 0.94·Toto + 0.06·ARMA (shrink 17%, clip 0.5).

## 0b. Numerical part 2 fill method + Decision repair margin (select_fill_gate.py)

- Search: 4 fill methods × 6 margins, evaluated with the full pipeline on Train (with the part-1 program above).
- Fitness: mean gain + 3 × mean negative gain.

Nested stratified CV (held-out joint-error reduction):
- fold 0 chose linear / 0.2 → held-out +0.230 (W/R 18/9);
- fold 1 chose phase_median / 0.1 → held-out +0.155 (13/14);
- fold 2 chose phase_median / 0.3 → held-out +0.338 (18/8).

| Rank | Train fitness | Fill | Margin |
|---|---|---|---|
| 1 | +0.1876 | phase_median | 0.3 |
| 2 | +0.1840 | phase_median | 0.2 |
| 3 | +0.1833 | snaive | 0.2 |
| 4 | +0.1821 | truncate | 0.2 |
| 5 | +0.1821 | linear | 0.2 |
| 6 | +0.1820 | truncate | 0.3 |
| 7 | +0.1820 | linear | 0.3 |
| 8 | +0.1819 | snaive | 0.3 |
| 9 | +0.1698 | truncate | None |
| 10 | +0.1698 | snaive | None |

## 1. Retrieval extraction instructions (tl2_evolve.py)

- Fitness: evidence F1 against `gt_evidence` on a 30-task train minibatch.
- Mutator: GPT-6-sol; 2 children per generation; a child is accepted only if F1 improves.

| Generation | Child F1 | Accepted? | Current best F1 |
|---|---|---|---|
| 0 (initial instructions) | — | — | 0.054 |
| 1 | 0.206, 0.014 | yes | 0.206 |
| 2 | 0.248, 0.276 | yes | 0.276 |
| 3 | 0.292, 0.258 | yes | 0.292 |
| 4 | 0.170, 0.464 | yes | 0.464 |

Full training set with the final instructions: F1 0.039 → 0.477, recall 5.4% → 48.3%, precision 9.1% → 67.0%. Only the final instruction text is saved (`artifacts/tl2_evolved_extraction_instructions.json`); intermediate texts were not stored.

## 2. Three-agent co-evolution (nrd4.py, full train, 3 seeds)

Team fitness = mean over 3 stratified train folds of (mean gain + 0.5 × mean negative gain) − 0.25 × fold std.
Generation 0 = exact Toto (fitness 0). A new team is kept only if its fitness beats the current team; otherwise the round rolls back.

### Seed 1

| Gen | Team fitness | Fold scores | Accepted? | Numerical calibrator (mult) | Decision (strength, trust gate) |
|---|---|---|---|---|---|
| 0 | +0.0000 | +0.000, +0.000, +0.000 | — | — | — |
| 1 | +0.0710 | +0.091, +0.025, +0.129 | **yes** | 1.000 | 0.68, 0.00 |
| 2 | +0.0754 | +0.080, +0.022, +0.170 | **yes** | 1.033 | 0.91, 0.00 |
| 3 | +0.0809 | +0.086, +0.022, +0.185 | **yes** | 1.033 | 1.00, 0.00 |
| 4 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 5 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 6 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 7 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 8 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 9 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 10 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 11 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 12 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 13 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 14 | +0.0809 | +0.086, +0.022, +0.185 | rolled back | 1.033 | 1.00, 0.00 |
| 15 | +0.0820 | +0.092, +0.022, +0.182 | **yes** | 1.022 | 1.00, 0.00 |

Retrieval validator self-evolution (per-correction credit, first round): 3.5879 → 5.2852 → 6.2187 → 6.2187 → 6.2187 → 6.2187
Numerical calibrator self-evolution (history-only error, first round): 0.6707 → 0.6707 → 0.6707 → 0.6707; last round: 0.6694 → 0.6694 → 0.6694 → 0.6694
Final validator weights: absmag +3.25, conf -0.07, wfrac +0.63, docbase -2.77, ratio -0.01, trust +1.02, bias -2.75

### Seed 2

| Gen | Team fitness | Fold scores | Accepted? | Numerical calibrator (mult) | Decision (strength, trust gate) |
|---|---|---|---|---|---|
| 0 | +0.0000 | +0.000, +0.000, +0.000 | — | — | — |
| 1 | +0.0807 | +0.082, +0.025, +0.185 | **yes** | 1.000 | 1.00, 0.07 |
| 2 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 3 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 4 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 5 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 6 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 7 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 8 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 9 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 10 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 11 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 12 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 13 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 14 | +0.0807 | +0.082, +0.025, +0.185 | rolled back | 1.000 | 1.00, 0.07 |
| 15 | +0.0820 | +0.092, +0.022, +0.182 | **yes** | 1.000 | 1.00, 0.07 |

Retrieval validator self-evolution (per-correction credit, first round): 0.0 → 3.9185 → 6.5352 → 6.8236 → 6.8236 → 7.6951
Numerical calibrator self-evolution (history-only error, first round): 0.6707 → 0.6707 → 0.6707 → 0.6707; last round: 0.6696 → 0.6696 → 0.6696 → 0.6696
Final validator weights: absmag +1.49, conf +0.58, wfrac +0.66, docbase -2.38, ratio +0.03, trust +0.34, bias -1.55

### Seed 3

| Gen | Team fitness | Fold scores | Accepted? | Numerical calibrator (mult) | Decision (strength, trust gate) |
|---|---|---|---|---|---|
| 0 | +0.0000 | +0.000, +0.000, +0.000 | — | — | — |
| 1 | +0.0790 | +0.082, +0.021, +0.185 | **yes** | 1.000 | 1.00, 0.16 |
| 2 | +0.0801 | +0.085, +0.021, +0.185 | **yes** | 1.000 | 1.00, 0.16 |
| 3 | +0.0802 | +0.086, +0.021, +0.185 | **yes** | 1.027 | 1.00, 0.16 |
| 4 | +0.0808 | +0.086, +0.022, +0.185 | **yes** | 1.027 | 1.00, 0.16 |
| 5 | +0.0809 | +0.086, +0.022, +0.185 | **yes** | 1.027 | 1.00, 0.05 |
| 6 | +0.0828 | +0.092, +0.022, +0.185 | **yes** | 1.027 | 1.00, 0.07 |
| 7 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 8 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 9 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 10 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 11 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 12 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 13 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 14 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |
| 15 | +0.0828 | +0.092, +0.022, +0.185 | rolled back | 1.027 | 1.00, 0.07 |

Retrieval validator self-evolution (per-correction credit, first round): 3.2516 → 5.0894 → 6.726 → 7.3474 → 7.5743 → 7.585
Numerical calibrator self-evolution (history-only error, first round): 0.6707 → 0.6702 → 0.6702 → 0.6702; last round: 0.6694 → 0.6694 → 0.6694 → 0.6694
Final validator weights: absmag +3.87, conf -1.18, wfrac +1.07, docbase -3.72, ratio +0.02, trust +0.77, bias -1.75

## 3. Numerical dictionary self-evolution (stage 1 of nrd11)

- Value per generation = mean over (frequency × seasonality) cells of [elite hindcast joint error − Toto hindcast joint error].
- Negative means the elites beat Toto. History only, no labels.

- seed 1: +0.092 → +0.016 → -0.006 → -0.009 → -0.024 → -0.025 → -0.030 → -0.033 → -0.067 → -0.069 → -0.069 → -0.069 → -0.069 → -0.069 → -0.069 → -0.069
- seed 2: +0.092 → +0.012 → +0.003 → +0.003 → -0.038 → -0.038 → -0.048 → -0.048 → -0.056 → -0.062 → -0.062 → -0.062 → -0.062 → -0.062 → -0.062 → -0.062
- seed 3: +0.092 → +0.081 → +0.008 → +0.008 → +0.008 → +0.008 → -0.019 → -0.032 → -0.032 → -0.032 → -0.032 → -0.046 → -0.046 → -0.052 → -0.054 → -0.054
