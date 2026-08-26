# Guarded Dynamic Combined Forecasting in Phase B

Date: 2026-08-24

## Outcome

Phase B now supports task-specific forecast-level combinations between one reviewed TSFM and one
Python statistical forecaster. It does not merge model parameters and it does not create new
checkpoints. The selector searches only a bounded portfolio:

```text
Top 2 eligible TSFMs x Top 3 eligible statistical methods
  -> asymmetric weighted blends
  -> clipped residual corrections
  -> history-only fold validation
  -> stable TSFM baseline protection
  -> Combined forecast or safe single-method fallback
```

The implementation is in `numerical_agent/evolution/numerical_selector.py`. The typed policy and
its self-evolution schema are in `numerical_agent/evolution/selector_evolution.py`.

## Combined operators

For TSFM forecast `F` and statistical forecast `S`, the bounded search supports:

1. **Weighted blend:** `w * F + (1 - w) * S`, with TSFM-heavy weights.
2. **Clipped residual correction:** `F + alpha * clip(S - F)`, where the clipping scale is
   calculated from the history available at the corresponding forecast cutoff.

The current Train-selected parent policy uses TSFM weights `0.8` and `0.9`, residual strengths
`0.1` and `0.25`, and a correction clip multiplier of `1.0`.

## Acceptance gates

A Combined candidate may replace its best parent and the stable Toto/TimesFM baseline only when:

- its median history-only fold score improves by at least 5%;
- it wins at least two folds;
- its normalized regret on every fold, including the latest fold, is at most 2%;
- both parents pass the existing reliability and catastrophic-MASE gates;
- the parent forecasts satisfy the minimum diversity threshold.

The same stable-baseline gate now applies to a single statistical or fixed Combined challenger.
If the challenger does not pass, the selector returns Toto, TimesFM, or the next eligible TSFM.

## Self-evolving policy fields

The Meta-Harness Engineer may propose typed changes to:

- the TSFM-heavy weight grid;
- residual-correction strengths and clipping multiplier;
- minimum fold wins;
- minimum median improvement;
- maximum worst-fold regret;
- the existing ranking and reliability thresholds.

It still cannot change method identities, TSFM checkpoints, the screening policy, task splits,
runtime bindings, scorer, cache, or labels. Legacy frozen DecisionPolicy files remain readable;
newly rendered policies contain the complete guarded-Combined schema.

## Exploratory cached 80/20 result

The implementation was checked without executing any model again by replaying the existing
80-Train/20-Dev DecisionCase artifacts from
`runs/numerical_selector/formal_80_20_fallback_20260823`. Candidate thresholds were searched on
Train only; the existing Dev partition was then used once for confirmation in this experiment.
No Public Test task was accessed by this comparison.

| Policy on the same 20 Dev cases | Mean MASE | Median MASE | Mean MAE | Mean sMAPE | Catastrophic rate |
|---|---:|---:|---:|---:|---:|
| Fixed Toto | 3.356899 | 2.947492 | 170.602391 | 34.606513 | 5% |
| Previously frozen selector report | 3.375291 | 2.135086 | 165.914357 | 32.856850 | 10% |
| Guarded dynamic Combined v1 | **3.224184** | **2.135086** | 166.271456 | 33.392261 | **5%** |

On these cached Dev cases, guarded dynamic Combined reduced mean MASE by 4.48% relative to the
previously frozen selector and by 3.95% relative to fixed Toto. It selected a dynamic Combined
forecast on 2 of 20 tasks: one weighted blend and one clipped residual correction. This is an
exploratory result on an already-used Dev partition, not a new held-out generalization claim.

## Dr-CiK-aligned point-metric rescore

On 2026-08-25, the same cached 20 Dev cases were rescored using the paper's point-forecast
definition: divide each task's MAE/RMSE by the mean absolute true future value, cap each scaled
task score at `5.0`, and aggregate over tasks. No model was run and sCRPS was not computed.

| Policy | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90 sMAE | P95 sMAE | sMAE/sRMSE clipped tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Toto | **0.304075** | 0.050534 | **0.523644** | 0.085192 | 0.568006 | 0.799188 | 0 / 0 |
| Previous selector without dynamic combinations | 0.331969 | 0.091199 | 0.526128 | 0.107282 | **0.492357** | **0.612788** | 0 / 0 |
| Guarded dynamic Combined v1 | 0.338401 | 0.091341 | 0.532460 | 0.107012 | **0.492357** | 0.612896 | 0 / 0 |

The earlier MASE gain does not survive the benchmark-aligned primary metric: dynamic Combined v1
has worse mean sMAE and mean sRMSE than both fixed Toto and the previous selector. Its upper-tail
sMAE is lower than Toto's, but the newly implemented Dev gate correctly rejects it because mean
sMAE does not improve and sRMSE regresses. MASE is therefore retained only as a diagnostic.

## Train-only evolution with a single Dev gate

On 2026-08-25, the selector protocol was changed so that generations never inspect Dev. Each LLM
proposal is expanded by trusted Python into at most 55 typed policies spanning weighted blend,
clipped residual correction, and both operators. The bounded search uses only 80 Train tasks.
After all three Train generations, the final Train winner is evaluated once on 20 Dev tasks.

| Policy | Train sMAE | Train sRMSE | Dev sMAE | Dev sRMSE | Result |
|---|---:|---:|---:|---:|---|
| Fixed Toto reference | 0.312303 | **0.505928** | 0.304075 | 0.523644 | Reference baseline |
| Safe-anchor parent | 0.319271 | 0.509747 | **0.278341** | **0.475353** | Frozen |
| Final Train winner | **0.309641** | **0.507612** | 0.287367 | 0.491571 | Rejected by Dev sRMSE gate |

Train generations 1 and 3 found admissible improvements. Generation 2 returned an invalid weight
and was rejected without changing the parent. The final Train winner was not frozen because it
failed the single read-only Dev gate. The accepted frozen policy therefore remained the safe-anchor
parent. Both learned policies beat fixed Toto on Dev, but the acceptance decision compares the
Train winner against the stronger current parent, not merely against Toto. This run took 88.85
seconds with cached forecasts and hindcasts.

## 99-task Public Regression Test

The 99-task split had already been consumed by earlier experiments, so this is explicitly a Public
Regression Test rather than a pristine generalization estimate. It was run only after the 80/20
policy was frozen, with zero LLM or mutation calls.

| Policy | Mean sMAE | Mean sRMSE | P90/P95 sMAE | sRMSE clipped | W/T/L vs Toto |
|---|---:|---:|---:|---:|---:|
| Fixed Toto | 0.460684 | **0.680614** | 1.145517 / 1.953672 | 2 | reference |
| Old full two-stage selector | 0.472474 | 0.688686 | not recorded in this report | not recorded | 41/7/51 |
| New safe-anchor full two-stage selector | **0.455015** | 0.683928 | **1.002484 / 1.713473** | 3 | **27/55/17** |

Relative to Toto, the new selector improves mean sMAE by 1.23% and materially reduces the upper
sMAE tail, but mean sRMSE is 0.49% worse and one additional task reaches the sRMSE clipping cap.
Relative to the old two-stage selector, mean sMAE improves by 3.69% and mean sRMSE by 0.69%.

No dynamic Combined forecast passed all history-only gates on these 99 tasks (`ensemble_rate = 0`).
The improvement comes from conservative single-method overrides plus safe fallback to Toto, not
from forecast blending. The next Combined iteration should therefore focus on better-calibrated
history-only gates rather than relaxing baseline protection.

Authoritative artifacts:

- `runs/numerical_selector/smae_train_only_combined_v3_80_20_20260825/selector_manifest.json`
- `runs/frozen_two_stage/public_regression_99_safe_anchor_combined_20260825/frozen_two_stage_results.json`
- `runs/frozen_two_stage/public_regression_99_safe_anchor_combined_20260825/evaluation_complete.json`
