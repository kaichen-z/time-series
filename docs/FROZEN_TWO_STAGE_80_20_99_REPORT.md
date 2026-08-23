# Frozen Two-Stage Numerical Agent: 80/20/99 Evaluation

Date: 2026-08-23 (Asia/Shanghai)

## Protocol

The 199 public Dr-CiK tasks were frozen into 80 Train, 20 Dev, and 99 Public Test
tasks. The Public Test labels were not used while evolving either component.

1. A single-agent screening loop evolved a task-conditioned dictionary on Train and
   accepted changes only through Dev gates.
2. A history-only Numerical Selector was evolved on Train and accepted only when its
   complete Dev reward improved.
3. Both policies were frozen by SHA-256 and evaluated once on all 99 Public Test tasks.
   The final Test run made zero LLM calls and zero mutation calls.

The candidate pool contained 103 candidates: 93 Python statistical forecasting
methods, five reviewed TSFM runtimes, and five Combined candidates.

## Phase A: task-conditioned screening

Frozen screening policy SHA-256:
`9fb0be569e2bc4597a206ba9e4b0fa44aa99f0abbb07bdee82c219e74fa95096`

Two screening generations were accepted. On the 20-task Dev split, the frozen policy:

- Retained the best available candidate on 20/20 tasks;
- Reduced the average active dictionary from 103 to 56.05 candidates;
- Increased active-candidate success rate from approximately 46.31% to 85.10%;
- Reduced Crash/Invalid exposure from approximately 46.07% to 1.16%;
- Preserved all three candidate families.

This establishes that the screening loop learned a safer and substantially smaller
task-conditioned active dictionary. It does not by itself establish better final
forecasts.

## Phase B: history-only Numerical Selector

Frozen decision policy SHA-256:
`9b3682bd46b7bf739f8ad8c02f8bd9d9384d167e9a771d2000c9991643c4d8a7`

One Selector generation was accepted. Relative to its Parent on Dev:

| Metric | Parent | Frozen Child |
|---|---:|---:|
| Coverage | 1.0000 | 1.0000 |
| Mean MASE | 4.421047 | 3.375291 |
| Median MASE | 2.369970 | 2.135086 |
| Mean MAE | 169.6828 | 165.9144 |
| Mean sMAPE | 38.4697 | 32.8569 |
| Catastrophic rate (MASE > 10) | 0.1500 | 0.1000 |

The accepted Child therefore improved the frozen Dev objective without losing
coverage.

## Phase C: one-time 99-task Public Test

| Method | Mean MASE | Median MASE | Mean RMSSE | Mean MAE | Mean sMAPE | Coverage | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current global ranker / Toto 2.0 | 2.883039 | 1.580000 | 2.799772 | 555.7281 | 45.6042 | 1.0000 | 0.0404 |
| Screening only | 2.883039 | 1.580000 | 2.799772 | 555.7281 | 45.6042 | 1.0000 | 0.0404 |
| Decision only | 3.184596 | 1.796324 | 2.876107 | 1340.1985 | 44.5240 | 1.0000 | 0.0505 |
| Full two-stage system | 3.184596 | 1.796324 | 2.876107 | 1340.1985 | 44.5240 | 1.0000 | 0.0505 |

Against the current global-ranker baseline, the full system achieved 41 wins, seven
ties, and 51 losses by per-task MASE. It reduced mean sMAPE by 2.37%, but worsened
mean MASE by 10.46%, median MASE by 13.69%, mean RMSSE by 2.73%, and mean MAE by
141.16%.

The large mean-MAE degradation is dominated by a severe error on `task_200`, where
the Selector chose `micn_multiscale_convolution`. However, removing that single task
does not reverse the MASE conclusion: the full system still has more per-task losses
than wins and a worse median MASE.

## Interpretation

The experiment supports the screening contribution but does not support the current
Selector as an accuracy improvement:

- Screening successfully removes unsafe or irrelevant candidates while retaining the
  task oracle, but it does not change the global baseline because Toto 2.0 remains
  active on every task.
- Decision-only and full two-stage results are identical, showing that every method
  selected by the current Selector already survives screening. The screening and
  decision components are therefore not yet functionally coupled at the final-choice
  boundary.
- The Selector's Dev gain did not generalize to Public Test. Its history-only
  hindcasts and ensemble gate remain vulnerable to future regime changes and rare
  catastrophic choices.

The 99-task Public Test is now consumed and must not be used to tune the next version.
Further Selector changes should be developed on Train/Dev or a new untouched split,
then evaluated on a new sealed test set.

## Artifacts

- Screening manifest: `runs/task_conditioned_screening/formal_80_20_all103_20260823/screening_manifest.json`
- Selector manifest: `runs/numerical_selector/formal_80_20_fallback_20260823/selector_manifest.json`
- Final report: `runs/frozen_two_stage/public_test_99_20260823/FINAL_TWO_STAGE_REPORT.md`
- Machine-readable results: `runs/frozen_two_stage/public_test_99_20260823/frozen_two_stage_results.json`
- Per-task results: `runs/frozen_two_stage/public_test_99_20260823/per_task_results.jsonl`
- Completion marker: `runs/frozen_two_stage/public_test_99_20260823/evaluation_complete.json`
