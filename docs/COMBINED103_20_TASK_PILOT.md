# 103-Candidate Combined Portfolio: 20-Task Pilot

Date: 2026-08-22 (Asia/Shanghai)

## Scope

This pilot evaluates one generation of sequential co-evolution over the first 20 tasks of the
entity-aware Dr-CiK Train partition:

- 16 Train tasks: `task_108`, `task_109`, `task_110`, `task_114`, `task_115`, `task_118`,
  `task_127`, `task_128`, `task_129`, `task_130`, `task_131`, `task_132`, `task_133`,
  `task_134`, `task_135`, and `task_136`;
- 4 mini-dev tasks: `task_142`, `task_147`, `task_149`, and `task_158`;
- no Public Test or hidden-test task was accessed.

The initial Git-tracked portfolio contains 103 executable candidates:

| Family | Count | Evolved artifact |
|---|---:|---|
| Python forecasting methods | 93 | `methods.py` |
| Reviewed TSFM policies | 5 | `policies.py` |
| TSFM/statistical Combined policies | 5 | `policies.py` |

Luna selected targets and generated Train-only failure summaries. Terra generated one bounded
repair per selected target. Every Child first ran on four Train tasks. At most three survivors
could proceed to all 16 Train tasks and then the frozen four-task mini-dev set. MASE was the
primary metric; MAE and sMAPE were diagnostic metrics. Mini-dev labels were used only by the
trusted acceptance gate and were never shown to the selector, Judge, or mutator.

The run used real local checkpoints for TimesFM 2.5, Moirai 2.0, Toto 2.0, Chronos-Bolt, and
Granite TTM R2. The complete run took approximately 34.6 minutes after all declared scientific
Python dependencies had been installed.

## Combined policies

| Combined candidate | TSFM parent | Statistical parent | Initial rule |
|---|---|---|---|
| `combined_timesfm_seasonal` | TimesFM 2.5 | seasonal naive | 0.65 / 0.35 blend |
| `combined_chronos_damped_trend` | Chronos-Bolt | Holt damped trend | 0.65 / 0.35 blend |
| `combined_moirai_croston_router` | Moirai 2.0 | Croston-SBA | zero-fraction router |
| `combined_toto_robust_router` | Toto 2.0 | robust LOESS trend | outlier-fraction router |
| `combined_granite_regime_profile` | Granite TTM R2 | median seasonal profile | 0.60 / 0.40 blend |

The model/checkpoint binding of a TSFM and both parent identities of a Combined policy are
immutable. Evolution may change bounded invocation, blending, or history-only routing fields.

## Parent TSFM and Combined results

### Train (16 tasks)

| Candidate | Success / 16 | Mean MASE | Mean MAE | Mean sMAPE |
|---|---:|---:|---:|---:|
| TimesFM 2.5 | 16 | 3.9064 | 325.0432 | 29.5269 |
| Moirai 2.0 | 16 | 3.8936 | 325.7467 | 45.3732 |
| Toto 2.0 | 16 | 3.9691 | 330.6113 | 44.6515 |
| Chronos-Bolt | 16 | 3.8916 | 326.3235 | 43.8211 |
| Granite TTM R2 | 16 | 6.1609 | 409.0918 | 55.7811 |
| TimesFM + seasonal naive | 16 | 3.8808 | 323.8806 | 29.4651 |
| Chronos + Holt damped trend | 16 | 4.4737 | 369.8155 | 50.2159 |
| Moirai / Croston-SBA router | 3 | 2.5351 | 210.6131 | 155.8526 |
| Toto / robust-trend router | 16 | 5.5230 | 465.7169 | 55.1605 |
| Granite + seasonal profile | 16 | 5.0571 | 365.0361 | 49.5340 |

The Moirai/Croston result is a specialist result over only three applicable tasks and is not
directly comparable with full-coverage averages.

### Mini-dev (4 tasks)

| Candidate | Success / 4 | Mean MASE | Mean MAE | Mean sMAPE |
|---|---:|---:|---:|---:|
| TimesFM 2.5 | 4 | 1.4907 | 18.4495 | 50.1289 |
| Moirai 2.0 | 4 | 1.4987 | 17.6361 | 67.0728 |
| Toto 2.0 | 4 | 1.5101 | 20.0189 | 64.1134 |
| Chronos-Bolt | 4 | 1.6265 | 24.2163 | 62.4652 |
| Granite TTM R2 | 4 | 1.9463 | 30.7101 | 74.9305 |
| TimesFM + seasonal naive | 4 | 1.5146 | 18.6586 | 50.0044 |
| Chronos + Holt damped trend | 4 | 1.5547 | 18.6658 | 78.1439 |
| Moirai / Croston-SBA router | 1 | 2.6848 | 127.7287 | 144.8139 |
| Toto / robust-trend router | 4 | 3.8364 | 114.3881 | 84.9959 |
| Granite + seasonal profile | 4 | 1.7145 | 25.0867 | 69.3166 |

Two simple combinations show useful but non-general effects. TimesFM plus seasonal naive is
slightly better than TimesFM on Train but slightly worse on mini-dev. Granite plus the seasonal
profile improves Granite on both splits, but remains weaker than the leading TSFMs. This supports
evolving combinations rather than assuming that every ensemble is beneficial.

## Accepted Python repairs

Two Python-method repairs survived screen, full Train, mini-dev, and sequential rebase checks:

| Method | Train mean MASE, Parent -> Child | Mini-dev mean MASE, Parent -> Child | Git commit |
|---|---:|---:|---|
| `scinet` | `1.45e68 -> 4.3147` | `3.23e157 -> 8.09e7` | `735cb72` |
| `itransformer` | `2.46e8 -> 10.1814` | `3.45e8 -> 13.3664` | `5606346` |

These are large repairs of numerically unstable implementations, not evidence that either Child
is now competitive. In particular, SCINet's absolute mini-dev MASE remains catastrophic. Future
runs should add a useful-quality or deployability gate in addition to relative Parent improvement.

Six other proposals were rejected or pruned. A notable SARIMA repair reduced the search from 144
models with 200 optimizer iterations to 16 models with 25 iterations. It eliminated the universal
timeout and reached 81.25% Train coverage, but did not improve the portfolio on mini-dev and was
therefore rejected.

## Policy evolution outcome

The policy selector chose three targets:

| Target | Proposed change | Result |
|---|---|---|
| Granite TTM R2 | periodic-only, 1024 context, standardized input | rejected: screen coverage fell from 1.00 to 0.75 |
| Chronos + Holt damped trend | replace blend with periodicity router | Train MASE `4.4737 -> 4.0403`, but mini-dev `1.5547 -> 1.7160`; rejected |
| Toto + robust trend | route by zero fraction at 0.10 | screen MASE unchanged; rejected |

No TSFM or Combined policy was committed. This is a valid negative result: the frozen mini-dev
gate prevented a Train-only improvement from becoming the next generation.

## Limitations and next changes

- This is a 20-task development pilot, not a held-out test result.
- Evolution is sequential within one generation: Python methods evolve first, then TSFM and
  Combined policies evolve against the updated Python module. It is not yet a single joint
  selector over all 103 candidates.
- The current policy gate forbids any reduction in coverage, which conflicts with the goal of
  learning specialized TSFM applicability. A specialist-aware coverage contract is needed.
- The current Python gate can accept a very large relative repair even when absolute performance
  remains unusable. Add an absolute/deployability gate before the larger experiment.
- The portfolio non-regression diagnostic is an oracle over candidates, not the performance of a
  learned history-only selector. Final system evaluation must use a deployable selector.

## Artifacts

The local run repository is:

`runs/method_evolution/combined103_20_valid_20260822/`

It contains the two accepted Git commits, `generation_001_targetwise.json`,
`generation_001_policies.json`, all model transcripts, trusted outcome caches, and the execution
trace.
