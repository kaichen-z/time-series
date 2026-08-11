# Frozen 30-Task Dr-CiK Evaluation

Date: 2026-08-07

This report records the first frozen 30-task evaluation of the Chronos-Bolt +
Codex Contract system on the public development split of `ServiceNow/Dr-CiK`.
It is a development result, not an official hidden-test Dr-CiK score.

## Protocol

- Dataset: `ServiceNow/Dr-CiK`, public development split
- Backbone: local `amazon/chronos-bolt-small`
- Agent: `codex-contract` with `gpt-5.6-sol`, high reasoning effort
- Selection seed: `20260807`
- Selection rule: cover every frequency x horizon-bin stratum, then allocate the
  remaining slots proportionally with stable SHA-256 ordering
- Excluded before selection: `task_42`, `task_163`, and `task_201`, because they
  were used during development
- Selection used no future values, ground-truth evidence, or document role labels
- Frozen manifest SHA-256:
  `c53272f87d4f1f8635cf281ca0d894092bbb5822c97752a1d54d8b7eab791761`

The subset contains 15 hourly, 6 daily, 3 minute-level, 3 five-minute, and 3
second-level tasks. Its forecast horizons cover all four configured horizon bins.

## Results

| Metric | Chronos baseline | Agent system | Change |
|---|---:|---:|---:|
| Mean MAE | 162.626963 | 156.362114 | -3.85% |
| Mean RMSE | 301.011687 | 296.465854 | -1.51% |

The local CRPS of the final probabilistic forecasts is `149.194873`. The runner
does not currently emit a directly comparable baseline CRPS, so no CRPS improvement
claim is made here. The reported sMAE, sRMSE, and sCRPS values are development
proxies rather than official Dr-CiK hidden-test scores.

Revision outcomes:

- Improved: 2 tasks
- Unchanged: 28 tasks
- Harmed: 0 tasks
- Revision rate: 6.67%

The two accepted revisions were:

| Task | Baseline MAE | Final MAE | Relative gain | Why revision was allowed |
|---|---:|---:|---:|---|
| `task_67` | 1088.688438 | 900.808324 | 17.26% | Verified normalization plus recurring seasonality; history-only harmonic candidate passed backtesting |
| `task_213` | 2.219885 | 2.154543 | 2.94% | Verified repaired anomaly plus daily seasonality; history-only harmonic candidate passed backtesting |

The aggregate improvement is therefore sparse: the median task is unchanged, and
most of the mean MAE reduction comes from `task_67`. This supports the safety of the
current revision gate more strongly than it supports broad forecasting gains.

## Retrieval and Contracts

- Mean retrieval precision: 0.7791
- Mean supporting-document recall: 0.4120
- Mean distractor avoidance: 0.7791
- Codex failures: 0/30

Contract distribution:

| Regime | Tasks |
|---|---:|
| `temporary_future_event` | 9 |
| `normal_seasonal` | 8 |
| `normal_nonseasonal` | 5 |
| `permanent_shift` | 4 |
| `explicit_future_values` | 3 |
| `insufficient_evidence` | 1 |

All three `explicit_future_values` contracts were withheld from numerical revision
because the explicit-value validator did not find independently corroborated,
horizon-aligned values. This is conservative and prevented unsupported numbers from
overriding Chronos, but it is also an important coverage limitation.

## Runtime and Reproducibility

The three evaluation shards required 29 new Codex calls and one cache hit from the
preflight task. Those 29 calls accumulated 4,408.476 seconds of model latency, or
152.016 seconds per new call on average. The preflight task itself was one additional
successful call before the sharded run. All 30 responses are content-addressed in the
local evaluation cache, and the unified report was regenerated entirely from cache.

Artifacts:

- Manifest: `outputs/eval-30-manifest.json`
- Summary: `outputs/eval-30-codex-contract/summary.json`
- Per-task report: `outputs/eval-30-codex-contract/run_report.jsonl`
- Auditable loop traces: `outputs/eval-30-codex-contract/loop_trace.jsonl`
- Forecasts: `outputs/eval-30-codex-contract/forecasts.jsonl`

Validation: 54 unit tests pass under `unittest`, and `git diff --check` reports no
whitespace errors.
