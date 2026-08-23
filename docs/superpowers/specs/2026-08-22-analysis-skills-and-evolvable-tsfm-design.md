# Analysis Skills and Evolvable TSFM Design

## Goal

Extend forecasting-method evolution from 93 mutable Python point forecasters to a
103-candidate portfolio made of 93 Python methods, five TSFM candidates, and five executable
Combined candidates, while giving every method reusable, history-only analysis skills.

## Analysis skill boundary

- Skills receive only historical values and frequency. They never receive a future label,
  retrieved document, GT evidence, forecast score, or Dev/Holdout metadata.
- The initial stable API contains `detect_periodicity`, `detect_outliers`, `detect_trend`,
  `detect_change_points`, `detect_intermittency`, `estimate_noise_scale`,
  `assess_stationarity`, `detect_recent_regime`, and `analyze_series`.
- Skills report measurements and confidence. They do not silently clean history or issue a
  forecast.
- Evolved Python methods may call the skills as injected globals. The forecasting function
  contract remains `method(history, horizon, frequency) -> horizon finite floats`.
- Skill source participates in the outcome-cache identity so changing a skill cannot reuse stale
  method scores.

## Evolvable TSFM boundary

The five flagship candidates are:

1. TimesFM 2.5 (`method_tsfm_0031`)
2. Moirai 2.0 (`method_tsfm_0017`)
3. Toto 2.0 (`method_tsfm_0014`)
4. Chronos-Bolt (`method_tsfm_0018`)
5. Granite Tiny Time Mixer R2 (`method_tsfm_0006`)

All five participate in Train, screen, Dev, and portfolio scoring and may be selected for
evolution. Their reviewed identity binding is immutable: checkpoint, revision, official adapter,
license, and model ID cannot change. Their invocation policy is evolvable:

- applicability rules derived from history-only skills;
- context-window selection;
- reviewed runtime-option choices within manifest limits;
- reversible preprocessing and inverse transformation;
- bounded output calibration;
- routing, blending, or residual correction with existing candidates.

The policy must remain executable and must call the reviewed TSFM runtime. A Child cannot replace
a TSFM with a similarly named statistical approximation. No LLM weight training occurs in this
loop.

## Evaluation and acceptance

- The initial candidate count is 93 Python + 5 TSFM + 5 Combined = 103.
- Python-method mutations remain source-level repairs/forks/deletions.
- TSFM mutations are typed policy changes whose model identity remains fixed.
- Combined policies bind one TSFM parent and one statistical parent. They may evolve their
  blend weight or history-only routing rule, but may not rename or substitute either parent.
- The five initial Combined candidates are TimesFM + seasonal naive, Chronos-Bolt + damped
  trend, Moirai + Croston-SBA, Toto + robust LOESS trend, and Granite TTM + median seasonal
  profile.
- All candidates use the same Train/screen/Dev protocol and trusted metrics.
- A TSFM policy Child must improve applicable-task MASE without a portfolio regression, runtime
  failure, identity change, or catastrophic Dev tail.
- Enabling the flagship portfolio is fail-closed: all five runtime bindings must resolve before a
  run can claim 103 candidates.
- No checkpoint is downloaded while parsing CLI arguments or constructing tests; downloads and
  model execution occur only during an explicitly enabled evaluation.
