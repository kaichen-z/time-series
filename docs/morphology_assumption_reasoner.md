# History-Only Morphology Reasoning

This layer adds a bounded analysis loop between task-conditioned screening and the
Numerical Selector:

```text
historical values
  -> Morphology Reasoner chooses reviewed analysis tools and windows
  -> Python executes the tools
  -> grounded Morphology Card
  -> assumption-guided Numerical Selector
  -> forecast
```

It is inspired by tool-using forecasting agents, but preserves this repository's
no-leak and deterministic-runtime boundaries. The LLM may choose only a reviewed
tool name and a historical `[left, right)` window. It cannot write executable code,
call a TSFM, inspect documents, or see future labels.

## Reviewed tool API

The callable tools are the history-only primitives from
`analysis_skills_template.py`:

- `detect_periodicity`
- `detect_outliers`
- `detect_trend`
- `detect_change_points`
- `detect_intermittency`
- `estimate_noise_scale`
- `assess_stationarity`
- `detect_recent_regime`

The controller requires at least one broad inspection and one distinct recent
inspection before accepting a final answer. It rejects unknown tools, invalid
windows, duplicate call IDs, invented evidence IDs, inactive candidate names, and
schema drift.

## Output contract

`MorphologyReasoner.reason(...)` returns a `MorphologyCard` containing:

- a short-term description;
- a long-term description;
- the exact executed tool calls and Python-produced observations;
- one to seven falsifiable `ForecastAssumption` records;
- the tool-call IDs grounding every assumption.

Pass `card.assumptions` to `select_assumption_guided_forecast(...,
assumptions=card.assumptions)`. Supplying assumptions disables the selector's old
deterministic assumption generator for that call; all baseline-protection and
Verifier gates remain unchanged.

## Train-only credit

`assign_tool_call_credit(...)` replays the card one observation at a time. An
assumption becomes available only after all of its cited calls have been observed.
Each step receives the marginal change in Dr-CiK-aligned sMAE and sRMSE relative to
the preceding step.

This evaluator is Train-only. `future_truth` is accepted only by the trusted credit
function and is not part of the Morphology Reasoner interface. Freeze any learned
Reasoner policy before Dev evaluation; Dev and Test remain read-only.

## Compatibility

The existing formal pipeline is unchanged unless callers explicitly construct a
`MorphologyReasoner` and supply its assumptions. Calls that omit `assumptions`
continue to use the repository's deterministic assumption generator, so existing
frozen policies and artifacts retain their prior behavior.
