# Consolidated findings through i3 episode 3

The visible folds contain 54 labeled Train tasks; the remaining 26 Train views have inputs only. Use only `views_train.json` and `traces/visible_tasks.jsonl` for local diagnosis. The seed fitness was 0.1948.

1. Brief single-correction upward spikes in 24-step hourly forecasts deserve a wider cap. The initial +50% cap underpredicted events with extracted multipliers 2 to 5. A two-thirds relative lift cap raised fitness to 0.2342, passing hidden checks.
2. A single full-horizon shutdown on a smooth, declining second-level series needs trend projection toward zero. The narrow rule affected only task_43 and raised fitness to 0.2484, passing hidden checks. Do not broaden it without more examples.
3. The extracted windows for nine visible short positive events include a final return-to-baseline step. Trimming only this step raised fitness to 0.2807. Once trimmed, allowing the full extracted 2–5x magnitude on the remaining event steps raised fitness to 0.3216. Both hidden checks passed and runtime errors were zero. The accepted method is `shared/best_harness.py`.
4. Overlap examples are sparse (four visible tasks) and their constituent corrections are mostly rejected by the seed validator. There is no supported composition rule yet. Task_146's accepted 5x holiday correction is erroneous across its window, but a rule for it alone would overfit.
5. A larger downside cap for zero-multiplier events and broad spike-cap changes harmed or gave only small gains. Avoid generic expansion of these rules.

Reusable audit: `shared/skills/correction_audit.py` and its README. Episode notes remain as the detailed submission record.
