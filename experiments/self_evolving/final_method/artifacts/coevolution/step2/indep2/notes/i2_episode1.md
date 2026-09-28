# Agent i2, episode 1

## Visible-trace analysis before submissions

- Reconstructed seed base program and the final document multipliers from the visible trace. The reconstructed seed fitness equals the recorded 0.1889437383 exactly. Two tasks (43 and 193) use repaired Toto; their repaired forecasts can be inverted from the final forecast and seed program.
- The current ARIMA blend is approximately 5.1%, with 20.15% shrink toward the last value. A 5% ARIMA blend with 10–15% shrink has slightly higher visible fitness (~0.195) even before adding another model.
- Adding 10% Chronos Bolt and reducing shrink to 10% yields predicted visible fitness ~0.209. Chronos diversification has a better visible risk-adjusted result than TimesFM at the same blend. This is a small perturbation of the known robust Toto + ARIMA structure, not a search over many unrelated methods.
- Clipping to the observed range or range + 5–20% reduces visible fitness. Do not use narrow clipping here.

## Submission 1

- 85% Toto, 5% ARIMA, 10% Chronos Bolt, 10% shrink: visible fitness 0.208087 (both folds improve); hidden check failed. The model diversification did not generalize according to the hidden gate, so do not promote it.

## Submission 2

- Preserve Toto/ARIMA mix and reduce shrink to 12%: visible fitness 0.1960; hidden check failed. Even this small numerical adjustment is not robust enough under the hidden gate.

## Submission 3 and group audit

- Shrink 18%: visible fitness 0.1917; hidden check failed. Repeated shrink reductions all fail hidden, despite visible gains.
- A 5% Chronos blend at original shrink gains mostly from a few groups and loses on all six store tasks. A 5% combined TimesFM seasonal blend has smaller but more balanced visible effects: positive mean change in both folds, though store tasks still lose. This is the next numerical trial.

## Submission 4 and final candidate rationale

- 5% seasonal TimesFM blend at original shrink: visible fitness 0.1932; hidden check failed. Do not promote.
- Accepted document corrections are mostly half-strength under the current validator. A small strength increase predicts a small gain in both visible folds, chiefly for city events; freeway and one sensor case lose. This is a separate mechanism from the numerical changes above, but the visible evidence is mixed. Testing only a 5% increase as the final episode submission.

## Submission 5 and episode summary

- Correction strength 1.05: visible fitness 0.1888, below seed; hidden check failed. Actual per-task output changed mainly task 158, unlike the simple offline scaling model. The evaluator appears to apply `strength` to a narrower set of corrections than inferred final/base factors, so that offline model must not be used for strength tuning.
- Five submissions total. None passed the hidden gate; shared best remains the seed at visible fitness 0.1889437383. The central lesson is that modest visible improvements from adding models or reducing shrink did not generalize to the hidden fold. Keep the Toto + small ARIMA baseline unless a new, independently justified mechanism is found.
