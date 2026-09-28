# i2, episode 4 working notes

- Read all prior i2 notes, consolidated notes, and the reusable inspection script. No other agents' notes were present.
- A smooth 3–8 sigma taper around the existing 5 sigma cutoff affected only two visible tasks (146 and 158); a local absolute-error proxy showed almost no net benefit, so it was not submitted.
- The 45x short event in visible task_183 has a near-zero baseline in three steps but an already elevated fourth base step. The accepted 50% cap changes the first three too little. A bounded additive uplift of up to eight calibration units for near-zero, extreme-multiplier events improved visible fitness from 0.2547 to 0.2626, hidden check passed. This affects one visible task, so evidence for generalization is limited.
- A half-strength acceptance path for short 1.2x–2x uplifts with at least five sigma absolute effect and poor cell backtest improved task_112 and visible fitness to 0.2638; hidden check passed. Only three forecast steps changed in one visible task.

Stopped after two submissions. `ws_i2/harness_i2.py` matches `shared/best_harness.py`, and its output is exactly H finite floats for all 80 Train views. The visible gains are concentrated in task_183 and task_112, so the hidden-check pass should not be read as a precise estimate of generalization.
