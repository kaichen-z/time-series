# i2, episode 1

Read the seed, the 54 visible traces, and the shared notes/skills directories (both were empty). Submitted three variants; stopped after the third.

## Observations

- Seed modifies 17 visible tasks, mainly short high-magnitude events and the two zero-multiplier cases. It has a fixed 50% bound in both directions.
- On visible traces, the large upward event corrections with multiplier 3–5 often correspond to actual peaks of roughly 2–4 times the base forecast. The seed bound leaves much of those peaks unmodeled.
- High event series in the visible traces have corrected-step base values over 100 times `sigma_main_calib`. A low-level 5x correction on task_146 is harmful and does not meet that gate. This is a heuristic; `sigma_main_calib` is not a direct raw-unit variability estimate.
- Several visible tasks repeat the same underlying forecast and truth with different documents or multipliers. Treat gains on paired tasks as less than independent evidence.

## Submissions

1. Raise the upward bound to +200% where accepted corrections apply to steps with `abs(base)/sigma_main_calib > 100`. Visible fitness 0.2244, hidden check pass, accepted.
2. For accepted multipliers at least 5, raise that bound to +300%. Visible fitness 0.2313, hidden check pass, accepted. This is the current shared best.
3. Reduce the bound for multiplier 3–4 to +150%. Visible fitness 0.2356, hidden check fail, rejected. The visible improvement came from four repeated event cases; do not carry this change forward.

The final local `harness_i2.py` is restored to submission 2. No further submissions in this episode.
