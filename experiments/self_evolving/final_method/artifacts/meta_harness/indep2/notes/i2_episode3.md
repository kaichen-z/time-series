# i2, episode 3

Read both earlier episode notes, the current best harness, all prior attempts,
and the allowed visible traces. Only i2 notes were present in `shared/notes/`.

## Observations

- The validator rejects a documented 1.5x increase on a short high-level event
  (task_137), even though a related event correction is useful. A short 2x
  event (task_134) was accepted only at the old 50% cap.
- In some low-level windows, a large multiplier implies a change of just a few
  `sigma_main_calib` units. Task_146's 5x correction increased a near-zero
  forecast and was harmful. This supports calibrating absolute implied change,
  not just multiplier size. It is limited evidence: task_179 and task_183 have
  low-level bases but large truth spikes, so a blanket level gate is unsound.
- Visible tasks have correlated pairs; task_134 and task_137 are closely
  related, and task_128/task_138 share the same event. Changes on these pairs
  should not be counted as independent validation.

## Submissions

1. Accept a 1.3–2x increase in a short window when every forecast step is
   above 100 calibration-scale units. Affected visible task_134 and task_137.
   Visible fitness 0.2541 versus 0.2507, hidden check pass, accepted.
2. Skip a nonzero correction at a step when its full absolute implied change
   is under 5 calibration-scale units. Affected visible tasks 121, 146, 158,
   and 179. Visible fitness 0.2547, hidden check pass, accepted. Improvements
   on 121 and 146 outweighed small losses on 158 and 179.

Stopped after two submissions. The final local `harness_i2.py` matches
`shared/best_harness.py` and returns exactly H finite floats on all 80 Train
views. Added `shared/skills/inspect_corrections.py` and its README.
