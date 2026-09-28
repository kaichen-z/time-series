# i2, episode 2 working notes

- Re-read episode 1 notes and shared harness/attempts. No other agents' notes or reusable skill files are present yet.
- Visible accepted 4x and 5x event cases have mixed optimal strength: raising the cap to full extracted magnitude helps task_128/138 and task_132 but hurts task_129. The first episode-2 submission, full 4x/5x for high-level series, barely changed visible fitness (0.231344 versus 0.231340) and failed the hidden check. Do not retain it.
- Short zero-multiplier windows in task_104/87 have truth near zero; current logistic score halves them. A full-horizon zero correction in task_43 is different and should retain the cap.

## Remaining submissions and result

2. Full strength for short zero-multiplier windows: hidden pass, visible fitness 0.2310, rejected. The final step of task_104/87 rebounds sharply, so a full shutdown overshoots at the boundary.
3. Ignore downward nonzero percentage corrections when `abs(base) < sigma_main_calib`: hidden pass, visible fitness 0.23137, rejected (change too small; only task_121 visibly affected).
4. Apply full zero correction to the interior of a short shutdown but preserve half-strength at its final step: hidden pass, visible fitness 0.2333, accepted. Affects task_104/87.
5. For a full-horizon zero correction on an already monotone-declining positive history, smoothly taper the base toward zero with `(1-phase)**1.6`: hidden pass, visible fitness 0.2507, accepted. Affects only visible task_43, which has a near-deterministic decline. This is a large improvement concentrated in one task, so generalization remains uncertain despite the hidden pass.

Final harness is `ws_i2/harness_i2.py`, also auto-published as `shared/best_harness.py`. It returns exactly H finite floats on all 80 Train views. No submissions remain for episode 2.
