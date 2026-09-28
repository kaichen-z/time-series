# Agent i3 consolidated findings, episodes 1–3

The current accepted config is `shared/best_config.json`, originating from episode 2's `0005_i3.json` (visible fitness 0.19477, hidden check pass). It adds a 10% TimesFM 2.5 share to the Toto/ARIMA base. The original seed scored 0.18894. The other submissions from these episodes did not replace it.

## Decisions that held up

- Keep document corrections active. Setting strength to zero substantially reduced visible fitness and failed the hidden check. Most visible benefit from short events would disappear.
- Keep `phase_median` history repair with the 0.3 proportional back-test margin. The only clear accepted repairs were task_43 and task_193; linear fill worsened task_193's forecast despite a stronger history back-test result. Changing the margin to 0.2 or 0.4 did not help.
- Keep the small accepted TimesFM share. Larger TimesFM, added Chronos, and less shrink improved visible fitness but failed hidden checks. Those are visible-fold overfits, not established improvements.
- Keep the present correction validator until there is a better signal. Both directions of confidence weighting, a stronger ratio penalty, and 15% global damping failed to improve on the accepted config.

## Cautions for later work

- Per-task gains are concentrated: history repair and some document corrections create large wins. Optimizing the 54 visible tasks can move fitness while making hidden performance worse.
- Some corrections with extreme multipliers are harmful, but the tested global ratio penalty did not isolate them. A specific outlier guard needs evidence that it retains beneficial short events and the full-window task_43 correction.
- `shared/skills/compare_visible.py` is a reusable visible-only diagnostic for identifying exactly which tasks change when a parameter crosses a validator threshold. It does not estimate hidden performance.

Detailed per-submission results and mechanisms remain in `i3_episode1.md`, `i3_episode2.md`, and `i3_episode3.md` for provenance.
