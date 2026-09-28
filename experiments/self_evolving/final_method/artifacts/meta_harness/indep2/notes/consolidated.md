# Correction harness findings through i2 episode 3

Reviewed all notes currently present: `i2_episode1.md`, `i2_episode2.md`, and
`i2_episode3.md`. No other agents' notes were present at consolidation.

- Strong short upward events in high-level series needed more than the seed's
  50% cap. The accepted harness allows up to +200%, or +300% for multipliers
  at least 5, when `abs(base)/sigma_main_calib > 100`.
- Full extracted 4x–5x strength overfit: it barely improved visible fitness
  and failed hidden check. Reducing the cap for 3x–4x events also failed the
  hidden check despite visible improvement. Keep the accepted bounds.
- Short zero-multiplier shutdowns improve with full suppression in the window
  interior and a half-strength final step. Full suppression of the final step
  hurt visible paired tasks 104/87.
- A zero target spanning the horizon on an already steadily declining positive
  history improved with a smooth taper. This is supported by one visible task
  (43), which accounts for much of the fitness increase. Generalization is
  uncertain even though the hidden check passed.
- A short modest upward event on a stable high-level series can be useful even
  when the magnitude-based validator rejects it. Episode 3's acceptance rule
  improved two correlated visible tasks and passed hidden check.
- Calibrating a correction's *absolute implied change* against
  `sigma_main_calib` helped skip weak, noisy adjustments. The current 5-scale
  gate improved two visible tasks and slightly harmed two. Low base level
  alone is insufficient as a rejection rule because true event spikes can
  arise from a near-zero base.
- Do not treat visible paired tasks as independent evidence. Avoid further
  threshold tuning to those pairs. `shared/skills/inspect_corrections.py`
  provides a compact, repeatable summary using only allowed data.

Current accepted harness: `shared/best_harness.py`; visible fitness 0.2547,
hidden check pass. These are submission results, not a measure of performance
on the hidden fold.
