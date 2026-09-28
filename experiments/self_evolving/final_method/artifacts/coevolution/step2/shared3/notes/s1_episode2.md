# s1 episode 2: reassessment of numerical tuning

Starting best was the seed (visible fitness 0.18894). Read all episode 1 notes and visible traces. The earlier 10–15% shrink and 5% Chronos changes improved visible fitness but failed hidden checks. They are **not generalizable evidence**. Changing the validator globally also has counterexamples: task 158 needs a broad staged correction, while task 77 has a confident broad correction that appears harmful.

## Repair threshold probe

Raised the phase-median repair back-test margin from 0.30 to 0.40. Visible fitness was 0.18893, hidden check passed, and the candidate was not accepted. The two main documented-repair wins (tasks 43 and 193) have much larger back-test reductions. The higher threshold provides no measurable gain here; keep 0.30.

## ARIMA probe and limitation

Prepared a 4% ARIMA / 96% Toto program from what was then the shared best. While the submission ran, another agent's 3% Moirai blend became the new shared best (visible fitness 0.19069, hidden pass). My submitted program omitted Moirai and therefore changed two things relative to the new best. It scored 0.18606 and failed the hidden check. This is **not** a clean test of a 4% ARIMA share on the new best, and should not be interpreted as one.

## What never worked in this line

- Visible-only shrink reductions to 10% or 15% and a 5% Chronos blend failed hidden checks in episode 1.
- Tight clipping of the numerical forecast to the historical range degraded the visible replay, particularly a repair-heavy task.
- Raising the repair margin to 0.40 did not improve the seed despite passing the hidden check.
- Replacing the new Moirai blend with a 4% ARIMA / 96% Toto program worsened visible fitness and failed the hidden check. This test was confounded by the concurrent best-config update.

No accepted improvement from this agent in episode 2. Another agent subsequently raised the accepted Moirai dose to 5% (visible fitness 0.19151, hidden pass); that is the current shared best. Two submissions used; I stopped before spending the remaining allowance on small visible-only tuning without a generalizable mechanism.
