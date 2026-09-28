# Shared evidence through episode 3

Current accepted best: attempt 0019, visible fitness 0.19151, hidden pass.
It adds about 5% Moirai to the Toto/ARIMA numerical mix and retains 20.15%
shrink, phase-median repair with margin 0.30, and the seed correction policy.
The 3% Moirai predecessor also improved visible fitness and passed hidden.

Strongest generalizable observation: a small Moirai blend is the only tested
change that improved the visible objective and passed hidden. Its visible
advantage is in gain magnitude, not number of improved tasks. Treat further
weight changes cautiously because several numerical replay gains hinge on one
large-error task.

History repair: lowering the back-test margin admitted harmful repairs;
raising it offered no gain. Linear fill improved task 193 back-test much more
than phase median but worsened its forecast. Keep phase median and 0.30.

Future corrections: short upward events can help substantially, while a
confident broad downward event on task 77 is harmful and staged broad
corrections on task 158 are useful. Simple changes to confidence, boilerplate,
window fraction, ratio, or global strength cannot separate these examples.
The ratio guard's modest hidden pass came with lower visible fitness.

Numerical failures: 10–15% shrink and 5% Chronos improved visible fitness
but failed hidden checks. Tight clipping harmed task 61. An ARIMA-only test
was confounded by a simultaneous shared-best update. Further visible-only
search around the mix lacks clear support.

Analysis script: `shared/skills/numerical_replay.py` screens numerical program
changes using only visible traces. It anchors corrections to the seed, excludes
13 repair-variant tasks, and is diagnostic rather than a replacement for the
evaluator. See `shared/skills/README.md`.

S0 episode 3 added a 2% kernel-ridge term (attempt 0022). Visible fitness rose to 0.19213, but the hidden check failed. This extends the pattern that small numerical gains on visible tasks do not reliably generalize. An initial s0 replay for that probe used the wrong trace anchor and was discarded; the retained `numerical_replay.py` correctly anchors to the seed program.

Episode 3 decision audit: lowering the validator trust coefficient from 1.017 to 0.75 passed hidden but lowered visible fitness from 0.19151 to 0.19030. It removed task 158's useful staged downward correction (gain -0.0961) while helping task 121 by only +0.0058. Strong short upward correction benefits occur in cells with low trust scores near 0.10–0.28, so a positive trust gate is poorly motivated. See `s2_episode3.md` and `shared/skills/visible_decision_audit.py`.
