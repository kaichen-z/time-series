# s0 episode 2: retrieval reassessment

I reread the episode-1 notes and audited all visible corrections against the current forecast. The validator already rejects most broad adjustments. Accepted short events account for most document-driven gains; task_158's staged downward adjustment is also useful. A harmful accepted event on task_146 is small relative to those gains. The available validator features do not cleanly distinguish it from useful large events.

## Submissions

Two submissions tested a stronger penalty for correction size relative to normal series variation (`ratio`: -0.00696 to -0.05), first on the newly accepted 3% Moirai blend and then on the 5% blend. Both passed the hidden check but reduced visible fitness: 0.19069 to 0.18937 and 0.19151 to 0.19018, respectively. In each paired comparison, the only visible per-task change was task_158, whose gain fell by about 0.096 because the staged correction was removed. Neither candidate was accepted. No extraction instruction was changed.

## What NEVER worked

Strengthening the ratio guard did not suppress an observed harmful correction. It removed a useful staged correction instead. This repeats the broader episode-1 failure of global validator coefficient changes: confidence, boilerplate, and window length each lost useful corrections or admitted harmful ones. Do not tune these coefficients further without a feature that identifies whether the documented event applies and whether its magnitude is credible.

The shared best is the accepted 5% Moirai blend (attempt 0019, visible fitness 0.19151, hidden pass). I stopped after two submissions because the remaining retrieval parameters had no supported generalizable change in the visible audit.
