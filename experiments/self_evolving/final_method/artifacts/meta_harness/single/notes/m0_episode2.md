# m0 episode 2

- Read prior episode notes and the visible trace before editing. The episode 1 best (0003) scored 0.23553 visible fitness.
- Visible `task_104` and `task_87` are duplicate instances of the same daily seasonal series: a zero-multiplier interval [8,13), with actual values near zero at steps 8–11 and a rebound at step 12. The seed's 50% downward cap leaves the shutdown interior too high.
- Submission 4 applied full zero throughout the extracted interval. Hidden check passed but fitness fell to 0.2352. The last extracted step is a restoration boundary; zeroing it was harmful.
- Submission 5 applied full zero to [start,end-1) and left the final step at the base forecast, only for one short daily zero-multiplier correction accepted by the original validator. Hidden check passed; visible fitness rose to 0.23931. Only tasks 104 and 87 changed, each gaining 0.102 over 0003. Since these tasks share a series and truth, this is effectively one observed pattern.
- Kept the seed validator and episode 1 wider cap for isolated hourly surges. Other visible correction patterns are mixed or have too few independent examples for further rules. Stopped after two submissions in this episode; three allowed submissions unused to avoid fitting single tasks.
