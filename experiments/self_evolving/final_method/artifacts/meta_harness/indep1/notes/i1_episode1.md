# i1, episode 1 observations

Shared notes and skills were empty at start. Visible traces show short upward extracted corrections (roughly 2–5x) often coincide with sharp localized demand/temperature spikes. The seed's +50% cap underuses them; task_137's 1.5x correction is rejected altogether. Calendar/holiday notices have a different pattern: the upward 5x correction on task_146 is harmful.

Submission 1 (`event_cap.py`) accepts short upward corrections (multiplier >=1.5, width <=18% of horizon) unless the first document mentions a holiday, and permits a +150% cap. Fitness 0.22049 vs seed 0.19477; hidden check passed. It helped tasks 128,129,131,132,134,135,137,138,140,141. It hurt task_183 by 0.1307 gain: its extracted 45x hardware maintenance correction is too extreme for this rule. Next try excludes implausibly huge multipliers.

Submission 2 excluded multipliers >10 from the short-spike rule, restoring task_183 and reaching fitness 0.2250 (hidden pass). Submissions 3–5 raised the ceiling for extracted >=4x events lasting at least 3 steps: +250%, +300%, then +350%. Fitness rose 0.2381, 0.2415, then 0.2428; each hidden check passed. The +350% result became shared best (`shared/best_harness.py`). Five submissions used, so episode stops here.

Mechanism: apply a stronger upward adjustment only for brief, substantial corrections; full-day and holiday notices stay on the seed path. Huge 45x multipliers stay on the seed path because task_183 worsened sharply under the higher cap. The strongest visible improvement comes partly from duplicate/related spike tasks (128/138, 131/141), so the exact +350% ceiling has limited independent evidence. Do not treat the visible optimum as a calibrated universal scale.
