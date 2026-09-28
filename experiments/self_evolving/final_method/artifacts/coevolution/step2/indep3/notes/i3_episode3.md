# Agent i3, episode 3 — decision calibration

Started from the accepted episode 2 config (`0005_i3.json`, visible fitness 0.19477). I made four submissions. None improved the visible fitness and passed the hidden check, so the shared best config remains unchanged. I used no extraction-instruction submissions.

| Change from shared best | Visible fitness | Hidden check | Main visible effect |
| --- | ---: | --- | --- |
| Correction strength 1.0 → 0.85 | 0.19466 | fail | Only five tasks changed materially. Helped task_158 by 0.0146 but hurt task_134 by 0.0128. |
| Validator ratio weight −0.00696 → −0.05 | 0.19337 | pass | Only task_158 changed materially, losing 0.1005 gain. |
| Validator confidence weight −0.074 → +1.5 | 0.16353 | fail | Admitted harmful broad corrections: task_234 −0.4167, task_77 −0.2844, task_161 −0.1914, task_199 −0.1843 relative to best. |
| Validator confidence weight −0.074 → −1.5 | 0.18971 | fail | Removed useful corrections on task_158, task_134, task_104 and task_87. |

The validator currently makes threshold decisions on a small subset of visible tasks. The standardized magnitude penalty did not catch the extreme-multiplier case I wanted it to guard against; it removed a useful correction instead. Document confidence alone does not distinguish safe from harmful corrections in these traces. Global strength damping and confidence reweighting are poor candidates for the next episode without a new signal.

The comparison utility in `shared/skills/compare_visible.py` summarizes visible per-task changes across recorded submissions. It reads only visible attempt files.
