# Agent i3, episode 1 — decision role

Started from shared seed (visible fitness 0.18894). Four submissions; none surpassed it, so I left the shared best config unchanged.

| Change from seed | Visible fitness | Hidden check | Observation |
| --- | ---: | --- | --- |
| Repair margin 0.3 → 0.4 | 0.1889 | pass | Identical visible task results. The margin operates on proportional back-test improvement: task_112 has an absolute improvement of 0.338, but only a 13.1% proportional improvement, so seed already excludes it. |
| Correction strength 1 → 0 | 0.1403 | fail | Corrections are useful overall. Removing them loses 1.461 joint-error gain on task_43 and ~0.08–0.17 on several short-event tasks. It helps only task_121 (+0.006) and task_146 (+0.026) materially. |
| Repair margin 0.3 → 0.2 | 0.1886 | fail | Admitting repairs with 20–30% back-test improvement gave a slight visible loss. |
| Repair fill phase_median → linear | 0.1826 | pass | Task_193's gain fell from 1.947 to 1.493, although linear fill had a stronger history back-test result. |

Visible traces: 13/54 tasks have repair variants. Under phase_median, only task_43 (89.3%), task_193 (83.0%), and task_205 (38.1%) have proportional back-test improvements above the seed's 30% margin. This explains why a 0.4 margin made no material difference (task_205 has negligible gain impact). Other fills can score better on history back-tests yet forecast worse; retain phase_median unless a broader mechanism is found.

Correction ablation shows accepted document events drive much of the visible gain. The validator/strength combination seems more valuable than relaxing repair. Avoid blanket damping or a high trust gate: many helpful event corrections occur in cells with trust around 0.10–0.14. The two visible tasks materially harmed by applied corrections (task_121 and task_146) do not justify a broad gate that would also reject multiple useful short events.
