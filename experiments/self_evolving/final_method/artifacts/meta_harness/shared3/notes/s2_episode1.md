# s2 episode 1: overlap and preserving the base

Read the seed, visible traces, and s1's note. Among the 54 visible tasks, 16 have no corrections; only four have any overlapping windows (`task_112`, `task_121`, `task_200`, and `task_231`, whose overlap is one step). In these cases the seed rejected nearly all the corrections. Thus changing overlap aggregation has essentially no visible evidence to support it. The base is usually better than small, broad correction multipliers. Retain the seed validator's rejection of those corrections.

Submission 1 (`0003_s2.py`) combined overlap averaging, a broader short-upward-event acceptance rule, and a proposed short-zero cap. Visible fitness 0.1972; hidden check failed. The zero-cap change was accidentally ineffective because the validator's 0.5 weight still limited its effect. The upward rule changed `task_137`; s1's seasonal large-shift rule covers it better.

Submissions 2 and 3 (`0007_s2.py`, `0010_s2.py`) tried increasing the validator weight for short explicit zero corrections on s1's improving shared method. Both tied visible fitness 0.2475 and passed hidden check, but the later 50% effect cap made them no-ops. This is a useful caution when modifying multi-stage correction logic.

Submission 4 (`0013_s2.py`) permitted a 68% reduction, including the final effect cap, for short (`<=20%` of H) extracted zero multipliers that had been half-accepted. It preserved a base component because document timing and baseline measurements are uncertain. Visible fitness increased from 0.2475 to 0.2477, hidden check passed, and the method became shared best at submission time. Only the two equivalent visible ATM-outage variants (`task_104`, `task_87`) changed materially. A simple local check of their raw MAE+RMSE favored roughly 65-70% reduction over 50%. Broad zero corrections, including `task_43`, remained unchanged.

No further submissions this episode. The visible improvement is small and based on two near-duplicate tasks; avoid widening the zero rule without evidence.
