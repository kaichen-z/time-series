# s2 episode 3: closed hourly slots and overlap reassessment

Read every shared note, the current best harness, all permitted visible traces, and the 80 unlabeled Train views. Only four visible tasks have genuinely overlapping correction windows. The current last-write-wins policy handles `task_112` reasonably: a later positive correction replaces an unhelpful negative one. The other overlapping cases have no accepted correction pair that supports a new aggregation rule. I made no overlap change.

One visible `hour|seas` correction raises a forecast in an hour that was exactly zero in both observed daily cycles (`task_112`, step 5). I submitted a narrow guard that suppresses a positive correction when *every* historical same-hour observation is zero. It changes only that single visible correction and no other Train view. Hidden check passed, but visible fitness moved only from 0.260121 to 0.260150, below the evaluator's 0.0001 promotion threshold. The submission was rejected; the shared best remains unchanged. This is too little support to keep the special case.

I also screened a broader rule that zeroes forecasts in hourly slots with at least three observed all-zero daily cycles. It would change only visible `task_179` and `task_183`, and local visible error worsens by 0.058 and 0.062 respectively. I did not submit it. This illustrates why a zero history profile does not imply the future event cannot activate that hour.

The final private `my_harness.py` is a copy of the current shared best, which was updated concurrently by s0's accepted attempt `0022_s0.py`. I added `shared/skills/overlap_diagnostics.py` and updated its README. One submission used this episode; no further rule has a strong enough mechanism or support to justify more threshold tuning.
