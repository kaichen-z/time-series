# i1, episode 3: redirection and consolidation

I reviewed all shared notes and six prior attempts. The seed remains the only accepted config. Episode 1's broad retrieval changes were harmful; episode 2's tiny STL-ETS addition improved both visible folds but failed hidden. This episode tested three distinct mechanisms, then stopped with two submissions unused rather than tuning against the same 54 visible tasks.

| Change | Visible fitness | Hidden check | Interpretation |
| --- | ---: | --- | --- |
| Strength 1.0 -> 1.2 | 0.1884 | fail | Almost all accepted short event corrections were already saturated; only task_158 worsened materially. The naive trace extrapolation was invalid because the pipeline caps/scales correction effects. |
| +0.03 combined TimesFM seasonal | 0.1912 | fail | Both visible folds improved, as with STL-ETS in episode 2. A small seasonal blend still did not generalize. |
| Shrink 0.20153 -> 0.18 | 0.1917 | fail | Distributed visible improvement did not pass hidden. The current shrink should stay. |

What NEVER worked as an accepted improvement across our experiments: lowering repair margin; broadening validator acceptance with its bias; penalizing long correction windows; removing the validator's trust term; linear fill; small STL-ETS and combined TimesFM seasonal additions; lower shrink; stronger document corrections. The linear fill alone passed hidden, but hurt visible enough to be rejected. Thus hidden failure is not limited to retrieval, and a visible gain across both folds is inadequate evidence.

The correction validator currently accepts many short, large events and rejects most broad percentage claims. On the visible traces, two repaired histories (task_43 and task_193) and task_61 account for much of the positive fitness. Attempting to amplify corrections is ineffective for the short events because their applied effects appear capped; a 20% strength increase changed only task_121 and task_158 in the per-task result.

I saved `shared/skills/trace_program_probe.py` for screening numerical ideas and documented its limits in `shared/skills/README.md`. It correctly anticipated the direction of the shrink trial on visible folds, but cannot reproduce repaired histories or predict hidden transfer. No extraction-instruction submission was used. Best config remains unchanged.
