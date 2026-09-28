# Consolidated correction findings through episode 3

The seed logistic validator is useful. Preserve its accept/half/reject decision and conservative treatment of broad, weak, or overlapping claims. Removing corrections entirely lowers visible fitness from about 0.195 to 0.14.

Short isolated hourly surges are the clearest supported exception. For one correction with multiplier 2–6 spanning at most half the horizon, submitted changes widened the amplitude bound, then treated the last extracted step as a restoration boundary and applied the full multiplier to earlier steps. The distinct visible surge series consistently show strong interior peaks and a last step near the base. Fitness rose from 0.19477 (seed) to 0.31156 (attempt 7), with hidden checks passing at each accepted stage.

One short isolated daily zero-multiplier event is best handled by fully suppressing all but the final extracted step. The final step is a restoration boundary. This improved two duplicate visible tasks; it is one observed series, not two independent examples.

Broad and weak claims are mixed or harmful in the visible trace. Prior attempts deliberately left them under the seed validator and 50% cap. The 45x claim, a 1.5x missed spike, and a full-horizon zero claim remain plausible opportunities, but each is supported by only one distinct visible example. Avoid tuning general rules from those alone.

Use `shared/skills/audit_visible.py` before interpreting per-task improvements: several task IDs repeat the same series and truth. The script reads only authorized visible labels and attempts.
