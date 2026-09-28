# m0 episode 4

- Read the consolidated notes, prior episode notes, and the visible correction trace before editing. Best entering the episode was attempt 7, visible fitness 0.31156, hidden check pass.
- Extended the restoration-boundary rule to every validated short hourly upward correction with multiplier at least 1.5, including multi-correction tasks and implausibly large extracted multipliers. Kept the original validator and all amplitude caps. The mechanism is the same as for isolated surges: extracted intervals include the first return-to-baseline step.
- Submission 8 passed the hidden check and improved visible fitness to 0.31191. Only two visible tasks changed, with distinct series: task_146 gained 0.0027 and task_183 gained 0.0074. Both had final extracted steps near baseline; their prior positive adjustment at that step caused avoidable error.
- Considered loosening validation for an isolated 1.5x surge and reshaping a full-horizon zero claim, but each had only one distinct visible example. No further submission was made because those rules would be tuned to individual tasks.
- Episode submissions used: 1 of 5. The accepted shared best is attempt 8.
