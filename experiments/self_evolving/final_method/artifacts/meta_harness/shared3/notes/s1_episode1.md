# s1 episode 1: calibrating localized positive corrections

Read the seed harness and the 54 visible traces. The seed accepts/halves/rejects by a fixed logistic score and caps every change at 50% of the base. This cap discards much of the signal in short positive events: several hour|seas tasks have forecast levels hundreds or thousands of times `sigma_main_calib`, and their documented multipliers correspond to large true spikes. In the visible examples, a multiplier of 1.5 was also rejected even though its solo correction helped.

Submission 1 (`s1_031238_2dff17`): for positive corrections covering at most 20% of the horizon, with `mean(abs(base_window))/sigma >= 100` and `docbase < 0.55`, accept the correction and raise the upward cap to 3.5x the base. Visible fitness rose from seed 0.1948 to 0.2323; hidden check passed. This helped hour|seas events but over-adjusted two hour|flat variants.

Submission 2 (`s1_031322_434e76`): restrict this rule to seasonal cells, shrink the event effect to `0.75*(m-1)+0.125`, and cap the effect at +300% (4x forecast). Visible fitness 0.2388; hidden check passed and it became shared best. Only eight visible tasks changed from seed, all improved. All 80 unlabeled views return exactly H finite values.

The improvement is concentrated in a few related visible event families; it is not evidence for a universal higher cap. Keep the original 50% bound for flat, broad, or low-level corrections. No further submissions this episode.
