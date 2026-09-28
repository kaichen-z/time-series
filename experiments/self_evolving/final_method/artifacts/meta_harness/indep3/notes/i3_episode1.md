# i3 episode 1 observations

- Seed visible fitness 0.1948. Four visible tasks have overlapping corrections, but the seed effectively leaves most conflicting spans unchanged; overlap is not a high-leverage first target.
- The seed's universal +50% cap underuses isolated, short upward events (2x to 5x) in 24-step hourly horizons. There are 9 visible and 4 input-only tasks matching a conservative eligibility rule: H <= 24, exactly one correction, duration <= 20% of H, multiplier 2 to 5.
- Submission 1 raised the cap for that rule to +100%. Visible fitness 0.2115; hidden check passed.
- Submission 2 raised it to max(+100%, half the implied relative increase), up to +200% at multiplier 5. Visible fitness 0.2256; hidden check passed. The larger extracted magnitude justifies more headroom, with substantial shrinkage.
- Raising the downside cap for zero-multiplier events looks weak: two short outages gained only about +0.0037 each locally at 70% versus 50%, and an all-horizon zero event became much worse. Retain the downside cap.
- Submission 3 used two-thirds of the extracted relative lift as the short-spike upside cap (minimum +100%). Visible fitness 0.2342 and hidden check passed. Nine visible tasks changed; their gains improved, while all other visible outputs stayed at the seed values.
- Do not generalize the wider cap to long horizons or multiple corrections based on these observations. In the visible 72-step holiday task, a 5x correction coexists with another correction and the seed's existing behavior is already uncertain. Overlap cases are too few and mostly rejected to justify a new composition rule here.
- Final accepted code is in shared/best_harness.py and i3's private my_harness.py. Three submissions used this episode.
