# m0 episode 1

- The seed validator already rejects most small, long corrections. On visible tasks, large single short hour events with multipliers 2–5 are usually useful, but all are clipped to +50% over the base. Their realized values are commonly 2–4 times base.
- A wider bound should only target isolated short surges; task_146 has multiple corrections and an incorrect 5x claim, while task_183 has a 45x claim and only a modest realized increase. This motivates excluding multi-correction tasks and implausibly large multipliers.
- Local raw MAE+RMSE comparison suggests a +150% bound for isolated events improves several visible tasks. This is only an approximation to the official scaled, capped score.
- Submission 1: fixed +150% upper bound for one short hourly claim with multiplier 2–6 passed hidden check and improved visible fitness 0.1948 → 0.2216.
- Submission 2: upper bound `max(1.5, 0.5*m)` passed hidden check and reached 0.2327.
- Submission 3: upper bound `max(1.5, m-2)` passed hidden check and reached 0.2355. It permits a 5x claim to move the base up to 4x, retaining stricter limits for smaller claims. Only short isolated hourly surges are affected; the seed validator and all other bounds remain.
- Stopped after three submissions. The remaining obvious gains involve weak or conflicting claims on a small number of visible tasks, so further rules risk fitting the visible fold.
