# i1, episode 1: retrieval observations

Initial best has visible fitness 0.18894. The instruction file is already detailed about target relevance, quoted timing, and recurrence, so instruction edits are costly and not the first lever.

Visible traces show the validator accepting short, large city events (multipliers 2–5), which correspond to localized increases in truth. It rejects most broad small percentage claims, several of which disagree with truth. Strong gains at task_43 and task_193 come from accepted history repair, so changing extraction or the repair gate risks these important cases.

Attempt 1: Lowering repair margin 0.30 to 0.20 slightly lowered visible fitness to 0.1886 and failed hidden check. The expected newly admitted task_179 repair did not help enough to compensate. Keep margin 0.30.

The seed's positive trust weight gives document corrections more influence when Toto's history back-test error is higher. That seems risky: task_121 has high trust error and an accepted correction in the wrong direction. A reduced trust coefficient with a higher intercept can preserve city and ATM decisions while reducing reliance on poor back-tests.

Attempt 2: trust coefficient 1.017 → 0 and bias -2.747 → -2.0 reduced visible fitness to 0.1876 and failed hidden. Keeping seed trust behavior is safer.

Attempt 3: bias +1 raised visible better count from 31 to 33 but cut fitness to 0.1737 and failed hidden. It let broad corrections through: task_234 gain fell to -0.436 and task_77 to -0.288. A validator change must penalize corrections spanning most of the horizon.

Attempt 4: bias +1 with wfrac weight -2 still fell to 0.1677 and failed hidden. It also reduced task_43 gain from 3.887 to 2.426, so horizon-wide corrections can be crucial when a documented lasting event is real. Reject this heuristic.

For history repair, task_193 is a large contributor. Its visible back-test errors are 0.278 with phase_median and 0.001 with linear fill, with comparable evidence for task_43. A linear fill may better bridge temporary disturbances without imposing a possibly inappropriate phase value. Test as final episode submission.

Attempt 5: linear fill passed hidden check but visible fitness fell to 0.1826, so it was not accepted. Despite task_193's stronger linear back-test, its forecast gain fell from 1.9472 to 1.4933. This is a useful warning that repair back-test ranking does not reliably rank future performance.

Episode summary: All five submissions were rejected; the initial shared best remained stronger. The existing validator's conservative treatment of broad effects and phase_median repair are important. Future work should avoid broad intercept changes and focus on mechanisms that distinguish documented, target-specific changes from boilerplate without penalizing genuine long-horizon states. No extraction-instruction submissions were used in this episode.
