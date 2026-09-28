# Agent i2, episode 2

## Reassessment

Episode 1's numerical perturbations (less shrink, Chronos/TimesFM blends) improved visible fitness but failed every hidden check. Do not retry numerical blend or shrink tuning on these 54 tasks; it did not generalize. A 5% correction-strength increase also lost visible fitness. Narrow clipping was already harmful offline. These directions NEVER worked in the tested forms.

## Submission 1

Changed history fill from phase median to linear interpolation. Hidden check passed, but visible fitness dropped from 0.18894 to 0.1826. Task 193's gain fell from 1.947 to 1.493 despite linear fill having lower history backtest error; thus that backtest is not a reliable selector for this repair type. Keep phase median.

## Submissions 2 and 3

- Margin 0.20 admitted additional moderate backtest improvements. Visible fitness 0.1886, slightly below seed; hidden check failed. The seed margin of 0.30 already captures the two major successful repairs. A looser threshold NEVER worked in this test.
- Seasonal naive fill gave essentially the same visible result as linear (0.1826) and passed hidden check, but lost about 0.45 gain on task 193. It did not clear the required visible gate.

## Document-control audit and stopping decision

The validator already rejects most weak corrections and accepts the strongly supported city-event corrections. Accepted document effects are positive for most city tasks and for the large ship repair. The clearest harmful accepted correction is a 5x freeway occupancy multiplier on task 146. Its extreme magnitude shares a feature pattern with helpful 5x city corrections, so a global validator weight change would likely discard useful events along with it. Other proposed threshold moves trade off beneficial and harmful tasks. No remaining global change has a clear generalization mechanism; I stopped at three submissions rather than spend two more on visible-fold tuning. Shared best remains the seed.
