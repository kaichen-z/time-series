# Agent i2, episode 4

## Reassessment

The shared best remains the seed (visible fitness 0.1889437). Read all prior notes and attempts before reconsidering the document validator and history repair gate. Prior episodes showed that visible improvements from lower shrink or model diversification consistently failed the hidden check. Linear and seasonal-naive repair fill passed hidden but lost visible fitness; a lower repair margin lost visible fitness and failed hidden. These tested numerical and repair changes NEVER worked as replacements for the seed.

## Visible-trace audit

- Reconstructed the seed base from the method forecasts and compared it with the final forecast. Accepted future corrections improve raw MAE on both ATM tasks, eight city tasks, and freeway task 158. They harm freeway task 146 and very slightly harm location task 121 and sensor task 179. Sensor task 183's 45x proposed multiplier is capped to an observed 1.5x and slightly improves raw MAE, so rejecting large stated magnitudes alone is not supported.
- The 5x multiplier helps city tasks 128 and 138 but harms freeway task 146. A single validator magnitude threshold cannot distinguish these from the features provided. Confidence and document-base features also overlap. The correction gate therefore has no visible, mechanistically clear global adjustment with a plausible hidden benefit.
- Repair variants exist for 13 visible tasks. The current margin accepts the two high-leverage repairs (ship task 43 and sensor task 193). Task 47's phase-median backtest gain is about 0.291 relative, just below the 0.30 gate; earlier lowering the gate to 0.20 reduced visible fitness and failed hidden. The other candidate repairs have weaker backtest evidence or no computable backtest.

## Decision and result

Made zero submissions this episode. The evidence supports preserving the seed rather than spending submissions on another threshold fitted to 54 visible tasks. No new accepted improvement was found. A future change needs a new relevance feature for document corrections or a repair-selection signal that predicts the future better than the existing backtest margin.
