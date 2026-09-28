# i2 consolidated findings through episode 3

## Current method and evidence

- Shared best remains the seed (visible fitness 0.1889437383): about 94.9% Toto and 5.1% ARIMA after weight normalization, 20.15% last-value shrink, phase-median history repair with margin 0.30, and the existing correction validator.
- The strongest visible gains include documented history repair on sensor task 193 and ship task 43. These large cases make overall fitness sensitive to repair choices; their gains do not justify broad numerical changes.
- The visible trace shows many document multipliers are rejected. Accepted large upward multipliers, including city 2x–5x and freeway 5x, appear in the final forecast as a factor of 1.5. Accepted downward multipliers can likewise appear as 0.5. This observed cap means tuning the validator mainly switches corrections on or off rather than applying the stated full magnitude. The audit script in `shared/skills/` reproduces the observed factors. A total task gain mixes the base forecast and document effects and should not be treated as a causal correction score.
- The harmful freeway task 146 has an accepted 5x event, but helpful city tasks 128/138 also have accepted 5x events. Their confidence and document-base features overlap with other beneficial and harmful examples. A global validator threshold chosen from these 54 tasks has no clear generalization case.

## Tested changes that NEVER worked as replacements for the seed

- Numerical blend/shrink tuning: 10% Chronos with less shrink, 5% seasonal TimesFM, and shrink reductions to 12% or 18% raised visible fitness but all failed the hidden check. Do not repeat searches of these parameters on the same visible tasks.
- Correction strength 1.05 lowered visible fitness and failed hidden. Its actual effect was narrower than offline scaling predicted, so simple scaling of final forecasts is not a sound proxy for strength tuning.
- Narrow history-range clipping lowered visible fitness in offline analysis.
- Linear and seasonal-naive history fill passed hidden but lost visible fitness (notably sensor task 193). A lower repair margin (0.20) lost a little visible fitness and failed hidden. Backtest improvement alone did not predict the best future forecast.

## Episode 3 decision

Reassessed the correction validator and magnitude calibrator instead of resuming numerical tuning. The observed correction cap and feature overlap leave no well-supported global change that is likely to improve hidden performance. No episode 3 submission was made; the five-submission limit remains unspent. The reusable correction audit is in `shared/skills/audit_visible_corrections.py` with a README. Preserve the seed until a mechanism can distinguish event relevance beyond the current validator features.
