# Time-series document intervention: decision ledger

Status date: 2026-10-20. This ledger separates reusable evidence from opened-split diagnostics.

## What survives

1. **A learned sparse numerical router is real mechanism evidence on random outer CV.**
   On 45,948 Time-MMD Train windows, the conservative point gate improves sMAE by 1.807% and sRMSE by 1.383%, positive in all three random outer folds. Same-coverage random-expert and random-position controls are strongly negative, so both when to route and which expert to choose matter.
2. **Generic residual calibration is reliable in random outer CV.**
   Horizon and frequency×horizon residual banks improve sCRPS by roughly 1.47–1.50%. Conditioning residuals on the selected expert is unnecessary.
3. **Unlabeled covariate shift is a prospective safety-gate hypothesis.**
   In six already-inspected chronological blocks, maximum standardized feature-mean shift ranks bad recent-window blocks well (Spearman rho=-0.886). This is post-hoc motivation only. The future protocol is frozen in `time_mmd_future_shift_gate_protocol.json`.

## What does not survive

1. **Temporal generalization of the router.**
   The event-time 80/20 holdout regresses 0.442% joint. A six-block walk-forward audit regresses 0.318% for expanding and 0.556% for recent-window training. Simple historical-sign release also fails.
2. **Time-MMD document increment.**
   Scalable document residual and selective-risk tests do not beat numeric-only/shuffled controls consistently. The earlier +2.10% result is only a 20-task semantic-sensitivity pilot.
3. **TimesX document veto.**
   Real documents add only 0.041 percentage points over numeric-only veto and 0.024 points over shuffled documents, are not fold-stable, and remain worse than the no-veto numerical gate.
4. **TimesX official-release additive signals.**
   EIA STEO, EIA weekly inventory on Dev, and USDA WASDE price/supply revisions fail temporal or held-out stability. Cocoa strict-support evidence is too sparse; cotton/wheat monthly routes remain abstain.

## Claim boundary

The defensible current story is:

> Sparse learned expert routing can exploit heterogeneous numerical forecast errors under exchangeable splits, but it fails under event-time regime shift. Documents show semantic selectivity without stable incremental forecasting value. Unlabeled shift-aware abstention is a frozen hypothesis for future evaluation, not a validated solution.

Do not claim document-grounded forecasting gains, temporal robustness, native model uncertainty, or a released production gate.

## Frozen next evaluation

- Do not reopen or retune on current Time-MMD Train/Dev/Test or TimesX Dev/Test.
- Use one genuinely future dated block.
- Apply the exact protocol and threshold in `time_mmd_future_shift_gate_protocol.json`.
- Primary success requires both sMAE and sRMSE non-regression with at least one strict improvement; zero routing fires fails.
- Report no-shift recent gate, expanding gate, and always-abstain baseline together with coverage and misrepair rate.

## Recommended paper structure

1. Random outer-CV mechanism result and causal controls.
2. Chronological stress test as the central negative finding.
3. Document ablations showing semantic sensitivity is not predictive utility.
4. Prospective shift-abstention protocol, clearly labeled future work / preregistered follow-up.

This is stronger and more reproducible than presenting the opened Test improvement or small document pilots as headline evidence.
