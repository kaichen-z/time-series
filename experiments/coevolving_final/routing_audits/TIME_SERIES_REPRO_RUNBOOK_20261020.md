# Time-series routing reproducibility runbook

Run from `joint_evolve/` in the existing locked environment. These commands reproduce analyses on already-opened data; they do not create fresh evidence.

## 1. Random outer-CV numerical routing and uncertainty

```bash
python native_uncertainty_risk_gate.py --dataset time_mmd \
  --out native_uncertainty_risk_gate_time_mmd.json
python native_uncertainty_ablation.py --dataset time_mmd \
  --out native_uncertainty_ablation_time_mmd.json
python point_gate_random_controls.py --dataset time_mmd \
  --out point_gate_random_controls_time_mmd.json
```

Use for: exchangeable-split routing mechanism, expert/position controls, and generic empirical residual calibration.

Do not use for: temporal robustness, model-native uncertainty, or document contribution.

## 2. Time-MMD temporal audits

```bash
python time_mmd_chronological_event_holdout.py \
  --out time_mmd_chronological_event_holdout.json
python time_mmd_temporal_frequency_release.py \
  --out time_mmd_temporal_frequency_release.json
python time_mmd_prequential_drift_audit.py \
  --out time_mmd_prequential_drift_audit.json
```

Use for: documenting temporal failure, frequency sign flips, expanding/recent-window failure, and the post-hoc association between unlabeled covariate shift and routing harm.

Do not optimize thresholds on these outputs. The only allowed future evaluation is specified by `time_mmd_future_shift_gate_protocol.json`.

## 3. TimesX document-veto audit

```bash
python timesx_document_veto_oof.py --out timesx_document_veto_oof.json
```

Use for: showing that real documents do not stably improve on numeric-only or shuffled-document controls.

Do not present the numerical router's gain as document gain.

## 4. Required reporting table

Every report must label each row with:

- dataset and split;
- whether the split was untouched when the rule was frozen;
- baseline and candidate sMAE/sRMSE/sCRPS;
- routing coverage and misrepair count;
- real/shuffled/no-document controls when documents are involved;
- whether the result is confirmatory, descriptive, or post-hoc.

Never merge random outer-CV, event-time holdout, opened Test, or small semantic pilots into a single aggregate.

## 5. Immutable current boundaries

- `TIME_SERIES_DECISION_LEDGER_20261020.md` is the claim ledger.
- `time_mmd_future_shift_gate_protocol.json` is the prospective protocol.
- Existing Time-MMD and TimesX splits are closed to further rule selection.
- A future result is valid only if the protocol file is unchanged and its SHA is reported before opening the new block.
