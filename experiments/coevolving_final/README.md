# Co-evolving forecasting: archived analysis / reproduction scripts (2026-10-20 snapshot)

Status: **archived analysis/reproduction scripts**, not a single end-to-end release of the paper's Table 1.
The co-evolved Dr-CiK Numerical/Decision modules used here are the ones already in
`experiments/self_evolving/final_method/unified/coevo_drcik/` (`evolved_forecast_drcik.py`,
`evolved_adjust_drcik.py`, `seed_forecast.py`, `seed_adjust.py`, byte-identical to the run that produced the
reported numbers). See "Known gaps" below.

No data, results, caches, model weights, logs or credentials are included. Every script reads inputs that you
prepare yourself (see "Data you must provide").

## Layout

```
routing_audits/                 numerical routing, uncertainty and document-ablation audits (Time-MMD, TimesX, Dr-CiK)
  TIME_SERIES_REPRO_RUNBOOK_20261020.md     commands and allowed use of each audit
  TIME_SERIES_DECISION_LEDGER_20261020.md   claim ledger: what survives / what does not
  time_mmd_future_shift_gate_protocol.json  frozen prospective protocol (report its SHA before opening a new block)
  LOCK.json                                 lock of the joint controller (chosen file + SHA below)
  formal_48/round_10/routing/candidate.py   the locked joint controller (sha256 42eac810...)
  run_evolution.py, seed_controller.py      three-role Codex evolution of the joint controller
  evaluate_controller.py, viewstore.py      Train-only evaluator and the frozen-view reader
  evaluate_v6_incumbent.py, v6_incumbent/   locked v6 Time-MMD / TimesX programs used as the incumbent
  native_uncertainty_*.py, point_gate_random_controls.py, risk_aware_statistical_gate.py,
  analyze_statistical_ensemble.py, time_mmd_*.py, timesx_document_veto_oof.py
drcik_reconstruction_v2/        Dr-CiK mixed-document reconstruction (v2) and joint-stage evolution ablation
  run_luna_reconstruction.py    LLM timeline reconstruction (uses final_method/scripts/tl2.py, configs/instr_evolved.json)
  build_reconstruction_views.py prepare / finish the reconstructed views
  score_scrps_mixed_v2.py, run_ablation_v2.py, run_joint_stage_evolution_ablation.py, export_decision_task_scatter.py
  *_REPORT.md, JOINT_STAGE_SUBMISSION_CURVE_NOTES.md   what each run showed (numbers as recorded)
```

## Reproducing

Requirements: Python 3.10+, `numpy`, `scipy`, `scikit-learn` (repo `requirements.txt`). LLM steps
(`run_luna_reconstruction.py`, `run_evolution.py`) need the `codex` CLI logged in; nothing in this folder
stores keys.

**Runs directly once the data below is in place** (commands from the runbook, run inside `routing_audits/`):

```bash
python native_uncertainty_risk_gate.py --dataset time_mmd --out native_uncertainty_risk_gate_time_mmd.json
python native_uncertainty_ablation.py  --dataset time_mmd --out native_uncertainty_ablation_time_mmd.json
python point_gate_random_controls.py   --dataset time_mmd --out point_gate_random_controls_time_mmd.json
python time_mmd_chronological_event_holdout.py --out time_mmd_chronological_event_holdout.json
python time_mmd_temporal_frequency_release.py  --out time_mmd_temporal_frequency_release.json
python time_mmd_prequential_drift_audit.py     --out time_mmd_prequential_drift_audit.json
python timesx_document_veto_oof.py             --out timesx_document_veto_oof.json
```

Read `TIME_SERIES_REPRO_RUNBOOK_20261020.md` for what each output may and may not be used for, and the ledger for
the claim boundary. Existing Time-MMD/TimesX splits are closed to further rule selection.

Dr-CiK reconstruction v2 (inside `drcik_reconstruction_v2/`): `run_luna_reconstruction.py` →
`build_reconstruction_views.py prepare|finish ...` → `score_scrps_mixed_v2.py` / `run_ablation_v2.py` /
`run_joint_stage_evolution_ablation.py` / `export_decision_task_scatter.py`. The later scripts read the
intermediate files written next to them by the earlier steps (`views_drcik_mixed_v2.json`,
`predictions_drcik_mixed_v2.json`, `toto_repair_q_v2.json`); those outputs are not committed.

## Data you must provide

| what | where the scripts look |
|---|---|
| Time-MMD / TimesX frozen Train windows (`time_mmd_train.pkl`, `timesx_train.pkl`) | `routing_audits/private/` |
| Dr-CiK frozen views, events, truth (`drcik_full/dry_mixed/...`) and native TSFM quantiles (`native_scrps/drcik/*_q.json`) | `$TS_DRCIK_AUX_ROOT` (default `data/drcik_aux`) |
| Dr-CiK reliability-masked TimesFM forecasts and masks | `routing_audits/artifacts/drcik_train_reliability/` |
| Dr-CiK public tasks (`<task_id>.json`) | `$TS_DRCIK_TASKS` (default `data/Dr-CiK_public/tasks`) |
| Dr-CiK native TSFM quantiles (`toto_q.json`, `timesfm_q.json`) | `$TS_DRCIK_QUANTILES` (default `data/native_scrps/drcik`) |

The Dr-CiK split (`splits/drcik_public_80_20_99_v3.json`) and the TL2 reconstruction code are taken from this repo.
Relative defaults are resolved from the current working directory.

## Known gaps

- No single entry point here regenerates every cell of the paper's Table 1. The Dr-CiK modules are in
  `final_method/unified/coevo_drcik/`; the Time-MMD/TimesX released-policy application to Test (routing + residual
  calibration as reported) is not packaged as one runnable script in this folder.
- The data-preparation steps that produce the private `.pkl` windows, the frozen view stores (`forecasts.f32`,
  `views_meta.json`) and the TSFM quantile files are not included.
