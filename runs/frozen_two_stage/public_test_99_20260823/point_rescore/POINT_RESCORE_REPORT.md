# Dr-CiK-Aligned Point Forecast Rescore

- Tasks: 99
- Baseline: `E_toto_reference`
- sCRPS: **not computed** (no probabilistic trajectories in this phase)
- Model calls: **0**; all forecasts were read from the frozen artifact
- Status: public-label development/regression metrics, not an official hidden-test score

| Row | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Coverage | W/T/L vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current_global_ranker | 0.460684 | 0.068927 | 0.680614 | 0.091953 | 1.145517/1.953672 | 0/2 | 1.0000 | 0/99/0 |
| B_screening_only | 0.460684 | 0.068927 | 0.680614 | 0.091953 | 1.145517/1.953672 | 0/2 | 1.0000 | 0/99/0 |
| C_decision_only | 0.472474 | 0.070226 | 0.688686 | 0.095567 | 1.058499/1.529734 | 0/3 | 1.0000 | 41/7/51 |
| D_full_two_stage | 0.472474 | 0.070226 | 0.688686 | 0.095567 | 1.058499/1.529734 | 0/3 | 1.0000 | 41/7/51 |
| E_toto_reference | 0.460684 | 0.068927 | 0.680614 | 0.091953 | 1.145517/1.953672 | 0/2 | 1.0000 | 0/99/0 |
