# Final Frozen Two-Stage Public Test Report

- Tasks: 99
- Screening SHA-256: `9fb0be569e2bc4597a206ba9e4b0fa44aa99f0abbb07bdee82c219e74fa95096`
- Decision SHA-256: `9b3682bd46b7bf739f8ad8c02f8bd9d9384d167e9a771d2000c9991643c4d8a7`
- LLM / mutation calls: 0 / 0

| Row | Mean MASE | Median MASE | Mean RMSSE | Mean MAE | Mean sMAPE | Coverage | Catastrophic | Methods | Families | Ensemble | W/T/L vs A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current_global_ranker | 2.883039 | 1.580000 | 2.799772 | 555.728113 | 45.604233 | 1.0000 | 0.0404 | 1 | 1 | 0.0000 | 0/99/0 |
| B_screening_only | 2.883039 | 1.580000 | 2.799772 | 555.728113 | 45.604233 | 1.0000 | 0.0404 | 1 | 1 | 0.0000 | 0/99/0 |
| C_decision_only | 3.184596 | 1.796324 | 2.876107 | 1340.198502 | 44.523957 | 1.0000 | 0.0505 | 35 | 3 | 0.4343 | 41/7/51 |
| D_full_two_stage | 3.184596 | 1.796324 | 2.876107 | 1340.198502 | 44.523957 | 1.0000 | 0.0505 | 35 | 3 | 0.4343 | 41/7/51 |
| E_toto_reference | 2.883039 | 1.580000 | 2.799772 | 555.728113 | 45.604233 | 1.0000 | 0.0404 | 1 | 1 | 0.0000 | 0/99/0 |
