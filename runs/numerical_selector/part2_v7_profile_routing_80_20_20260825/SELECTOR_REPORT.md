# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: []
- Screening SHA-256: `feb155c2ab5abee4f6e6a8b2b50080fd2ca94bc2c09d0b43e5547dd5ce9cfc7b`
- Decision SHA-256: `aad5dcba2759991e050ab4158506a252d35c837c53081dc3197ee1a2a5ac837c`
- Dev accepted: `False`
- Final Dev gate: Dev sRMSE increased
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.318685 | 0.046803 | 0.507280 | 0.061440 | 0.565992/0.914741 | 0/0 | 2.567957 | 201.277290 | 35.535585 | 0.105312 | 13 | 2 | 0.1750 |
| Dev | 1.0000 | 0.277232 | 0.044055 | 0.476544 | 0.074772 | 0.500820/0.557040 | 0/0 | 3.204404 | 164.185254 | 32.415204 | 0.063134 | 4 | 2 | 0.0500 |
