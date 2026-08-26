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
| Train | 1.0000 | 0.319276 | 0.046780 | 0.509210 | 0.061449 | 0.563712/0.914741 | 0/0 | 2.576616 | 202.502636 | 35.267701 | 0.105816 | 15 | 2 | 0.2125 |
| Dev | 1.0000 | 0.281844 | 0.044521 | 0.480494 | 0.074572 | 0.492357/0.557040 | 0/0 | 3.292765 | 164.319118 | 33.389580 | 0.067284 | 6 | 2 | 0.0500 |
