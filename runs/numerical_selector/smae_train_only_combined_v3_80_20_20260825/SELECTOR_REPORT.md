# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: []
- Screening SHA-256: `9fb0be569e2bc4597a206ba9e4b0fa44aa99f0abbb07bdee82c219e74fa95096`
- Decision SHA-256: `904ece3a47e339d5657e46e6fc4f283475abf0a5f6c16bcdec36781549e0284b`
- Dev accepted: `False`
- Final Dev gate: Dev sRMSE increased
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.319271 | 0.046686 | 0.509747 | 0.061180 | 0.563712/0.847740 | 0/0 | 2.570319 | 202.570654 | 34.786710 | 0.104882 | 19 | 3 | 0.2250 |
| Dev | 1.0000 | 0.278341 | 0.043786 | 0.475353 | 0.074503 | 0.492357/0.557040 | 0/0 | 3.202455 | 164.167821 | 32.552845 | 0.064061 | 8 | 2 | 0.1000 |
