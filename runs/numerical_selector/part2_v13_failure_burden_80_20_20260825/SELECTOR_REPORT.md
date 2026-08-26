# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: []
- Screening SHA-256: `5727d1c9bce4fd1ac509593fff455597fdedc3b11c8fa18f39b97e00facf37a4`
- Decision SHA-256: `97dad2ef6f235ae58631154239a5daf1d765b22c20f5a19f53f3498fadea9c69`
- Dev accepted: `False`
- Final Dev gate: Dev sRMSE increased
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.314688 | 0.045601 | 0.509561 | 0.060921 | 0.535545/0.840952 | 0/0 | 2.600768 | 201.871199 | 34.394513 | 0.101848 | 18 | 3 | 0.2000 |
| Dev | 1.0000 | 0.283871 | 0.044369 | 0.484658 | 0.074250 | 0.492357/0.557040 | 0/0 | 3.302257 | 164.317069 | 33.149848 | 0.068509 | 8 | 2 | 0.1500 |
