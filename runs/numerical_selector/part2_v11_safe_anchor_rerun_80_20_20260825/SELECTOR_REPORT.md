# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: []
- Screening SHA-256: `2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2`
- Decision SHA-256: `992b87aa82177456573072661d94ba30e31b3643d21bfe83cd49850ee3b9ab06`
- Dev accepted: `False`
- Final Dev gate: Dev sRMSE increased
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.324975 | 0.047085 | 0.516161 | 0.061744 | 0.564302/0.881486 | 0/0 | 2.658557 | 203.051578 | 35.269483 | 0.110269 | 20 | 3 | 0.2375 |
| Dev | 1.0000 | 0.279145 | 0.043796 | 0.479898 | 0.074302 | 0.492357/0.557040 | 0/0 | 3.214052 | 164.184712 | 32.290128 | 0.064344 | 6 | 2 | 0.0500 |
