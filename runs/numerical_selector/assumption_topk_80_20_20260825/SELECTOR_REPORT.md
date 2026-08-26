# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: [1]
- Screening SHA-256: `2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2`
- Decision SHA-256: `9be0b5383a7cc1dc0cb94839084e10859eae0241a99311436f88b3c372e70875`
- Dev accepted: `True`
- Final Dev gate: Train proposal passed all read-only Dev gates
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.242075 | 0.022318 | 0.424717 | 0.042049 | 0.491282/0.533017 | 0/0 | 2.488225 | 198.928377 | 30.396965 | 0.045802 | 7 | 3 | 0.0125 |
| Dev | 1.0000 | 0.268222 | 0.037638 | 0.463206 | 0.068154 | 0.462338/0.545792 | 0/0 | 3.261772 | 163.402283 | 35.604355 | 0.056675 | 4 | 2 | 0.0000 |
