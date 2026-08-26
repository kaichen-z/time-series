# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: [1]
- Screening SHA-256: `2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2`
- Decision SHA-256: `c0bc6eea7e75897ecc6c19465b1fb9c1a5c02a99349189780fbbdfa9710f6d09`
- Dev accepted: `True`
- Final Dev gate: Train proposal passed all read-only Dev gates
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.235637 | 0.021042 | 0.411959 | 0.039325 | 0.445881/0.532969 | 0/0 | 2.487323 | 200.157874 | 29.683178 | 0.041136 | 11 | 2 | 0.1125 |
| Dev | 1.0000 | 0.271525 | 0.036926 | 0.466565 | 0.067000 | 0.457583/0.535258 | 0/0 | 3.326491 | 169.585345 | 35.711423 | 0.059617 | 8 | 2 | 0.1500 |
