# Frozen Numerical Selector Report

- Train / Dev: 80 / 20
- Accepted generations: [1]
- Screening SHA-256: `2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2`
- Decision SHA-256: `f9fcd056db3044bf3664055b90a2be7c02e2b520028c0842d014b19dc5d2a06e`
- Dev accepted: `True`
- Final Dev gate: Train proposal passed all read-only Dev gates
- Public Test accessed: `False`

| Split | Coverage | Mean sMAE | sMAE SE | Mean sRMSE | sRMSE SE | P90/P95 sMAE | Clipped sMAE/sRMSE | Mean MASE | Mean MAE | Mean sMAPE | Oracle regret | Methods | Families | Ensemble | Assumptions | Verifier pool | Pool families | Assumption kinds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.241722 | 0.022131 | 0.423312 | 0.041541 | 0.491282/0.533017 | 0/0 | 2.490968 | 201.102360 | 29.095191 | 0.045706 | 8 | 3 | 0.0875 | 1.54 | 2.95 | 2.08 | 5 |
| Dev | 1.0000 | 0.253887 | 0.032767 | 0.446229 | 0.066635 | 0.440712/0.454035 | 0/0 | 3.264619 | 169.095824 | 32.355960 | 0.044168 | 4 | 2 | 0.0500 | 1.60 | 3.05 | 2.30 | 5 |
