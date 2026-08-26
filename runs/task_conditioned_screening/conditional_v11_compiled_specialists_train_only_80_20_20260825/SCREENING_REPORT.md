# Frozen Task-Conditioned Screening Report

- Candidates: 103
- Train / Dev: 80 / 20
- Accepted generations: [1, 2, 3, 6]
- Candidate SHA-256: `2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2`
- Frozen SHA-256: `2277acd9ef0b5e219960e84aa1b02fbfa13bf1b54cf106e26b00c5f3fc2babc2`
- Public Test accessed: `False`
- Final constraints met: `True`
- Dev evaluations: 1
- Final Dev gate: accepted: screening improved on Train without a Dev regression

| Split | Coverage | Active success | Failure exposure | N/A exposure | Oracle retention | Mean regret | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.8006 | 0.0647 | 0.1347 | 1.0000 | 0.0000 | 0.4596 |
| Dev | 1.0000 | 0.7856 | 0.0876 | 0.1267 | 0.9000 | 0.0062 | 0.4597 |

| Split | Mean active | Min / Max | Unique dictionaries | Pairwise Jaccard | Conditioned Statistical / TSFM / Combined |
|---|---:|---:|---:|---:|---:|
| Train | 47.34 | 46 / 49 | 7 | 0.9719 | 3 / 1 / 1 |
| Dev | 47.35 | 46 / 49 | 6 | 0.9731 | 2 / 1 / 1 |
