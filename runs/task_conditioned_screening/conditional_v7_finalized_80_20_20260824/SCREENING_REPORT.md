# Rejected Task-Conditioned Screening Candidate Report

- Candidates: 103
- Train / Dev: 80 / 20
- Accepted generations: [1, 2, 3, 4, 5, 6, 7, 14]
- Candidate SHA-256: `feb155c2ab5abee4f6e6a8b2b50080fd2ca94bc2c09d0b43e5547dd5ce9cfc7b`
- Frozen SHA-256: `None`
- Public Test accessed: `False`
- Final constraints met: `False`

| Split | Coverage | Active success | Failure exposure | N/A exposure | Oracle retention | Mean regret | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.8354 | 0.0215 | 0.1431 | 1.0000 | 0.0000 | 0.5479 |
| Dev | 1.0000 | 0.8195 | 0.0442 | 0.1363 | 1.0000 | 0.0000 | 0.5485 |

| Split | Mean active | Min / Max | Unique dictionaries | Pairwise Jaccard | Conditioned Statistical / TSFM / Combined |
|---|---:|---:|---:|---:|---:|
| Train | 56.44 | 54 / 64 | 14 | 0.9355 | 13 / 1 / 1 |
| Dev | 56.50 | 54 / 64 | 9 | 0.9365 | 13 / 1 / 1 |
