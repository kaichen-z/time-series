# Rejected Task-Conditioned Screening Candidate Report

- Candidates: 103
- Train / Dev: 80 / 20
- Accepted generations: [1, 2, 3, 4, 5, 6, 7, 14]
- Candidate SHA-256: `d4af26d636513721f35de933af1cae59fe8c91e1bcd615783402648148acf2b2`
- Frozen SHA-256: `None`
- Public Test accessed: `False`
- Final constraints met: `False`

| Split | Coverage | Active success | Failure exposure | N/A exposure | Oracle retention | Mean regret | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.8165 | 0.0445 | 0.1390 | 1.0000 | 0.0000 | 0.5642 |
| Dev | 1.0000 | 0.8002 | 0.0672 | 0.1326 | 1.0000 | 0.0000 | 0.5636 |

| Split | Mean active | Min / Max | Unique dictionaries | Pairwise Jaccard | Conditioned Statistical / TSFM / Combined |
|---|---:|---:|---:|---:|---:|
| Train | 58.11 | 56 / 64 | 12 | 0.9452 | 11 / 1 / 1 |
| Dev | 58.05 | 56 / 64 | 9 | 0.9473 | 11 / 1 / 0 |
