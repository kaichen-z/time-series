# Rejected Task-Conditioned Screening Candidate Report

- Candidates: 103
- Train / Dev: 80 / 20
- Accepted generations: []
- Candidate SHA-256: `3e3776f5b8fcdd55af3c27fb56e3a20062a5b2467cb8e18edc6cda8fb1924011`
- Frozen SHA-256: `None`
- Public Test accessed: `False`
- Final constraints met: `False`
- Dev evaluations: 1
- Final Dev gate: rejected: insufficient task-conditioned diversity

| Split | Coverage | Active success | Failure exposure | N/A exposure | Oracle retention | Mean regret | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.6751 | 0.2199 | 0.1051 | 1.0000 | 0.0000 | 0.7601 |
| Dev | 1.0000 | 0.6104 | 0.2892 | 0.1004 | 1.0000 | 0.0000 | 0.7587 |

| Split | Mean active | Min / Max | Unique dictionaries | Pairwise Jaccard | Conditioned Statistical / TSFM / Combined |
|---|---:|---:|---:|---:|---:|
| Train | 78.29 | 78 / 80 | 4 | 0.9934 | 4 / 0 / 0 |
| Dev | 78.15 | 78 / 79 | 3 | 0.9963 | 2 / 0 / 0 |
