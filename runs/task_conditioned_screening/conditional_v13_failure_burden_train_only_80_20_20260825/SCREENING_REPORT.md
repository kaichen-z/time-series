# Frozen Task-Conditioned Screening Report

- Candidates: 103
- Train / Dev: 80 / 20
- Accepted generations: [1, 3, 6]
- Candidate SHA-256: `5727d1c9bce4fd1ac509593fff455597fdedc3b11c8fa18f39b97e00facf37a4`
- Frozen SHA-256: `5727d1c9bce4fd1ac509593fff455597fdedc3b11c8fa18f39b97e00facf37a4`
- Public Test accessed: `False`
- Final constraints met: `True`
- Dev evaluations: 1
- Final Dev gate: accepted: screening improved on Train without a Dev regression

| Split | Coverage | Active success | Failure exposure | N/A exposure | Oracle retention | Mean regret | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.0000 | 0.6636 | 0.2336 | 0.1028 | 1.0000 | 0.0000 | 0.7601 |
| Dev | 1.0000 | 0.5916 | 0.3101 | 0.0983 | 0.9000 | 0.0062 | 0.7607 |

| Split | Mean active | Min / Max | Unique dictionaries | Pairwise Jaccard | Conditioned Statistical / TSFM / Combined |
|---|---:|---:|---:|---:|---:|
| Train | 78.29 | 77 / 80 | 6 | 0.9841 | 2 / 1 / 1 |
| Dev | 78.35 | 77 / 80 | 6 | 0.9836 | 2 / 1 / 1 |
