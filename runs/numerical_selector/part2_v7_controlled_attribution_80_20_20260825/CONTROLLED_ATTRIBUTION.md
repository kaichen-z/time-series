# Controlled Part 1 vs Part 2 Downstream Attribution

The same frozen Decision policy and the same saved candidate forecasts/hindcasts are used in both arms. Only the active dictionary changes.

| Split | Policy | Mean sMAE | Mean sRMSE | P90/P95 sMAE | Coverage |
|---|---|---:|---:|---:|---:|
| Train | part1 | 0.319271 | 0.509747 | 0.563712/0.847740 | 1.0000 |
| Train | part2_v7 | 0.319758 | 0.508754 | 0.563712/0.847740 | 1.0000 |

- Train active-set changes: 80
| Dev | part1 | 0.278341 | 0.475353 | 0.492357/0.557040 | 1.0000 |
| Dev | part2_v7 | 0.278341 | 0.475353 | 0.492357/0.557040 | 1.0000 |

- Dev active-set changes: 20
