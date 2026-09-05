# Dr-CiK 80/20/99 v3 Toto-balance audit

V3 repartitions the same 199 public labeled tasks into exactly 80 Train, 20 Dev, and 99 Public Test
tasks. Entities are disjoint across partitions. Selection used the frozen seven-baseline profile
plus Toto 2.0 per-task capped sMAE and capped sRMSE; it did not use document labels or text.

| Split | Train mean | Dev mean | Test mean | Relative mean gap |
|---|---:|---:|---:|---:|
| v1 seven-baseline aggregate | 0.528712 | 0.524750 | 0.592232 | 12.05% |
| v1 Toto sMAE | 0.312302 | 0.304074 | 0.460684 | 40.65% |
| v1 Toto sRMSE | 0.505927 | 0.523643 | 0.680614 | 29.38% |
| v2 seven-baseline aggregate | 0.559225 | 0.559650 | 0.560525 | 0.23% |
| v2 Toto sMAE | 0.355945 | 0.303690 | 0.425495 | 31.61% |
| v2 Toto sRMSE | 0.519478 | 0.610416 | 0.652134 | 22.31% |
| **v3 seven-baseline aggregate** | **0.561925** | **0.561000** | **0.558071** | **0.69%** |
| **v3 Toto sMAE** | **0.388409** | **0.392069** | **0.381407** | **2.77%** |
| **v3 Toto sRMSE** | **0.601756** | **0.587150** | **0.590347** | **2.46%** |

V3 passes all three predeclared 5% relative-mean gates. It has 44 Train entities, 12 Dev entities,
and 57 Public-Test entities. The manifest digest is
`5f6ddec2ae460b292629f78e76784db1d58da79df1fcbbe993b5e9aa1d93b835`.

This is an internal difficulty-controlled protocol: because its assignment used public labels and
Toto outcomes, its 99-task result must not be presented as untouched external test performance.
The official unlabeled Hidden-80 remains the final test boundary.
