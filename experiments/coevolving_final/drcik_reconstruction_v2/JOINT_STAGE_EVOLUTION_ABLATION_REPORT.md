# Dr-CiK reconstruction-v2 joint-stage evolution ablation

Date: 2026-10-21

This deterministic, zero-model-call replay compares the archived joint
co-evolution stage's unique `0000_seed` numerical/decision program pair with
its final accepted program pair, using the exact same reconstruction-v2 views,
official task files, and point-metric implementation.

| Program state | Dev sMAE | Dev sRMSE | Dev sCRPS | Public Test sMAE | Public Test sRMSE | Public Test sCRPS |
|---|---:|---:|---:|---:|---:|---:|
| Joint-stage seed | 0.268876 | 0.449450 | 0.212581 | 0.298370 | 0.462113 | 0.249795 |
| Joint-stage evolved | **0.263541** | **0.435976** | **0.208992** | **0.291873** | **0.451074** | **0.246159** |

Relative to the joint-stage seed, the evolved pair improves Dev
sMAE/sRMSE/sCRPS by 1.98%/3.00%/1.69% (2.62% point-joint) and public-Test
sMAE/sRMSE/sCRPS by 2.18%/2.39%/1.46% (2.31% point-joint). The two arms
differ on 160 of 199 tasks.

## Scope and interpretation

This is a valid same-input, same-metric ablation of the **additional archived
joint program-evolution stage**. It is not a full `without all co-evolution`
counterfactual: the `0000_seed` pair already incorporates programs evolved in
earlier stages. The paper should therefore label the rows `joint-stage seed`
and `joint-stage evolved`, not `without co-evolution` and `full`.

## Reproducibility checks

- The evolved replay exactly reproduces the cached reconstruction-v2
  predictions: maximum absolute difference 0; 0/199 changed tasks.
- The final numerical program is byte-identical to accepted attempt
  `0020_n1b.py`; the final decision program is byte-identical to accepted
  attempt `0013_d2a.py`.
- Result JSON SHA-256:
  `44b3b744198b4ed666f9afede0de4887bfa25396da6b63c9f833d4b63860649a`.
- Runner SHA-256:
  `a4add082df31b196c3bdb40a13decda257c39663d53146dd19a9c58d3da5b2a0`.
