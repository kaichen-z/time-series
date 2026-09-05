# Dr-CiK 80/20/99 v2 Accuracy-Balance Audit

Date: 2026-09-05

## Result

The v2 assignment reduces the relative gap in mean task difficulty from 12.05% to 0.23% while
retaining exact 80/20/99 sizes and entity disjointness. Difficulty is the per-task median capped
sMAE across seven frozen no-context baselines.

| Split | v1 tasks/entities | v1 mean/median/P90 | v2 tasks/entities | v2 mean/median/P90 |
|---|---:|---:|---:|---:|
| Train | 80 / 47 | 0.5287 / 0.3505 / 0.9570 | 80 / 50 | 0.5592 / 0.3580 / 0.9570 |
| Dev | 20 / 10 | 0.5248 / 0.3320 / 0.5190 | 20 / 10 | 0.5597 / 0.3060 / 0.5280 |
| Test | 99 / 56 | 0.5922 / 0.3360 / 1.5410 | 99 / 53 | 0.5605 / 0.3360 / 1.3030 |

V2 substantially removes the mean Train/Test difficulty shift. Test P90 also falls from 1.541 to
1.303, although its upper tail remains harder than Train and Dev; that residual is reported rather
than triggering repeated reshuffling.

## Per-baseline capped sMAE

| Baseline | v1 Train | v1 Dev | v1 Test | v2 Train | v2 Dev | v2 Test |
|---|---:|---:|---:|---:|---:|---:|
| ARIMA | 0.6182 | 0.5656 | 0.7132 | 0.6644 | 0.6731 | 0.6542 |
| ETS | 0.7942 | 0.4941 | 0.6246 | 0.7699 | 0.5714 | 0.6286 |
| SES | 0.6392 | 0.4454 | 0.6305 | 0.6597 | 0.4425 | 0.6145 |
| Chronos | 0.4797 | 0.5217 | 0.5216 | 0.5263 | 0.5472 | 0.4788 |
| Aurora | 0.5530 | 0.5907 | 0.6863 | 0.5995 | 0.6029 | 0.6463 |
| Moirai | 0.4014 | 0.4241 | 0.5146 | 0.4843 | 0.4412 | 0.4442 |
| Seasonal Naive | 0.4507 | 0.3720 | 0.5784 | 0.4884 | 0.4143 | 0.5394 |

No single method defines the split. The primary task score is the median of the seven values;
model-specific percentile quintiles are secondary distribution strata.

## Provenance and interpretation

- Baseline source commit: `1d0d9690e6d81fd00d344700216cfd40f35638f5`
- Accuracy profile: `splits/drcik_public_baseline_accuracy_v1.json`
- V2 manifest: `splits/drcik_public_80_20_99_v2.json`
- V2 manifest SHA-256: `16aff5b76f04faf034c89b668a772880c472b5e71635d91690f95e585a140207`
- Assignment seed/trials: `20260905` / `32768`
- Entity minimum: one entity per two tasks in every partition

Because v2 uses model errors derived from public future labels, this is an internal balanced
protocol. It is unsuitable as evidence of untouched generalization. The 80 human-authored hidden
tasks scored by the Dr-CiK maintainers remain the final test.
