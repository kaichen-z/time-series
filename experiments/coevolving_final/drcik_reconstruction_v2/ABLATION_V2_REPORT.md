# Dr-CiK reconstruction-v2 component ablation

Date: 2026-10-21

All variants reuse the exact reconstruction-v2 views, cached TSFM forecasts,
official task files, and frozen forecast/adjust programs. No model calls were
made. Metrics are task-mean capped normalized MAE/RMSE, matching the existing
Dr-CiK point-metric protocol.

| Variant | Dev sMAE / sRMSE | Public Test sMAE / sRMSE | Tasks changed vs Full |
|---|---:|---:|---:|
| Full v2 | **0.264 / 0.436** | 0.292 / 0.451 | -- |
| without history repair | 0.279 / **0.431** | 0.311 / 0.465 | 10 |
| without model combination | 0.270 / 0.472 | **0.290 / 0.448** | 197 |
| without document corrections | 0.331 / 0.502 | 0.341 / 0.551 | 94 |
| without validator (apply all cards) | 0.420 / 0.601 | 0.341 / 0.522 | 125 |

## Interpretation

- Document corrections are the largest positive component: bypassing the
  adjustment program substantially worsens both Dev and public Test.
- The selective validator is necessary: applying every cached correction with
  the historical +/-50% safety clip performs much worse than Full.
- History repair improves public-Test sMAE and sRMSE, although Dev sRMSE is
  slightly better without it. Its effect is sparse (10 tasks changed).
- Model combination is not a uniformly positive component. It is clearly
  beneficial on Dev, but Toto-only is slightly better on the already-consulted
  public Test. This is a useful negative/mixed ablation result, not evidence to
  claim universal benefit.
- A `without co-evolution` row was not run. There is no unique current-v2
  counterfactual program pair: choosing an earlier forecast and/or adjustment
  checkpoint would add an arbitrary model-selection decision rather than remove
  a well-defined runtime component.

## Reproducibility checks

- Full predictions reproduce the cached v2 predictions exactly (maximum
  absolute difference 0; 0/199 changed tasks).
- The no-history-repair variant reproduces the existing cached control exactly
  (maximum absolute difference 0; 0/199 changed tasks).
- Result JSON SHA-256:
  `ed661b8dec264936ea49018d9615fbc4ca79db37f912e553db0ed9b341a3e77a`.
- Runner SHA-256:
  `0eb1268b3bf4e64d0c5f8729782c13cbfb3e039dc07e3e0056e239651f7dbbef`.

## Disclosure

This is an exploratory reconstruction on a public Test set already consulted by
the historical workflow. It is suitable for mechanism analysis with explicit
disclosure, not as sealed confirmation evidence.
