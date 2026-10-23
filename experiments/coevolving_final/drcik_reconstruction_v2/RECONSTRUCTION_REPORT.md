# Dr-CiK mixed pipeline reconstruction v2

Date: 2026-10-18

## Outcome

The archived mixed pipeline was first replayed without modification.  The
frozen programs reproduce the recorded scores exactly (task-mean normalized
metrics):

| Run | Dev sMAE | Dev sRMSE | Public Test sMAE | Public Test sRMSE |
|---|---:|---:|---:|---:|
| Archived mixed result | 0.268700 | 0.418423 | 0.284285 | 0.429837 |
| No-history-repair control | 0.279144 | 0.430899 | 0.310776 | 0.464947 |
| Reconstruction v2 | 0.263541 | 0.435976 | 0.291873 | 0.451074 |

Reconstruction v2 still improves over the no-repair control on public Test
(-0.018903 sMAE, -0.013873 sRMSE), but it is worse than the archived mixed
result (+0.007587 sMAE, +0.021237 sRMSE).  On Dev, v2 lowers sMAE but raises
sRMSE relative to the archived result.

## What was regenerated

- Re-ran the frozen evolved extraction instruction with `gpt-6-luna` on all
  199 official public tasks (80 Train, 20 Dev, 99 public Test).
- Preserved the archived Haiku cards, raw numerical forecasts, coevolution
  view fields, and frozen forecast/adjust programs.
- Rebuilt the historical `phase_median` history-repair branch using the
  archived interval kinds and the archived 30% history-only gate
  (`val_rep < 0.7 * val_raw`).
- Re-ran Toto on repaired histories and then produced all 99 public-Test
  predictions with the frozen programs.

The new extraction produced 51 histories with an applicable repair candidate;
11 passed the historical gate: 4 Train, 2 Dev, and 5 public Test.  The final
runner reported zero runtime errors.

## Provenance

- Official Dr-CiK repository revision:
  `00fbe820ff7a221e4aca71883219ef27f8223050`
- Official task files: 199/199 raw SHA checks and 199/199 canonical
  label-free content SHA checks passed.
- Canonical aggregate content SHA-256:
  `8efea4bfc2e552248faefd8965ac24ad749f4332a0714051b2620aa7692af58f`
- Frozen instruction SHA-256:
  `d2a917a0eee8cc47dad08f261dc03ab2ce6d83339935995d83039601d762df58`
- New Luna timelines SHA-256:
  `6a21b7dbc0c584ee4a379f690cd7e5f5643fc642abdd24dbfa60bf9ea7ceb76e`
- Frozen forecast program SHA-256:
  `aaeb46fe035c51e8beeaf758530bd8045dbb19a45d6b2f0bf7ca0e82b320d103`
- Frozen adjust program SHA-256:
  `c69e9512f0c20e39660895a33e7f76088fb34229eafff8cc44b77afb0bc8f8e2`
- Reconstruction views SHA-256:
  `35f5b2ee02e2fdb3aa2a70cd9ff2792056799d630229a735a890b5f620e6d942`
- Reconstruction predictions SHA-256:
  `85b3a08d57f9db2c1264f235a4b6d790a0cf9768534500df80785cc5fb9938e7`

The original Luna cache is still absent.  Therefore v2 is a new stochastic
reconstruction, not a byte-identical replay of the old extraction.  The public
Test set was already consulted by the historical workflow, so this remains an
exploratory reconstruction/ablation and must not be presented as a sealed
confirmation result.
