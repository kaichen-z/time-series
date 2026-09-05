# Accuracy-Stratified Dr-CiK 80/20/99 Split v2

Date: 2026-09-05
Status: approved in chat
Scope: the 199 labeled synthetic Dr-CiK `public_dev` tasks

## 1. Problem

The frozen v1 manifest is deterministic and entity-disjoint, but it balances only frequency,
horizon bucket, reasoning hops, and origin. The last two fields are constant across the public 199,
so v1 does not directly balance empirical forecast difficulty. A fixed seven-model baseline panel
shows mean task-level median capped sMAE of 0.5287 / 0.5247 / 0.5922 on Train-80 / Dev-20 /
Public-99. Public-99 is about 12% harder than Train-80 on this proxy and has a materially heavier
tail.

## 2. Goal

Create `splits/drcik_public_80_20_99_v2.json` with exactly 80 Train, 20 Dev, and 99 internal Test
tasks while:

- keeping every entity wholly inside one partition;
- balancing frequency, horizon bucket, and empirical difficulty;
- deriving difficulty from a frozen, heterogeneous baseline panel rather than the current Toto or
  any evolved Champion;
- making all label use and source provenance explicit and reproducible;
- preserving `splits/drcik_public_80_20_99_v1.json` byte-for-byte; and
- leaving the official Hidden-80 as the only final, locally unscored test.

## 3. Scientific Boundary

V2 is a balanced internal development/test protocol, not fresh generalization evidence. Its
difficulty signal is computed from public future labels through published baseline forecasts.
Therefore the manifest must declare `selection_uses_future_values: true` and
`selection_uses_model_metrics: true`. It must never be described as an untouched test.

The fixed panel is bound to GitHub commit
`1d0d9690e6d81fd00d344700216cfd40f35638f5` and contains seven complete no-context methods:

- `arima`
- `ets`
- `ses`
- `chronos`
- `aurora`
- `moirai`
- `seasonal_naive`

Toto and all locally evolved policies are excluded so the split cannot be optimized for the system
being developed. The old Public-99 remains available only for historical v1 comparisons.

## 4. Accuracy Profile

Commit `splits/drcik_public_baseline_accuracy_v1.json` as the small, offline input artifact. It
contains, for every public task, each panel member's capped sMAE parsed from the published baseline
logs, plus source commit, source paths, per-source SHA-256 digests, panel membership, schema version,
and a canonical artifact digest.

For each model independently, sort all 199 tasks by `(capped_smae, task_id)` and convert rank to a
percentile in `[0, 1]`. The task's primary difficulty score is the median raw capped sMAE across the
seven models; its stratification score is the median of the seven percentiles and is bucketed into
ten deterministic deciles. The robust raw median retains tail magnitude, while percentile buckets
prevent a weak or high-variance baseline from dominating the categorical distribution objective.

Input validation fails closed on missing/extra task IDs, missing/extra panel members, booleans,
non-finite values, or scores outside `[0, 5]`.

## 5. Assignment

Candidate assignments use a fixed seed, stable SHA-256 ordering, exact subset-sum feasibility, and
entity-level grouping. They retain the v1 contract of exact task counts and deterministic
task/entity membership.

The deterministic objective balances:

1. frequency and horizon-bin distributions;
2. aggregate panel-difficulty deciles;
3. each baseline's own difficulty quintiles; and
4. the mean aggregate difficulty percentile in each partition.

Every partition must contain at least one entity for every two tasks. Candidates are compared
lexicographically by passing the 5% relative mean-difficulty gate, lowest relative mean-difficulty
gap, maximum normalized bin deviation, total normalized distribution deviation, and stable
membership signature. All objective components are stored in the manifest. The number of
deterministic trials is a versioned constant.

## 6. Manifest Contract

V2 uses schema version 2 and preserves the consumer-facing
`partitions.<train|dev|public_test>.task_ids` structure. It additionally records:

- `difficulty_profile_schema` and profile SHA-256;
- source baseline commit and panel names;
- `stratification_features`, including aggregate decile and per-model quintiles;
- explicit selection-use booleans;
- per-partition metadata and difficulty distributions;
- objective components and trial count; and
- a canonical manifest SHA-256 covering every field except the digest itself.

Existing runners do not switch defaults in this change. V2 is selected explicitly until later
experiments deliberately migrate.

## 7. Tests and Acceptance

Tests are written before production changes and must prove:

- exact 80/20/99 sizes, total coverage, unique IDs, and entity disjointness;
- deterministic output under input and profile reordering;
- strict accuracy-profile validation and canonical digest verification;
- task percentile and bucket calculations with deterministic tie-breaking;
- explicit label/model-metric declarations in v2;
- unchanged v1 generation and committed v1 digest;
- unchanged task-ID consumer compatibility; and
- lower aggregate difficulty imbalance than v1 on the real 199-task data.

The committed v2 is accepted only if the maximum relative gap between the three partitions' mean
aggregate difficulty scores is at most 5%, compared with roughly 12% for v1. The audit also reports
each baseline's per-partition mean capped sMAE and aggregate median/P90/max. Residual model-specific
differences are reported; the split is generated once and is not repeatedly reshuffled to improve a
particular model.

## 8. Deliverables

- accuracy profile parser/builder and deterministic scoring helpers;
- accuracy-aware split generator and CLI options;
- committed offline accuracy profile;
- committed v2 split manifest;
- v1-versus-v2 balance report;
- focused tests and full split-test verification.
