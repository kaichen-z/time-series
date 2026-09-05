# Accuracy-Stratified 80/20/99 Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build and freeze an entity-disjoint Dr-CiK 80/20/99 v2 split whose public-task difficulty is balanced with a fixed seven-baseline accuracy panel.

**Architecture:** A new accuracy-profile module parses and validates frozen baseline sMAE evidence, converts each model's scores to deterministic percentile buckets, and emits a compact provenance-bound artifact. The existing split module gains an opt-in v2 path that balances metadata and accuracy buckets at entity level while leaving its v1 API and output unchanged.

**Tech Stack:** Python 3.12 standard library, pytest, JSON, SHA-256

**Spec:** `docs/superpowers/specs/2026-09-03-morphology-balanced-80-20-99-split-design.md`

## Global Constraints

- V2 contains exactly 80 Train, 20 Dev, and 99 Public Test task IDs.
- No entity may cross partitions.
- Accuracy provenance is fixed to commit `1d0d9690e6d81fd00d344700216cfd40f35638f5` and panel `arima`, `ets`, `ses`, `chronos`, `aurora`, `moirai`, `seasonal_naive`.
- V2 declares `selection_uses_future_values: true` and `selection_uses_model_metrics: true`.
- V1 remains byte-for-byte unchanged and remains the default when no accuracy profile is supplied.
- The maximum relative gap among mean raw-median capped-sMAE difficulty scores must be at most 5%.
- Every partition contains at least one entity for every two tasks.
- Official Hidden-80 tasks and labels are never used.

---

### Task 1: Provenance-Bound Accuracy Profile

**Files:**
- Create: `evolving_loop/accuracy_profile.py`
- Create: `tests/test_drcik_accuracy_profile.py`

**Interfaces:**
- Consumes: baseline log paths and the exact set of public task IDs.
- Produces: `parse_baseline_log(text: str) -> dict[str, float]`, `build_accuracy_profile(scores_by_model: Mapping[str, Mapping[str, float]], *, source_commit: str, source_files: Mapping[str, Mapping[str, str]]) -> dict`, `validate_accuracy_profile(profile: Mapping[str, object], expected_task_ids: Collection[str] | None = None) -> dict`, and `task_difficulty_features(profile: Mapping[str, object]) -> dict[str, dict[str, object]]`.

- [x] **Step 1: Write profile parsing and validation tests**

  Add literal fixtures proving that duplicate task rows, incomplete panel coverage, non-finite/out-of-range scores, digest changes, and model/task reordering are handled correctly. Assert the hand-derived four-task percentile, panel median, decile, and per-model quintile values.

- [x] **Step 2: Run tests and verify RED**

  Run: `pytest -q tests/test_drcik_accuracy_profile.py`

  Expected: collection fails because `evolving_loop.accuracy_profile` does not exist.

- [x] **Step 3: Implement the minimal profile module**

  Use strict regex parsing for lines containing `task_<n> ... sMAE=<finite>`, reject duplicate task IDs, require the exact panel, canonicalize JSON with sorted keys and compact separators, and calculate the artifact SHA-256 without its digest field. Use the median raw panel sMAE as primary task difficulty. Rank each model by `(score, task_id)`, use `rank / (task_count - 1)`, and derive aggregate deciles and per-model quintiles with bounded integer indices.

- [x] **Step 4: Run profile tests and verify GREEN**

  Run: `pytest -q tests/test_drcik_accuracy_profile.py`

  Expected: all profile tests pass.

- [x] **Step 5: Commit the profile module**

  ```bash
  git add evolving_loop/accuracy_profile.py tests/test_drcik_accuracy_profile.py
  git commit -m "feat(split): add baseline accuracy profile"
  ```

### Task 2: Accuracy-Aware Entity Assignment

**Files:**
- Modify: `evolving_loop/split_manifest.py`
- Modify: `tests/test_drcik_split_manifest.py`

**Interfaces:**
- Consumes: `task_difficulty_features(profile)` keyed by every source task ID.
- Produces: `build_accuracy_stratified_split_manifest(records: Sequence[dict], accuracy_profile: Mapping[str, object], *, seed: int, train_size: int, dev_size: int, public_test_size: int, trials: int = 32768) -> dict`.
- Preserves: `build_split_manifest(...) -> dict` and its schema-v1 output.

- [x] **Step 1: Write failing assignment tests**

  Extend the synthetic fixture to at least twelve entity groups. Prove exact sizes, entity disjointness, input-order determinism, v2 selection declarations, profile binding, objective fields, and unchanged v1 output. Add a controlled hard/easy fixture where accuracy stratification produces a smaller mean-difficulty gap than the v1 assignment.

- [x] **Step 2: Run tests and verify RED**

  Run: `pytest -q tests/test_drcik_split_manifest.py`

  Expected: failure because `build_accuracy_stratified_split_manifest` and the v2 fields do not exist.

- [x] **Step 3: Implement deterministic v2 assignment**

  Reuse entity extraction, exact subset-sum, and stable hashing. Add generic distribution scoring for `frequency`, `horizon_bin`, `difficulty_decile`, and `<model>_quintile`. Reject assignments with fewer than one entity per two tasks, and fail generation if no candidate passes the 5% mean-difficulty gate. Compare passing candidates by relative mean-difficulty gap, maximum normalized bin deviation, total normalized deviation, and membership signature. Serialize per-partition difficulty summaries, using the shared linearly interpolated quantile implementation for P90, and objective evidence into schema v2.

- [x] **Step 4: Add opt-in CLI arguments**

  Add `--accuracy-profile` and `--trials`. With no profile, retain the current v1 call and bytes. With a profile, validate exact task coverage and call the v2 builder.

- [x] **Step 5: Run split tests and verify GREEN**

  Run: `pytest -q tests/test_drcik_accuracy_profile.py tests/test_drcik_split_manifest.py`

  Expected: all focused tests pass.

- [x] **Step 6: Commit the assignment implementation**

  ```bash
  git add evolving_loop/split_manifest.py tests/test_drcik_split_manifest.py
  git commit -m "feat(split): balance public tasks by accuracy"
  ```

### Task 3: Freeze Real Profile, Manifest, and Audit

**Files:**
- Create: `splits/drcik_public_baseline_accuracy_v1.json`
- Create: `splits/drcik_public_80_20_99_v2.json`
- Modify: `splits/README.md`
- Create: `docs/results/DRCIK_80_20_99_V2_BALANCE_REPORT.md`

**Interfaces:**
- Consumes: seven `runs/baselines/*_dev.log` files from the frozen baseline commit and 199 local public task JSON files.
- Produces: an offline accuracy evidence artifact and a frozen v2 manifest with canonical digests.

- [x] **Step 1: Merge the frozen baseline evidence commit**

  Run: `git merge --no-edit origin/main`

  Expected: the clean feature branch gains commit `1d0d969...` baseline artifacts without changing the user's dirty main worktree.

- [x] **Step 2: Build the accuracy profile**

  Run:

  ```bash
  python -m evolving_loop.accuracy_profile \
    --baselines-dir runs/baselines \
    --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
    --source-commit 1d0d9690e6d81fd00d344700216cfd40f35638f5 \
    --output splits/drcik_public_baseline_accuracy_v1.json
  ```

  Expected: 199 tasks, seven panel members, verified per-source and artifact digests.

- [x] **Step 3: Generate the frozen v2 manifest**

  Run:

  ```bash
  python -m evolving_loop.split_manifest \
    --tasks-path external/Dr-CiK/full-download/Dr-CiK_public/tasks \
    --accuracy-profile splits/drcik_public_baseline_accuracy_v1.json \
    --output splits/drcik_public_80_20_99_v2.json \
    --seed 20260905 \
    --trials 32768
  ```

  Expected: exact `80/20/99`, entity disjointness, and maximum mean difficulty gap no greater than 5%.

- [x] **Step 4: Write the v1-versus-v2 audit**

  Document task/entity counts, all seven models' mean capped sMAE by split, aggregate mean/median/P90/max, v1 and v2 gaps, provenance, leakage boundary, and the command lines above. Update `splits/README.md` without changing the recommended defaults for old experiments.

- [x] **Step 5: Verify artifacts and regressions**

  Run:

  ```bash
  pytest -q tests/test_drcik_accuracy_profile.py tests/test_drcik_split_manifest.py
  python -m json.tool splits/drcik_public_baseline_accuracy_v1.json >/dev/null
  python -m json.tool splits/drcik_public_80_20_99_v2.json >/dev/null
  git diff --exit-code f3f419a32270d8dfc6e2c0208d5dd8cab38cf024 -- splits/drcik_public_80_20_99_v1.json
  ```

  Expected: tests and JSON checks pass; v1 diff is empty.

- [x] **Step 6: Commit generated artifacts and documentation**

  ```bash
  git add splits/drcik_public_baseline_accuracy_v1.json splits/drcik_public_80_20_99_v2.json splits/README.md docs/results/DRCIK_80_20_99_V2_BALANCE_REPORT.md docs/superpowers/specs/2026-09-03-morphology-balanced-80-20-99-split-design.md docs/superpowers/plans/2026-09-05-accuracy-stratified-80-20-99-split.md
  git commit -m "docs(split): freeze accuracy-balanced v2"
  ```
