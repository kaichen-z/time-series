# Task-local Numerical Shortlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-alternative/group shortlist path with a complete global Numerical Dictionary, a history-only 6--10 candidate shortlist for each task, local hindcast over only that shortlist, and an Anchor-heavy final blend with at most two specialists.

**Architecture:** Keep the global Dictionary, task shortlist, hindcast diagnostics, and final blend as four separate authorities. A versioned global supply exposes every selectable candidate; a new shortlist module deterministically selects eight candidates per task when available (hard bounds 6--10, including Anchor) using history-only applicability and Train-fitted priors. Existing task-local tournament logic remains the final safety gate, and P2 freezes every task's shortlist alongside its diagnostics so P3 only consumes closed Numerical artifacts.

**Tech Stack:** Python dataclasses, canonical JSON/SHA-256 contracts, existing `ScreeningPolicy`, `TaskProfile`, `ForecastStore`, task-local OOF fitting, Evolution V2 Numerical QD, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-task-local-numerical-shortlist-design.md`

## Global Constraints

- The complete global Dictionary has no candidate-count limit.
- Each task shortlist contains 6--10 candidates including the protected Anchor; target size is exactly 8 when at least 8 safe executable candidates exist.
- Fewer than 6 candidates is permitted only when fewer than 6 safe executable candidates exist and must set `shortlist_underfilled=true`.
- Shortlist selection uses history, Dictionary status/applicability, and Train-fitted priors only; future values, Dev labels, and Public evidence are forbidden.
- Local hindcast runs only for shortlisted candidates.
- Final output is Anchor alone or Anchor plus at most two specialists; Anchor weight is at least 0.5.
- Missing, invalid, unstable, or insufficient evidence falls back byte-for-byte to the Anchor forecast.
- Existing schema-v1 Numerical supply and schema-v1/v2 task-local releases remain explicitly readable; new artifacts never silently downgrade to legacy group selection.
- P3 consumes the frozen P2 registry and does not rerun Numerical selection.
- Do not modify or stage the interrupted P3 files `evolving_loop/v2/real/bridges.py` and `tests/test_evolution_v2_real_cooperative.py` during this plan.

---

### Task 1: Canonical per-task shortlist contracts

**Files:**
- Create: `numerical_agent/evolution/task_shortlist.py`
- Modify: `numerical_agent/evolution/__init__.py`
- Test: `tests/test_task_shortlist.py`

**Interfaces:**
- Consumes: `FilterDictionary`, `ScreeningPolicy`, `TaskProfile`, and canonical payload helpers.
- Produces:

```python
@dataclass(frozen=True)
class CandidatePriorV1:
    candidate_name: str
    family: str
    success_rate: float
    mean_joint: float
    p90_joint: float
    morphology_scores: tuple[tuple[str, float], ...]

@dataclass(frozen=True)
class TaskShortlistPolicyV1:
    schema_version: int = 1
    minimum_candidates: int = 6
    target_candidates: int = 8
    maximum_candidates: int = 8

@dataclass(frozen=True)
class TaskCandidateShortlistV1:
    schema_version: int
    task_input_sha256: str
    dictionary_sha256: str
    policy_sha256: str
    candidate_names: tuple[str, ...]
    exclusion_reasons: tuple[tuple[str, str], ...]
    shortlist_underfilled: bool
    public_test_accessed: bool

def build_task_candidate_shortlist(
    *, dictionary: FilterDictionary, profile: TaskProfile,
    task_input_sha256: str, anchor_name: str,
    available_names: Sequence[str], priors: Sequence[CandidatePriorV1],
    policy: TaskShortlistPolicyV1,
) -> TaskCandidateShortlistV1: ...
```

- [ ] **Step 1: Write contract RED tests.** Assert exact schema parsing, canonical bytes/fingerprints, unique lowercase-normalized candidate IDs, ordered exclusion reasons, finite prior scores, exact `6/8/10` policy constants, Anchor-first ordering, and an always-false Public field.

```python
def test_shortlist_contract_is_canonical_and_anchor_first():
    value = TaskCandidateShortlistV1(
        1, "1" * 64, "2" * 64, "3" * 64,
        ("anchor", "a", "b", "c", "d", "e"), (), False, False,
    )
    assert value.candidate_names[0] == "anchor"
    assert TaskCandidateShortlistV1.from_payload(value.to_payload()) == value
    assert value.canonical_bytes() == canonical_json_bytes(value.to_payload())
```

- [ ] **Step 2: Run RED.** Run `python -m pytest -q tests/test_task_shortlist.py`; expect import failure because `task_shortlist.py` does not exist.

- [ ] **Step 3: Implement strict immutable contracts.** Reject NaN/inf, unknown families/statuses, duplicate names after NFKC/casefold, wrong SHA values, noncanonical ordering, `public_test_accessed=True`, or policy bounds other than `6/8/10`.

- [ ] **Step 4: Implement deterministic task selection.** Materialize history-only applicability through the existing screening API; force the Anchor first; sort eligible candidates by morphology-specific prior, success rate descending, mean/p90 joint ascending, family-diversity preference, and canonical name. Select exactly 8 if possible, otherwise all safe candidates up to 8, and record every excluded candidate with one stable reason: `unsafe_status`, `not_applicable`, `unavailable`, or `ranked_out`.

```python
rank_key = (
    morphology_score,
    -prior.success_rate,
    prior.mean_joint,
    prior.p90_joint,
    canonical_name,
)
```

- [ ] **Step 5: Add behavior tests.** Prove different `TaskProfile` histories may produce different shortlists, future values are absent from the API, family diversity wins only as a deterministic tie-break, unsafe entries never fill an underfull list, and fewer than six eligible candidates sets `shortlist_underfilled`.

- [ ] **Step 6: Run GREEN.** Run `python -m pytest -q tests/test_task_shortlist.py`; require all tests pass, then run `git diff --check`.

- [ ] **Step 7: Commit.** Stage only Task 1 files and commit `feat(numerical): add task shortlist contracts`.

---

### Task 2: Version the complete global Numerical supply

**Files:**
- Modify: `evolving_loop/package_numerical_supply.py`
- Modify: `evolving_loop/package_numerical_evolution.py`
- Modify: `evolving_loop/run_package_coevolution.py`
- Test: `tests/test_package_numerical_supply.py`
- Test: `tests/test_package_numerical_evolution.py`

**Interfaces:**
- Consumes: existing `NumericalAlternativeSpec` and schema-v1 `NumericalSupplyRelease` payloads.
- Produces: schema-v2 `NumericalSupplyRelease` semantics in which `alternatives` is the complete ordered selectable catalog excluding the Anchor; repeated families and more than four alternatives are valid, candidate IDs remain unique, and all source/runtime/cross-fit commitments remain mandatory.

- [ ] **Step 1: Write migration RED tests.** Construct a schema-v2 release with 12 alternatives across repeated families, round-trip it canonically, and require all 12 identities to survive. Retain tests proving schema-v1 rejects a fifth alternative and repeated family.

```python
release = NumericalSupplyRelease(
    schema_version=2,
    version="n001",
    parent_sha256=parent.fingerprint,
    anchor_release_payload=anchor.to_payload(),
    alternatives=twelve_cross_fitted_specs,
    atlas_release_sha256=None,
    source_fingerprints=source_hashes,
    runtime_fingerprints=runtime_hashes,
)
assert len(parse_numerical_supply_release(release.to_payload()).alternatives) == 12
```

- [ ] **Step 2: Run RED.** Run the two exact migration tests; expect rejection from the four-alternative and one-per-family guards.

- [ ] **Step 3: Implement explicit version semantics.** Admit only schema versions 1 and 2. Preserve every schema-v1 check unchanged. For schema v2, remove cardinality and unique-family restrictions while retaining unique candidate IDs, supported families/materializers, fitted policy alignment, five cross-fitted Build policies for non-seed releases, and canonical input order.

- [ ] **Step 4: Stop collapsing candidates by family.** Replace `by_family = {item.family: item ...}` in package binding/evolution with ordered identity-based lookup. Any family diversity choice belongs to `build_task_candidate_shortlist`, not the release schema. Do not truncate the v2 release when generating children.

- [ ] **Step 5: Update producers deliberately.** Make new production evolution output schema v2. Leave smoke/test fixtures on schema v1 unless a test explicitly exercises v2. Assert parent/child versioning never changes the candidate catalog implicitly.

- [ ] **Step 6: Run GREEN.** Run:

```bash
python -m pytest -q \
  tests/test_package_numerical_supply.py \
  tests/test_package_numerical_evolution.py \
  tests/test_run_package_coevolution.py
```

- [ ] **Step 7: Commit.** Stage only Task 2 files and commit `feat(numerical): expose complete supply catalog`.

---

### Task 3: Fit Train-only candidate priors

**Files:**
- Modify: `numerical_agent/evolution/task_shortlist.py`
- Modify: `numerical_agent/evolution/task_local_evolution.py`
- Test: `tests/test_task_shortlist.py`
- Test: `tests/test_task_local_evolution.py`

**Interfaces:**
- Consumes: exact Train task IDs, `TaskLocalTaskRow` forecast outcomes, morphology keys, and `GroupFoldManifest`.
- Produces:

```python
def fit_candidate_priors(
    rows: Sequence[TaskLocalTaskRow], *, task_ids: Sequence[str],
    morphology_keys: Mapping[str, str], candidate_names: Sequence[str],
) -> tuple[CandidatePriorV1, ...]: ...

def fit_oof_shortlist_priors(
    rows: Sequence[TaskLocalTaskRow], manifest: GroupFoldManifest,
    *, candidate_names: Sequence[str],
) -> tuple[Mapping[int, tuple[CandidatePriorV1, ...]], tuple[CandidatePriorV1, ...]]: ...
```

- [ ] **Step 1: Write RED tests for prior fitting.** Verify deterministic ordering, success/failed forecast accounting, mean and p90 joint errors, morphology-specific scores, no diagnostic/hindcast dependency, and fitting from only the supplied task IDs.

- [ ] **Step 2: Add an OOF leakage test.** Mutate held-out fold truths and prove priors for that fold are unchanged because they are fit exclusively on complementary fold IDs. Mutate a complementary Train truth and prove the corresponding prior fingerprint changes.

- [ ] **Step 3: Run RED.** Run the new prior tests; expect missing functions.

- [ ] **Step 4: Implement prior fitting.** Use full-horizon Train forecast outcomes already owned by the Host; assign unsuccessful/missing rows worst finite ranks, never NaN/inf. Compute fold-specific priors from complement IDs and final runtime priors from all Train IDs. Do not call `diagnose_candidate` here.

- [ ] **Step 5: Replace group supply fitting authority.** In new task-local release construction, store the shortlist policy and fitted prior payload/fingerprint. Keep `fit_group_candidate_supply` only for reading/evaluating legacy schema-v1/v2 releases; new production paths must not call it.

- [ ] **Step 6: Run GREEN.** Run `python -m pytest -q tests/test_task_shortlist.py tests/test_task_local_evolution.py` and require all tests pass.

- [ ] **Step 7: Commit.** Stage only Task 3 files and commit `feat(numerical): fit shortlist priors`.

---

### Task 4: Execute per-task shortlist before local hindcast

**Files:**
- Modify: `numerical_agent/evolution/task_local_ensemble.py`
- Modify: `numerical_agent/evolution/task_local_evolution.py`
- Modify: `numerical_agent/run_task_local_ensemble_evolution.py`
- Modify: `numerical_agent/evaluate_frozen_task_local_ensemble.py`
- Test: `tests/test_task_local_ensemble.py`
- Test: `tests/test_task_local_evolution.py`
- Test: `tests/test_task_local_ensemble_cli.py`

**Interfaces:**
- Consumes: schema-v3 task-local release containing `TaskShortlistPolicyV1`, complete candidate priors, and Dictionary fingerprint.
- Produces:

```python
@dataclass(frozen=True)
class TaskLocalEnsembleReleaseV3:
    schema_version: Literal[3]
    anchor_release_sha256: str
    anchor_name: str
    tournament_policy: TaskLocalTournamentPolicy
    shortlist_policy: TaskShortlistPolicyV1
    candidate_priors: tuple[CandidatePriorV1, ...]
    dictionary_sha256: str
    grouping_fingerprint: str
    oof_report_sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    metric_policy_fingerprint: str
    lineage: tuple[str, ...]
    confidence_evidence: HierarchicalEvidenceBank | None

def materialize_task_shortlist_rows(
    store: ForecastStore, task: DataTask, shortlist: TaskCandidateShortlistV1,
    families: Mapping[str, str], *, split: str,
    hindcast_config: HindcastConfig,
) -> tuple[TaskLocalTaskRow, ...]: ...
```

- [ ] **Step 1: Write schema-v3 RED tests.** Round-trip the new release and prove legacy parser dispatch returns old schema-v1/v2 objects only for old payloads. Reject missing Dictionary/prior/policy fingerprints and reject a v3 payload containing legacy `group_supplies` or `default_candidate_names`.

- [ ] **Step 2: Write execution-order RED test.** Use a spy `ForecastStore`; build a Dictionary with 20 executable methods and a task shortlist of 8; assert full forecasts and `diagnose_candidate` are called only for those 8 names.

- [ ] **Step 3: Run RED.** Run the exact new tests and require missing v3/materializer failures.

- [ ] **Step 4: Implement versioned release parsing.** Add a top-level parser that dispatches exact schema-v1/v2 fields to `TaskLocalEnsembleRelease` and exact schema-v3 fields to `TaskLocalEnsembleReleaseV3`. Consumers must branch explicitly; no optional-field guessing.

- [ ] **Step 5: Refactor materialization into two phases.** For every task: profile history, build and persist its shortlist, then forecast/diagnose only ordered shortlist names. Preserve per-candidate failure rows and distinguish `not_shortlisted` from `shortlisted_runtime_failure` without manufacturing diagnostics.

- [ ] **Step 6: Reuse the existing final tournament unchanged.** Pass the shortlist names/forecasts/diagnostics to `execute_confidence_task_local_ensemble` or `execute_task_local_ensemble`. Assert selected names contain Anchor plus no more than two specialists and weights retain Anchor at `>=0.5`; return the exact Anchor forecast on fallback.

- [ ] **Step 7: Update formal CLI artifacts.** Write canonical `task_shortlists/<task-input-sha>.json` files plus a shortlist index whose entries bind task ID, task-input fingerprint, shortlist fingerprint, and diagnostics fingerprint. The run manifest binds Dictionary, shortlist policy, candidate priors, and index fingerprints.

- [ ] **Step 8: Run GREEN.** Run:

```bash
python -m pytest -q \
  tests/test_task_shortlist.py \
  tests/test_task_local_ensemble.py \
  tests/test_task_local_evolution.py \
  tests/test_task_local_ensemble_cli.py
```

- [ ] **Step 9: Commit.** Stage only Task 4 files and commit `feat(numerical): run per-task local shortlist`.

---

### Task 5: Freeze shortlist evidence into P2 and P3 handoff

**Files:**
- Modify: `evolving_loop/package_numerical_supply.py`
- Modify: `evolving_loop/v2/numerical_qd/adapters.py`
- Modify: `evolving_loop/v2/numerical_qd/artifacts.py`
- Modify: `evolving_loop/v2/numerical_qd/persistence.py`
- Modify: `evolving_loop/v2/kernel.py`
- Test: `tests/test_package_numerical_supply.py`
- Test: `tests/test_evolution_v2_numerical_adapters.py`
- Test: `tests/test_evolution_v2_numerical_runner.py`
- Test: `tests/test_evolution_v2_real_numerical.py`

**Interfaces:**
- Consumes: schema-v2 complete supply, per-task shortlist artifacts, and local hindcast diagnostics.
- Produces: every `NumericalForecastPackage` and frozen registry entry binds `task_shortlist`, `shortlist_policy`, `dictionary`, and `hindcast_diagnostics` fingerprints; `NumericalQDRunStore.load_active_frozen_pair()` verifies and restores them before P3 handoff.

- [ ] **Step 1: Write P2 closure RED tests.** Build a 12-candidate v2 supply over two tasks with different histories. Assert different task shortlist fingerprints, 6--10 candidates including Anchor, only shortlisted diagnostics, and final Anchor-plus-at-most-two selection.

- [ ] **Step 2: Write mutation rejection tests.** Change shortlist order, candidate identity, Dictionary fingerprint, diagnostics fingerprint, or final selected weight after sealing. Require the frozen-pair loader or Kernel handoff validation to reject each mutation.

- [ ] **Step 3: Run RED.** Run the new adapter/persistence tests; expect missing artifact kind/closure commitments.

- [ ] **Step 4: Add canonical artifact kinds.** Add `TASK_SHORTLIST`, `SHORTLIST_POLICY`, and `SHORTLIST_INDEX` to the Numerical artifact catalog with exact payload fields. Include their SHAs in the frozen registry envelope and evaluation cache key.

- [ ] **Step 5: Bind packages to local evidence.** `bound_numerical_package` must accept the already verified task shortlist instead of retaining one candidate per family from `release.alternatives`. Preserve complete release identity, but expose only shortlisted candidates in `active_candidate_names` and only the final selected Anchor/specialists in `selection_decision`.

- [ ] **Step 6: Harden frozen-pair restore.** Recompute each task-input fingerprint, reparse each shortlist, confirm its Dictionary/policy identities, verify diagnostics are a subset of shortlist names, and confirm final selected names/weights satisfy Anchor constraints. Keep P3 cache-only: restoration performs no forecasts or hindcasts.

- [ ] **Step 7: Run GREEN.** Run:

```bash
python -m pytest -q \
  tests/test_package_numerical_supply.py \
  tests/test_evolution_v2_numerical_adapters.py \
  tests/test_evolution_v2_numerical_runner.py \
  tests/test_evolution_v2_real_numerical.py
```

- [ ] **Step 8: Commit.** Stage only Task 5 files and commit `feat(numerical): seal task shortlist evidence`.

---

### Task 6: Regression, migration, and operational documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-13-task-local-numerical-shortlist-design.md` only if implementation reveals a required explicit clarification
- Test: existing numerical, package, cooperative, and real bridge suites

**Interfaces:**
- Consumes: Tasks 1--5 completed commits.
- Produces: documented canonical pipeline and a passing focused regression gate.

- [ ] **Step 1: Add one end-to-end regression.** Exercise a complete Dictionary with at least 12 alternatives, two distinct task histories, two task-local shortlists, shortlist-only hindcast, final Anchor-heavy selection, P2 frozen-pair reload, and P3 read-only consumption. Assert `public_test_accessed is False` at every artifact boundary.

- [ ] **Step 2: Add explicit legacy migration coverage.** Load representative schema-v1 Numerical supply and schema-v1/v2 task-local releases; prove byte-stable read/replay. Prove new production output uses supply v2 and task-local v3.

- [ ] **Step 3: Document the runtime flow.** Add this exact conceptual sequence to `README.md` and list shortlist/fallback artifact locations:

```text
complete global Dictionary
  -> task history-only shortlist (Anchor + 5--9, target total 8)
  -> shortlist-only local hindcast
  -> Anchor + zero/one/two specialists
```

- [ ] **Step 4: Run the focused gate.** Run all tests named in Tasks 1--5 plus cooperative bridge tests. Require zero failures.

- [ ] **Step 5: Run the existing Numerical/package regression gate.** Run:

```bash
python -m pytest -q \
  tests/test_numerical_champion_loop.py \
  tests/test_package_numerical_supply.py \
  tests/test_package_numerical_evolution.py \
  tests/test_run_package_coevolution.py \
  tests/test_evaluate_frozen_package_bundle.py \
  tests/test_evolution_v2_numerical_adapters.py \
  tests/test_evolution_v2_numerical_cli.py \
  tests/test_evolution_v2_numerical_runner.py
```

- [ ] **Step 6: Verify repository hygiene.** Run `git diff --check`; verify no Public task IDs/bodies, generated caches, credentials, or interrupted P3 files are staged.

- [ ] **Step 7: Commit.** Stage only Task 6 documentation/tests and commit `docs(numerical): document local shortlist flow`.
