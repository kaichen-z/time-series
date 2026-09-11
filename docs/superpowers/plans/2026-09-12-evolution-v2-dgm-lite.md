# Evolution V2 DGM-Lite Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve a branching archive of executable source selection policies, evaluate them through Project 3, and automatically canary, activate, rollback, and resume.

**Architecture:** Keep a single pure candidate `policy.py` behind an audited JSON subprocess. A fixed Host bridge proposes legal Project 3 Bundles and scores source behavior with two Train folds and sealed meta-validation. A Host-owned source authority persists canary state and exact rollback independently of the existing Bundle pointer.

**Tech Stack:** Python 3.11+, dataclasses, ast, subprocess, canonical V2 JSON/SHA-256, existing budget/store helpers, Project 2 frozen Numerical artifacts, Project 3 proposal/pipeline APIs, pytest. No new dependencies or network clients.

**Spec:** `docs/superpowers/specs/2026-09-12-evolution-v2-dgm-lite-design.md`

## Global Constraints

- This is a research prototype; target implementation plus verification is 120 minutes.
- Editable candidate source consists of exactly `policy.py` exporting `choose_arm(request) -> str`.
- Kernel, evaluator, validation, source archive, publication, rollback, task materialization, and scoring code are never editable candidate files.
- Candidate execution uses a separate Python subprocess with JSON input/output; no candidate import occurs in the Host.
- Isolation is an editable-source and data/API boundary, not an OS security sandbox.
- Train meta-CV uses exactly two folds over four Train tasks; sealed meta-validation uses one disjoint Dev task; Public input is rejected before proposals.
- Source requests and archive sampling receive only Train aggregates and source lineage; no Dev values, labels, task IDs, forecasts, or held-out acceptance credit.
- One epoch proposes at most two source Children; one selected finalist may receive meta-validation and one canary epoch.
- Source safety/integrity and canary failures are terminal; static/unit-passing nonwinning source variants remain Train-ranked stepping stones.
- Promotion and rollback are automatic Host decisions; there is no human approval flag or pause.
- Smoke hard limit is 120 seconds, subprocess timeout is 2 seconds, source limit is 8192 UTF-8 bytes, and request/response limits are 16384 bytes each.
- Persist at closed source-candidate, finalist-validation, and canary boundaries; resume never repeats a completed stage.
- Do not add OS sandboxing, TOCTOU/race defenses, adversarial escape research, per-forecast writes, long formal/80-20 runs, or repeated full regression gates.

## Execution and file ownership

Planning is ready before Project 3 completes; code begins only after its focused
compatibility gate passes. The controller runs five tasks in order with one
implementer per task, gives each task one scoped spec/quality review, and uses
other available agents for independent review/preparation only. Workers do not
spawn subagents. Use evidence from the implementer's actual test run instead of
asking reviewers to repeat it. One final integrated gate is sufficient.

New production files live under `evolving_loop/v2/source/`; no Project 3 file or
existing Kernel behavior needs modification. Source artifacts themselves live
under each run's `sources/objects/`. New tests are
`tests/test_evolution_v2_source_{runtime,archive,meta,authority,runner,cli}.py`.
Task 3 owns `tests/build_evolution_v2_source_fixture.py`; Task 5 adds its CLI.
Only Task 5 touches `README.md`, adding one guide link.

Every contract uses strict known fields, finite JSON, `to_payload`,
`from_payload`, `canonical_bytes`, and `fingerprint`. Detailed payloads are in
the linked spec. Python-injected dependencies are never serialized.

### Task 1: Canonical source contract and isolated execution

**Time:** 20 minutes.

**Files:** Create `evolving_loop/v2/source/{__init__,contracts,runtime}.py` and
`tests/test_evolution_v2_source_runtime.py`.

**Consumes:** `canonical_v2_bytes`, `fingerprint_payload`, `require_sha256`.
**Produces:** `SourceVariantV2`, `SourceRequestV2`,
`audit_source(variant) -> None`, and
`run_policy(variant, request, *, timeout_seconds=2.0) -> str`.
`SourceVariantV2.seed(source, protocol_fingerprint, runtime_fingerprint)` creates
the exact seed payload. `SourceVariantV2.child(parent, source, operator)` copies
commitments and records the actual parent fingerprint.

- [ ] Write the failing behavior tests, including canonical round-trip:

```python
def test_policy_executes_actual_branch_in_child_process():
    text = 'def choose_arm(request):\n    return request["enabled_arms"][1]\n'
    variant = SourceVariantV2.seed(text, "a" * 64, "b" * 64)
    request = SourceRequestV2(1, ("numerical", "decision"), 0, 17,
                             {"numerical": 0.0, "decision": 0.1})
    assert run_policy(variant, request) == "decision"
    assert SourceVariantV2.from_payload(variant.to_payload()) == variant


def test_kernel_path_edit_is_rejected():
    variant = SourceVariantV2.seed('def choose_arm(request):\n    return "decision"\n',
                                   "a" * 64, "b" * 64)
    payload = variant.to_payload()
    payload["files"]["../kernel.py"] = "changed = True\n"
    with pytest.raises(ValueError):
        SourceVariantV2.from_payload(payload)


@pytest.mark.parametrize("text", [
    'import os\ndef choose_arm(request):\n    return "decision"\n',
    'def choose_arm(request):\n    return request.__class__\n',
    'def choose_arm(request):\n    return open("active_source.json")\n',
])
def test_ordinary_authority_access_is_rejected(text):
    with pytest.raises(ValueError):
        audit_source(SourceVariantV2.seed(text, "a" * 64, "b" * 64))
```

- [ ] Run `pytest -q tests/test_evolution_v2_source_runtime.py`; confirm RED on missing API.
- [ ] Implement exact schemas and the AST allowlist from the spec; reject oversized source, unknown request fields, output not in enabled arms, worker exception, and timeout. The fixed worker core is:

```python
scope = {"__builtins__": {}}
exec(compile(audited_source, "policy.py", "exec"), scope)
answer = scope["choose_arm"](request_payload)
# The Host validates the decoded output against enabled_arms.
```

Only the fixed worker uses `exec`; the AST rejects candidate calls. Launch the
worker with `subprocess.run([sys.executable, "-I", "-c", WORKER], input=...,
capture_output=True, timeout=2.0, cwd=tempdir, env=minimal_env)`. Validate the
output byte limit before parsing. The admitted grammar cannot emit arbitrary
stdout. Add one timeout-path test by injecting the subprocess runner result or
exception; no elaborate resource-attack test is required.

- [ ] Run the same test file; confirm GREEN.
- [ ] Commit only Task 1 files: `feat(source): add isolated policy runtime`.

### Task 2: Branching source archive and deterministic proposals

**Time:** 20 minutes.

**Files:** Create `evolving_loop/v2/source/archive.py` and
`tests/test_evolution_v2_source_archive.py`; update `source/__init__.py`.

**Consumes:** Task 1 contracts/audit and V2 canonical write helpers.
**Produces:** `SourceArchiveV2(root)`, `.add(variant) -> str`,
`.close(source_sha256, *, status, train_gain=0.0, task_cost=0) -> None`,
`.sample(draw_counter) -> SourceVariantV2`, `.snapshot_sha256() -> str`, and
`propose_sources(parent, *, draw_counter, limit=2) -> tuple[SourceVariantV2, ...]`.
`status` is `eligible`, `terminal`, or `validated`; sampler never consumes the
held-out status as a reward. Invalid sources are recorded before terminal close.

- [ ] Write failing branch and restart tests:

```python
def test_nonactive_stepping_stone_remains_a_parent(tmp_path):
    seed = SourceVariantV2.seed('def choose_arm(request):\n    return "numerical"\n',
                                "a" * 64, "b" * 64)
    archive = SourceArchiveV2(tmp_path)
    archive.add(seed)
    archive.close(seed.fingerprint(), status="eligible")
    children = propose_sources(seed, draw_counter=0)
    for child in children:
        archive.add(child)
        archive.close(child.fingerprint(), status="eligible", train_gain=0.1)
    seen = {archive.sample(i).fingerprint() for i in range(3)}
    assert seen == {seed.fingerprint(), *(c.fingerprint() for c in children)}
    branch = propose_sources(children[0], draw_counter=1, limit=1)[0]
    archive.add(branch)
    assert branch.parent_source_sha256 == children[0].fingerprint()
    restored = SourceArchiveV2(tmp_path)
    assert restored.snapshot_sha256() == archive.snapshot_sha256()
```

Also assert a terminal source is never sampled and Dev-key metadata is rejected.

- [ ] Run `pytest -q tests/test_evolution_v2_source_archive.py`; confirm RED.
- [ ] Implement canonical objects plus append-only events. Validate parent existence, object digest, and event prefix on load. The sampling key and cycle are exactly:

```python
key = (-train_gain, task_cost, child_count, depth, source_sha256)
parent = ordered_eligible[draw_counter % len(ordered_eligible)]
```

Within tied quality/cost groups prefer novel source-text digests before the
remaining key fields. Seed templates return enabled arm index zero, last enabled
arm, or branch between two indexes based on Train reward comparison. Select two
distinct changed source texts deterministically; derive the seed child's
`operator` from the template identity. Never generate a literal disabled arm.
The unit test may use literal single-arm seed text because it tests lineage.
Pass static audit and a valid request/output unit probe before `eligible`.

- [ ] Run Task 2 tests; confirm GREEN and commit `feat(source): add branching archive`.

### Task 3: Real pipeline meta-CV and sealed validation

**Time:** 30 minutes.

**Files:** Create `evolving_loop/v2/source/meta.py`,
`tests/test_evolution_v2_source_meta.py`, and
`tests/build_evolution_v2_source_fixture.py`.

**Consumes:** `run_policy`; fixed Project 3 `propose_bundle_candidate`,
`CooperativePipelineAdapter.evaluate`, and `sanitize_train_feedback` signatures
listed in the spec. Project 3 seed/catalog/agents are trusted Host injections.
**Produces:** `SourceMetaEvaluatorV2(case)`,
`.train(variant) -> SourceTrainResultV2`,
`.validate(parent_source, finalist_source) -> SourceValidationV2`,
`.canary(parent_source, finalist_source, *, epoch_seed) -> SourceValidationV2`,
and `.replay(variant, train_result) -> bool`.
Define both result contracts in `meta.py`: Train result fields are source SHA,
fold gains, mean gain, both capped means, invalid/catastrophic counts, task cost,
feasible flag, and deterministic execution fingerprint. Validation result fields
are parent/finalist SHA, passed boolean, reason, both aggregate outcomes,
commitment SHA, and replay/evaluation fingerprints; this object stays Host-only.

- [ ] Build `build_source_case(root)` in the fixture module: reuse the Project 3 canonical seed and deterministic real-agent factories; four Train tasks with entity-disjoint folds `(0,1)` and `(2,3)`, one disjoint Dev task, and all four enabled arms. Return a Host case exposing `seed_source`, `improving_source`, `neutral_source`, `train_tasks`, `dev_tasks`, `pipeline_trace`, `policy_requests`, and `evaluator`. Make the deterministic prompts produce different actual legal selected forecasts so `improving_source` wins by measured pipeline scores. The helper never assigns artificial fitness values.
- [ ] Write failing isolation/fitness tests:

```python
def test_source_fitness_uses_real_pipeline_and_train_only_requests(tmp_path):
    case = build_source_case(tmp_path)
    result = case.evaluator.train(case.improving_source)
    assert result.feasible and result.mean_gain > 1e-12
    assert len(result.fold_gains) == 2
    assert {entry["agent"] for entry in case.pipeline_trace} >= {
        "numerical", "retrieval", "decision"}
    for request in case.policy_requests:
        assert set(request) == {"schema_version", "enabled_arms", "step", "seed",
                                "train_reward_by_arm"}


def test_meta_validation_stays_host_only(tmp_path):
    case = build_source_case(tmp_path)
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    assert evidence.passed
    assert all("dev" not in str(request).lower() for request in case.policy_requests)
```

Add one failed held-out validation fixture with unchanged Train tasks, and reject
Public/cross-split entity overlap before the first source call.

- [ ] Run `pytest -q tests/test_evolution_v2_source_meta.py`; confirm RED.
- [ ] Implement complementary-fold Train-only arm reward scans and one held-out evaluation per fold, resetting the frozen seed for each source. Cache aggregate evaluations by Bundle/task/split/protocol/runtime/seed identities. Use:

```python
joint = (mean_capped_smae + mean_capped_srmse) / 2
gain = (parent_joint - child_joint) / max(abs(parent_joint), 1e-12)
passes = (complete and finite and not added_invalid_or_catastrophic
          and child_smae <= parent_smae + 1e-12
          and child_srmse <= parent_srmse + 1e-12)
```

Train feasibility applies to both folds. Finalist ranking uses mean gain, then
task cost and source SHA. Compare the finalist against the active source under
the same Train experiment as well as requiring positive gain over the seed.
Validation additionally requires strict joint improvement; canary requires only
nonregression. Select the finalist Bundle from all Train inputs before opening
Dev. Reuse the Project 3 aggregate field mappings; do not guess new score keys.

- [ ] Run Task 3 tests plus `tests/test_evolution_v2_cooperative_proposals.py`
and `tests/test_evolution_v2_cooperative_pipeline.py` once; confirm GREEN.
- [ ] Commit `feat(source): add automatic meta evaluation`.

### Task 4: Host publication, one-epoch canary, and exact resume

**Time:** 25 minutes.

**Files:** Create `evolving_loop/v2/source/{authority,runner}.py`,
`tests/test_evolution_v2_source_authority.py`, and
`tests/test_evolution_v2_source_runner.py`; update `contracts.py` for config,
checkpoint, and completion contracts, and `__init__.py` for exports.

**Consumes:** Tasks 1–3; fixed `BudgetPlan`/`BudgetLedger`/`ResourceUse` and V2 store helpers.
**Produces:** `SourceAuthorityV2(root, seed_source)`,
`.begin_canary(candidate, evidence) -> None`,
`.finish_canary(result) -> str` (returns active source SHA), and
`run_source_evolution(output_dir, config, case, *, resume=False, stop_after=None)
-> SourceRunResultV2`. `stop_after` names `candidate:1`, `candidate:2`,
`validation`, or `canary`, and stops only after that closed checkpoint.
`SourceConfigV2` exact fields: schema_version, seed, max_candidates (1 or 2),
hard_limit_seconds (120 smoke), subprocess_timeout_seconds (2),
protocol_fingerprint, runtime_fingerprint, and resource_ceilings.

- [ ] Write failing lifecycle tests, using real meta-evaluation from Task 3 and a narrow failing-canary injection for rollback:

```python
def test_canary_failure_restores_exact_pointer_bytes(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    host = SourceAuthorityV2(tmp_path / "authority", case.seed_source)
    before = (tmp_path / "authority/active_source.json").read_bytes()
    evidence = case.evaluator.validate(case.seed_source, case.improving_source)
    host.begin_canary(case.improving_source, evidence)
    assert (tmp_path / "authority/active_source.json").read_bytes() == before
    failed = dataclasses.replace(evidence, passed=False, reason="canary_regression")
    assert host.finish_canary(failed) == case.seed_source.fingerprint()
    assert (tmp_path / "authority/active_source.json").read_bytes() == before


def test_closed_validation_resume_matches_full_result(tmp_path):
    case = build_source_case(tmp_path / "inputs")
    config = SourceConfigV2.smoke(seed=17)
    full = run_source_evolution(tmp_path / "full", config, case)
    run_source_evolution(tmp_path / "resumed", config, case, stop_after="validation")
    resumed = run_source_evolution(tmp_path / "resumed", config, case, resume=True)
    assert resumed.canonical_bytes() == full.canonical_bytes()
```

Also test automatic successful activation, failed validation preserving the
Parent, complete resume doing no evaluation/writes, and altered source/input
digest rejecting resume before evaluation.

- [ ] Run both Task 4 files; confirm RED.
- [ ] Implement one fixed Host state machine:

```python
phases = ("search", "validation", "canary_pending", "complete")
# Closed stages persist their identity and budget closure before next phase.
# select finalist -> seal validation -> replay -> begin canary -> finish canary
# Failure closes evidence, retains/restores Parent, and writes terminal summary.
```

Snapshot exact previous pointer bytes before canary; only the Host writes active
state and sealed evidence. Hash evidence and reread it before canary publication.
Terminalize static/integrity/canary failures; ordinary meta-validation rejection
preserves the Train-ranked stepping stone without revealing Dev scores. Charge
actual source invocations/task work; do not put nondeterministic wall time into
semantic result identities. Completed resume verifies referenced objects and
returns the existing result without writes. A budget boundary before canary
returns a checkpointed result, and a later invocation supplies fresh bounded
wall time without clearing accumulated cost. An in-flight crash is not a
supported resume checkpoint and cannot publish.

- [ ] Run both Task 4 files; confirm GREEN and commit `feat(source): add canary and resume`.

### Task 5: Offline CLI smoke, guide, and final integrated evidence

**Time:** 25 minutes including final focused review.

**Files:** Create `evolving_loop/v2/source/{cli,__main__}.py`,
`configs/evolution_v2/source/smoke.json`,
`tests/test_evolution_v2_source_cli.py`, and `docs/evolution-v2-dgm-lite.md`;
update fixture helper with manifest export/load and `README.md` with one link.

**Consumes:** `SourceConfigV2`, fixture builder, and `run_source_evolution`.
**Produces:** `main(argv=None) -> int`, invoked via `python -m evolving_loop.v2.source`.
The exact commands and profile limits are in the spec. Input manifest contains
schema_version, content hashes and paths for frozen Project 3 seed files, exact
Train/Dev memberships, two Train folds, and protocol/runtime commitments. Paths
are input-only metadata, not source artifact fingerprints. Reject overlap with
output and mismatched file hashes before running.

- [ ] Write a failing subprocess CLI test:

```python
def test_cli_smoke_and_completed_resume(tmp_path):
    base = [sys.executable, "-m", "evolving_loop.v2.source"]
    subprocess.run(base + ["make-smoke-inputs", "--output-dir", str(tmp_path / "in")],
                   check=True, capture_output=True)
    command = base + ["evolve", "--config", "configs/evolution_v2/source/smoke.json",
                      "--input-manifest", str(tmp_path / "in/manifest.json"),
                      "--output-dir", str(tmp_path / "run")]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    result_file = tmp_path / "run/evaluation_complete.json"
    before = result_file.read_bytes()
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    assert result_file.read_bytes() == before
    result = json.loads(before)
    assert result["public_test_accessed"] is False
    assert result["status"] == "source_evolution_complete"
    assert result["activated"] >= 1
```

- [ ] Run Task 5 test; confirm RED.
- [ ] Implement CLI argparse dispatch and canonical fixture materialization. Repeating `evolve` auto-detects a matching source manifest; no approval option. Completion fields are schema_version, status, active_source_sha256, archive_snapshot_sha256, proposed, eligible, activated, rolled_back, completed_stage_ids, and public_test_accessed. Epoch/checkpoint result may use status `source_evolution_checkpointed`; only actual terminal completion writes `evaluation_complete.json`.
- [ ] Write the operator guide with runnable commands, editable boundary, evidence locations, exact resume/rollback behavior, offline-fixture limitation, and deferred OS/general-Harness scope. Do not imply a measured real-dataset improvement.
- [ ] Run one final gate, under three minutes; prior compatibility evidence from Task 3 suffices unless its integration changed:

```bash
pytest -q tests/test_evolution_v2_source_runtime.py \
  tests/test_evolution_v2_source_archive.py \
  tests/test_evolution_v2_source_meta.py \
  tests/test_evolution_v2_source_authority.py \
  tests/test_evolution_v2_source_runner.py \
  tests/test_evolution_v2_source_cli.py
git diff --check
```

The CLI test is the one required offline smoke; do not launch a duplicate smoke
manually. Final reviewer reads this result and the full P4 diff. Any fix reruns
only covering tests. Update SDD progress with commits, observed counts, command
output, remaining limitations, and both review verdicts.

- [ ] Commit `feat(source): expose research smoke command`.

## Completion boundary

Completion requires actual executable source variation, a retained nonactive
branch, Train-only sampling, sealed automatic meta-validation, automatic
successful canary activation, exact rollback on failed canary, and closed-stage
resume equality. Reports must distinguish plans/tests from observed experiment
results. Five tasks total: 20 + 20 + 30 + 25 + 25 = 120 minutes; if work threatens
the target, cut optional experiments and broad hardening first, preserving these
research invariants.
