# Toto-Balanced Three-Agent Co-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the package-native Numerical, Retrieval, and Decision feedback loop compatible with the frozen Toto-balanced v3 authority, then run and verify two-coordinate-cycle evolution and one sealed Public-99 regression evaluation.

**Architecture:** Keep the existing one-coordinate-at-a-time controller and immutable bundle contracts. Add an explicit, fingerprinted opt-in for the v3 label-informed regression split, extend the already-tested task-feedback manager from interaction smoke to formal evolution, and preserve the separate one-shot Public evaluator.

**Tech Stack:** Python 3.12, frozen dataclasses, canonical JSON/SHA-256 identities, pytest, Codex CLI `gpt-5.6-luna`, Dr-CiK capped sMAE/sRMSE.

**Spec:** `docs/superpowers/specs/2026-09-06-toto-balanced-three-agent-coevolution-design.md`

## Global Constraints

- The split file must be `splits/drcik_public_80_20_99_v3.json` with file SHA-256 `e2cc50dc95b3bdcd0dcc4b0df8d7eac3ff59ad9ce18268aa3d663e182733307d`.
- Label-informed partitioning is permitted only through an explicit CLI opt-in recorded in run identity; ground-truth evidence and document labels remain forbidden.
- Evolution loads 80 Train and 20 Dev tasks only; Public-99 remains inaccessible until a final bundle is sealed.
- Formal evolution uses two ordered `Numerical -> Retrieval -> Decision` cycles, three Children per coordinate, seed `20260903`, `gpt-5.6-luna`, and reasoning effort `medium`.
- Task feedback is host-validated, sanitized, artifact-bound, and injected only into the second Numerical coordinate.
- Existing forecast, coverage, failure, tail, fold, lineage, and single-coordinate acceptance gates are not weakened.
- Public-99 is a frozen, label-informed regression set and may be evaluated once; it is not described as an unseen test set.

---

### Task 1: Create isolated execution worktree and establish the baseline

**Files:**
- Read: `docs/superpowers/specs/2026-09-06-toto-balanced-three-agent-coevolution-design.md`
- Read: `evolving_loop/run_package_coevolution.py`
- Read: `evolving_loop/evaluate_frozen_package_bundle.py`
- Test: `tests/test_run_package_coevolution.py`
- Test: `tests/test_package_artifacts.py`
- Test: `tests/test_package_coordinate_e2e.py`
- Test: `tests/test_evaluate_frozen_package_bundle.py`

**Interfaces:**
- Consumes: current commit `d94ddb5` and the project-local ignored `.worktrees/` directory.
- Produces: branch `feature/toto-balanced-triad-run` in `.worktrees/toto-balanced-triad-run` with a recorded green or explicitly diagnosed baseline.

- [ ] **Step 1: Create the isolated worktree**

```bash
git worktree add .worktrees/toto-balanced-triad-run -b feature/toto-balanced-triad-run d94ddb5
```

- [ ] **Step 2: Run focused baseline tests**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_run_package_coevolution.py \
  tests/test_package_artifacts.py \
  tests/test_package_coordinate_e2e.py \
  tests/test_evaluate_frozen_package_bundle.py
```

Expected: all selected tests pass before compatibility changes.

### Task 2: Add an explicit v3 label-informed split authority

**Files:**
- Modify: `evolving_loop/run_package_coevolution.py`
- Modify: `tests/test_run_package_coevolution.py`

**Interfaces:**
- Produces: CLI flag `--allow-label-informed-regression-split`.
- Produces: `_validated_split(path, *, allow_label_informed_regression_split: bool = False)`.
- Consumes: manifest fields `selection_uses_future_values`, `selection_uses_model_metrics`, `selection_uses_gt_evidence`, `selection_uses_document_labels`, `toto_difficulty_model`, and `toto_difficulty_profile_sha256`.

- [ ] **Step 1: Write failing tests for default rejection and explicit v3 acceptance**

Add tests equivalent to:

```python
def test_v3_label_informed_split_requires_explicit_regression_opt_in():
    v3_split = Path("splits/drcik_public_80_20_99_v3.json")
    with pytest.raises(ValueError, match="label-informed"):
        _validated_split(v3_split)

    payload, train, dev = _validated_split(
        v3_split,
        allow_label_informed_regression_split=True,
    )
    assert payload["selection_uses_future_values"] is True
    assert payload["selection_uses_model_metrics"] is True
    assert len(train) == 80
    assert len(dev) == 20
```

Also assert that the opt-in still rejects `selection_uses_gt_evidence=True`, `selection_uses_document_labels=True`, a missing Toto model identity, or a non-SHA-256 difficulty profile.

- [ ] **Step 2: Verify the new tests fail for the intended reason**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_run_package_coevolution.py -k 'label_informed or validated_split'
```

Expected: failure because the parser/signature does not yet expose the opt-in and v3 remains rejected.

- [ ] **Step 3: Implement the minimal closed opt-in**

Add the boolean parser flag, include it in `_configuration_identity` and the run manifest, and pass it to `_validated_split`. Keep the default label-free rule. Under explicit opt-in, accept future-value selection only when model-metric stratification, a non-empty Toto model identifier, and a 64-character lowercase hexadecimal Toto profile fingerprint are present; never permit ground-truth evidence or document-label selection.

- [ ] **Step 4: Verify focused tests pass**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_run_package_coevolution.py -k 'label_informed or validated_split or parser'
```

- [ ] **Step 5: Commit the authority change**

```bash
git add evolving_loop/run_package_coevolution.py tests/test_run_package_coevolution.py
git commit -m "fix(evolution): admit frozen v3 split"
```

### Task 3: Enable task feedback in formal two-cycle evolution

**Files:**
- Modify: `evolving_loop/run_package_coevolution.py`
- Modify: `tests/test_run_package_coevolution.py`
- Test: `tests/test_package_task_feedback.py`
- Test: `tests/test_package_artifacts.py`

**Interfaces:**
- Consumes: `_InteractionFeedbackManager`, `PackageTaskFeedbackLedger`, and `PackageProposalFeedback.task_evidence`.
- Produces: formal `--feedback-mode task` support for exactly `--cycles 2 --children-per-coordinate 3`.
- Preserves: one-cycle smoke rejects task feedback; cycle-1 Numerical receives `None`; cycle-2 Numerical receives the persisted generation-3 projection.

- [ ] **Step 1: Write failing formal-mode and orchestration tests**

Add tests equivalent to:

```python
def test_formal_mode_allows_task_feedback(required_args):
    args = build_parser().parse_args([
        *required_args,
        "--cycles", "2",
        "--children-per-coordinate", "3",
        "--feedback-mode", "task",
    ])
    _validate_mode(args)

def test_formal_second_numerical_receives_cycle_one_projection(tmp_path):
    controller = deterministic_formal_controller(tmp_path, feedback_mode="task")
    controller.run()
    assert controller.numerical_feedback_generations == [None, 3]
    assert controller.loaded_projection == controller.persisted_projection
```

The deterministic fixture must also prove the projection is bound to the accepted Parent fingerprint and that Public membership is absent.

- [ ] **Step 2: Verify RED**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_run_package_coevolution.py \
  tests/test_package_artifacts.py \
  tests/test_package_task_feedback.py \
  -k 'formal or feedback or cycle'
```

Expected: formal task mode fails at `_validate_mode`, or the second Numerical phase lacks the projection.

- [ ] **Step 3: Extend the existing manager without changing its wire contract**

Construct `PackageTaskFeedbackLedger` whenever `feedback_mode == "task"`. Attach the feedback manager to formal phases as well as interaction-smoke phases. Persist the cycle-1 projection after Decision generation 2, load it only for Numerical generation 3, and retain the existing empty-treatment artifact when no accepted non-seed Retrieval trace exists.

- [ ] **Step 4: Bind formal feedback to run and resume authority**

Keep `feedback_mode` in configuration identity and `run_manifest.json`. Confirm checkpoint resume rejects missing, changed, empty-when-required, or wrong-Parent generation-3 feedback bytes before proposal execution.

- [ ] **Step 5: Verify GREEN and the broader package boundary**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_task_evidence_feedback.py \
  tests/test_package_task_feedback.py \
  tests/test_package_candidate_proposal.py \
  tests/test_run_package_coevolution.py \
  tests/test_package_artifacts.py \
  tests/test_package_coordinate_e2e.py
```

- [ ] **Step 6: Commit the feedback extension**

```bash
git add evolving_loop/run_package_coevolution.py tests/test_run_package_coevolution.py tests/test_package_artifacts.py
git commit -m "feat(evolution): enable formal task feedback"
```

### Task 4: Run and audit a real v3 interaction smoke

**Files:**
- Write generated artifacts only: `runs/package_coevolution/smoke_toto_balanced_v3_luna_20260906/`
- Write external authority only: `runs/package_coevolution_authority/smoke_toto_balanced_v3_luna_20260906/`
- Write log only: `tmp/smoke_toto_balanced_v3_luna_20260906.log`

**Interfaces:**
- Consumes: v3 split, v3 Toto release, v3 forecast cache, Retrieval `v000`, and the formal runtime configuration.
- Produces: a two-cycle one-Child-per-coordinate smoke with `full_chain_exercised=true` and `public_test_accessed=false`.

- [ ] **Step 1: Provision a temporary writable Codex home**

Create a task-scoped directory under `/private/tmp`, link only the existing `auth.json`, and set `CODEX_HOME` for nested proposal calls. Do not copy credentials into project artifacts.

```bash
mkdir -p /private/tmp/codex-v3-triad-20260906
ln -s /Users/yyoraa/.codex/auth.json \
  /private/tmp/codex-v3-triad-20260906/auth.json
```

- [ ] **Step 2: Run the v3 interaction smoke**

Run from the isolated worktree with `--interaction-smoke --cycles 2 --children-per-coordinate 1 --feedback-mode task --allow-label-informed-regression-split`, the approved model/runtime arguments, and fresh output/authority directories.

```bash
cd /Users/yyoraa/time-series/.worktrees/toto-balanced-triad-run
CODEX_HOME=/private/tmp/codex-v3-triad-20260906 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
/Users/yyoraa/time-series/.venv/bin/python -m evolving_loop.run_package_coevolution \
  --repo /Users/yyoraa/time-series/runs/method_evolution/v001 \
  --split-file /Users/yyoraa/time-series/splits/drcik_public_80_20_99_v3.json \
  --tasks-file /Users/yyoraa/time-series/external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --numerical-champion-release /Users/yyoraa/time-series/runs/champion_evolution/gpt56sol_high_toto_balanced_v3_20260906_g3_r2/champion_release.json \
  --forecast-store /Users/yyoraa/time-series/runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906 \
  --retrieval-seed-release /Users/yyoraa/time-series/runs/retrieval_releases/package_nrd_20260903/v000 \
  --retrieval-skills /Users/yyoraa/time-series/runs/retrieval_releases/package_nrd_20260903/v000/skills.json \
  --output-dir /Users/yyoraa/time-series/runs/package_coevolution/smoke_toto_balanced_v3_luna_20260906 \
  --authority-dir /Users/yyoraa/time-series/runs/package_coevolution_authority/smoke_toto_balanced_v3_luna_20260906 \
  --model gpt-5.6-luna --reasoning-effort medium \
  --cycles 2 --children-per-coordinate 1 --seed 20260903 \
  --tsfm-runtimes chronos,timesfm --chronos-device-map cpu \
  --model-cache-dir /Users/yyoraa/time-series/outputs/model-cache \
  --tsfm-workers-config /Users/yyoraa/time-series/tmp/toto2_worker_smoke.json \
  --acknowledged-model-licenses CC-BY-NC-4.0 \
  --interaction-smoke --feedback-mode task \
  --allow-label-informed-regression-split \
  2>&1 | tee /Users/yyoraa/time-series/tmp/smoke_toto_balanced_v3_luna_20260906.log
```

- [ ] **Step 3: Verify smoke artifacts**

Assert `evaluation_complete.json.status == "complete"`, `full_chain_exercised is True`, exactly six ordered coordinate records exist, task-feedback generation 3 exists, and every Public-access marker is false.

- [ ] **Step 4: Apply the debugging/TDD loop only if the smoke fails**

Capture the first causal failure boundary, reproduce it with the narrowest existing test module, add one regression test that fails for that behavior, implement the minimum correction, rerun the focused and package suites, commit the fix, and restart with new output and authority directories.

### Task 5: Audit the design against primary research before Public access

**Files:**
- Create: `docs/results/TOTO_BALANCED_V3_TRIAD_DESIGN_REVIEW.md`

**Interfaces:**
- Consumes: Dr-CiK, Last-Mile Forecasting, PostTime, forecast-aware retrieval/reranking, and relevant time-series routing/ensemble primary papers or official repositories.
- Produces: an evidence-backed assessment of point-in-time retrieval, evidence-to-forecast mechanisms, model selection, coordinate credit assignment, adaptive Dev use, and label-informed split interpretation.

- [ ] **Step 1: Read primary sources and official implementations**

Record publication identifiers, protocol details, and exact source links. Do not use secondary summaries for technical claims.

- [ ] **Step 2: Compare each design boundary with the literature**

Classify each boundary as supported, defensible but unvalidated, or incorrect/high-risk. Explicitly evaluate repeated Dev feedback and the absence of paired-coordinate proposals.

- [ ] **Step 3: Write the review before starting Public-99**

The report must distinguish changes required for correctness before the formal run from future ablations that must not be selected using Public-99.

### Task 6: Run formal v3 three-agent co-evolution

**Files:**
- Write generated artifacts only: `runs/package_coevolution/gpt56luna_medium_toto_balanced_v3_g2_20260906/`
- Write external authority only: `runs/package_coevolution_authority/gpt56luna_medium_toto_balanced_v3_g2_20260906/`
- Write log only: `tmp/gpt56luna_medium_toto_balanced_v3_g2_20260906.log`

**Interfaces:**
- Consumes: the smoke-verified code and unchanged registered v3 authority.
- Produces: a sealed `final_bundle.json`, `evaluation_complete.json`, six-or-early-stop coordinate records, candidate evidence, task feedback, and replay-bound checkpoints.

- [ ] **Step 1: Start a fresh formal run**

Use `--cycles 2 --children-per-coordinate 3 --feedback-mode task --allow-label-informed-regression-split`, `gpt-5.6-luna`, `medium`, seed `20260903`, and the approved Chronos/TimesFM runtime settings.

```bash
cd /Users/yyoraa/time-series/.worktrees/toto-balanced-triad-run
CODEX_HOME=/private/tmp/codex-v3-triad-20260906 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
/Users/yyoraa/time-series/.venv/bin/python -m evolving_loop.run_package_coevolution \
  --repo /Users/yyoraa/time-series/runs/method_evolution/v001 \
  --split-file /Users/yyoraa/time-series/splits/drcik_public_80_20_99_v3.json \
  --tasks-file /Users/yyoraa/time-series/external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --numerical-champion-release /Users/yyoraa/time-series/runs/champion_evolution/gpt56sol_high_toto_balanced_v3_20260906_g3_r2/champion_release.json \
  --forecast-store /Users/yyoraa/time-series/runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906 \
  --retrieval-seed-release /Users/yyoraa/time-series/runs/retrieval_releases/package_nrd_20260903/v000 \
  --retrieval-skills /Users/yyoraa/time-series/runs/retrieval_releases/package_nrd_20260903/v000/skills.json \
  --output-dir /Users/yyoraa/time-series/runs/package_coevolution/gpt56luna_medium_toto_balanced_v3_g2_20260906 \
  --authority-dir /Users/yyoraa/time-series/runs/package_coevolution_authority/gpt56luna_medium_toto_balanced_v3_g2_20260906 \
  --model gpt-5.6-luna --reasoning-effort medium \
  --cycles 2 --children-per-coordinate 3 --seed 20260903 \
  --tsfm-runtimes chronos,timesfm --chronos-device-map cpu \
  --model-cache-dir /Users/yyoraa/time-series/outputs/model-cache \
  --tsfm-workers-config /Users/yyoraa/time-series/tmp/toto2_worker_smoke.json \
  --acknowledged-model-licenses CC-BY-NC-4.0 \
  --feedback-mode task --allow-label-informed-regression-split \
  2>&1 | tee /Users/yyoraa/time-series/tmp/gpt56luna_medium_toto_balanced_v3_g2_20260906.log
```

- [ ] **Step 2: Monitor without mutating run authority**

Inspect the append-only trace, log, and process state. Resume only with byte-identical arguments and the independently retained authority. Any code or runtime fix requires a new run ID.

- [ ] **Step 3: Verify formal completion**

Assert `formal_run=true`, `public_test_accessed=false`, final-bundle identity matches the completion marker, coordinate ordering is valid, accepted steps change one principal fingerprint, rejected steps preserve Parent bytes, and every opened stage matches the registered schedule.

### Task 7: Run Public-99 exactly once and publish the report

**Files:**
- Write generated artifacts only: `runs/package_coevolution_public/gpt56luna_medium_toto_balanced_v3_g2_public99_20260906/`
- Modify: `docs/results/TOTO_BALANCED_V3_TRIAD_DESIGN_REVIEW.md`

**Interfaces:**
- Consumes: the verified sealed formal bundle only.
- Produces: one `public_report.json`, `public_forecasts.jsonl`, and `evaluation_complete.json` with `public_result_may_feed_evolution=false`.

- [ ] **Step 1: Invoke the separate frozen evaluator**

Use the exact formal evolution directory, its sealed `final_bundle.json`, the v3 split, identical repo/cache/runtime inputs, and a fresh Public output directory.

```bash
cd /Users/yyoraa/time-series/.worktrees/toto-balanced-triad-run
CODEX_HOME=/private/tmp/codex-v3-triad-20260906 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
/Users/yyoraa/time-series/.venv/bin/python -m evolving_loop.evaluate_frozen_package_bundle \
  --evolution-dir /Users/yyoraa/time-series/runs/package_coevolution/gpt56luna_medium_toto_balanced_v3_g2_20260906 \
  --final-bundle /Users/yyoraa/time-series/runs/package_coevolution/gpt56luna_medium_toto_balanced_v3_g2_20260906/final_bundle.json \
  --split-file /Users/yyoraa/time-series/splits/drcik_public_80_20_99_v3.json \
  --tasks-file /Users/yyoraa/time-series/external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --repo /Users/yyoraa/time-series/runs/method_evolution/v001 \
  --forecast-store /Users/yyoraa/time-series/runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906 \
  --output-dir /Users/yyoraa/time-series/runs/package_coevolution_public/gpt56luna_medium_toto_balanced_v3_g2_public99_20260906 \
  --model gpt-5.6-luna --reasoning-effort medium \
  --tsfm-runtimes chronos,timesfm --chronos-device-map cpu \
  --model-cache-dir /Users/yyoraa/time-series/outputs/model-cache \
  --tsfm-workers-config /Users/yyoraa/time-series/tmp/toto2_worker_smoke.json \
  --acknowledged-model-licenses CC-BY-NC-4.0
```

- [ ] **Step 2: Verify one-shot Public evidence**

Assert exactly 99 task IDs were evaluated, no proposal or mutation artifact was written, `selection_used_public=false`, and the report fingerprint matches the completion marker.

- [ ] **Step 3: Run final regression verification**

Run:

```bash
/Users/yyoraa/time-series/.venv/bin/pytest -q \
  tests/test_task_evidence_feedback.py \
  tests/test_package_task_feedback.py \
  tests/test_package_candidate_proposal.py \
  tests/test_run_package_coevolution.py \
  tests/test_package_artifacts.py \
  tests/test_package_coordinate_e2e.py \
  tests/test_evaluate_frozen_package_bundle.py
git diff --check
```

- [ ] **Step 4: Complete the result report and commit tracked work**

Report the initial Toto bundle, accepted coordinate increments, final Dev result, Public-99 regression result, W/T/L and failure/tail diagnostics, paper-alignment findings, limitations, and exact artifact paths.

```bash
git add docs/results/TOTO_BALANCED_V3_TRIAD_DESIGN_REVIEW.md
git commit -m "docs(results): report v3 triad run"
```
