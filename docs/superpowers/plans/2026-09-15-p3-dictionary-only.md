# P3 Dictionary-Only Numerical Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every new real Evolution V2 run evolve a P3 Numerical Selector over the complete executable P2 Dictionary instead of choosing P2-preselected task-local bundles.

**Architecture:** P2 frozen pairs are reduced to one immutable executable Dictionary closure: the union of their compatible release specifications and per-task forecast/diagnostic cache. Each canonical Selector Genome materializes its own schema-v2 Numerical release and 100-task frozen registry from that closure. Existing P3 bundle ownership remains unchanged because a Selector child is represented by the resulting atomic `(release, registry)` pair.

**Tech Stack:** Python 3 dataclasses, canonical JSON/SHA-256 artifacts, pytest, existing Evolution V2 Numerical package and cooperative runner APIs.

**Spec:** `docs/superpowers/specs/2026-09-15-p3-dictionary-only-design.md`

## Global Constraints

- New real runs have one active numerical mode: `p3_dictionary`.
- Old P2 frozen artifacts remain readable but cannot be proposed directly in P3.
- Candidate selection uses history-only diagnostics; future values and Public evidence are forbidden.
- A shortlist contains at most eight candidates and always includes the protected Anchor.
- A final package contains Anchor plus zero, one, or two specialists, and Anchor weight is at least `0.5`.
- The existing metrics, verifier, Train/Dev split, artifact validation, and promotion Host remain fixed.

---

### Task 1: Canonical P3 Selector Genome

**Files:**
- Create: `evolving_loop/v2/cooperative/numerical_dictionary.py`
- Modify: `evolving_loop/v2/cooperative/__init__.py`
- Test: `tests/test_evolution_v2_p3_dictionary.py`

**Interfaces:**
- Produces: `DictionarySelectorGenomeV2`, `seed_selector_genome()`, and `mutate_selector_genome(parent, step)`.
- The genome fingerprint becomes part of every P3 Numerical release identity.

- [ ] **Step 1: Write the failing contract and mutation tests**

```python
def test_selector_genome_round_trips_and_rejects_unknown_fields():
    seed = seed_selector_genome()
    assert DictionarySelectorGenomeV2.from_payload(seed.to_payload()) == seed
    assert seed.target_candidates == 8

def test_selector_mutation_changes_one_owned_policy_coordinate():
    parent = seed_selector_genome()
    child = mutate_selector_genome(parent, 0)
    changed = {k for k in parent.to_payload() if parent.to_payload()[k] != child.to_payload()[k]}
    assert changed == {"generation", "parent_selector_sha256", "morphology_weight"}
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_p3_dictionary.py`

Expected: collection fails because `numerical_dictionary` does not exist.

- [ ] **Step 3: Implement the strict artifact and deterministic mutation cycle**

The exact schema is:

```python
DictionarySelectorGenomeV2(
    schema_version: int,
    generation: int,
    parent_selector_sha256: str | None,
    target_candidates: int,
    morphology_weight: float,
    success_weight: float,
    mean_error_weight: float,
    p90_error_weight: float,
    family_diversity_weight: float,
)
```

All weights are finite and non-negative; `target_candidates` is in `[6, 8]`.
The seed uses target `8` and weights `(1.0, 1.0, 1.0, 0.5, 0.25)`.
Mutation cycles through one weight or target size and records exact parent lineage.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_evolution_v2_p3_dictionary.py`

Expected: PASS.

### Task 2: Complete Executable Dictionary Closure

**Files:**
- Modify: `evolving_loop/v2/cooperative/numerical_dictionary.py`
- Test: `tests/test_evolution_v2_p3_dictionary.py`

**Interfaces:**
- Consumes: every compatible `FrozenNumericalArtifactsV2` emitted by P2 and the exact Host task universe.
- Produces: `P3NumericalDictionaryV2` and `build_p3_numerical_dictionary(pairs, tasks)`.

- [ ] **Step 1: Write failing closure tests**

```python
def test_dictionary_closure_unions_members_instead_of_preserving_p2_shortlists():
    closure = build_p3_numerical_dictionary((pair_a, pair_b), tasks)
    assert closure.candidate_names == ("anchor", "method_a", "method_b")
    assert closure.package_inputs_for(tasks[0]).keys() == {
        "anchor", "method_a", "method_b"
    }

def test_dictionary_closure_rejects_anchor_or_runtime_drift():
    with pytest.raises(ValueError, match="Anchor"):
        build_p3_numerical_dictionary((pair_a, changed_anchor_pair), tasks)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_p3_dictionary.py`

Expected: failure because the closure API is missing.

- [ ] **Step 3: Implement closure validation and union**

Require identical task IDs, task fingerprints, protected Anchor forecasts,
runtime fingerprints, and compatible candidate specifications. Union candidate
specifications and per-task ranked alternatives by candidate identity. Reject
conflicting bytes rather than picking one. The closure payload contains only
canonical identities and a `public_test_accessed=false` flag; live packages are
kept beside, not inside, the canonical payload.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_evolution_v2_p3_dictionary.py`

Expected: PASS.

### Task 3: Materialize Selector Genomes into Frozen Numerical Pairs

**Files:**
- Modify: `evolving_loop/v2/cooperative/numerical_dictionary.py`
- Modify: `evolving_loop/v2/cooperative/adapters.py`
- Modify: `evolving_loop/v2/cooperative/proposals.py`
- Test: `tests/test_evolution_v2_p3_dictionary.py`
- Test: `tests/test_evolution_v2_cooperative_adapters.py`
- Test: `tests/test_evolution_v2_cooperative_proposals.py`

**Interfaces:**
- Produces: `materialize_selector_pair(dictionary, genome, tasks)` and a `NumericalCoordinateAdapter` that proposes the next materialized Selector child using `step`.
- Consumes: history-only `CandidateDiagnostics` already cached in the P2 executable closure.

- [ ] **Step 1: Write failing selection and package tests**

```python
def test_different_selector_genomes_change_shortlists_from_one_dictionary():
    first = materialize_selector_pair(closure, seed_selector_genome(), tasks)
    second = materialize_selector_pair(
        closure, replace(seed_selector_genome(), family_diversity_weight=10.0), tasks
    )
    assert first.release.source_fingerprints["p3_dictionary"] == closure.fingerprint()
    assert first.release.source_fingerprints["p3_selector"] != second.release.source_fingerprints["p3_selector"]
    assert shortlist_names(first, tasks[0]) != shortlist_names(second, tasks[0])

def test_materialized_selector_pair_obeys_anchor_and_cardinality_bounds():
    pair = materialize_selector_pair(closure, seed_selector_genome(), tasks)
    package = pair.registry.package_for(tasks[0])
    assert len(package.active_candidate_names) <= 8
    assert package.protected_baseline.name in package.selection_decision.selected
    assert len(package.selection_decision.selected) <= 3
    assert package.selection_decision.weights[
        package.selection_decision.selected.index(package.protected_baseline.name)
    ] >= 0.5
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_p3_dictionary.py tests/test_evolution_v2_cooperative_adapters.py tests/test_evolution_v2_cooperative_proposals.py`

Expected: failure because Selector materialization and step-aware numerical proposals are missing.

- [ ] **Step 3: Implement deterministic weighted ranking and materialization**

Rank eligible candidates by weighted morphology error, failure/success signal,
mean error, p90 error, and dynamic family repetition penalty; break ties by
candidate identity. Keep the Anchor first and select `target_candidates` up to
the fixed maximum eight. Rebuild a schema-v2 release containing the complete
Dictionary, bind `p3_dictionary` and `p3_selector` fingerprints, then use the
existing task-local tournament to create the final package and frozen registry.

Change `NumericalCoordinateAdapter.propose(parent, step)` so it materializes the
deterministic next Selector Genome. Change cooperative proposal construction to
pass `step`. It must never return a raw P2 pair.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_evolution_v2_p3_dictionary.py tests/test_evolution_v2_cooperative_adapters.py tests/test_evolution_v2_cooperative_proposals.py`

Expected: PASS.

### Task 4: Replace the Real P2-to-P3 Handoff

**Files:**
- Modify: `evolving_loop/v2/real/runner.py`
- Modify: `evolving_loop/v2/real/bridges.py`
- Modify: `evolving_loop/v2/real/contracts.py`
- Test: `tests/test_evolution_v2_real_cooperative.py`
- Test: `tests/test_evolution_v2_real_runner.py`
- Test: `tests/test_evolution_v2_real_contracts.py`

**Interfaces:**
- P2 seal produces `p3_dictionary_sha256`, not a list of P2 numerical alternatives.
- `run_real_cooperative` accepts the complete P3 Dictionary closure and seeds P3 from `seed_selector_genome()`.

- [ ] **Step 1: Write failing real bridge tests**

```python
def test_real_bridge_seeds_p3_with_a_selector_materialization(monkeypatch, tmp_path):
    def observe(_output, _config, seed, _tasks, adapters, **_kwargs):
        assert seed["numerical"].release.source_fingerprints["p3_dictionary"]
        assert seed["numerical"].release.source_fingerprints["p3_selector"]
        assert adapters["numerical"].mode == "p3_dictionary"
        raise Observed

def test_real_bridge_never_publishes_raw_p2_alternatives():
    assert not hasattr(host, "numerical_alternatives")
```

- [ ] **Step 2: Run real bridge tests and verify RED**

Run: `pytest -q tests/test_evolution_v2_real_cooperative.py tests/test_evolution_v2_real_runner.py tests/test_evolution_v2_real_contracts.py`

Expected: failures showing the old P2-pair seed and `numerical_alternatives` handoff.

- [ ] **Step 3: Implement the Dictionary-only handoff and resume checks**

Replace `_publish_p2_numerical_alternatives` with a publisher that loads every
compatible P2 frozen pair, constructs the Dictionary closure, and stores it on
the Host for P3. On resume, reconstruct the exact same closure and verify its
fingerprint. `run_real_cooperative` materializes the seed Selector and its
bounded mutation sequence before invoking the unchanged cooperative runner.
Add `numerical_mode="p3_dictionary"` to proposal-space commitments and reject
any other new-run value. Existing on-disk closure readers stay unchanged for
legacy reporting only.

- [ ] **Step 4: Run real bridge tests and verify GREEN**

Run: `pytest -q tests/test_evolution_v2_real_cooperative.py tests/test_evolution_v2_real_runner.py tests/test_evolution_v2_real_contracts.py`

Expected: PASS.

### Task 5: Full Regression and Smoke Verification

**Files:**
- Modify: `docs/evolution-v2-cooperative-bundle.md`
- Modify: `docs/evolution-v2-numerical-qd.md`
- Modify: `docs/superpowers/specs/2026-09-13-task-local-numerical-shortlist-design.md`
- Modify: `docs/superpowers/specs/2026-09-12-evolution-v2-cooperative-bundle-design.md`

**Interfaces:**
- Documents state that P2 creates the complete executable Dictionary and P3 is the only active selection authority.

- [ ] **Step 1: Run the complete relevant regression suite**

Run:

```bash
pytest -q \
  tests/test_evolution_v2_p3_dictionary.py \
  tests/test_evolution_v2_cooperative_contracts.py \
  tests/test_evolution_v2_cooperative_adapters.py \
  tests/test_evolution_v2_cooperative_proposals.py \
  tests/test_evolution_v2_cooperative_pipeline.py \
  tests/test_evolution_v2_cooperative_runner.py \
  tests/test_evolution_v2_real_cooperative.py \
  tests/test_evolution_v2_real_runner.py \
  tests/test_evolution_v2_real_contracts.py \
  tests/test_evolution_v2_frozen_shortlist.py \
  tests/test_evolution_v2_shortlist_runtime.py
```

Expected: PASS with no warnings.

- [ ] **Step 2: Update active architecture documentation**

Replace statements that P3 consumes or chooses P2-preselected frozen supplies.
Document the sole active `p3_dictionary` path and label old P2 shortlist text
as legacy compatibility behavior.

- [ ] **Step 3: Run a deterministic end-to-end smoke**

Run:

```bash
pytest -q \
  tests/test_evolution_v2_real_cli.py::test_production_ports_complete_root_and_resume_byte_identically \
  tests/test_evolution_v2_p3_dictionary.py::test_dictionary_selector_real_stack_smoke
```

Expected: both tests pass. The second test must execute the real stage-port
P2-to-P3 handoff with deterministic local agents, inspect the P3
proposal-space artifact and one task package, and assert that both
`p3_dictionary` and `p3_selector` fingerprints are present while
`public_test_accessed` remains false.

- [ ] **Step 4: Run `git diff --check` and inspect the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only planned source, tests, and docs plus the
user's pre-existing `runs/` artifacts are present.
