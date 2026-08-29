# Real Morphology Provider Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in history-only Morphology provider that safely feeds grounded assumptions
into two-stage Retrieval while preserving the empty provider as the default.

**Architecture:** The host projects executed Coding candidates into immutable `name + family`
records, and `MorphologyAdapter` invokes the existing `MorphologyReasoner.reason()` with only those
records and the numeric task view. `_factory()` centrally constructs either the empty or real
provider from CLI configuration, so root, legacy, source-worker, coordinate, and frozen-inference
paths share one compatibility rule. Retrieval Round 1 stays assumption-blind; only four sanitized
fields can reach Round 2.

**Tech Stack:** Python 3.10+, dataclasses, runtime-checkable Protocols, argparse, pytest, existing
`FakeLLMClient`, existing SHA-256 manifest utilities

**Spec:** `docs/superpowers/specs/2026-08-29-real-morphology-provider-integration-design.md`

## Global Constraints

- The feature is opt-in through `--morphology-provider reasoner`; the exact default is `empty`.
- Morphology receives history, frequency, horizon, candidate name, and candidate family only.
- Morphology must not receive documents, labels, forecasts, hindcast metrics, generated source,
  candidate assumptions, or prompts.
- Round 1 remains assumption-blind; Round 2 receives only `assumption_id`, `kind`, `claim`, and
  `failure_condition` plus host-validated named gaps.
- Existing Retrieval releases remain Retrieval-owned and do not absorb Morphology source.
- Non-transient Morphology failures preserve the numeric host and verified Round 1, record the
  typed failure, and skip Round 2.
- `TransientLLMError` retains the existing retry/abort behavior.
- Do not copy or overwrite uncommitted files from either existing worktree.
- Every production behavior starts with a failing test and an observed expected failure.

## Execution Workspace

The current `main` worktree and the existing
`.worktrees/morphology-guided-numerical-loop` worktree are both dirty. Before Task 1, use the
`using-git-worktrees` skill to create a separate worktree and branch from the committed feature
head `af832561af3b26bf79c274197b277425506a2e8c`:

```bash
git worktree add \
  /Users/yyoraa/time-series/.worktrees/real-morphology-provider \
  -b feature/real-morphology-provider \
  af832561af3b26bf79c274197b277425506a2e8c
```

Bring the approved design and this plan into that branch without staging any other worktree's
changes:

```bash
morph_design_sha=$(git -C /Users/yyoraa/time-series log -1 --format=%H -- \
  docs/superpowers/specs/2026-08-29-real-morphology-provider-integration-design.md)
morph_plan_sha=$(git -C /Users/yyoraa/time-series log -1 --format=%H -- \
  docs/superpowers/plans/2026-08-29-real-morphology-provider-integration.md)
git cherry-pick "$morph_design_sha" "$morph_plan_sha"
```

Verify that `numerical_agent/evolution/morphology.py` and
`tests/test_evolution_morphology.py` are tracked in the isolated branch before starting. All paths
and commands below are relative to the isolated worktree.

---

### Task 1: Replace the preliminary adapter with the real safe boundary

**Files:**
- Modify: `evolving_loop/morphology_adapter.py:1-64`
- Modify: `tests/test_two_stage_retrieval.py:1-330`

**Interfaces:**
- Produces: `MorphologyCandidate(name: str, family: str)`
- Produces: `MorphologyProvider.assumptions(task, candidates) -> tuple[RetrievalAssumption, ...]`
- Produces: `MorphologyAdapter(reasoner).assumptions(task, candidates)`
- Produces: a fixed Numerical-to-Retrieval assumption-kind translation
- Consumes: an injected object exposing
  `reason(history, frequency, horizon, active_names, families) -> MorphologyCard`

- [ ] **Step 1: Write the failing adapter-boundary test**

Update the import and replace the existing adapter test with a reasoner-shaped fake:

```python
from evolving_loop.morphology_adapter import MorphologyAdapter, MorphologyCandidate


class _RecordingMorphologyReasoner:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.kind = "trend"

    def reason(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            assumptions=(
                SimpleNamespace(
                    assumption_id="a_trend",
                    kind=self.kind,
                    claim="The historical trend continues into the horizon.",
                    failure_condition="A future event reverses the trend.",
                    candidate_names=("numeric",),
                    prior_confidence=0.8,
                    supporting_call_ids=("private_tool_result",),
                ),
            )
        )


def test_morphology_adapter_calls_reasoner_with_only_numeric_view_and_safe_candidates():
    reasoner = _RecordingMorphologyReasoner()

    assumptions = MorphologyAdapter(reasoner).assumptions(
        _task(),
        (MorphologyCandidate("numeric", "generated"),),
    )

    assert reasoner.kwargs == {
        "history": tuple(float(value) for value in range(1, 21)),
        "frequency": "D",
        "horizon": 2,
        "active_names": ("numeric",),
        "families": {"numeric": "generated"},
    }
    assert [item.to_payload() for item in assumptions] == [
        {
            "assumption_id": "a_trend",
            "kind": "trend_persistence",
            "claim": "The historical trend continues into the horizon.",
            "failure_condition": "A future event reverses the trend.",
        }
    ]
```

Add focused validation tests:

```python
@pytest.mark.parametrize(
    ("name", "family"),
    (("", "generated"), ("not-a-python-name", "generated"), ("numeric", "")),
)
def test_morphology_candidate_rejects_invalid_safe_identity(name, family):
    with pytest.raises(RetrievalContractError):
        MorphologyCandidate(name, family)


def test_morphology_adapter_rejects_duplicate_candidate_names():
    with pytest.raises(RetrievalContractError, match="duplicate Morphology candidate"):
        MorphologyAdapter(_RecordingMorphologyReasoner()).assumptions(
            _task(),
            (
                MorphologyCandidate("numeric", "generated"),
                MorphologyCandidate("numeric", "statistical"),
            ),
        )
```

Add a parametrized projection test for the complete kind bridge:

```python
@pytest.mark.parametrize(
    ("morphology_kind", "retrieval_kind"),
    (
        ("seasonality", "seasonality"),
        ("trend", "trend_persistence"),
        ("intermittency", "other"),
        ("regime", "regime_persistence"),
        ("noise", "other"),
        ("level", "level_persistence"),
    ),
)
def test_morphology_adapter_translates_every_numerical_assumption_kind(
    morphology_kind, retrieval_kind
):
    reasoner = _RecordingMorphologyReasoner()
    reasoner.kind = morphology_kind

    assumption = MorphologyAdapter(reasoner).assumptions(
        _task(), (MorphologyCandidate("numeric", "generated"),)
    )[0]

    assert assumption.kind == retrieval_kind
```

- [ ] **Step 2: Run the new tests and verify the expected RED state**

Run:

```bash
python -m pytest -q \
  tests/test_two_stage_retrieval.py::test_morphology_adapter_calls_reasoner_with_only_numeric_view_and_safe_candidates \
  tests/test_two_stage_retrieval.py::test_morphology_candidate_rejects_invalid_safe_identity \
  tests/test_two_stage_retrieval.py::test_morphology_adapter_rejects_duplicate_candidate_names \
  tests/test_two_stage_retrieval.py::test_morphology_adapter_translates_every_numerical_assumption_kind
```

Expected: collection or import fails because `MorphologyCandidate` does not exist and the adapter
still requires `.run(task)`.

- [ ] **Step 3: Implement the minimal safe adapter**

Replace the preliminary interface in `evolving_loop/morphology_adapter.py` with:

```python
@dataclass(frozen=True)
class MorphologyCandidate:
    name: str
    family: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise RetrievalContractError(
                "Morphology candidate name must be a Python identifier"
            )
        if not isinstance(self.family, str) or not self.family.strip():
            raise RetrievalContractError("Morphology candidate family must not be empty")


@runtime_checkable
class MorphologyProvider(Protocol):
    def assumptions(
        self,
        task: ContextTask,
        candidates: tuple[MorphologyCandidate, ...],
    ) -> tuple[RetrievalAssumption, ...]: ...


@dataclass(frozen=True)
class MorphologyAdapter:
    reasoner: object

    def __post_init__(self) -> None:
        if not callable(getattr(self.reasoner, "reason", None)):
            raise RetrievalContractError(
                "Numerical morphology reasoner requires reason(...)"
            )

    def assumptions(
        self,
        task: ContextTask,
        candidates: tuple[MorphologyCandidate, ...],
    ) -> tuple[RetrievalAssumption, ...]:
        names = tuple(item.name for item in candidates)
        if not names or len(names) != len(set(names)):
            raise RetrievalContractError(
                "Morphology candidates must be nonempty with no duplicate Morphology candidate"
            )
        numeric = task.numeric_view()
        reason = getattr(self.reasoner, "reason")
        card = reason(
            history=numeric.history_values,
            frequency=numeric.frequency,
            horizon=numeric.prediction_length,
            active_names=names,
            families={item.name: item.family for item in candidates},
        )
        raw_assumptions = (
            card.get("assumptions")
            if isinstance(card, Mapping)
            else getattr(card, "assumptions", None)
        )
        if isinstance(raw_assumptions, (str, bytes)) or not isinstance(
            raw_assumptions, Sequence
        ):
            raise RetrievalContractError("Morphology Card requires assumptions")
        parsed = tuple(self._assumption(item) for item in raw_assumptions)
        identities = tuple(item.assumption_id for item in parsed)
        if not identities or len(identities) != len(set(identities)):
            raise RetrievalContractError(
                "Morphology assumptions must be nonempty and unique"
            )
        return parsed
```

Keep `_assumption()` as the only projection function. Add the translation table at module scope
and replace the existing static method with:

```python
_RETRIEVAL_KIND_BY_MORPHOLOGY_KIND = {
    "seasonality": "seasonality",
    "trend": "trend_persistence",
    "intermittency": "other",
    "regime": "regime_persistence",
    "noise": "other",
    "level": "level_persistence",
}


@staticmethod
def _assumption(raw: object) -> RetrievalAssumption:
    if isinstance(raw, Mapping):
        get = raw.get
    else:
        get = lambda field: getattr(raw, field, None)
    raw_kind = get("kind")
    kind = (
        _RETRIEVAL_KIND_BY_MORPHOLOGY_KIND.get(raw_kind, raw_kind)
        if isinstance(raw_kind, str)
        else raw_kind
    )
    return RetrievalAssumption.from_payload(
        {
            "assumption_id": get("assumption_id"),
            "kind": kind,
            "claim": get("claim"),
            "failure_condition": get("failure_condition"),
        }
    )
```

This mapping is semantic and fixed: Morphology emits future-facing persistence assumptions from
historical shape classes, while Retrieval requires its narrower query taxonomy. Unknown values
still flow into `RetrievalAssumption.from_payload()` and fail closed.

Export all three public boundary types:

```python
__all__ = ["MorphologyAdapter", "MorphologyCandidate", "MorphologyProvider"]
```

- [ ] **Step 4: Run the focused adapter tests and verify GREEN**

Run the Step 2 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Run the existing Morphology Reasoner tests**

Run:

```bash
python -m pytest -q tests/test_evolution_morphology.py
```

Expected: the existing history-only reasoner suite passes unchanged.

- [ ] **Step 6: Commit Task 1**

```bash
git add evolving_loop/morphology_adapter.py tests/test_two_stage_retrieval.py
git commit -m "feat(retrieval): add morphology boundary"
```

### Task 2: Pass safe Coding candidate identities through the Harness

**Files:**
- Modify: `evolving_loop/harness.py:15-280`
- Modify: `tests/test_two_stage_retrieval.py:120-480`
- Modify: `tests/test_retrieval_e2e.py` at every Morphology test double
- Modify: `tests/test_evolving_cli.py` at every explicit Morphology test double

**Interfaces:**
- Consumes: `MorphologyCandidate` and the two-argument `MorphologyProvider.assumptions()`
- Produces: `_morphology_candidates(coding) -> tuple[MorphologyCandidate, ...]`
- Preserves: the existing `morphology_provider_failed:<ExceptionType>` failure path

- [ ] **Step 1: Write the failing Harness projection test**

Add to `tests/test_two_stage_retrieval.py`:

```python
def test_harness_projects_only_candidate_name_and_family_to_morphology_provider():
    observed: dict[str, object] = {}

    class Provider:
        def assumptions(self, task, candidates):
            observed["task"] = task
            observed["candidates"] = candidates
            return (
                RetrievalAssumption(
                    "a_trend",
                    "trend_persistence",
                    "The trend persists.",
                    "A reversal invalidates the trend.",
                ),
            )

    retrieval = _agent([_round(_chain())])
    result = _harness(
        retrieval,
        [_decision(), _decision()],
        morphology=Provider(),
    ).run(_task())

    assert observed["task"] is not None
    assert observed["candidates"] == (
        MorphologyCandidate("numeric", "generated"),
    )
    assert vars(observed["candidates"][0]) == {
        "name": "numeric",
        "family": "generated",
    }
    assert result.forecast == (21.0, 22.0)
```

Add a conflicting duplicate test by returning two Coding candidates with the same name and
different sources; assert the run records
`morphology_provider_failed:RetrievalContractError`, makes no Round-2 call, and preserves the
selected numeric forecast.

- [ ] **Step 2: Run the Harness tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_two_stage_retrieval.py::test_harness_projects_only_candidate_name_and_family_to_morphology_provider \
  tests/test_two_stage_retrieval.py::test_two_stage_prompt_boundaries_and_decision_gap_projection \
  tests/test_two_stage_retrieval.py::test_morphology_runtime_failure_is_recorded_and_skips_round2
```

Expected: failures show that the Harness still calls `assumptions(task)` and does not create safe
candidate records.

- [ ] **Step 3: Implement deterministic candidate projection**

Import `MorphologyCandidate` and add this helper in `evolving_loop/harness.py`:

```python
def _morphology_candidates(
    coding: CodingEvolutionResult,
) -> tuple[MorphologyCandidate, ...]:
    projected: list[MorphologyCandidate] = []
    families: dict[str, str] = {}
    for item in coding.candidates:
        candidate = MorphologyCandidate(item.program.name, item.program.source)
        prior = families.get(candidate.name)
        if prior is None:
            families[candidate.name] = candidate.family
            projected.append(candidate)
        elif prior != candidate.family:
            raise RetrievalContractError(
                "duplicate Morphology candidate has conflicting families"
            )
    if not projected:
        raise RetrievalContractError("Morphology requires executed Coding candidates")
    return tuple(projected)
```

Keep projection and provider invocation inside the existing non-transient failure boundary:

```python
try:
    safe_candidates = _morphology_candidates(coding)
    assumptions = tuple(
        RetrievalAssumption.from_payload(
            item.to_payload() if isinstance(item, RetrievalAssumption) else item
        )
        for item in self.morphology.assumptions(task, safe_candidates)
    )
```

- [ ] **Step 4: Update all provider test doubles to the new exact signature**

Every test provider becomes:

```python
def assumptions(self, task, candidates):
    del task, candidates
    return ()
```

Change `_NumericalMorphology` into a `.reason(**kwargs)` fake or reuse
`_RecordingMorphologyReasoner`. Do not loosen the runtime-checkable Protocol to accept the old
one-argument signature.

Add one end-to-end Harness test that wraps the actual
`MorphologyReasoner(FakeLLMClient(...))`. Give its fake LLM a full-history tool action, a distinct
recent-window tool action, and a valid final `trend` assumption naming `numeric`; configure
provisional Decision to return a named gap and request more evidence. Assert that Retrieval makes
exactly two calls and that its Round-2 prompt contains `trend_persistence` but none of
`candidate_names`, `supporting_call_ids`, `prior_confidence`, or tool observations. This is the
acceptance test that the repository's real reasoner—not only an adapter-shaped fake—can trigger
Round 2.

- [ ] **Step 5: Run the focused two-stage and E2E suites**

Run:

```bash
python -m pytest -q \
  tests/test_two_stage_retrieval.py \
  tests/test_retrieval_e2e.py \
  tests/test_evolving_cli.py -k 'factory and morphology or two_stage'
```

Expected: all selected tests pass. The existing prompt-boundary test continues to prove Round 1
has no assumptions and Round 2 has only the four approved fields.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  evolving_loop/harness.py \
  tests/test_two_stage_retrieval.py \
  tests/test_retrieval_e2e.py \
  tests/test_evolving_cli.py
git commit -m "feat(retrieval): pass safe morphology inputs"
```

### Task 3: Add opt-in CLI construction across every unified worker path

**Files:**
- Modify: `evolving_loop/cli.py:150-4305`
- Modify: `tests/test_evolving_cli.py:40-330,2560-2815`

**Interfaces:**
- Produces: CLI option `--morphology-provider empty|reasoner`, default `empty`
- Produces: `_configured_morphology_provider(args, llm) -> MorphologyProvider`
- Changes: `_factory()` auto-constructs a provider only when the parsed configuration explicitly
  contains `morphology_provider`; programmatic callers without that field still fail closed
- Consumes: `MorphologyAdapter` and `numerical_agent.evolution.morphology.MorphologyReasoner`

- [ ] **Step 1: Write failing parser and factory tests**

Extend `test_retrieval_topology_controls_are_explicit_for_both_interfaces()`:

```python
assert legacy.morphology_provider == "empty"
reasoner = parser.parse_args([*prefix, "--morphology-provider", "reasoner"])
assert reasoner.morphology_provider == "reasoner"
```

Add a construction test:

```python
from evolving_loop.morphology_adapter import MorphologyAdapter
from numerical_agent.evolution.morphology import MorphologyReasoner


def test_two_stage_factory_constructs_opt_in_history_only_reasoner(tmp_path):
    release = write_retrieval_release(tmp_path / "releases", RetrievalGenome.seed())
    args = SimpleNamespace(
        setting="llm_only",
        retrieval_mode="two-stage",
        retrieval_release_path=release.path,
        morphology_provider="reasoner",
    )

    harness = _factory(
        args,
        FakeLLMClient([]),
        SkillLibrary(tmp_path / "coding.json"),
        RetrievalSkillLibrary(tmp_path / "retrieval.json"),
        DecisionSkillLibrary(tmp_path / "decision.json"),
        None,
        isolate_library=True,
    )(HarnessPolicy())

    assert isinstance(harness.morphology, MorphologyAdapter)
    assert isinstance(harness.morphology.reasoner, MorphologyReasoner)
```

Keep `test_factory_rejects_two_stage_without_morphology_provider()` unchanged to prove
programmatic callers without parsed configuration still fail closed.

- [ ] **Step 2: Run the new CLI tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_evolving_cli.py::test_retrieval_topology_controls_are_explicit_for_both_interfaces \
  tests/test_evolving_cli.py::test_two_stage_factory_constructs_opt_in_history_only_reasoner \
  tests/test_evolving_cli.py::test_factory_rejects_two_stage_without_morphology_provider
```

Expected: parser and construction tests fail because the option and centralized builder do not
exist.

- [ ] **Step 3: Add the parser option and centralized builder**

In `_add_retrieval_topology_arguments()` add:

```python
parser.add_argument(
    "--morphology-provider",
    choices=("empty", "reasoner"),
    default="empty",
    help=(
        "Use the compatibility empty provider or the opt-in history-only "
        "MorphologyReasoner for two-stage Retrieval."
    ),
)
```

Update `_ConservativeMorphologyProvider.assumptions()` to the two-argument protocol and add:

```python
def _configured_morphology_provider(args, llm) -> MorphologyProvider:
    configured = getattr(args, "morphology_provider", None)
    if configured == "empty":
        return _ConservativeMorphologyProvider()
    if configured == "reasoner":
        from numerical_agent.evolution.morphology import MorphologyReasoner

        return MorphologyAdapter(MorphologyReasoner(llm))
    raise ValueError("morphology_provider must be empty or reasoner")
```

In `_factory()`, before validating the two-stage provider:

```python
if retrieval_mode == "two-stage" and morphology_provider is None:
    if getattr(args, "morphology_provider", None) is not None:
        morphology_provider = _configured_morphology_provider(args, llm)
```

Remove direct `_ConservativeMorphologyProvider()` construction from Retrieval evolution,
Decision coordinate evolution, alternate coordinate evolution, and Retrieval frozen inference;
let `_factory()` use the parsed option consistently.

- [ ] **Step 4: Propagate the option to source workers**

Add `morphology_provider` to both `runtime_keys` tuples used by source evolution and source frozen
inference.

Change the source-inference test to create a real empty Retrieval release with
`write_retrieval_release()`. Its `fake_source_inference()` must reconstruct `worker_args`, call
`_factory()` without an explicit provider, build the Harness, assert
`isinstance(harness.morphology, _ConservativeMorphologyProvider)`, and return
`{"status": "provider_constructed"}`. Then assert:

```python
assert result == {"status": "provider_constructed"}
assert captured["morphology_provider"] == "empty"
```

Change the source-evolution test's fake subprocess runner to handle both operations explicitly:

```python
def capture_source_worker(command, *args, **kwargs):
    if list(command[:3]) == ["git", "diff", "--quiet"]:
        return subprocess.CompletedProcess(command, 0, "", "")
    if "evolving_loop.source_evolution.source_eval" in command:
        config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        captured.update(config["runtime"])
        evaluation = {
            "train": {
                "system_reward": 0.0,
                "module_rewards": {},
                "diagnostics": {"mean_smae": 1.0, "mean_srmse": 1.0},
                "failure_traces": [],
            },
            "dev": {
                "system_reward": 0.0,
                "module_rewards": {},
                "diagnostics": {"mean_smae": 1.0, "mean_srmse": 1.0},
            },
        }
        return subprocess.CompletedProcess(
            command, 0, json.dumps(evaluation) + "\n", ""
        )
    return real_run(command, *args, **kwargs)


class EvaluatingEngine:
    def __init__(self, repo_root, evaluate, config):
        del repo_root, config
        self.evaluate = evaluate

    def evolve(self, seed_patch=""):
        del seed_patch
        assert isinstance(self.evaluate(cli_module.Path.cwd()), SourceEvaluation)
        return "", ()
```

Assert `_source_evolve_command()` succeeds and
`captured["morphology_provider"] == "empty"`. This test verifies serialized worker configuration;
the source-inference test above verifies centralized worker-side construction.

- [ ] **Step 5: Run all CLI construction tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_evolving_cli.py -k \
  'retrieval_topology or morphology_provider or two_stage_factory or source_inference_propagates_two_stage or source_evolution_worker_receives_two_stage'
```

Expected: all selected tests pass.

- [ ] **Step 6: Run the CLI and Harness regression suites**

Run:

```bash
python -m pytest -q \
  tests/test_evolving_cli.py \
  tests/test_evolving_harness.py \
  tests/test_two_stage_retrieval.py \
  tests/test_frozen_inference.py \
  tests/test_retrieval_frozen_inference.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add evolving_loop/cli.py tests/test_evolving_cli.py
git commit -m "feat(cli): enable real morphology provider"
```

### Task 4: Bind provider identity to experiment artifacts and document usage

**Files:**
- Modify: `evolving_loop/cli.py` near `_sha256_sources()`, Retrieval hashes, trace payloads, and
  command summaries
- Modify: `evolving_loop/frozen_inference.py:345-608`
- Modify: `evolving_loop/source_evolution/frozen_runner.py:1-75`
- Modify: `tests/test_evolving_cli.py:785-835,1093-1225`
- Modify: `tests/test_frozen_inference.py:100-138`
- Modify: `README.md:164-190,381-387`
- Modify: `docs/SELF_EVOLUTION_FRAMEWORK.md:117-164`

**Interfaces:**
- Produces: `_morphology_provider_binding(args) -> dict[str, str]`
- Produces artifact shape:
  `{"name": "conservative_empty_v1|history_only_reasoner_v1", "source_sha256": "<64 hex>"}`
- Produces: `run_frozen_inference(..., summary_metadata=...)` with collision-safe metadata merged
  before atomic `summary.json` publication
- Preserves Retrieval release schema and ownership

- [ ] **Step 1: Write failing binding and artifact tests**

Add:

```python
def test_morphology_provider_binding_distinguishes_empty_and_reasoner_sources():
    empty = cli_module._morphology_provider_binding(
        SimpleNamespace(morphology_provider="empty")
    )
    reasoner = cli_module._morphology_provider_binding(
        SimpleNamespace(morphology_provider="reasoner")
    )

    assert empty["name"] == "conservative_empty_v1"
    assert reasoner["name"] == "history_only_reasoner_v1"
    assert len(empty["source_sha256"]) == 64
    assert len(reasoner["source_sha256"]) == 64
    assert empty["source_sha256"] != reasoner["source_sha256"]
```

In
`test_retrieval_evolution_publishes_only_accepted_release_and_keeps_traces_in_runs`, pass
`--morphology-provider reasoner` and assert:

```python
assert output["morphology_provider"]["name"] == "history_only_reasoner_v1"
trace = json.loads(Path(args.trace_path).read_text(encoding="utf-8"))
assert trace["morphology_provider"] == output["morphology_provider"]
assert len(trace["hashes"]["harness_sha256"]) == 64
```

Add to `tests/test_frozen_inference.py`:

```python
def test_frozen_summary_publishes_supplied_component_metadata(tmp_path):
    binding = {"name": "history_only_reasoner_v1", "source_sha256": "a" * 64}

    class Harness:
        def run(self, task, *, allow_skill_writes=True):
            assert allow_skill_writes is False
            return _result(task.numeric.task_id)

    summary = run_frozen_inference(
        HarnessPolicy(),
        [_task(public=False)],
        lambda _: Harness(),
        output_dir=tmp_path,
        samples=1,
        score_public=False,
        summary_metadata={"morphology_provider": binding},
    )

    published = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["morphology_provider"] == binding
    assert published["morphology_provider"] == binding
```

Use the test module's existing `_task()` and `_result()` helpers rather than creating a second
result shape. Add a second assertion or focused test that
`summary_metadata={"artifact_kind": "bad"}`
raises `ValueError` before publication, so callers cannot replace host-owned summary fields.

- [ ] **Step 2: Run the new artifact tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_evolving_cli.py::test_morphology_provider_binding_distinguishes_empty_and_reasoner_sources \
  tests/test_evolving_cli.py::test_retrieval_evolution_publishes_only_accepted_release_and_keeps_traces_in_runs \
  tests/test_frozen_inference.py::test_frozen_summary_publishes_supplied_component_metadata
```

Expected: failures show that the binding helper and artifact fields do not exist.

- [ ] **Step 3: Implement provider binding**

Add after `_sha256_sources()`:

```python
def _morphology_provider_binding(args) -> dict[str, str]:
    configured = getattr(args, "morphology_provider", "empty")
    paths = ["evolving_loop/cli.py", "evolving_loop/morphology_adapter.py"]
    if configured == "empty":
        name = "conservative_empty_v1"
    elif configured == "reasoner":
        name = "history_only_reasoner_v1"
        paths.extend(
            [
                "numerical_agent/evolution/morphology.py",
                "numerical_agent/evolution/analysis_skills_template.py",
                "numerical_agent/evolution/assumptions.py",
            ]
        )
    else:
        raise ValueError("morphology_provider must be empty or reasoner")
    return {"name": name, "source_sha256": _sha256_sources(*paths)}
```

Replace every hard-coded `conservative_empty_v1` hash input with the complete binding. Add the
binding to the Retrieval evolution trace/return value, coordinate evolution return values,
ordinary `run_command()` summaries, generic Genome/Source evolution summaries, and every frozen
inference summary. Do not add it to Retrieval release `manifest.json`.

Extend `run_frozen_inference()` without coupling it to Morphology:

```python
def run_frozen_inference(
    ...,
    summary_metadata: Mapping[str, object] | None = None,
) -> dict:
    ...
    metadata = dict(summary_metadata or {})
    reserved = set(metadata) & set(summary)
    if reserved:
        raise ValueError(
            "frozen summary metadata collides with reserved fields: "
            + ", ".join(sorted(reserved))
        )
    summary.update(metadata)
```

Perform this merge before `payloads` is encoded, so both the returned mapping and atomically
published `summary.json` contain identical metadata. Add
`summary_metadata={"morphology_provider": _morphology_provider_binding(args)}` to the CLI's direct
frozen-inference calls. In `source_evolution/frozen_runner.py`, import the same binding helper and
pass identical summary metadata to the inner frozen runner; its serialized runtime already carries
the provider choice from Task 3.

- [ ] **Step 4: Update documentation**

Document both commands:

```bash
# Existing compatible behavior; Round 2 skips without assumptions.
python -m evolving_loop --evolution retrieval \
  --retrieval-mode two-stage \
  --morphology-provider empty \
  ...

# Opt-in history-only tool reasoning; may trigger gap-directed Round 2.
python -m evolving_loop --evolution retrieval \
  --retrieval-mode two-stage \
  --morphology-provider reasoner \
  ...
```

State explicitly that `reasoner` is this repository's KairosAgent-inspired adapter, not the
authors' training code or a reproduction of T-STAR/SFT/GRPO/cross-modal fusion. State that no
forecasting improvement is claimed until a frozen experiment is run.

- [ ] **Step 5: Run focused verification**

Run:

```bash
python -m pytest -q \
  tests/test_evolution_morphology.py \
  tests/test_two_stage_retrieval.py \
  tests/test_evolving_harness.py \
  tests/test_evolving_cli.py \
  tests/test_retrieval_e2e.py \
  tests/test_retrieval_runner.py \
  tests/test_retrieval_verifier.py \
  tests/test_frozen_inference.py \
  tests/test_retrieval_frozen_inference.py
git diff --check
```

Expected: all selected tests pass and `git diff --check` prints nothing.

- [ ] **Step 6: Run full repository verification**

Run:

```bash
python -m pytest -q
git diff --check
```

Expected: the full suite passes with zero failures, and the diff check is clean. If unrelated
environment-dependent tests fail, preserve the exact output, run the smallest diagnostic command
that identifies the external dependency, and do not describe the full suite as passing.

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  evolving_loop/cli.py \
  tests/test_evolving_cli.py \
  README.md \
  docs/SELF_EVOLUTION_FRAMEWORK.md
git commit -m "docs: bind morphology experiment identity"
```

## Final Review Gate

After all four tasks:

1. Run `git status --short --branch` and confirm no user-owned files from the two pre-existing
   dirty worktrees entered this branch.
2. Run the full verification commands again in the same turn used to report completion.
3. Use the `requesting-code-review` skill to review the complete range from the post-cherry-pick
   base through `HEAD` against the approved spec.
4. Fix every Critical and Important finding with a new failing regression test where behavior
   changes, then rerun focused and full verification.
5. Use `finishing-a-development-branch` only after review and verification are clean.
