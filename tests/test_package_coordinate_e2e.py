from __future__ import annotations

import json
from dataclasses import replace

from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.llm import FakeLLMClient, LLMResponse
from evolving_loop.co_evolution import (
    CoEvolutionConfig,
    HarnessPolicy,
    embed_retrieval_release,
)
from evolving_loop.coordinate_evolution import (
    DecisionEvolutionPhaseAdapter,
    RetrievalEvolutionPhaseAdapter,
)
from evolving_loop.data import ContextTask, Document
from evolving_loop.decision_agent.agent import DecisionAgent
from evolving_loop.package_coordinate_evolution import (
    PackageCoordinateBundle,
    PackageCoordinateController,
)
from evolving_loop.package_decision_evolution import (
    PackageDecisionEvaluator,
    PackageDecisionEvolutionEngine,
)
from evolving_loop.package_registry import FrozenNumericalPackageRegistry
from evolving_loop.package_retrieval_evolution import PackageRetrievalEvaluator
from evolving_loop.retrieval_agent.evolution import (
    RetrievalEvolutionConfig,
    RetrievalEvolutionEngine,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
    write_retrieval_release,
)
from evolving_loop.retrieval_agent.skill_library import RetrievalSkillLibrary
from evolving_loop.retrieval_agent.two_stage_agent import TwoStageRetrievalAgent
from numerical_agent.evolution.execution import Task
from numerical_agent.evolution.numerical_handoff import task_input_fingerprint
from numerical_agent.evolution.screening import profile_task
from tests.test_package_decision_evolution import (
    _decision_task,
    _safe_default_package,
)
from tests.test_package_retrieval_evolution import _frozen_registry


_RETRIEVAL_DEPENDENCIES = {
    "retrieval_factory": "1" * 64,
    "decision_factory": "2" * 64,
    "bridge_runtime": "3" * 64,
}


def _context_task(index: int, *, split: str) -> ContextTask:
    template = ContextTask(
        numeric=replace(
            # Reuse the frozen package's exact history and resolved target.
            # Identity and entity diversity are supplied below.
            _decision_task().numeric,
            task_id=f"{split}-{index:03d}",
            entity_name=(
                f"Train Entity {index // 8:02d}"
                if split == "train"
                else f"Dev Entity {index // 2:02d}"
            ),
        ),
        target_name="sales",
        target_description="Daily sales",
        history_timestamps=tuple(
            f"2026-01-{position + 1:02d}" for position in range(36)
        ),
        future_timestamps=("2026-02-06", "2026-02-07"),
        documents=(),
        gt_evidence=(),
        labels_public=True,
    )
    entity = template.numeric.entity_name
    support = (
        f"{entity} sales remain at one and two units during the forecast window."
    )
    cycle = (
        f"A scheduled seasonal cycle will persist and increase {entity} sales by "
        "1 unit from 2026-02-06 through 2026-02-07."
    )
    return replace(
        template,
        documents=(
            Document("support", support, role="supporting"),
            Document("cycle_support", cycle, role="supporting"),
            Document(
                "distractor",
                "Another entity's inventory changes during an unrelated window.",
                role="distractor",
            ),
        ),
        gt_evidence=(support,),
    )


def _champion_package(task: ContextTask):
    template = _safe_default_package()
    numeric = task.numeric
    profile = profile_task(
        Task(
            numeric.task_id,
            numeric.history_values,
            numeric.prediction_length,
            numeric.frequency,
            (),
        )
    )
    fingerprints = {
        **dict(template.component_fingerprints),
        "task_input": task_input_fingerprint(
            task_id=numeric.task_id,
            history=numeric.history_values,
            frequency=numeric.frequency,
            horizon=numeric.prediction_length,
        ),
    }
    return replace(
        template,
        task_profile=profile,
        component_fingerprints=fingerprints,
    )


def _chain(
    *,
    chain_id: str,
    document_id: str,
    content: str,
    start: str,
    end: str,
    addressed_assumption_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "chain_id": chain_id,
        "claim": content,
        "entity_match": True,
        "target_match": True,
        "temporal_relation": "overlaps_future",
        "mechanism": "future_driver",
        "direction": "up" if addressed_assumption_ids else "stable",
        "magnitude_kind": "absolute" if addressed_assumption_ids else "none",
        "magnitude_value": 1.0 if addressed_assumption_ids else None,
        "start_timestamp": start,
        "end_timestamp": end,
        "citations": [{"document_id": document_id, "exact_quote": content}],
        "missing_links": [],
        "used_skill_ids": [],
        "addressed_assumption_ids": list(addressed_assumption_ids),
        "stance": "supports",
        "numeric_eligible": True,
    }


class _DeterministicRetrievalLLM:
    def complete(self, *, system, messages, temperature=0.0) -> LLMResponse:
        del system, temperature
        payload = json.loads(messages[0]["content"])
        documents = {
            item["document_id"]: item["content"] for item in payload["documents"]
        }
        start, end = payload["target"]["forecast_window"]
        if "assumptions" in payload:
            chains = [
                _chain(
                    chain_id="cycle_chain",
                    document_id="cycle_support",
                    content=documents["cycle_support"],
                    start=start,
                    end=end,
                    addressed_assumption_ids=("assumption_001",),
                )
            ]
        else:
            chains = [
                _chain(
                    chain_id="support_chain",
                    document_id="support",
                    content=documents["support"],
                    start=start,
                    end=end,
                )
            ]
        return LLMResponse(
            json.dumps(
                {
                    "evidence_chains": chains,
                    "counterevidence": [],
                    "missing_information": [],
                    "sufficient": True,
                }
            )
        )


class _DeterministicDecisionLLM:
    def __init__(self, *, specialist: bool, selected: list[str]) -> None:
        self.specialist = specialist
        self.selected = selected
        self.calls = 0

    def complete(self, *, system, messages, temperature=0.0) -> LLMResponse:
        del system, messages, temperature
        if not self.specialist:
            candidate = "safe_anchor"
            payload = {
                "selected_candidate_id": candidate,
                "supporting_document_ids": [],
                "rationale": "Retain the frozen safe default.",
                "request_more_retrieval": False,
                "gaps": [],
                "used_skill_names": [],
            }
        elif self.calls == 0:
            candidate = "specialist"
            payload = {
                "selected_candidate_id": candidate,
                "supporting_document_ids": ["support"],
                "rationale": "Request the named seasonal discriminator.",
                "request_more_retrieval": True,
                "gaps": [
                    {
                        "assumption_id": "assumption_001",
                        "gap_type": "continuation_or_reversal",
                        "missing_information": "Evidence the seasonal cycle persists",
                        "priority": "high",
                    }
                ],
                "used_skill_names": [],
            }
        else:
            candidate = "specialist"
            payload = {
                "selected_candidate_id": candidate,
                "supporting_document_ids": ["cycle_support"],
                "rationale": "Verified evidence supports the materialized specialist.",
                "request_more_retrieval": False,
                "gaps": [],
                "used_skill_names": [],
            }
        self.calls += 1
        self.selected.append(candidate)
        return LLMResponse(json.dumps(payload))


def _retrieval_proposals(parent: RetrievalGenome) -> list[str]:
    payloads: list[dict[str, object]] = []
    for index, scope in enumerate(("A", "B", "C"), start=1):
        payload = parent.to_payload()
        payload.update({"version": f"v{index:03d}", "parent": parent.version})
        if scope == "A":
            payload["round1_prompt"] = (
                f"{parent.round1_prompt}\nUse verified evidence to prefer specialist."
            )
        elif scope == "B":
            payload["max_citations_per_chain"] = 2
        else:
            payload["round2_strategy"] = "gap_first"
        payloads.append(payload)
    return [json.dumps(payload) for payload in payloads]


def _accepted_audit(registry: FrozenNumericalPackageRegistry) -> dict[str, object]:
    return {
        "state": "accepted",
        "train_dev_split_sha256": registry.fingerprint,
        "verifier_sha256": "4" * 64,
        "evaluator_sha256": "5" * 64,
        "metric_sha256": METRIC_POLICY_FINGERPRINT,
        "metric_cap": 5.0,
        "train_summary": {"task_count": 80},
        "dev_summary": {"task_count": 20},
        "acceptance_reason": "package Train/Dev gates passed",
    }


def test_package_coordinate_evolution_closes_deterministic_80_20_loop(
    tmp_path, monkeypatch
) -> None:
    train = tuple(_context_task(index, split="train") for index in range(80))
    dev = tuple(_context_task(index, split="dev") for index in range(20))
    entries = tuple((task, _champion_package(task)) for task in (*train, *dev))
    registry = _frozen_registry(entries)
    replica = _frozen_registry(entries)
    selected_candidates: list[str] = []
    active_genome: list[RetrievalGenome] = []
    accepted_results = []
    accepted_release_paths = []

    skills = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False).clone(
        read_only=True
    )

    def retrieval_factory(genome: RetrievalGenome, library):
        active_genome[:] = [genome]
        return TwoStageRetrievalAgent(
            _DeterministicRetrievalLLM(),
            genome,
            library,
        )

    def retrieval_decision_factory():
        specialist = bool(
            active_genome
            and "prefer specialist" in active_genome[0].round1_prompt
        )
        return DecisionAgent(
            _DeterministicDecisionLLM(
                specialist=specialist,
                selected=selected_candidates,
            )
        )

    retrieval_evaluator = PackageRetrievalEvaluator(
        registry,
        retrieval_factory,
        retrieval_decision_factory,
        dependency_fingerprints=_RETRIEVAL_DEPENDENCIES,
    )
    seed = RetrievalGenome.seed()
    retrieval_engine = RetrievalEvolutionEngine(
        FakeLLMClient(_retrieval_proposals(seed)),
        retrieval_evaluator,
        RetrievalEvolutionConfig(
            generations=1,
            promote=1,
            train_folds=4,
            strict_gain_target="final",
            transient_retries=0,
        ),
        skill_library=skills,
    )
    releases = tmp_path / "releases"
    seed_release = write_retrieval_release(releases, seed)
    parent_policy = embed_retrieval_release(
        HarnessPolicy(),
        seed_release,
        changelog="Frozen package Retrieval seed.",
    )

    def publish_accepted(result):
        accepted_results.append(result)
        release = _write_accepted_retrieval_release(
            releases,
            result.release_genome,
            audit=_accepted_audit(registry),
        )
        accepted_release_paths.append(release.path)
        return release.path

    retrieval_phase = RetrievalEvolutionPhaseAdapter(
        retrieval_engine,
        parent_release_path=seed_release.path,
        accepted_release_path=publish_accepted,
    )

    def decision_retrieval_factory(policy: HarnessPolicy):
        return TwoStageRetrievalAgent(
            _DeterministicRetrievalLLM(),
            policy.retrieval_genome,
            skills,
        )

    def decision_factory(policy: HarnessPolicy):
        return DecisionAgent(
            _DeterministicDecisionLLM(
                specialist="prefer specialist" in policy.decision_prompt,
                selected=selected_candidates,
            ),
            prompt=policy.decision_prompt,
        )

    decision_evaluator = PackageDecisionEvaluator(
        registry,
        decision_retrieval_factory,
        decision_factory,
        dependency_fingerprints=_RETRIEVAL_DEPENDENCIES,
    )
    decision_engine = PackageDecisionEvolutionEngine(
        FakeLLMClient(
            [
                json.dumps(
                    {
                        "decision_prompt": (
                            "prefer specialist using verified seasonal evidence"
                        ),
                        "changelog": "Use the materialized specialist when verified.",
                    }
                )
            ]
        ),
        decision_evaluator,
        CoEvolutionConfig(
            generations=1,
            children_per_generation=1,
            mode="genome",
            target="decision",
            screening_tolerance=1e-12,
        ),
    )
    decision_phase = DecisionEvolutionPhaseAdapter(
        decision_engine,
        accepted_release_path=lambda _policy: accepted_release_paths[0],
    )
    parent = PackageCoordinateBundle(
        generation=0,
        parent_sha256=None,
        numerical_manifest_sha256=registry.fingerprint,
        policy=parent_policy,
        runtime_fingerprints={
            "bridge_runtime": _RETRIEVAL_DEPENDENCIES["bridge_runtime"],
            "retrieval_runtime": _RETRIEVAL_DEPENDENCIES["retrieval_factory"],
            "decision_runtime": _RETRIEVAL_DEPENDENCIES["decision_factory"],
            "retrieval_verifier": retrieval_evaluator.verifier_hash,
            "metric_policy": METRIC_POLICY_FINGERPRINT,
        },
    )

    numerical_calls = 0

    def forbidden_numerical_loop(*_args, **_kwargs):
        nonlocal numerical_calls
        numerical_calls += 1
        raise AssertionError("package evolution must not rerun Numerical")

    monkeypatch.setattr(
        "numerical_agent.evolution.numerical_loop.run_numerical_loop",
        forbidden_numerical_loop,
    )
    selected, trace = PackageCoordinateController(
        retrieval_phase,
        decision_phase,
    ).run(parent, train, dev)

    assert len(train) == 80 and len(dev) == 20
    assert registry.fingerprint == replica.fingerprint
    assert tuple(step.target for step in trace) == ("retrieval", "decision")
    assert all(step.accepted for step in trace)
    assert all(not step.public_test_accessed for step in trace)
    assert selected.numerical_manifest_sha256 == registry.fingerprint
    assert selected.policy.retrieval_genome.version == "v001"
    assert selected.policy.version == "v002"
    assert accepted_results[0].accepted is True
    assert accepted_results[0].parent_dev.task_count == 20
    assert numerical_calls == 0
    assert set(selected_candidates) <= {"safe_anchor", "specialist"}
