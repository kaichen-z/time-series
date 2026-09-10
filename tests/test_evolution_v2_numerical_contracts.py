from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload
from evolving_loop.v2.numerical_qd.contracts import (
    ConstraintReportV2,
    MorphologyCellV2,
    MutationOperatorStatsV2,
    NumericalEvaluationV2,
    NumericalGenomeV2,
    NumericalInventoryV2,
    NumericalMemberV2,
    NumericalMutationPolicyV2,
    NumericalObjectiveVectorV2,
    NumericalProposerPromptV2,
    NumericalQDEntryV2,
)


SHA = "a" * 64
OTHER_SHA = "b" * 64
OPERATORS = ("add", "combine", "crossover", "fork", "policy_tune", "quarantine",
             "remove", "repair", "route", "specialize")


def payloads():
    cell = dict(trend="low", seasonality="none", intermittency="low",
                regime="stable", horizon="short", family="statistical")
    objectives = dict(mean_capped_smae=0.5, mean_capped_srmse=0.6,
                      p95_capped_srmse=0.9, mean_raw_joint_error=0.7,
                      normalized_execution_cost=0.1)
    constraints = dict(feasible=True, violations=[])
    member = dict(member_id="anchor", family="statistical", source_sha256=SHA,
                  policy_sha256=OTHER_SHA, parent_ids=[], applicability_cells=[],
                  status="active")
    stats = dict(attempts=4, feasible=3, promotions=2, insertions=1, credit=5)
    genome = dict(schema_version=1, generation=0, parent_genome_sha256s=[],
                  mutation_operator="add", inventory_sha256=SHA,
                  screening_policy_sha256=SHA, combined_policy_sha256=SHA,
                  mutation_policy_sha256=SHA, proposer_prompt_sha256=SHA,
                  runtime_fingerprints={"python": SHA}, protocol_fingerprint=SHA)
    prompt = dict(schema_version=1, template="Propose one allowed mutation.",
                  response_schema="numerical_mutation_batch_v1", max_response_bytes=4096,
                  allowed_mutation_operators=list(OPERATORS), parent_prompt_sha256=None)
    evaluation = dict(
        schema_version=1, genome_sha256=SHA, supply_sha256=SHA, registry_sha256=SHA,
        task_subset_sha256=SHA, split_sha256=SHA, metric_policy_sha256=SHA,
        descriptor_policy_sha256=SHA, execution_adapter_sha256=SHA,
        runtime_fingerprints={"python": SHA}, protocol_fingerprint=SHA,
        split="train", bracket="explore", rung=0, task_ids=["task-a", "task-b"],
        task_statuses={"task-a": "passed", "task-b": "passed"},
        objectives=objectives, constraints=constraints, cells=[cell],
        train_diagnostic_categories=[], cache_sha256s=[SHA],
        resource_use=dict(wall_seconds=1.0, task_executions=2, llm_calls=0,
                          input_tokens=0, output_tokens=0, gpu_seconds=0.0,
                          subprocesses=0, artifact_bytes=0))
    entry = dict(schema_version=1, genome_sha256=SHA, evaluation_sha256=SHA,
                 cell=cell, task_ids=["task-a"], objectives=objectives,
                 constraints=constraints, train_diagnostic_categories=[])
    return {
        MorphologyCellV2: cell,
        NumericalObjectiveVectorV2: objectives,
        ConstraintReportV2: constraints,
        NumericalMemberV2: member,
        NumericalInventoryV2: dict(schema_version=1, members=[member]),
        NumericalGenomeV2: genome,
        MutationOperatorStatsV2: stats,
        NumericalMutationPolicyV2: dict(schema_version=1,
                                      operators={op: dict(stats) for op in OPERATORS}),
        NumericalProposerPromptV2: prompt,
        NumericalEvaluationV2: evaluation,
        NumericalQDEntryV2: entry,
    }


CLASSES = tuple(payloads())


@pytest.mark.parametrize("cls", CLASSES)
def test_all_artifacts_round_trip_exact_canonical_payload(cls):
    payload = copy.deepcopy(payloads()[cls])
    artifact = cls.from_payload(payload)
    assert artifact.to_payload() == payload
    assert cls.from_payload(artifact.to_payload()) == artifact
    assert artifact.canonical_bytes() == canonical_v2_bytes(payload)
    assert artifact.fingerprint() == hashlib.sha256(artifact.canonical_bytes()).hexdigest()
    assert not hasattr(artifact, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        setattr(artifact, next(iter(payload)), None)


SCHEMA_CASES = [(cls, name) for cls, payload in payloads().items() for name in payload]


@pytest.mark.parametrize("cls,field", SCHEMA_CASES)
def test_every_missing_field_is_rejected(cls, field):
    payload = copy.deepcopy(payloads()[cls])
    del payload[field]
    with pytest.raises(ValueError, match="exact schema"):
        cls.from_payload(payload)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("extra", ["unknown", "dev_metrics", 1])
def test_every_artifact_rejects_unknown_or_nonstring_fields(cls, extra):
    with pytest.raises(ValueError):
        cls.from_payload(payloads()[cls] | {extra: None})


def mutate_containers(value):
    if isinstance(value, dict):
        for nested in list(value.values()):
            mutate_containers(nested)
        value["injected"] = {"future_values": [123]}
    elif isinstance(value, list):
        for nested in value:
            mutate_containers(nested)
        value.append("injected")


@pytest.mark.parametrize("cls", CLASSES)
def test_mutating_input_and_exported_nested_containers_cannot_change_artifact(cls):
    payload = copy.deepcopy(payloads()[cls])
    artifact = cls.from_payload(payload)
    before = artifact.canonical_bytes()
    mutate_containers(payload)
    mutate_containers(artifact.to_payload())
    assert artifact.canonical_bytes() == before


def test_numerical_genome_is_canonical_and_deeply_immutable():
    genome = NumericalGenomeV2.from_payload(payloads()[NumericalGenomeV2])
    assert NumericalGenomeV2.from_payload(genome.to_payload()) == genome
    assert genome.fingerprint() == fingerprint_payload(genome.to_payload())
    with pytest.raises(TypeError):
        genome.runtime_fingerprints["python"] = "f" * 64
    policy = NumericalMutationPolicyV2.from_payload(payloads()[NumericalMutationPolicyV2])
    with pytest.raises(TypeError):
        policy.operators["add"] = MutationOperatorStatsV2(0, 0, 0, 0, 0)
    with pytest.raises(AttributeError):
        policy.operators["add"].credit = 0
    evaluation = NumericalEvaluationV2.from_payload(payloads()[NumericalEvaluationV2])
    with pytest.raises(TypeError):
        evaluation.resource_use["task_executions"] = 100
    with pytest.raises(TypeError):
        evaluation.task_statuses["task-a"] = "invalid"


@pytest.mark.parametrize("field", tuple(payloads()[NumericalObjectiveVectorV2]))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_objective_contract_rejects_nonfinite_or_boolean_numbers(field, value):
    with pytest.raises((TypeError, ValueError)):
        NumericalObjectiveVectorV2(**(payloads()[NumericalObjectiveVectorV2] | {field: value}))


SHA_CASES = [(cls, field) for cls, payload in payloads().items()
             for field in payload if field.endswith("_sha256") or field == "protocol_fingerprint"]


@pytest.mark.parametrize("cls,field", SHA_CASES)
@pytest.mark.parametrize("value", ["bad", "A" * 64, "g" * 64, True, 1])
def test_all_scalar_identity_fields_require_canonical_sha256(cls, field, value):
    with pytest.raises((TypeError, ValueError)):
        cls.from_payload(payloads()[cls] | {field: value})


INVALID_CASES = [
    (NumericalMemberV2, "family", "neural"),
    (NumericalMemberV2, "status", "deleted"),
    (NumericalMemberV2, "member_id", ""),
    (NumericalMemberV2, "parent_ids", ["z", "a"]),
    (NumericalMemberV2, "parent_ids", ["a", "a"]),
    (NumericalMemberV2, "parent_ids", ["anchor"]),
    (NumericalMemberV2, "applicability_cells", [OTHER_SHA, SHA]),
    (NumericalMemberV2, "applicability_cells", [SHA, SHA]),
    (NumericalMemberV2, "applicability_cells", ["bad"]),
    (NumericalGenomeV2, "parent_genome_sha256s", [OTHER_SHA, SHA]),
    (NumericalGenomeV2, "parent_genome_sha256s", [SHA, SHA]),
    (NumericalGenomeV2, "parent_genome_sha256s", ["bad"]),
    (NumericalGenomeV2, "mutation_operator", "rewrite_kernel"),
    (NumericalGenomeV2, "runtime_fingerprints", {}),
    (NumericalGenomeV2, "runtime_fingerprints", {"python": "bad"}),
    (NumericalGenomeV2, "runtime_fingerprints", {"": SHA}),
    (ConstraintReportV2, "feasible", 1),
    (ConstraintReportV2, "violations", ["unknown"]),
    (ConstraintReportV2, "violations", ["coverage"]),
    (ConstraintReportV2, "feasible", False),
    (NumericalProposerPromptV2, "template", ""),
    (NumericalProposerPromptV2, "template", "x" * 65537),
    (NumericalProposerPromptV2, "response_schema", "arbitrary_python"),
    (NumericalProposerPromptV2, "max_response_bytes", 0),
    (NumericalProposerPromptV2, "max_response_bytes", True),
    (NumericalProposerPromptV2, "max_response_bytes", 1048577),
    (NumericalProposerPromptV2, "allowed_mutation_operators", []),
    (NumericalProposerPromptV2, "allowed_mutation_operators", ["add", "add"]),
    (NumericalProposerPromptV2, "allowed_mutation_operators", ["rewrite_kernel"]),
    (NumericalEvaluationV2, "split", "dev"),
    (NumericalEvaluationV2, "split", "public"),
    (NumericalEvaluationV2, "bracket", "other"),
    (NumericalEvaluationV2, "rung", 3),
    (NumericalEvaluationV2, "task_ids", ["task-b", "task-a"]),
    (NumericalEvaluationV2, "task_ids", ["task-a", "task-a"]),
    (NumericalEvaluationV2, "task_ids", []),
    (NumericalEvaluationV2, "task_statuses", {"task-a": "passed"}),
    (NumericalEvaluationV2, "task_statuses", {"task-a": "oops", "task-b": "passed"}),
    (NumericalEvaluationV2, "runtime_fingerprints", {"python": "bad"}),
    (NumericalEvaluationV2, "cache_sha256s", ["bad"]),
    (NumericalEvaluationV2, "cache_sha256s", [SHA, SHA]),
    (NumericalEvaluationV2, "train_diagnostic_categories", ["dev_metrics"]),
    (NumericalQDEntryV2, "train_diagnostic_categories", ["future_values"]),
    (NumericalQDEntryV2, "task_ids", []),
]


@pytest.mark.parametrize("cls,field,value", INVALID_CASES)
def test_invalid_contract_values_are_rejected(cls, field, value):
    with pytest.raises((TypeError, ValueError)):
        cls.from_payload(payloads()[cls] | {field: value})


@pytest.mark.parametrize("field", tuple(payloads()[MorphologyCellV2]))
def test_every_cell_dimension_rejects_unknown_values(field):
    with pytest.raises(ValueError):
        MorphologyCellV2.from_payload(payloads()[MorphologyCellV2] | {field: "unknown"})


@pytest.mark.parametrize("cls,field", [(MutationOperatorStatsV2, field) for field in payloads()[MutationOperatorStatsV2]]
                         + [(NumericalGenomeV2, "generation"), (NumericalEvaluationV2, "rung")])
@pytest.mark.parametrize("value", [-1, True, 1.0])
def test_counters_are_nonnegative_integers(cls, field, value):
    with pytest.raises(ValueError):
        cls.from_payload(payloads()[cls] | {field: value})


@pytest.mark.parametrize("cls", [cls for cls, payload in payloads().items() if "schema_version" in payload])
@pytest.mark.parametrize("value", [True, 1.0, 0, 2])
def test_schema_version_is_exact_integer_one(cls, value):
    with pytest.raises(ValueError):
        cls.from_payload(payloads()[cls] | {"schema_version": value})


def test_inventory_rejects_duplicate_ids_and_no_active_member():
    member = payloads()[NumericalMemberV2]
    for members in ([], [member, member], [member | {"status": "quarantined"}],
                    [member | {"status": "specialized"}]):
        with pytest.raises(ValueError):
            NumericalInventoryV2(1, members)


def test_constraint_names_are_closed_sorted_unique_and_consistent():
    for name in ("coverage", "nonfinite_forecast", "wrong_horizon", "new_invalid_task",
                 "new_catastrophic_task", "joint_regret", "protocol_mismatch",
                 "runtime_mismatch", "cache_mismatch", "ownership", "scope",
                 "future_leakage", "dev_leakage", "document_role_leakage", "public_leakage"):
        assert ConstraintReportV2(False, [name]).violations == (name,)
    for violations in (["coverage", "coverage"], ["scope", "coverage"]):
        with pytest.raises(ValueError):
            ConstraintReportV2(False, violations)


def test_nested_artifacts_reject_unknown_and_missing_fields():
    cases = [(NumericalInventoryV2, ("members", 0)),
             (NumericalMutationPolicyV2, ("operators", "add")),
             (NumericalEvaluationV2, ("objectives",)),
             (NumericalEvaluationV2, ("constraints",)),
             (NumericalEvaluationV2, ("cells", 0)),
             (NumericalEvaluationV2, ("resource_use",)),
             (NumericalQDEntryV2, ("cell",)),
             (NumericalQDEntryV2, ("objectives",)),
             (NumericalQDEntryV2, ("constraints",))]
    for cls, path in cases:
        original = copy.deepcopy(payloads()[cls])
        nested = original
        for key in path:
            nested = nested[key]
        for field in ("extra", *nested):
            changed = copy.deepcopy(original)
            target = changed
            for key in path:
                target = target[key]
            if field == "extra":
                target[field] = True
            else:
                del target[field]
            with pytest.raises(ValueError, match="exact schema"):
                cls.from_payload(changed)


def test_policy_rejects_empty_unknown_and_invalid_operator_stats():
    for operators in ({}, {"rewrite_kernel": payloads()[MutationOperatorStatsV2]},
                      {"add": {"credit": 1}}):
        with pytest.raises(ValueError):
            NumericalMutationPolicyV2(1, operators)


def test_evaluation_cells_must_be_sorted_by_fingerprint_and_unique():
    cell = payloads()[MorphologyCellV2]
    other = cell | {"family": "program"}
    ordered = sorted([cell, other], key=fingerprint_payload)
    payload = payloads()[NumericalEvaluationV2]
    assert NumericalEvaluationV2.from_payload(payload | {"cells": ordered})
    for cells in ([cell, cell], list(reversed(ordered)), []):
        with pytest.raises(ValueError):
            NumericalEvaluationV2.from_payload(payload | {"cells": cells})


@pytest.mark.parametrize("field,value", [("wall_seconds", float("inf")),
                                         ("gpu_seconds", float("nan")),
                                         ("task_executions", True),
                                         ("artifact_bytes", -1)])
def test_evaluation_resource_use_is_strict_and_finite(field, value):
    payload = copy.deepcopy(payloads()[NumericalEvaluationV2])
    payload["resource_use"][field] = value
    with pytest.raises(ValueError):
        NumericalEvaluationV2.from_payload(payload)


@pytest.mark.parametrize("value", [Path("source.py"), lambda: None, object()])
def test_live_objects_are_rejected_in_nested_payloads(value):
    payload = payloads()[NumericalGenomeV2] | {"runtime_fingerprints": {"python": value}}
    with pytest.raises((TypeError, ValueError)):
        NumericalGenomeV2.from_payload(payload)


def test_canonical_artifacts_are_identical_in_two_fresh_interpreters():
    fixtures = {cls.__name__: payload for cls, payload in payloads().items()}
    script = """
import json, sys
from evolving_loop.v2.numerical_qd import contracts
for name, payload in sorted(json.loads(sys.argv[1]).items()):
    artifact = getattr(contracts, name).from_payload(payload)
    print(artifact.canonical_bytes().hex(), artifact.fingerprint())
"""
    commands = [sys.executable, "-c", script, json.dumps(fixtures)]
    first = subprocess.run(commands, check=True, capture_output=True)
    second = subprocess.run(commands, check=True, capture_output=True)
    assert first.stdout == second.stdout
    assert len(first.stdout.splitlines()) == len(CLASSES)
    for line in first.stdout.splitlines():
        raw_hex, digest = line.split()
        assert hashlib.sha256(bytes.fromhex(raw_hex.decode())).hexdigest() == digest.decode()
