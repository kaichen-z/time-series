"""Hostile information-flow and non-vacuous Numerical QD algorithm gates."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from evolving_loop.v2 import cli as v2_cli
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.numerical_qd import hyperband, map_elites, mutation, nsga2
from evolving_loop.v2.numerical_qd.proposers import (
    LLMProposalProvider,
    REQUEST_KEYS,
    primitive_proposer_request,
)
from tests.build_evolution_v2_numerical_fixture import build as build_cli_fixture
from tests.test_evolution_v2_numerical_hyperband import (
    ADAPTER,
    DESCRIPTOR,
    METRIC,
    PROTOCOL,
    RUNTIME,
    SPLIT,
    manifest,
    sha,
    state,
)
from tests.test_evolution_v2_numerical_map_elites import Draws, entry
from tests.test_evolution_v2_numerical_mutation import feedback as mutation_feedback
from tests.test_evolution_v2_numerical_mutation import parent_state
from tests.test_evolution_v2_numerical_proposers import request_args
from tests.test_evolution_v2_numerical_runner import run_fixture


ROOT = Path(__file__).resolve().parents[1]


def _expect_raises(error, operation):
    try:
        operation()
    except error:
        return
    names = (error,) if isinstance(error, type) else error
    raise AssertionError(
        "expected " + " or ".join(candidate.__name__ for candidate in names)
    )


def _all_file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "location,key,value",
    [
        ("root", "future_values", [917_263_541.125]),
        ("feedback", "dev_metrics", "DEV_22f776a9"),
        ("cell", "public_ids", "PUBLIC_34346e91"),
        ("feedback", "evaluator_labels", "LABEL_696ff9df"),
        ("budget", "path", Path("PATH_f3187ed5")),
        ("genome", "kernel", object()),
        ("state", "store", object()),
        ("root", "callback", lambda: None),
    ],
)
def test_hostile_capability_or_evaluator_sentinel_is_rejected_before_provider(
    location, key, value
):
    """Removing exact/deep validation would call the external provider."""

    class CountingClient:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            raise AssertionError("hostile input reached the provider capability")

    args = request_args()
    targets = {
        "root": args,
        "state": args["parent_state"],
        "cell": args["selected_cells"][0],
        "feedback": args["train_feedback"][0],
        "budget": args["remaining_budget"],
        "genome": args["parent_genome"],
    }
    targets[location][key] = value
    client = CountingClient()
    _expect_raises(
        (TypeError, ValueError), lambda: LLMProposalProvider(client).propose(args)
    )
    assert client.calls == 0


def _numerical_artifact_groups(root: Path) -> dict[str, dict[str, bytes]]:
    all_files = _all_file_bytes(root)
    numerical = {
        name: raw for name, raw in all_files.items() if name.startswith("numerical_qd/")
    }
    json_objects = {}
    for name, raw in numerical.items():
        if name.endswith((".json", ".jsonl")):
            for index, line in enumerate(raw.splitlines()):
                if line:
                    json_objects[f"{name}:{index}"] = json.loads(line)

    def selected(predicate):
        return {
            name: canonical_v2_bytes(payload)
            for name, payload in json_objects.items()
            if predicate(payload)
        }

    return {
        "proposer_request": selected(
            lambda payload: isinstance(payload, dict) and set(payload) == REQUEST_KEYS
        ),
        "qd_object": selected(
            lambda payload: isinstance(payload, dict)
            and {"capacity", "cells", "entries", "insertion_log"} <= set(payload)
        ),
        "checkpoint": {
            name: raw for name, raw in all_files.items() if name.endswith("checkpoint.json")
        },
        "prompt": selected(
            lambda payload: isinstance(payload, dict)
            and {"template", "response_schema", "parent_prompt_sha256"} <= set(payload)
        ),
        "mutation_policy": selected(
            lambda payload: isinstance(payload, dict)
            and set(payload) == {"schema_version", "operators"}
        ),
        "progress": selected(
            lambda payload: isinstance(payload, dict) and set(payload) == {"numerical_qd_step"}
        ),
        "cache_row": {
            name: raw
            for name, raw in numerical.items()
            if name.startswith("numerical_qd/results/")
        },
        "source": {name: raw for name, raw in numerical.items() if name.endswith(".py")},
    }


@pytest.mark.parametrize(
    "sentinel_kind,sentinel",
    [
        ("future", 917_263_541.125),
        ("evaluator_label", "EVALUATOR_LABEL_4c943e21"),
    ],
)
def test_trusted_evaluator_sees_sentinel_but_every_numerical_artifact_excludes_it(
    tmp_path, monkeypatch, sentinel_kind, sentinel
):
    """Bypassing the trusted evaluator or persisting its input fails by kind."""

    inputs = tmp_path / "inputs"
    build_cli_fixture(inputs)
    config_path = inputs / "smoke.json"
    config = json.loads(
        (ROOT / "configs/evolution_v2/numerical_qd/smoke.json").read_bytes()
    )
    config_path.write_bytes(canonical_v2_bytes(config))
    task_path = inputs / "task_manifest.json"
    tasks = json.loads(task_path.read_bytes())
    if sentinel_kind == "future":
        tasks["train"][0]["future_values"] = [sentinel, sentinel]
    else:
        tasks["train"][0]["gt_evidence"] = [sentinel]
    task_path.write_bytes(canonical_v2_bytes(tasks))

    from evolving_loop.v2.numerical_qd import runner

    evaluated, requests = [], []
    original_evaluate = runner.evaluate_numerical_child
    original_request = runner.primitive_proposer_request

    def trusted_evaluator(adapter, child, committed, **kwargs):
        host_task = next(
            task for task in adapter.tasks if task.numeric.task_id == committed.task_id
        )
        if sentinel_kind == "future":
            if sentinel in host_task.numeric.future_values:
                evaluated.append(committed.task_id)
        elif sentinel in host_task.gt_evidence:
            evaluated.append(committed.task_id)
        return original_evaluate(adapter, child, committed, **kwargs)

    def proposer_boundary(**payload):
        request = original_request(**payload)
        assert str(sentinel).encode() not in canonical_v2_bytes(request)
        requests.append(request)
        return request

    monkeypatch.setattr(runner, "evaluate_numerical_child", trusted_evaluator)
    monkeypatch.setattr(runner, "primitive_proposer_request", proposer_boundary)
    output = tmp_path / "run"
    assert v2_cli.main(
        [
            "numerical-evolve",
            "--config",
            str(config_path),
            "--seed-supply",
            str(inputs / "seed_supply.json"),
            "--task-manifest",
            str(task_path),
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert evaluated and requests

    groups = _numerical_artifact_groups(output)
    encoded = str(sentinel).encode()
    for kind, artifacts in groups.items():
        assert artifacts, f"run produced no {kind} artifacts"
        assert all(encoded not in raw for raw in artifacts.values()), kind


def test_real_run_artifacts_exclude_environment_path_public_and_dev_sentinels(
    tmp_path, monkeypatch
):
    """Leaking Host context into any durable Numerical artifact fails this scan."""

    environment = "ENV_SECRET_c848257a"
    dev = "DEV_KERNEL_ONLY_b1c13fa0"
    output = tmp_path / "PATH_HOST_ONLY_d221503b" / "run"
    monkeypatch.setenv("EVOLUTION_V2_HOSTILE_SECRET", environment)

    from evolving_loop.v2.kernel import EvolutionKernel

    original = EvolutionKernel.close_evaluation

    def inject_dev(kernel, parent, child, **kwargs):
        comparison = kwargs["dev_comparison"]
        if comparison["parent_metrics"] or comparison["candidate_metrics"]:
            kwargs["dev_comparison"] = comparison | {
                "candidate_metrics": comparison["candidate_metrics"] | {"sentinel": dev}
            }
        return original(kernel, parent, child, **kwargs)

    monkeypatch.setattr(EvolutionKernel, "close_evaluation", inject_dev)
    run_fixture(output, task_budget=920)

    artifacts = _all_file_bytes(output)
    numerical = {name: raw for name, raw in artifacts.items() if name.startswith("numerical_qd/")}
    decoded = {name: raw.decode("utf-8") for name, raw in artifacts.items()}
    numerical_decoded = {name: raw.decode("utf-8") for name, raw in numerical.items()}

    assert any("proposals/" in name for name in numerical)
    assert any("/results/" in f"/{name}" for name in numerical)
    assert any(name.endswith("checkpoint.json") for name in artifacts)
    assert any('"entries"' in raw and '"cells"' in raw for raw in numerical_decoded.values())
    assert any('"operators"' in raw and '"credit"' in raw for raw in numerical_decoded.values())
    assert any('"response_schema"' in raw and '"template"' in raw for raw in numerical_decoded.values())
    assert any('"numerical_qd_step"' in raw for raw in numerical_decoded.values())
    assert any(name.endswith(".py") for name in numerical)
    assert any('"cache_key"' in raw for raw in numerical_decoded.values())

    forbidden = (environment, output.parent.name)
    for name, raw in decoded.items():
        assert all(sentinel not in raw for sentinel in forbidden), name
    locations = [name for name, raw in decoded.items() if dev in raw]
    assert locations
    assert all(name.startswith("acceptance/") for name in locations)
    assert all(dev not in raw for raw in numerical_decoded.values())


def _guard_constraint_first_ordering():
    feasible = entry("a", score=99.0)
    infeasible = entry("b", score=0.0, violations=("coverage",))
    assert nsga2.constraint_compare(feasible, infeasible) == -1


def test_gate_constraint_first_ordering():
    """Objective-first ranking would let an infeasible child win."""

    _guard_constraint_first_ordering()


def _guard_empty_category_normalization():
    pools = {
        "underexplored": (),
        "elite": ("elite",),
        "failure_matched": ("failure",),
        "stepping_stone": (),
    }
    assert map_elites._sample_category(pools, Draws((50, 29))) == "elite"
    assert map_elites._sample_category(pools, Draws((50, 30))) == "failure_matched"
    assert map_elites._sample_category(pools, Draws((50, 49))) == "failure_matched"


def test_gate_empty_parent_categories_are_renormalized():
    """Sampling over the unfiltered 100-point mass would select an empty pool."""

    _guard_empty_category_normalization()


def _guard_sha_tie_break():
    tied = tuple(entry(marker) for marker in "abc")
    expected = tuple(sorted(tied, key=lambda item: item.fingerprint()))[:2]
    assert nsga2.select_survivors(tuple(reversed(tied)), 2) == expected


def test_gate_sha_is_the_final_survivor_tie_break():
    """Input order or reverse-SHA tie breaking changes retained QD bytes."""

    _guard_sha_tie_break()


def _guard_cache_key_binding():
    identity = {
        "candidate": sha("candidate"),
        "task": sha("task"),
        "split": SPLIT,
        "metric": METRIC,
        "descriptor": DESCRIPTOR,
        "runtime": RUNTIME,
        "protocol": PROTOCOL,
        "adapter": ADAPTER,
    }
    baseline = hyperband.evaluation_cache_key(**identity)
    replacements = {
        "candidate": sha("changed candidate"),
        "task": sha("changed task"),
        "split": sha("changed split"),
        "metric": sha("changed metric"),
        "descriptor": sha("changed descriptor"),
        "runtime": {"python": sha("changed runtime")},
        "protocol": sha("changed protocol"),
        "adapter": sha("changed adapter"),
    }
    for field, changed in replacements.items():
        assert hyperband.evaluation_cache_key(**(identity | {field: changed})) != baseline


def test_gate_cache_key_binds_every_execution_dependency():
    """Dropping any one identity permits an invalid successful cache hit."""

    _guard_cache_key_binding()


def _guard_rung_checkpoint_binding():
    current = state()
    wrong = replace(manifest(), resource=32)
    _expect_raises(
        ValueError,
        lambda: hyperband._validate_next_manifest(current, wrong),
    )


def test_gate_rung_checkpoint_binds_the_exact_next_manifest():
    """Skipping next-rung validation permits replay under a different resource."""

    _guard_rung_checkpoint_binding()


def _guard_train_only_policy_credit():
    parent = parent_state()
    hostile = mutation_feedback(split="dev", inserted=True)
    _expect_raises(ValueError, lambda: mutation.record_train_outcome(parent, hostile))
    assert mutation.record_train_outcome(
        parent, mutation_feedback(inserted=True)
    ).mutation_policy.operators["fork"].credit == 1


def test_gate_only_train_insertions_credit_mutation_policy():
    """Accepting a Dev outcome would contaminate mutation memory."""

    _guard_train_only_policy_credit()


MUTATIONS = {
    "constraint_first": "nsga2._compare = lambda left, right: 0",
    "category_normalization": "map_elites._sample_category = lambda pools, stream: ('underexplored' if stream.randbelow(100) < 40 else 'elite')",
    "sha_tie_break": "nsga2._crowding_order = lambda front: tuple(sorted(front, key=lambda item: item.fingerprint(), reverse=True))",
    "cache_key_binding": "original = hyperband.evaluation_cache_key\nhyperband.evaluation_cache_key = lambda candidate, task, split, metric, descriptor, runtime, protocol, adapter: original(candidate, task, split, metric, descriptor, runtime, protocol, ADAPTER)",
    "rung_checkpoint_binding": "hyperband._validate_next_manifest = lambda state, manifest: None",
    "train_only_credit": "mutation.TrainMutationFeedbackV2 = type('MutantFeedback', (), {'from_payload': staticmethod(lambda payload: type('Outcome', (), payload)())})",
}


@pytest.mark.parametrize(
    "mutation_name,guard",
    [
        ("constraint_first", "_guard_constraint_first_ordering"),
        ("category_normalization", "_guard_empty_category_normalization"),
        ("sha_tie_break", "_guard_sha_tie_break"),
        ("cache_key_binding", "_guard_cache_key_binding"),
        ("rung_checkpoint_binding", "_guard_rung_checkpoint_binding"),
        ("train_only_credit", "_guard_train_only_policy_credit"),
    ],
)
def test_named_algorithm_gate_kills_isolated_mutation(mutation_name, guard):
    """Each required bad algorithm dies in a fresh interpreter, not shared state."""

    script = f"""
import sys
sys.path.insert(0, {str(ROOT / "tests")!r})
from tests import test_evolution_v2_numerical_safety as gates
from evolving_loop.v2.numerical_qd import hyperband, map_elites, mutation, nsga2
from tests.test_evolution_v2_numerical_hyperband import ADAPTER
{MUTATIONS[mutation_name]}
try:
    gates.{guard}()
except AssertionError:
    print('MUTATION_CAUGHT:{mutation_name}')
else:
    raise SystemExit('mutation survived: {mutation_name}')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == f"MUTATION_CAUGHT:{mutation_name}"
