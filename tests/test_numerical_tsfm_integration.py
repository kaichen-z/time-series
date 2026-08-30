from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from common.evolution_core.contracts import metric_policy_metadata
from numerical_agent import main as numerical_main
from numerical_agent.dictionary import ToolDictionary
from numerical_agent.main import main
from numerical_agent.tsfm.deployment import TSFMDeployment
from numerical_agent.tsfm.protocol import WorkerResponse


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _file_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_mixed_fake_worker_curation_and_frozen_evaluation_cover_runtime_contracts(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """Breaks if family routing, typed availability, history-only choice, or Frozen mutates."""

    adapter_requests: list[str] = []
    broker_instances = []
    runtime_validation_calls: list[tuple[str, ...]] = []

    class NoDownloadBroker:
        def __init__(
            self, _commands, *, timeout_seconds, parent_environment, redactor
        ) -> None:
            assert timeout_seconds == 300.0
            assert parent_environment["PATH"]
            self.redactor = redactor
            self.close_count = 0
            broker_instances.append(self)

        def request(self, _worker_key, request):
            adapter_requests.append(request.provider)
            if request.provider == "dedicated":
                return WorkerResponse.failure(
                    request.request_id,
                    "unavailable",
                    "checkpoint_unavailable",
                    "fixture checkpoint is intentionally absent",
                )
            step = request.history[-1] - request.history[-2]
            return WorkerResponse.success(
                request.request_id,
                [
                    request.history[-1] + step * offset
                    for offset in range(1, request.horizon + 1)
                ],
            )

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(numerical_main, "WorkerBroker", NoDownloadBroker)

    def validate_runtime(self, environment_keys=None, *, parent_environment=None):
        del parent_environment
        runtime_validation_calls.append(
            tuple(self.commands) if environment_keys is None else tuple(environment_keys)
        )

    monkeypatch.setattr(TSFMDeployment, "validate_runtime", validate_runtime)

    deployment_path = tmp_path / "workers.json"
    _write_json(
        deployment_path,
        {
            "schema_version": 1,
            "environments": {
                "timesfm_v1": {"interpreter": sys.executable},
                "uni2ts": {"interpreter": sys.executable},
                "granite_tsfm": {"interpreter": sys.executable},
                "transformers_recent": {"interpreter": sys.executable},
                "toto2": {"interpreter": sys.executable},
            },
        },
    )

    rejected = TSFMDeployment.load(deployment_path)
    accepted = TSFMDeployment.load(
        deployment_path,
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )
    assert "method_tsfm_0003" not in rejected.enabled_manifest_ids
    assert "method_tsfm_0003" in accepted.enabled_manifest_ids

    methods = [
        {
            "method_id": "statistical_fixture",
            "family": "statistical",
            "description": "A future-favored constant that loses on observed hindcasts.",
            "implementation_spec": {"prediction": 0.0},
        },
        *(
            {
                "method_id": method_id,
                "family": "foundation",
                "description": "A manifest-bound integration fixture.",
                "implementation_spec": {},
            }
            for method_id in (
                "method_tsfm_0001",  # legacy
                "method_tsfm_0003",  # uni2ts
                "method_tsfm_0006",  # granite
                "method_tsfm_0008",  # transformer_generation
                "method_tsfm_0014",  # dedicated, typed unavailable
                "method_tsfm_0005",  # precise catalog unavailability
            )
        ),
    ]
    base_methods = tmp_path / "base-methods.json"
    _write_json(
        base_methods,
        ToolDictionary.from_legacy_report_payload({
            "schema_version": 1,
            "dictionary_id": "mixed-integration.v000",
            "parent_dictionary_id": None,
            "generation": 0,
            "methods": methods,
        }).to_payload(),
    )

    task = {
        "item_id": "trend",
        "history": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "horizon": 2,
        "frequency": "D",
    }
    experiment_path = tmp_path / "experiment.json"
    _write_json(
        experiment_path,
        {
            "evolution": {
                "schema_version": 2,
                **metric_policy_metadata(),
                "generations": 1,
                "children_per_generation": 1,
                "resume": False,
            },
            "curation": {
                "schema_version": 2,
                **metric_policy_metadata(),
                "allowed_families": ["statistical", "foundation"],
                "accepted_max_smae": 5.0,
                "accepted_max_srmse": 5.0,
                "specialized_max_smae": 5.0,
                "specialized_max_srmse": 5.0,
                "selection_folds": 2,
                "selection_horizon": 1,
            },
            "tasks": {"train": [task], "dev": [task]},
            # The future favors the statistical constant. Selection must still use history.
            "labels": {"train": {"trend": [0.0, 0.0]}, "dev": {"trend": [0.0, 0.0]}},
        },
    )

    curation_output = tmp_path / "curation"
    assert main(
        [
            "curate",
            "--experiment-config", str(experiment_path),
            "--base-methods", str(base_methods),
            "--provider", "fake",
            "--output-dir", str(curation_output),
            "--tsfm-workers-config", str(deployment_path),
            "--acknowledged-model-licenses", "CC-BY-NC-4.0",
        ]
    ) == 0
    capsys.readouterr()

    assert set(adapter_requests) == {
        "legacy",
        "uni2ts",
        "granite",
        "transformer_generation",
        "dedicated",
    }
    evaluation = json.loads(
        (curation_output / "method_evaluations.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    diagnostics = evaluation["diagnostics"]
    assert diagnostics["selected_method_ids"] == {"trend": "method_tsfm_0001"}
    assert diagnostics["oracle_score"] == 0.0
    dedicated = diagnostics["per_method"]["method_tsfm_0014"]
    assert dedicated["unavailable_count"] == 1
    assert dedicated["sample_errors"] == [
        "checkpoint_unavailable: fixture checkpoint is intentionally absent"
    ]

    working_dictionary = curation_output / "working_dictionary.json"
    working = json.loads(working_dictionary.read_text(encoding="utf-8"))
    moment = next(
        record for record in working["methods"]
        if record["definition"]["method_id"] == "method_tsfm_0005"
    )
    # The capped pair ties the all-missing Parent at the cap, so Pareto acceptance
    # rejects the child and the published working dictionary remains exact Parent.
    assert moment["candidate"] is None
    assert moment["status"] == "unimplemented"

    frozen_tasks = tmp_path / "frozen-tasks.jsonl"
    frozen_tasks.write_text(
        json.dumps(
            {
                "benchmark_id": "public_trend",
                "series": {
                    "history_values": task["history"],
                    "future_values": [0.0, 0.0],
                },
                "task_metadata": {"frequency": "D", "prediction_length": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_split = tmp_path / "frozen-split.json"
    _write_json(
        frozen_split,
        {
            "manifest_sha256": "fixture-public-manifest",
            "partitions": {"public_test": {"task_ids": ["public_trend"]}},
        },
    )
    before = _file_hashes(curation_output)
    frozen_output = tmp_path / "frozen"
    assert main(
        [
            "evaluate-frozen",
            "--tasks-file", str(frozen_tasks),
            "--split-file", str(frozen_split),
            "--experiment-config", str(experiment_path),
            "--dictionary", str(working_dictionary),
            "--output-dir", str(frozen_output),
            "--tsfm-workers-config", str(deployment_path),
            "--acknowledged-model-licenses", "CC-BY-NC-4.0",
        ]
    ) == 0
    capsys.readouterr()

    assert _file_hashes(curation_output) == before
    assert {path.name for path in frozen_output.iterdir()} == {
        "frozen_test_forecasts.jsonl",
        "frozen_test_report.json",
    }
    assert all(instance.close_count == 1 for instance in broker_instances)
    assert runtime_validation_calls == [
        (
            "timesfm_v1",
            "uni2ts",
            "granite_tsfm",
            "transformers_recent",
            "toto2",
        ),
        (
            "timesfm_v1",
            "uni2ts",
            "granite_tsfm",
            "transformers_recent",
            "toto2",
        ),
    ]
