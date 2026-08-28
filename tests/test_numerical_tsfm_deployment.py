from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import venv

import pytest

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.providers import RuntimeUnavailableError
from numerical_agent.tsfm.broker import WorkerBroker, WorkerCommand, WorkerMethodRuntime
from numerical_agent.tsfm.deployment import (
    TSFMDeployment,
    parse_acknowledged_licenses,
)
from numerical_agent.tsfm.manifests import ManifestRegistry
from numerical_agent.tsfm.protocol import WorkerRequest, WorkerResponse


def _write_config(path: Path, environments: dict[str, object]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "environments": environments}),
        encoding="utf-8",
    )
    return path


def _candidate(method_id: str) -> MethodCandidate:
    manifest = ManifestRegistry.load_default()[method_id]
    return MethodCandidate(
        method_id=method_id,
        provider="tsfm_worker",
        implementation_kind="tsfm_checkpoint",
        implementation=manifest.candidate_binding(),
    )


def test_deployment_builds_only_fixed_reviewed_worker_commands(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "workers.json",
        {
            "uni2ts": {"interpreter": sys.executable},
            "timer_legacy": {"interpreter": sys.executable},
        },
    )

    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    expected_interpreter = os.path.normpath(str(Path(sys.executable).expanduser()))
    assert deployment.commands["uni2ts"].argv == (
        expected_interpreter,
        "-m",
        "numerical_agent.tsfm.worker_main",
        "--adapter",
        "uni2ts",
    )
    assert deployment.commands["timer_legacy"].argv[-1] == "transformer_generation"
    assert deployment.enabled_manifest_ids == frozenset(
        {
            "method_tsfm_0003",
            "method_tsfm_0007",
            "method_tsfm_0013",
            "method_tsfm_0017",
            "method_tsfm_0019",
        }
    )


def test_gated_workers_are_disabled_without_exact_local_acknowledgement(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path / "workers.json",
        {
            "uni2ts": {"interpreter": sys.executable},
            "granite_tsfm": {"interpreter": sys.executable},
        },
    )

    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=(),
    )

    assert deployment.enabled_manifest_ids == frozenset({"method_tsfm_0006"})


def test_explicit_empty_manifest_registry_denies_all_worker_environments(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path / "workers.json",
        {"timesfm_v1": {"interpreter": sys.executable}},
    )

    with pytest.raises(ValueError, match="unknown worker environments"):
        TSFMDeployment.load(config, manifests=ManifestRegistry({}))


def test_filtered_manifest_registry_enables_only_its_reviewed_cards(
    tmp_path: Path,
) -> None:
    default = ManifestRegistry.load_default()
    manifests = ManifestRegistry({"method_tsfm_0003": default["method_tsfm_0003"]})
    config = _write_config(
        tmp_path / "workers.json",
        {"uni2ts": {"interpreter": sys.executable}},
    )

    deployment = TSFMDeployment.load(
        config,
        manifests=manifests,
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    assert deployment.enabled_manifest_ids == frozenset({"method_tsfm_0003"})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("CC-BY-NC-4.0, CC-BY-NC-4.0", "duplicate"),
        ("CC-BY-NC-4.0,,NXAI Community License", "empty"),
        ("made-up-license", "unknown"),
    ],
)
def test_license_acknowledgement_list_is_strict(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as raised:
        parse_acknowledged_licenses(value, ManifestRegistry.load_default())
    if value == "made-up-license":
        assert value not in str(raised.value)


def test_license_acknowledgement_list_trims_exact_identifiers() -> None:
    assert parse_acknowledged_licenses(
        " CC-BY-NC-4.0 , NXAI Community License ",
        ManifestRegistry.load_default(),
    ) == frozenset({"CC-BY-NC-4.0", "NXAI Community License"})
    assert parse_acknowledged_licenses("", ManifestRegistry.load_default()) == frozenset()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 1}, "missing fields"),
        ({"schema_version": 1, "environments": {}, "model": "unsafe"}, "unknown fields"),
        (
            {
                "schema_version": 1,
                "environments": {"unknown": {"interpreter": sys.executable}},
            },
            "unknown worker environments",
        ),
        (
            {
                "schema_version": 1,
                "environments": {
                    "uni2ts": {
                        "interpreter": sys.executable,
                        "checkpoint": "attacker/model",
                    }
                },
            },
            "unknown fields",
        ),
        (
            {
                "schema_version": 1,
                "environments": {"uni2ts": {"command": [sys.executable, "-c", "bad"]}},
            },
            "missing fields",
        ),
        (
            {
                "schema_version": 1,
                "environments": {"uni2ts": {"interpreter": "relative/python"}},
            },
            "absolute",
        ),
    ],
)
def test_deployment_rejects_unreviewed_or_incomplete_configuration(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    config = tmp_path / "workers.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TSFMDeployment.load(config, manifests=ManifestRegistry.load_default())


def test_deployment_rejects_duplicate_environment_keys(tmp_path: Path) -> None:
    config = tmp_path / "workers.json"
    config.write_text(
        '{"schema_version":1,"environments":{'
        f'"uni2ts":{{"interpreter":{json.dumps(sys.executable)}}},'
        f'"uni2ts":{{"interpreter":{json.dumps(sys.executable)}}}'
        "}}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate field"):
        TSFMDeployment.load(config, manifests=ManifestRegistry.load_default())


def test_deployment_rejects_missing_and_non_executable_interpreters(
    tmp_path: Path,
) -> None:
    missing = _write_config(
        tmp_path / "missing.json",
        {"uni2ts": {"interpreter": str(tmp_path / "absent-python")}},
    )
    with pytest.raises(ValueError, match="does not exist"):
        TSFMDeployment.load(missing, manifests=ManifestRegistry.load_default())

    non_executable = tmp_path / "python"
    non_executable.write_text("not executable", encoding="utf-8")
    non_executable.chmod(0o600)
    config = _write_config(
        tmp_path / "nonexec.json",
        {"uni2ts": {"interpreter": str(non_executable)}},
    )
    with pytest.raises(ValueError, match="not executable"):
        TSFMDeployment.load(config, manifests=ManifestRegistry.load_default())


def test_runtime_validation_rejects_a_stripped_virtual_environment(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "stripped-worker"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    interpreter = environment / "bin" / "python"
    interpreter.symlink_to(sys.executable)
    config = _write_config(
        tmp_path / "workers.json",
        {"uni2ts": {"interpreter": str(interpreter)}},
    )

    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    with pytest.raises(ValueError, match="requires an explicit virtual environment"):
        deployment.validate_runtime()


def test_runtime_validation_rejects_missing_reviewed_dependencies(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "empty-worker"
    venv.EnvBuilder(with_pip=False).create(environment)
    interpreter = environment / "bin" / "python"
    config = _write_config(
        tmp_path / "workers.json",
        {"uni2ts": {"interpreter": str(interpreter)}},
    )

    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    with pytest.raises(
        ValueError,
        match=r"missing reviewed dependencies: .*numerical_agent\.tsfm\.worker_main",
    ):
        deployment.validate_runtime()


def test_runtime_validation_rejects_a_non_venv_system_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "system-prefix"
    (environment / "bin").mkdir(parents=True)
    interpreter = environment / "bin" / "python"
    interpreter.symlink_to(sys.executable)
    config = _write_config(
        tmp_path / "workers.json",
        {"uni2ts": {"interpreter": str(interpreter)}},
    )
    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    monkeypatch.setattr(
        "numerical_agent.tsfm.deployment.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_prefix": str(environment),
                    "executable": str(interpreter),
                    "missing": [],
                    "prefix": str(environment),
                }
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires an explicit virtual environment"):
        deployment.validate_runtime()


def test_runtime_validation_rejects_system_site_packages_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "system-site-worker"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    interpreter = environment / "bin" / "python"
    config = _write_config(
        tmp_path / "workers.json",
        {"uni2ts": {"interpreter": str(interpreter)}},
    )
    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    monkeypatch.setattr(
        "numerical_agent.tsfm.deployment.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_prefix": sys.base_prefix,
                    "executable": str(interpreter),
                    "missing": [],
                    "prefix": str(environment),
                }
            ),
        ),
    )

    with pytest.raises(ValueError, match="must isolate system site-packages"):
        deployment.validate_runtime()


def test_runtime_validation_probes_exact_lazy_backend_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "worker"
    venv.EnvBuilder(with_pip=False).create(environment)
    interpreter = environment / "bin" / "python"
    config = _write_config(
        tmp_path / "workers.json",
        {"uni2ts": {"interpreter": str(interpreter)}},
    )
    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=("CC-BY-NC-4.0",),
    )

    def probe(argv, **_kwargs):
        dependency = "uni2ts.model.moirai2"
        missing = [dependency] if dependency in argv[-1] else []
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_prefix": sys.base_prefix,
                    "executable": str(interpreter),
                    "missing": missing,
                    "prefix": str(environment),
                }
            ),
        )

    monkeypatch.setattr("numerical_agent.tsfm.deployment.subprocess.run", probe)

    with pytest.raises(ValueError, match="uni2ts.model.moirai2"):
        deployment.validate_runtime()


def test_runtime_validation_probes_tabpfn_license_error_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "tabpfn-worker"
    venv.EnvBuilder(with_pip=False).create(environment)
    interpreter = environment / "bin" / "python"
    config = _write_config(
        tmp_path / "workers.json",
        {"tabpfn_ts": {"interpreter": str(interpreter)}},
    )
    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=(
            "TabPFN-3 Non-Commercial License; Apache-2.0 code",
        ),
    )

    def probe(argv, **_kwargs):
        required_symbols = json.loads(argv[-1])
        required = {
            "TabPFNHuggingFaceGatedRepoError",
            "TabPFNLicenseError",
        }
        present = set(required_symbols.get("tabpfn.errors", ()))
        missing = [
            f"tabpfn.errors:{name}" for name in sorted(required - present)
        ]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "base_prefix": sys.base_prefix,
                    "executable": str(interpreter),
                    "missing": missing,
                    "prefix": str(environment),
                }
            ),
        )

    monkeypatch.setattr("numerical_agent.tsfm.deployment.subprocess.run", probe)

    deployment.validate_runtime()


def test_worker_failure_redacts_token_bearing_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "fixture-super-secret-token"
    monkeypatch.setenv("HF_TOKEN", token)

    class FailingBroker:
        def request(self, worker_key, request):
            del worker_key
            return WorkerResponse.failure(
                request.request_id,
                "unavailable",
                "checkpoint_unavailable",
                f"download failed using {token}",
            )

    runtime = WorkerMethodRuntime(
        FailingBroker(),  # type: ignore[arg-type]
        manifests=ManifestRegistry.load_default(),
        enabled_manifest_ids={"method_tsfm_0001"},
    )

    with pytest.raises(RuntimeUnavailableError) as raised:
        runtime.forecast(_candidate("method_tsfm_0001"), [1.0], 1, "D")
    assert token not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


def test_worker_inherits_needed_token_but_broker_response_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "fixture-inherited-worker-token"
    monkeypatch.setenv("HF_TOKEN", token)
    fixture_worker = Path(__file__).parent / "fixtures/tsfm_worker.py"
    broker = WorkerBroker(
        {
            "fixture": WorkerCommand(
                (sys.executable, str(fixture_worker), "inherited_token_failure")
            )
        },
        timeout_seconds=5.0,
    )
    try:
        response = broker.request(
            "fixture",
            WorkerRequest(
                request_id="secret-test",
                provider="fixture",
                checkpoint="fixture/checkpoint",
                history=(1.0,),
                horizon=1,
                frequency="D",
            ),
        )
    finally:
        broker.close()

    assert response.message == "upstream included [REDACTED]"
    assert token not in response.to_json()


def test_worker_environment_is_controlled_and_response_is_recursively_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    short = "overlap-secret"
    long = f"prefix-{short}-suffix"
    excluded = "excluded-aws-credential"
    monkeypatch.setenv("HF_TOKEN", short)
    monkeypatch.setenv("TABPFN_TOKEN", long)
    monkeypatch.setenv("GITHUB_PAT", "excluded-github-credential")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "excluded-legacy-hf-credential")
    monkeypatch.setenv("SSH_PRIVATE_KEY", "excluded-ssh-credential")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", excluded)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    fixture_worker = Path(__file__).parent / "fixtures/tsfm_worker.py"
    broker = WorkerBroker(
        {
            "fixture": WorkerCommand(
                (sys.executable, str(fixture_worker), "environment_snapshot")
            )
        },
        timeout_seconds=5.0,
    )
    request = WorkerRequest(
        request_id="controlled-environment",
        provider="fixture",
        checkpoint="fixture/checkpoint",
        history=(1.0,),
        horizon=2,
        frequency=excluded,
    )
    try:
        first = broker.request("fixture", request)
        monkeypatch.delenv("HF_TOKEN")
        monkeypatch.delenv("TABPFN_TOKEN")
        monkeypatch.setenv("RENAMED_TOKEN", short)
        second = broker.request("fixture", request)
    finally:
        broker.close()

    for response in (first, second):
        serialized = response.to_json()
        assert response.metadata["credentials_present"] is True
        assert response.metadata["excluded_present"] is False
        assert response.metadata["safe_environment"] == {
            "HOME": os.environ["HOME"],
            "PATH": os.environ["PATH"],
            "HF_HOME": str(tmp_path / "hf-cache"),
        }
        assert short not in serialized
        assert long not in serialized
        assert excluded not in serialized
        assert "excluded-github-credential" not in serialized
        assert "excluded-legacy-hf-credential" not in serialized
        assert "excluded-ssh-credential" not in serialized
        assert serialized.count("[REDACTED]") >= 3


def test_malformed_worker_response_redacts_snapshotted_secret_after_parent_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "malformed-snapshotted-secret"
    monkeypatch.setenv("HF_TOKEN", token)
    fixture_worker = Path(__file__).parent / "fixtures/tsfm_worker.py"
    broker = WorkerBroker(
        {
            "fixture": WorkerCommand(
                (sys.executable, str(fixture_worker), "malformed_secret_status")
            )
        },
        timeout_seconds=5.0,
    )
    monkeypatch.delenv("HF_TOKEN")
    try:
        with pytest.raises(RuntimeUnavailableError) as raised:
            broker.request(
                "fixture",
                WorkerRequest(
                    request_id="malformed-secret",
                    provider="fixture",
                    checkpoint="fixture/checkpoint",
                    history=(1.0,),
                    horizon=1,
                    frequency="D",
                ),
            )
    finally:
        broker.close()

    assert token not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


def test_worker_redacts_direct_failure_response_before_serializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main

    token = "fixture-direct-response-token"
    monkeypatch.setenv("HF_TOKEN", token)

    class DirectFailureAdapter:
        def forecast(self, request):
            return WorkerResponse.failure(
                request.request_id,
                "unavailable",
                "checkpoint_unavailable",
                f"adapter response included {token}",
            )

    monkeypatch.setattr(worker_main, "_load_adapter", lambda _name: DirectFailureAdapter())
    request = WorkerRequest(
        request_id="direct-secret",
        provider="fixture",
        checkpoint="fixture/checkpoint",
        history=(1.0,),
        horizon=1,
        frequency="D",
    )
    output = io.StringIO()

    worker_main.serve("legacy", io.StringIO(request.to_json() + "\n"), output)

    assert token not in output.getvalue()
    assert WorkerResponse.from_json(output.getvalue()).message.endswith("[REDACTED]")


def test_worker_redacts_nested_success_metadata_before_serializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.tsfm import worker_main

    first = "worker-success-secret"
    second = "second-worker-success-secret"
    monkeypatch.setenv("HF_TOKEN", first)
    monkeypatch.setenv("TABPFN_TOKEN", second)

    class DirectSuccessAdapter:
        def forecast(self, request):
            return WorkerResponse.success(
                request.request_id,
                [1.0],
                {first: {"nested": [first, {"value": second}]}},
            )

    monkeypatch.setattr(worker_main, "_load_adapter", lambda _name: DirectSuccessAdapter())
    request = WorkerRequest(
        request_id="success-secret",
        provider="fixture",
        checkpoint="fixture/checkpoint",
        history=(1.0,),
        horizon=1,
        frequency="D",
    )
    output = io.StringIO()

    worker_main.serve("legacy", io.StringIO(request.to_json() + "\n"), output)

    assert first not in output.getvalue()
    assert second not in output.getvalue()
    response = WorkerResponse.from_json(output.getvalue())
    assert response.metadata == {
        "[REDACTED]": {"nested": ["[REDACTED]", {"value": "[REDACTED]"}]}
    }


def test_example_config_is_valid_after_replacing_placeholder_paths(tmp_path: Path) -> None:
    example = json.loads(
        (Path(__file__).parents[1] / "configs/tsfm_workers.example.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(example) == {"schema_version", "environments"}
    for entry in example["environments"].values():
        assert set(entry) == {"interpreter"}
        entry["interpreter"] = sys.executable
    config = tmp_path / "workers.json"
    config.write_text(json.dumps(example), encoding="utf-8")

    deployment = TSFMDeployment.load(
        config,
        manifests=ManifestRegistry.load_default(),
        acknowledged_licenses=(
            "CC-BY-NC-4.0",
            "research/non-commercial; official terms ambiguous",
            "NXAI Community License",
            "TabPFN-3 Non-Commercial License; Apache-2.0 code",
            "CC-BY-NC-SA-4.0",
        ),
    )

    assert len(deployment.commands) == 11
    assert len(deployment.enabled_manifest_ids) == 17
