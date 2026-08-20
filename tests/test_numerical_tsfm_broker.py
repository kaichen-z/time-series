from __future__ import annotations

import dataclasses
import errno
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.providers import RuntimeRegistry, RuntimeUnavailableError
from numerical_agent.tsfm.broker import WorkerBroker, WorkerCommand, WorkerMethodRuntime
from numerical_agent.tsfm.manifests import ManifestRegistry, TSFMManifest
from numerical_agent.tsfm.protocol import WorkerRequest, WorkerResponse


FIXTURE_WORKER = str(Path(__file__).parent / "fixtures" / "tsfm_worker.py")
IMMUTABLE_REVISION = "a" * 40


def worker_candidate() -> MethodCandidate:
    return MethodCandidate(
        method_id="fixture_model",
        provider="tsfm_worker",
        implementation_kind="tsfm_checkpoint",
        implementation={
            "manifest_id": "fixture_model",
            "worker_environment": "fixture",
            "runtime_family": "fixture",
            "checkpoint": "fixture/checkpoint",
            "model_id": "fixture/checkpoint",
            "runtime_options": {"literal": True},
            "point_reduction": "direct",
            "license_id": "Apache-2.0",
            "license_acknowledgement_required": False,
        },
    )


def runtime_for(mode: str, *args: str, timeout_seconds: float = 5.0):
    broker = WorkerBroker(
        {"fixture": WorkerCommand((sys.executable, FIXTURE_WORKER, mode, *args))},
        timeout_seconds=timeout_seconds,
    )
    manifests = ManifestRegistry(
        {
            "fixture_model": TSFMManifest.from_payload(
                {
                    "method_id": "fixture_model",
                    "checkpoint": "fixture/checkpoint",
                    "worker_environment": "fixture",
                    "adapter": "fixture",
                    "license_id": "Apache-2.0",
                    "license_acknowledgement_required": False,
                    "point_reduction": "direct",
                    "status": "experimental_unverified",
                    "reason_code": "",
                    "runtime_options": {"literal": True},
                    "official_source_ids": ["source_fixture"],
                }
            )
        }
    )
    return broker, WorkerMethodRuntime(
        broker,
        manifests=manifests,
        enabled_manifest_ids={"fixture_model"},
    )


def test_worker_runtime_round_trips_literal_values():
    broker, runtime = runtime_for("success")
    try:
        assert tuple(runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")) == (
            11.0,
            12.0,
        )
    finally:
        broker.close()


def test_worker_runtime_rejects_request_id_mismatch():
    broker, runtime = runtime_for("request_id_mismatch")
    try:
        with pytest.raises(RuntimeUnavailableError, match="request ID"):
            runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
    finally:
        broker.close()


@pytest.mark.parametrize("mode", ["malformed", "non_finite"])
def test_worker_runtime_rejects_malformed_or_non_finite_output(mode: str):
    broker, runtime = runtime_for(mode)
    try:
        with pytest.raises(RuntimeUnavailableError, match="invalid response"):
            runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
    finally:
        broker.close()


def test_worker_runtime_terminates_a_timed_out_process(tmp_path: Path):
    pid_path = tmp_path / "worker.pid"
    broker, runtime = runtime_for("timeout", str(pid_path), timeout_seconds=0.2)
    try:
        with pytest.raises(RuntimeUnavailableError, match="timed out"):
            runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
        pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError as error:
                if error.errno == errno.ESRCH:
                    break
                raise
            time.sleep(0.01)
        else:
            pytest.fail("timed-out worker process was not terminated")
    finally:
        broker.close()


def test_worker_runtime_lazily_restarts_after_a_crash(tmp_path: Path):
    state_path = tmp_path / "crashed-once"
    broker, runtime = runtime_for("crash_once", str(state_path))
    try:
        with pytest.raises(RuntimeUnavailableError, match="exited"):
            runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
        assert int(state_path.read_text(encoding="utf-8")) > 0
        assert runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D") == (
            11.0,
            12.0,
        )
    finally:
        broker.close()


def test_worker_broker_close_is_idempotent():
    broker, runtime = runtime_for("success")
    descriptors = (broker._wake_read_fd, broker._wake_write_fd)
    runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
    broker.close()
    broker.close()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("failure_step", ["pipe", "read", "write"])
def test_worker_broker_closes_wake_descriptors_when_pipe_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_step: str,
) -> None:
    real_pipe = os.pipe
    real_set_blocking = os.set_blocking
    captured_descriptors: list[int] = []
    set_blocking_calls = 0

    def injected_pipe() -> tuple[int, int]:
        if failure_step == "pipe":
            raise OSError("injected pipe failure")
        descriptors = real_pipe()
        captured_descriptors.extend(descriptors)
        return descriptors

    def injected_set_blocking(descriptor: int, blocking: bool) -> None:
        nonlocal set_blocking_calls
        set_blocking_calls += 1
        if (failure_step == "read" and set_blocking_calls == 1) or (
            failure_step == "write" and set_blocking_calls == 2
        ):
            raise OSError(f"injected {failure_step} set_blocking failure")
        real_set_blocking(descriptor, blocking)

    monkeypatch.setattr(os, "pipe", injected_pipe)
    monkeypatch.setattr(os, "set_blocking", injected_set_blocking)
    before_count = len(os.listdir("/dev/fd"))

    try:
        with pytest.raises(OSError, match="injected"):
            WorkerBroker(
                {"fixture": WorkerCommand((sys.executable, "worker.py"))},
                timeout_seconds=1.0,
            )
        after_count = len(os.listdir("/dev/fd"))
        descriptor_closed = []
        for descriptor in captured_descriptors:
            try:
                os.fstat(descriptor)
            except OSError:
                descriptor_closed.append(True)
            else:
                descriptor_closed.append(False)

        assert (after_count, descriptor_closed) == (
            before_count,
            [True] * len(captured_descriptors),
        )
    finally:
        for descriptor in captured_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize(("failure_stream", "failure_call"), [("stdin", 1), ("stdout", 2)])
def test_worker_broker_reaps_process_when_pipe_nonblocking_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_stream: str,
    failure_call: int,
) -> None:
    token = "post-popen-setup-secret"
    monkeypatch.setenv("HF_TOKEN", token)
    broker = WorkerBroker(
        {
            "fixture": WorkerCommand(
                (sys.executable, FIXTURE_WORKER, "success")
            )
        },
        timeout_seconds=5.0,
    )
    real_popen = subprocess.Popen
    real_set_blocking = os.set_blocking
    processes: list[subprocess.Popen[str]] = []
    set_blocking_calls = 0

    def tracked_popen(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        process = real_popen(*args, **kwargs)  # type: ignore[call-overload]
        processes.append(process)
        return process

    def injected_set_blocking(descriptor: int, blocking: bool) -> None:
        nonlocal set_blocking_calls
        set_blocking_calls += 1
        if set_blocking_calls == failure_call:
            raise OSError(f"injected {failure_stream} failure containing {token}")
        real_set_blocking(descriptor, blocking)

    monkeypatch.setattr(subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(os, "set_blocking", injected_set_blocking)
    try:
        with pytest.raises(RuntimeUnavailableError) as raised:
            broker.request(
                "fixture",
                WorkerRequest(
                    request_id=f"{failure_stream}-setup-failure",
                    provider="fixture",
                    checkpoint="fixture/checkpoint",
                    history=(1.0,),
                    horizon=1,
                    frequency="D",
                ),
            )

        assert token not in str(raised.value)
        assert "[REDACTED]" in str(raised.value)
        assert broker._states["fixture"].process is None
        assert len(processes) == 1
        process = processes[0]
        assert process.poll() is not None
        assert process.stdin is not None and process.stdin.closed
        assert process.stdout is not None and process.stdout.closed
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2.0)
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    stream.close()
        broker.close()


def test_worker_runtime_and_registry_own_an_idempotent_close_contract() -> None:
    class CloseCountingBroker:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    broker = CloseCountingBroker()
    manifests = ManifestRegistry.load_default()
    runtime = WorkerMethodRuntime(
        broker,  # type: ignore[arg-type]
        manifests=manifests,
        enabled_manifest_ids={"method_tsfm_0001"},
    )
    registry = RuntimeRegistry({"tsfm_worker": runtime, "worker_alias": runtime})

    with registry as entered:
        assert entered is registry
        assert registry.resolve(
            MethodCandidate(
                "other",
                "missing",
                "python_code",
                {"code": ""},
            )
        ).available is False
    registry.close()
    runtime.close()

    assert broker.close_count == 1


def test_registry_context_closes_worker_process_and_wake_descriptors() -> None:
    broker, runtime = runtime_for("success")
    descriptors = (broker._wake_read_fd, broker._wake_write_fd)

    with RuntimeRegistry({"tsfm_worker": runtime}):
        assert runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D") == (
            11.0,
            12.0,
        )
        process = broker._states["fixture"].process
        assert process is not None
        pid = process.pid

    assert broker._states["fixture"].process is None
    with pytest.raises(OSError):
        os.kill(pid, 0)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_protocol_dataclasses_are_frozen_and_validate_strict_json():
    request = WorkerRequest(
        request_id="request-1",
        provider="fixture",
        checkpoint="fixture/checkpoint",
        history=(1.0, 2.0),
        horizon=2,
        frequency="D",
        runtime_options={"literal": True},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.horizon = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="unexpected fields"):
        WorkerRequest.from_json(request.to_json()[:-1] + ',"command":"unsafe"}')
    with pytest.raises(ValueError, match="protocol version"):
        WorkerRequest.from_json(
            request.to_json().replace('"protocol_version":1', '"protocol_version":true')
        )
    with pytest.raises(ValueError, match="duplicate field"):
        WorkerRequest.from_json(request.to_json()[:-1] + ',"horizon":99}')
    with pytest.raises(ValueError, match="finite"):
        WorkerResponse.from_json(
            '{"protocol_version":1,"request_id":"request-1","status":"success",'
            '"values":[NaN],"metadata":{}}'
        )


def test_protocol_carries_only_empty_or_immutable_checkpoint_revisions() -> None:
    request = WorkerRequest(
        request_id="request-attested",
        provider="fixture",
        checkpoint="fixture/checkpoint",
        checkpoint_revision=IMMUTABLE_REVISION,
        history=(1.0, 2.0),
        horizon=2,
        frequency="D",
    )

    assert WorkerRequest.from_json(request.to_json()).checkpoint_revision == (
        IMMUTABLE_REVISION
    )
    legacy_payload = request.to_payload()
    legacy_payload.pop("checkpoint_revision")
    assert WorkerRequest.from_payload(legacy_payload).checkpoint_revision == ""

    for invalid in ("main", "A" * 40, "a" * 39, "a" * 65, " " + "a" * 40):
        with pytest.raises(ValueError, match="checkpoint_revision"):
            dataclasses.replace(request, checkpoint_revision=invalid)


def test_runtime_support_requires_enabled_manifest_and_fixed_candidate_shape():
    broker, runtime = runtime_for("success")
    try:
        assert runtime.supports(worker_candidate())
        disabled = dataclasses.replace(worker_candidate(), method_id="other")
        assert not runtime.supports(disabled)
    finally:
        broker.close()


@pytest.mark.parametrize(
    ("field", "substitute"),
    [
        ("checkpoint", "attacker/checkpoint"),
        ("model_id", "attacker/checkpoint"),
        ("worker_environment", "attacker_environment"),
        ("runtime_family", "attacker_adapter"),
        ("runtime_options", {"literal": True, "revision": "attacker"}),
        ("checkpoint_revision", IMMUTABLE_REVISION),
        ("point_reduction", "mean"),
        ("license_id", "attacker-license"),
        ("license_acknowledgement_required", True),
        ("manifest_id", "attacker_manifest"),
    ],
)
def test_worker_runtime_rejects_candidate_manifest_substitution(
    field: str, substitute: object
) -> None:
    broker, runtime = runtime_for("success")
    implementation = dict(worker_candidate().implementation)
    implementation[field] = substitute
    candidate = dataclasses.replace(
        worker_candidate(), implementation=implementation
    )
    try:
        assert not runtime.supports(candidate)
        with pytest.raises(ValueError, match="enabled TSFM worker manifest"):
            runtime.forecast(candidate, [1.0], 2, "D")
    finally:
        broker.close()


def test_worker_runtime_rejects_provider_substitution() -> None:
    broker, runtime = runtime_for("success")
    candidate = dataclasses.replace(worker_candidate(), provider="attacker")
    try:
        assert not runtime.supports(candidate)
    finally:
        broker.close()


def test_worker_request_is_constructed_only_from_registry_manifest(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    broker, runtime = runtime_for("record_request", str(request_path))
    try:
        assert runtime.forecast(worker_candidate(), [1.0], 2, "D") == (11.0, 12.0)
    finally:
        broker.close()

    request = __import__("json").loads(request_path.read_text(encoding="utf-8"))
    assert request["provider"] == "fixture"
    assert request["checkpoint"] == "fixture/checkpoint"
    assert request["checkpoint_revision"] == ""
    assert request["runtime_options"] == {"literal": True}


def test_broker_rejects_ambiguous_commands_and_non_finite_timeouts():
    with pytest.raises(ValueError, match="argv"):
        WorkerCommand(sys.executable)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout"):
        WorkerBroker(
            {"fixture": WorkerCommand((sys.executable, FIXTURE_WORKER, "success"))},
            timeout_seconds=math.nan,
        )


def test_worker_timeout_covers_partial_stdout_line():
    broker, runtime = runtime_for("partial_line", timeout_seconds=0.2)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeUnavailableError, match="timed out"):
            runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
        assert time.monotonic() - started < 1.0
    finally:
        broker.close()


def test_close_interrupts_a_request_blocked_writing_to_worker(tmp_path: Path):
    pid_path = tmp_path / "never-read.pid"
    broker = WorkerBroker(
        {
            "fixture": WorkerCommand(
                (sys.executable, FIXTURE_WORKER, "never_read", str(pid_path))
            )
        },
        timeout_seconds=10.0,
    )
    request = WorkerRequest(
        request_id="blocked-write",
        provider="fixture",
        checkpoint="fixture/checkpoint",
        history=(1.0,),
        horizon=1,
        frequency="D",
        runtime_options={"padding": "x" * 2_000_000},
    )
    request_thread = threading.Thread(
        target=lambda: _ignore_runtime_unavailable(broker, request), daemon=True
    )
    request_thread.start()
    deadline = time.monotonic() + 2.0
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_path.exists()

    close_thread = threading.Thread(target=broker.close, daemon=True)
    close_thread.start()
    close_thread.join(timeout=0.75)
    try:
        assert not close_thread.is_alive(), "close blocked behind worker request I/O"
        request_thread.join(timeout=0.75)
        assert not request_thread.is_alive()
    finally:
        if close_thread.is_alive():
            os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
        close_thread.join(timeout=2.0)
        request_thread.join(timeout=2.0)
        broker.close()


def _ignore_runtime_unavailable(broker: WorkerBroker, request: WorkerRequest) -> None:
    try:
        broker.request("fixture", request)
    except RuntimeUnavailableError:
        pass


def test_extreme_integer_response_is_invalid_and_discards_worker(tmp_path: Path):
    state_path = tmp_path / "extreme-integer.pid"
    broker, runtime = runtime_for("extreme_integer_once", str(state_path))
    try:
        with pytest.raises(RuntimeUnavailableError, match="invalid response"):
            runtime.forecast(worker_candidate(), [1.0, 2.0], 2, "D")
        first_pid = int(state_path.read_text(encoding="utf-8"))
        response = broker.request(
            "fixture",
            WorkerRequest(
                request_id="after-extreme-integer",
                provider="fixture",
                checkpoint="fixture/checkpoint",
                history=(1.0, 2.0),
                horizon=2,
                frequency="D",
            ),
        )
        assert response.values == (11.0, 12.0)
        assert response.metadata["pid"] != first_pid
    finally:
        broker.close()
