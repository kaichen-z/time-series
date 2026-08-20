"""Persistent subprocess broker for dependency-isolated TSFM workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
import selectors
import subprocess
import threading
import time
import uuid

from ..dictionary import MethodCandidate
from ..providers import RuntimeUnavailableError
from .manifests import ManifestRegistry
from .protocol import WorkerRequest, WorkerResponse
from .security import (
    SecretRedactor,
    controlled_worker_environment,
)


@dataclass(frozen=True)
class WorkerCommand:
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or not all(isinstance(arg, str) and arg for arg in self.argv)
        ):
            raise ValueError("worker command argv must contain non-empty strings")


@dataclass
class _WorkerState:
    command: WorkerCommand
    process: subprocess.Popen[str] | None = None
    request_lock: threading.Lock = field(default_factory=threading.Lock)
    process_lock: threading.Lock = field(default_factory=threading.Lock)
    stdout_buffer: bytearray = field(default_factory=bytearray)


class _WorkerIOTimeout(Exception):
    pass


class _WorkerExited(Exception):
    pass


class _BrokerClosed(Exception):
    pass


class WorkerBroker:
    """Own and serialize one persistent subprocess per configured worker key."""

    _MAX_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        commands: Mapping[str, WorkerCommand],
        *,
        timeout_seconds: float,
        parent_environment: Mapping[str, str] | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("worker timeout must be positive")
        if not commands:
            raise ValueError("at least one worker command must be configured")
        if not all(isinstance(key, str) and key for key in commands):
            raise ValueError("worker command keys must be non-empty strings")
        if not all(isinstance(command, WorkerCommand) for command in commands.values()):
            raise ValueError("worker commands must be WorkerCommand instances")
        self._states = {key: _WorkerState(command) for key, command in commands.items()}
        self._timeout_seconds = float(timeout_seconds)
        source_environment = dict(
            os.environ if parent_environment is None else parent_environment
        )
        self._worker_environment = controlled_worker_environment(source_environment)
        self._redactor = redactor or SecretRedactor.from_environment(
            source_environment
        )
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        wake_read_fd, wake_write_fd = os.pipe()
        try:
            os.set_blocking(wake_read_fd, False)
            os.set_blocking(wake_write_fd, False)
        except BaseException:
            for descriptor in (wake_read_fd, wake_write_fd):
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
            raise
        self._wake_read_fd = wake_read_fd
        self._wake_write_fd = wake_write_fd

    def request(self, worker_key: str, request: WorkerRequest) -> WorkerResponse:
        state = self._states.get(worker_key)
        if state is None:
            raise RuntimeUnavailableError(f"worker {worker_key!r} is not configured")
        with state.request_lock:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeUnavailableError("worker broker is closed")
            deadline = time.monotonic() + self._timeout_seconds
            process = self._ensure_process(worker_key, state)
            try:
                payload = (request.to_json() + "\n").encode("utf-8")
                line = self._exchange(state, process, payload, deadline)
            except _WorkerIOTimeout as error:
                self._discard(state)
                raise RuntimeUnavailableError(
                    f"worker {worker_key!r} timed out after "
                    f"{self._timeout_seconds:g} seconds"
                ) from error
            except (
                _BrokerClosed,
                _WorkerExited,
                BrokenPipeError,
                OSError,
                ValueError,
            ) as error:
                self._discard(state)
                with self._lifecycle_lock:
                    closed = self._closed
                if closed:
                    raise RuntimeUnavailableError("worker broker is closed") from error
                raise RuntimeUnavailableError(
                    f"worker {worker_key!r} exited without a response "
                    f"(code {process.returncode})"
                ) from error
            try:
                response = WorkerResponse.from_json(line)
            except ValueError as error:
                self._discard(state)
                raise RuntimeUnavailableError(
                    f"worker {worker_key!r} returned an invalid response: "
                    f"{self._redactor.redact_text(error)}"
                ) from None
            if response.request_id != request.request_id:
                self._discard(state)
                raise RuntimeUnavailableError(
                    f"worker {worker_key!r} returned a mismatched request ID"
                )
            return self._redactor.sanitize_response(response)

    @property
    def redactor(self) -> SecretRedactor:
        return self._redactor

    def _ensure_process(
        self, worker_key: str, state: _WorkerState
    ) -> subprocess.Popen[str]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeUnavailableError("worker broker is closed")
            with state.process_lock:
                if state.process is not None and state.process.poll() is None:
                    return state.process
                stale_process = state.process
                state.process = None
                state.stdout_buffer.clear()
                self._terminate_and_close(stale_process)
                process: subprocess.Popen[str] | None = None
                try:
                    process = subprocess.Popen(
                        list(state.command.argv),
                        shell=False,
                        text=True,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        bufsize=1,
                        env=dict(self._worker_environment),
                    )
                    assert process.stdin is not None
                    assert process.stdout is not None
                    os.set_blocking(process.stdin.fileno(), False)
                    os.set_blocking(process.stdout.fileno(), False)
                except OSError as error:
                    self._terminate_and_close(process)
                    raise RuntimeUnavailableError(
                        f"worker {worker_key!r} could not be started: "
                        f"{self._redactor.redact_text(error)}"
                    ) from None
                assert process is not None
                state.process = process
                return process

    def _exchange(
        self,
        state: _WorkerState,
        process: subprocess.Popen[str],
        payload: bytes,
        deadline: float,
    ) -> str:
        assert process.stdin is not None
        assert process.stdout is not None
        stdin_fd = process.stdin.fileno()
        stdout_fd = process.stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._wake_read_fd, selectors.EVENT_READ, "close")
            selector.register(stdin_fd, selectors.EVENT_WRITE, "worker")
            written = 0
            while written < len(payload):
                self._wait_for_io(selector, deadline)
                try:
                    written += os.write(stdin_fd, payload[written:])
                except BlockingIOError:
                    continue
            selector.unregister(stdin_fd)
            selector.register(stdout_fd, selectors.EVENT_READ, "worker")
            while b"\n" not in state.stdout_buffer:
                self._wait_for_io(selector, deadline)
                try:
                    chunk = os.read(stdout_fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise _WorkerExited
                state.stdout_buffer.extend(chunk)
                if len(state.stdout_buffer) > self._MAX_RESPONSE_BYTES:
                    raise ValueError("worker response exceeds size limit")
            line, _, remainder = state.stdout_buffer.partition(b"\n")
            state.stdout_buffer[:] = remainder
            try:
                return line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("worker response is not valid UTF-8") from error
        finally:
            selector.close()

    @staticmethod
    def _wait_for_io(selector: selectors.BaseSelector, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _WorkerIOTimeout
        while True:
            try:
                ready = selector.select(remaining)
                break
            except InterruptedError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _WorkerIOTimeout
        if not ready:
            raise _WorkerIOTimeout
        if any(key.data == "close" for key, _events in ready):
            raise _BrokerClosed

    @staticmethod
    def _terminate(process: subprocess.Popen[str] | None) -> None:
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=1.0)

    @classmethod
    def _terminate_and_close(cls, process: subprocess.Popen[str] | None) -> None:
        cls._terminate(process)
        cls._close_streams(process)

    @staticmethod
    def _close_streams(process: subprocess.Popen[str] | None) -> None:
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _detach(self, state: _WorkerState) -> subprocess.Popen[str] | None:
        with state.process_lock:
            process = state.process
            state.process = None
            state.stdout_buffer.clear()
            return process

    def _discard(self, state: _WorkerState) -> None:
        self._terminate_and_close(self._detach(state))

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        try:
            os.write(self._wake_write_fd, b"\0")
        except (BlockingIOError, OSError):
            pass
        detached: dict[int, subprocess.Popen[str]] = {}
        for state in self._states.values():
            process = self._detach(state)
            if process is not None:
                self._terminate(process)
                detached[id(state)] = process
        for state in self._states.values():
            with state.request_lock:
                self._close_streams(detached.get(id(state)))
        for descriptor in (self._wake_read_fd, self._wake_write_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> "WorkerBroker":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class WorkerMethodRuntime:
    """Adapt enabled, manifest-bound candidates to the MethodRuntime contract."""

    def __init__(
        self,
        broker: WorkerBroker,
        *,
        manifests: ManifestRegistry,
        enabled_manifest_ids: set[str] | frozenset[str],
    ) -> None:
        self._broker = broker
        self._closed = False
        redactor = getattr(broker, "redactor", None)
        self._redactor = (
            redactor
            if isinstance(redactor, SecretRedactor)
            else SecretRedactor.from_environment()
        )
        self._manifests = manifests
        self._enabled_manifest_ids = frozenset(enabled_manifest_ids)
        if not self._enabled_manifest_ids <= set(manifests):
            raise ValueError("enabled TSFM workers must have reviewed manifests")
        if any(
            manifests[method_id].status != "experimental_unverified"
            for method_id in self._enabled_manifest_ids
        ):
            raise ValueError("only experimental manifests can use TSFM workers")

    def supports(self, candidate: MethodCandidate) -> bool:
        if self._closed:
            return False
        if "checkpoint_revision" in candidate.implementation:
            return False
        if candidate.method_id not in self._enabled_manifest_ids:
            return False
        try:
            manifest = self._manifests[candidate.method_id]
        except KeyError:
            return False
        return manifest.status == "experimental_unverified" and manifest.matches_candidate(
            candidate, provider="tsfm_worker"
        )

    def forecast(
        self,
        candidate: MethodCandidate,
        history: Sequence[float],
        horizon: int,
        frequency: str,
    ) -> tuple[float, ...]:
        if not self.supports(candidate):
            raise ValueError("candidate is not an enabled TSFM worker manifest")
        manifest = self._manifests[candidate.method_id]
        request = WorkerRequest(
            request_id=uuid.uuid4().hex,
            provider=manifest.adapter,
            checkpoint=manifest.checkpoint,
            history=tuple(history),
            horizon=horizon,
            frequency=frequency,
            runtime_options=dict(
                manifest.candidate_binding()["runtime_options"]  # type: ignore[arg-type]
            ),
        )
        response = self._redactor.sanitize_response(
            self._broker.request(manifest.worker_environment, request)
        )
        reason_code = response.reason_code
        message = response.message
        if response.status == "unavailable":
            raise RuntimeUnavailableError(f"{reason_code}: {message}")
        if response.status == "invalid_request":
            raise ValueError(f"{reason_code}: {message}")
        if response.status == "runtime_error":
            raise RuntimeError(f"{reason_code}: {message}")
        if len(response.values) != horizon:
            raise RuntimeUnavailableError(
                "worker returned an invalid response: forecast has the wrong horizon length"
            )
        return response.values

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._broker.close()

    def __enter__(self) -> "WorkerMethodRuntime":
        if self._closed:
            raise RuntimeError("TSFM worker runtime is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
