from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.evolution_core.contracts import EvaluationReport, MutationContext
from common.llm import (
    ClaudeCLIClient,
    CodexCLIClient,
    FakeLLMClient,
    JsonExtractionError,
    QwenClient,
)
from common.sandbox import ALLOWED_IMPORTS
from common.tracing import configure
from numerical_agent.adapters.dictionary_curation import DictionaryMutator
from numerical_agent.codegen import (
    SANDBOX_PROVIDER,
    LLMMethodImplementer,
    SandboxMethodRuntime,
)
from numerical_agent.config import DictionaryCurationConfig
from numerical_agent.dictionary import (
    MethodCandidate,
    MethodDefinition,
    MethodRecord,
    ToolDictionary,
)
from numerical_agent.main import LLM_BACKENDS, _providers, build_parser
from numerical_agent.persistence import MethodSourceArtifactStore
from numerical_agent.prompts import ALLOWED_IMPORTS_TEXT
from numerical_agent.providers import ImplementationContext, SanitizedMethodFeedback


NAIVE_CODE = "def forecast(history, horizon, frequency):\n    return [float(history[-1])] * horizon\n"


def scripted(code: str) -> FakeLLMClient:
    return FakeLLMClient([json.dumps({"code": code})])


def definition(method_id: str = "ses") -> MethodDefinition:
    return MethodDefinition(
        method_id,
        "statistical",
        "Simple exponential smoothing.",
        assumptions=("assumes no trend",),
        failure_conditions=("fails on seasonal series",),
    )


def test_implement_returns_a_sandbox_candidate_carrying_the_generated_code() -> None:
    implementer = LLMMethodImplementer(scripted(NAIVE_CODE))

    candidate = implementer.implement(
        definition(), ImplementationContext("statistical_base_methods_v000", 1)
    )

    assert candidate.method_id == "ses"
    assert candidate.provider == SANDBOX_PROVIDER
    assert candidate.implementation_kind == "python_code"
    assert candidate.implementation["code"] == NAIVE_CODE
    assert candidate.version == 1


def test_implement_sends_the_method_description_to_the_model() -> None:
    client = scripted(NAIVE_CODE)

    LLMMethodImplementer(client).implement(
        definition(), ImplementationContext("statistical_base_methods_v000", 3)
    )

    user = client.calls[0]["messages"][0]["content"]
    assert "ses" in user
    assert "Simple exponential smoothing." in user
    assert "assumes no trend" in user
    assert "fails on seasonal series" in user
    assert client.calls[0]["temperature"] == 0.0


def test_implement_sends_child_diversity_context_to_the_model() -> None:
    client = scripted(NAIVE_CODE)

    LLMMethodImplementer(client).implement(
        definition(),
        ImplementationContext(
            "statistical_base_methods_v000",
            3,
            child_index=2,
            diversity_instruction="Prefer a robust alternative parameterization.",
        ),
    )

    user = client.calls[0]["messages"][0]["content"]
    assert '"child_index": 2' in user
    assert "robust alternative parameterization" in user


def test_revise_versions_the_candidate_and_forwards_sanitized_feedback() -> None:
    parent = MethodCandidate("ses", SANDBOX_PROVIDER, "python_code", {"code": "old"})
    client = scripted(NAIVE_CODE)

    child = LLMMethodImplementer(client).revise(
        parent,
        SanitizedMethodFeedback(
            "ses",
            {"mean_error": 91.25},
            ("high_error",),
            ("IndexError: list index out of range",),
        ),
    )

    assert child.version == 2
    assert child.parent_version == 1
    assert child.implementation["code"] == NAIVE_CODE
    user = client.calls[0]["messages"][0]["content"]
    assert "old" in user
    assert "91.25" in user
    assert "high_error" in user
    # The real error text must reach the model, not just aggregate counts.
    assert "IndexError: list index out of range" in user


def test_implement_rejects_a_response_without_code() -> None:
    implementer = LLMMethodImplementer(FakeLLMClient([json.dumps({"notcode": 1})]))

    with pytest.raises(ValueError, match="code"):
        implementer.implement(definition(), ImplementationContext("d0", 1))


def test_implement_rejects_a_response_that_is_not_json() -> None:
    implementer = LLMMethodImplementer(FakeLLMClient(["sorry, no JSON here"]))

    with pytest.raises(JsonExtractionError):
        implementer.implement(definition(), ImplementationContext("d0", 1))


def test_mutator_keeps_a_first_model_failure_retryable() -> None:
    record = MethodRecord(definition())
    parent = ToolDictionary("d0", None, 0, (record,))
    mutator = DictionaryMutator(
        DictionaryCurationConfig(),
        LLMMethodImplementer(FakeLLMClient(["not json"])),
    )
    context = MutationContext(1, EvaluationReport("d0", "train", {"smape": 1.0}, 1))

    children = mutator.propose(parent, context, 1)

    assert children[0].methods[0].status == "unimplemented"
    assert children[0].methods[0].implementation_attempts == 1


def test_runtime_supports_only_sandbox_candidates_carrying_code() -> None:
    runtime = SandboxMethodRuntime()

    assert runtime.supports(
        MethodCandidate("m", SANDBOX_PROVIDER, "python_code", {"code": NAIVE_CODE})
    )
    assert not runtime.supports(
        MethodCandidate("m", "fake", "fixture", {"prediction": 10.0})
    )
    assert not runtime.supports(
        MethodCandidate("m", SANDBOX_PROVIDER, "python_code", {"code": "   "})
    )


def test_runtime_executes_generated_code_through_the_sandbox() -> None:
    candidate = MethodCandidate("m", SANDBOX_PROVIDER, "python_code", {"code": NAIVE_CODE})

    forecast = SandboxMethodRuntime().forecast(candidate, [1.0, 2.0, 7.0], 3, "D")

    assert forecast == (7.0, 7.0, 7.0)


def test_runtime_reports_unsafe_code_as_a_runtime_failure() -> None:
    unsafe = MethodCandidate(
        "m",
        SANDBOX_PROVIDER,
        "python_code",
        {"code": "import os\ndef forecast(history, horizon, frequency):\n    return [1.0] * horizon\n"},
    )

    # A disallowed import must surface as a normal failure so the method stays
    # quarantined and therefore revisable, rather than escaping the run.
    with pytest.raises(RuntimeError, match="disallowed import"):
        SandboxMethodRuntime().forecast(unsafe, [1.0], 2, "D")


def traced(tmp_path: Path) -> Path:
    """Point the shared tracer at a temporary log and return it."""
    log = tmp_path / "trace.jsonl"
    configure(log)
    return log


def events(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_successful_call_is_traced_and_transcribed(tmp_path: Path) -> None:
    log = traced(tmp_path)
    implementer = LLMMethodImplementer(scripted(NAIVE_CODE), transcript_dir=tmp_path / "t")

    implementer.implement(definition(), ImplementationContext("d0", 1))

    end = [event for event in events(log) if event["event_type"] == "method_end"]
    assert end[0]["task_id"] == "ses"
    assert end[0]["detail"]["ok"] is True
    assert end[0]["detail"]["stage"] == "implement"
    transcript = (tmp_path / "t" / "ses.implement.md").read_text(encoding="utf-8")
    assert "# system" in transcript and "# response" in transcript
    assert NAIVE_CODE.strip().splitlines()[0] in transcript


def test_a_failed_call_records_the_real_error_before_it_is_swallowed(tmp_path: Path) -> None:
    log = traced(tmp_path)
    implementer = LLMMethodImplementer(
        FakeLLMClient(["sorry, no JSON"]), transcript_dir=tmp_path / "t"
    )

    with pytest.raises(JsonExtractionError):
        implementer.implement(definition(), ImplementationContext("d0", 1))

    end = [event for event in events(log) if event["event_type"] == "method_end"]
    assert end[0]["detail"]["ok"] is False
    assert "JsonExtractionError" in end[0]["detail"]["error"]
    # The raw response must survive so a bad answer can be inspected afterwards.
    assert "sorry, no JSON" in (tmp_path / "t" / "ses.implement.md").read_text(encoding="utf-8")


def test_sandbox_failures_are_traced_with_the_real_reason(tmp_path: Path) -> None:
    log = traced(tmp_path)
    unsafe = MethodCandidate(
        "ses",
        SANDBOX_PROVIDER,
        "python_code",
        {"code": "import scipy\ndef forecast(history, horizon, frequency):\n    return [1.0] * horizon\n"},
    )

    with pytest.raises(RuntimeError):
        SandboxMethodRuntime().forecast(unsafe, [1.0], 2, "D")

    failures = [event for event in events(log) if event["event_type"] == "sandbox_failed"]
    # The adapter keeps only the exception class name, so the detail must be captured here.
    assert failures[0]["detail"]["error"] == "disallowed import: scipy"
    assert failures[0]["task_id"] == "ses"


def test_transcripts_refuse_a_method_id_that_escapes_the_directory(tmp_path: Path) -> None:
    traced(tmp_path)
    implementer = LLMMethodImplementer(scripted(NAIVE_CODE), transcript_dir=tmp_path / "t")

    implementer.implement(
        MethodDefinition("../escaped", "statistical", "hostile id"),
        ImplementationContext("d0", 1),
    )

    assert not (tmp_path / "escaped.implement.md").exists()
    assert not list(tmp_path.glob("**/*.implement.md"))


def test_prompt_advertises_exactly_the_sandbox_allow_list() -> None:
    advertised = {
        token.strip().strip(",")
        for token in ALLOWED_IMPORTS_TEXT.replace("and ", "").split()
    }

    assert advertised == set(ALLOWED_IMPORTS)


def parsed(*extra: str):
    return build_parser().parse_args(
        [
            "curate",
            "--experiment-config",
            "x",
            "--base-methods",
            "y",
            "--output-dir",
            "z",
            *extra,
        ]
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (None, CodexCLIClient),
        ("codex", CodexCLIClient),
        ("claude", ClaudeCLIClient),
        ("qwen", QwenClient),
    ],
)
def test_llm_provider_builds_the_requested_backend(backend, expected) -> None:
    extra = ["--provider", "llm"] + (["--llm-backend", backend] if backend else [])

    implementer, registry = _providers("llm", parsed(*extra))

    # Constructing only: QwenClient loads its weights lazily, so no model is fetched.
    assert isinstance(implementer.llm, expected)
    assert registry.resolve(
        MethodCandidate("m", SANDBOX_PROVIDER, "python_code", {"code": NAIVE_CODE})
    ).available


def test_every_advertised_backend_is_constructible() -> None:
    for backend in LLM_BACKENDS:
        implementer, _ = _providers(
            "llm", parsed("--provider", "llm", "--llm-backend", backend)
        )
        assert implementer.llm is not None


def test_llm_provider_keeps_each_config_default_when_flags_are_unset() -> None:
    implementer, _ = _providers("llm", parsed("--provider", "llm", "--llm-backend", "codex"))

    # Unset flags must not overwrite the dataclass defaults with None.
    assert implementer.llm.config.reasoning_effort == "high"
    assert implementer.llm.config.timeout_seconds == 900


def test_provider_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported approved provider"):
        _providers("bogus", parsed("--provider", "bogus"))


def dictionary_with(code: str | None) -> ToolDictionary:
    candidate = (
        MethodCandidate("ses", SANDBOX_PROVIDER, "python_code", {"code": code})
        if code is not None
        else None
    )
    return ToolDictionary(
        "d1",
        None,
        1,
        (
            MethodRecord(definition("ses"), candidate),
            MethodRecord(definition("arima_auto")),
        ),
    )


def test_store_writes_generated_code_beside_the_json_artifact(tmp_path: Path) -> None:
    store = MethodSourceArtifactStore(tmp_path)
    payload = dictionary_with(NAIVE_CODE).to_payload()

    destination = store.save_artifact("generation_001_child_x", payload)

    assert (tmp_path / "generation_001_child_x" / "methods" / "ses.py").read_text() == NAIVE_CODE
    # The unimplemented method must not leave an empty stub behind.
    assert not (tmp_path / "generation_001_child_x" / "methods" / "arima_auto.py").exists()
    restored = ToolDictionary.from_payload(json.loads(destination.read_text()))
    assert restored == ToolDictionary.from_payload(payload)


def test_store_refuses_a_method_id_that_escapes_the_directory(tmp_path: Path) -> None:
    store = MethodSourceArtifactStore(tmp_path)
    escaping = ToolDictionary(
        "d1",
        None,
        1,
        (
            MethodRecord(
                MethodDefinition("../escaped", "statistical", "hostile id"),
                MethodCandidate("../escaped", SANDBOX_PROVIDER, "python_code", {"code": NAIVE_CODE}),
            ),
        ),
    )

    store.save_artifact("generation_001_child_x", escaping.to_payload())

    assert not (tmp_path.parent / "escaped.py").exists()
    assert list(tmp_path.rglob("*.py")) == []
