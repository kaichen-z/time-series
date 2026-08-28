from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import evolving_loop.retrieval_agent.evolution as evolution_module
from common.data import Task
from common.llm import FakeLLMClient, LLMResponse, TransientLLMError
from evolving_loop.data import ContextTask, Document
from evolving_loop.retrieval_agent.evolution import (
    RetrievalCheckpointError,
    RetrievalEvaluation,
    RetrievalEvolutionConfig,
    RetrievalEvolutionEngine,
    RetrievalEvolutionError,
    RetrievalEvolutionResult,
    RetrievalForecastingFailure,
    RetrievalGenerationTrace,
    build_inference_cache_key,
    combine_retrieval_evaluations,
    parse_scoped_child,
)
from evolving_loop.retrieval_agent.policy import (
    RetrievalGenome,
    _write_accepted_retrieval_release,
)
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalApplicability,
    RetrievalSkill,
    RetrievalSkillError,
    RetrievalSkillLibrary,
    RetrievalSkillOperation,
    _migrate_legacy_for_operator,
)


_OPERATOR_AUTHORITY_KEY = b"task-8-test-operator-authority-key-32-bytes"


def _staged_checkpoint(
    checkpoint: Path,
    encoded: bytes,
    *,
    suffix: str,
) -> tuple[Path, tuple[int, int]]:
    staged = checkpoint.with_name(f"{checkpoint.name}.{suffix}")
    staged.write_bytes(encoded)
    metadata = staged.stat()
    return staged, (metadata.st_dev, metadata.st_ino)


def _tasks(prefix: str, count: int, *, entity_offset: int = 0) -> tuple[ContextTask, ...]:
    result = []
    for index in range(count):
        entity = f"entity_{entity_offset + index // 8:03d}"
        result.append(
            ContextTask(
                numeric=Task(
                    task_id=f"{prefix}_{index:03d}",
                    history_values=(1.0, 2.0, 3.0),
                    future_values=(4.0, 5.0),
                    prediction_length=2,
                    frequency="D",
                    seasonal_period=None,
                    entity_name=entity,
                ),
                target_name="volume",
                target_description="private resolved target",
                history_timestamps=("2026-01-01", "2026-01-02", "2026-01-03"),
                future_timestamps=("2026-01-04", "2026-01-05"),
                documents=(
                    Document(
                        f"{prefix}_doc_{index:03d}",
                        "A public document.",
                        role="PRIVATE_RELEVANCE_LABEL",
                    ),
                ),
                gt_evidence=("PRIVATE_GT_EVIDENCE",),
                labels_public=True,
            )
        )
    return tuple(result)


def _evaluation(
    version: str,
    tasks: tuple[ContextTask, ...],
    *,
    error: float = 1.0,
    **overrides: object,
) -> RetrievalEvaluation:
    values: dict[str, object] = {
        "version": version,
        "task_count": len(tasks),
        "mean_final_smae": error,
        "mean_final_srmse": error,
        "mean_contextual_oracle_smae": error,
        "mean_contextual_oracle_srmse": error,
        "p90_smae": error,
        "p95_smae": error,
        "supporting_recall": 0.90,
        "distractor_avoidance": 0.90,
        "exact_quote_validity": 1.0,
        "complete_chain_rate": 0.80,
        "invalid_count": 0,
        "catastrophic_count": 0,
        "task_traces": tuple(
            {
                "task_id": task.numeric.task_id,
                "entity_name": task.numeric.entity_name,
                "final_smae": error,
                "final_srmse": error,
                "contextual_oracle_smae": error,
                "contextual_oracle_srmse": error,
                "failure_example": "TRAIN_OR_DEV_EVALUATOR_SECRET",
            }
            for task in tasks
        ),
    }
    values.update(overrides)
    return RetrievalEvaluation(**values)


def _proposal(parent: RetrievalGenome, version: str, scope: str) -> dict[str, object]:
    payload = parent.to_payload()
    payload.update({"version": version, "parent": parent.version})
    if scope == "A":
        payload["round1_prompt"] = f"{parent.round1_prompt}\nScoped change {version}."
    elif scope == "B":
        payload["max_citations_per_chain"] = (
            parent.max_citations_per_chain % 8
        ) + 1
    elif scope == "C":
        payload["round2_strategy"] = (
            "gap_first"
            if parent.round2_strategy != "gap_first"
            else "causal_chain_first"
        )
    else:
        raise AssertionError(scope)
    return payload


def _responses(generations: int) -> list[str]:
    parent = RetrievalGenome.seed()
    responses: list[str] = []
    for generation in range(generations):
        children = []
        for index, scope in enumerate(("A", "B", "C"), start=1):
            version = f"v{generation * 3 + index:03d}"
            payload = _proposal(parent, version, scope)
            responses.append(json.dumps(payload))
            children.append(RetrievalGenome.from_payload(payload))
        parent = children[0]
    return responses


@dataclass(frozen=True)
class _EvaluationCall:
    version: str
    task_ids: tuple[str, ...]
    entities: tuple[str, ...]
    stage: str
    persist: bool
    writers_enabled: bool
    evolver_enabled: bool
    cache_keys: tuple[object, ...]


class _FakeEvaluator:
    evaluator_hash = "fake-evaluator-v1"
    verifier_hash = "fake-verifier-v1"

    def __init__(
        self,
        *,
        errors: dict[str, float] | None = None,
        dev_overrides: dict[str, object] | None = None,
        transient_stage: str | None = None,
        failing_version: str | None = None,
    ) -> None:
        self.errors = errors or {}
        self.dev_overrides = dev_overrides or {}
        self.transient_stage = transient_stage
        self.failing_version = failing_version
        self.calls: list[_EvaluationCall] = []
        self._transient_raised = False

    def evaluate(
        self,
        genome: RetrievalGenome,
        tasks: tuple[ContextTask, ...],
        *,
        stage: str,
        persist: bool,
        writers_enabled: bool,
        evolver_enabled: bool,
        cache_keys: tuple[object, ...],
        **_unused: object,
    ) -> RetrievalEvaluation:
        self.calls.append(
            _EvaluationCall(
                genome.version,
                tuple(task.numeric.task_id for task in tasks),
                tuple(sorted({task.numeric.entity_name for task in tasks})),
                stage,
                persist,
                writers_enabled,
                evolver_enabled,
                cache_keys,
            )
        )
        if self.transient_stage == stage and not self._transient_raised:
            self._transient_raised = True
            raise TransientLLMError("temporary evaluator outage")
        if genome.version == self.failing_version and "screen_train" in stage:
            raise ValueError("forecasting candidate failed")
        error = self.errors.get(genome.version, 1.0)
        overrides = (
            dict(self.dev_overrides)
            if stage == "child_dev" and genome.version != "v000"
            else {}
        )
        trace_final_smae = overrides.pop("trace_final_smae", None)
        result = _evaluation(genome.version, tasks, error=error, **overrides)
        if trace_final_smae is not None:
            result = replace(
                result,
                task_traces=tuple(
                    {**trace, "final_smae": trace_final_smae}
                    for trace in result.task_traces
                ),
            )
        return result


def _engine(
    evaluator: _FakeEvaluator,
    responses: list[str],
    *,
    checkpoint_path: Path | None = None,
    generations: int = 1,
    transient_retries: int = 1,
    dataset_split_hash: str = "split-v1",
) -> tuple[RetrievalEvolutionEngine, FakeLLMClient]:
    llm = FakeLLMClient(responses)
    config = RetrievalEvolutionConfig(
        generations=generations,
        screen_tasks=8,
        promote=2,
        train_folds=4,
        random_seed=17,
        transient_retries=transient_retries,
        checkpoint_path=checkpoint_path,
        dataset_split_hash=dataset_split_hash,
        verifier_hash="verifier-v1",
        evaluator_hash="evaluator-v1",
        metric_hash="metric-v1",
        metric_cap=5.0,
    )
    return RetrievalEvolutionEngine(llm, evaluator, config), llm


def _complete_train_checkpoint(
    engine: RetrievalEvolutionEngine,
    parent: RetrievalGenome,
    train: tuple[ContextTask, ...],
    dev: tuple[ContextTask, ...],
    *,
    generation_count: int | None = None,
) -> tuple[tuple[ContextTask, ...], tuple[tuple[ContextTask, ...], ...]]:
    """Drive the real engine through a chosen Train generation prefix."""
    engine._validate_inputs(parent, train, dev)
    screen, remaining_folds = engine._partition_train(train)
    engine._scientific_inputs = engine._science_signature(parent, train, dev)
    assert engine._load_checkpoint(
        parent, train, dev, screen, remaining_folds
    ) is None
    count = engine.config.generations if generation_count is None else generation_count
    for generation in range(count):
        engine._run_generation(generation, screen, remaining_folds)
    return screen, remaining_folds


def test_checkpoint_payload_validator_runs_before_tainted_checkpoint_commit(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    responses = _responses(1)
    tainted = json.loads(responses[0])
    tainted["round1_prompt"] += " prefix_task_100_suffix"
    responses[0] = json.dumps(tainted)
    config = RetrievalEvolutionConfig(
        generations=1,
        screen_tasks=8,
        promote=2,
        train_folds=4,
        random_seed=17,
        checkpoint_path=checkpoint,
        dataset_split_hash="split-v1",
    )

    def reject_public_id(payload: object) -> None:
        if "task_100" in json.dumps(payload, sort_keys=True):
            raise RetrievalEvolutionError("Public Regression payload rejected")

    engine = RetrievalEvolutionEngine(
        FakeLLMClient(responses),
        _FakeEvaluator(),
        config,
        _checkpoint_payload_validator=reject_public_id,
    )

    with pytest.raises(RetrievalEvolutionError, match="Public Regression"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )

    encoded = checkpoint.read_text(encoding="utf-8") if checkpoint.exists() else ""
    assert "task_100" not in encoded


def test_terminal_evaluator_exception_is_sanitized_before_trace_and_checkpoint(
    tmp_path: Path,
) -> None:
    secret = "SECRET_TRUSTED_SCORER_EXCEPTION_PAYLOAD"

    class SecretEvaluator(_FakeEvaluator):
        def evaluate(self, *_args, **_kwargs):
            raise RuntimeError(secret)

    checkpoint = tmp_path / "checkpoint.json"
    engine, _llm = _engine(
        SecretEvaluator(),
        _responses(1),
        checkpoint_path=checkpoint,
    )

    with pytest.raises(RetrievalForecastingFailure) as captured:
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )

    assert captured.value.__cause__ is None
    assert secret not in repr(captured.value.args)
    assert secret not in checkpoint.read_text(encoding="utf-8")
    assert secret not in json.dumps(engine._trace)


def test_operator_checkpoint_authority_recovers_pending_commit_without_a_gap(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    first = b"first trusted checkpoint\n"
    first_digest = hashlib.sha256(first).hexdigest()

    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    staged, staged_identity = _staged_checkpoint(
        checkpoint, first, suffix="pending"
    )
    token = authority.prepare(
        first_digest, checkpoint_identity=staged_identity
    )
    staged.rename(checkpoint)
    authority.close()

    pending = json.loads(authority_path.read_text(encoding="utf-8"))
    assert pending["pending"]["checkpoint_sha256"] == first_digest
    reopened = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
        expected_authority_anchor=(
            pending["authority_epoch"], pending["authority_head"]
        ),
    )
    reopened.close()
    committed = json.loads(authority_path.read_text(encoding="utf-8"))
    assert committed["checkpoint_sha256"] == first_digest
    assert committed["authority_epoch"] == token.authority_epoch
    assert committed["pending"] is None


def test_operator_checkpoint_authority_rejects_fabricated_protected_journal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"caller checkpoint\n")
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    authority_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
                "authority_epoch": 9,
                "authority_head": "1" * 64,
                "pending": None,
                "journal_mac": "2" * 64,
            }
        ),
        encoding="utf-8",
    )
    authority_head_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
                "authority_epoch": 9,
                "authority_head": "1" * 64,
                "head_mac": "3" * 64,
            }
        ),
        encoding="utf-8",
    )
    authority_path.chmod(0o600)
    authority_head_path.chmod(0o600)

    with pytest.raises(RetrievalCheckpointError, match="authentic|authority"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority_path,
            authority_head_path,
            authentication_key=_OPERATOR_AUTHORITY_KEY,
        )


def test_operator_checkpoint_authority_rejects_missing_or_wrong_key(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    encoded = b"trusted checkpoint\n"
    staged, staged_identity = _staged_checkpoint(
        checkpoint, encoded, suffix="trusted"
    )
    token = authority.prepare(
        hashlib.sha256(encoded).hexdigest(),
        checkpoint_identity=staged_identity,
    )
    staged.rename(checkpoint)
    authority.commit(token)
    authority.close()

    with pytest.raises(RetrievalCheckpointError, match="key|authentic|authority"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority_path,
            authority_head_path,
            authentication_key=None,
        )
    with pytest.raises(RetrievalCheckpointError, match="authentic|authority"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority_path,
            authority_head_path,
            authentication_key=b"wrong-task-8-operator-authority-key-32",
        )


def test_operator_checkpoint_authority_rejects_replayed_old_epoch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    first = b"first trusted checkpoint\n"
    first_staged, first_identity = _staged_checkpoint(
        checkpoint, first, suffix="first"
    )
    token = authority.prepare(
        hashlib.sha256(first).hexdigest(),
        checkpoint_identity=first_identity,
    )
    first_staged.rename(checkpoint)
    authority.commit(token)
    old_journal = authority_path.read_bytes()

    second = b"second trusted checkpoint\n"
    second_staged, second_identity = _staged_checkpoint(
        checkpoint, second, suffix="second"
    )
    token = authority.prepare(
        hashlib.sha256(second).hexdigest(),
        checkpoint_identity=second_identity,
    )
    checkpoint.rename(checkpoint.with_name("replayed-old-first.json"))
    second_staged.rename(checkpoint)
    authority.commit(token)
    current_state = json.loads(authority_path.read_text(encoding="utf-8"))
    current_anchor = (
        current_state["authority_epoch"], current_state["authority_head"]
    )
    authority.close()
    current_head = authority_head_path.read_bytes()

    authority_path.write_bytes(old_journal)
    with pytest.raises(RetrievalCheckpointError, match="replay|head|epoch|authority"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority_path,
            authority_head_path,
            authentication_key=_OPERATOR_AUTHORITY_KEY,
            expected_authority_anchor=current_anchor,
        )
    assert authority_head_path.read_bytes() == current_head


def test_operator_checkpoint_authority_rejects_complete_set_rollback_in_fresh_process(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )

    first = b"first trusted checkpoint\n"
    staged_first, first_identity = _staged_checkpoint(
        checkpoint, first, suffix="first-staged"
    )
    first_token = authority.prepare(
        hashlib.sha256(first).hexdigest(),
        checkpoint_identity=first_identity,
    )
    staged_first.rename(checkpoint)
    authority.commit(first_token)

    old_checkpoint = checkpoint.with_name("epoch-one-checkpoint.json")
    old_journal = authority_path.with_name("epoch-one-journal.json")
    old_head = authority_head_path.with_name("epoch-one-head.json")
    os.link(checkpoint, old_checkpoint)
    os.link(authority_path, old_journal)
    os.link(authority_head_path, old_head)

    second = b"second trusted checkpoint\n"
    staged_second, second_identity = _staged_checkpoint(
        checkpoint, second, suffix="second-staged"
    )
    second_token = authority.prepare(
        hashlib.sha256(second).hexdigest(),
        checkpoint_identity=second_identity,
    )
    checkpoint.rename(checkpoint.with_name("displaced-epoch-one-checkpoint.json"))
    staged_second.rename(checkpoint)
    authority.commit(second_token)
    current = json.loads(authority_path.read_text(encoding="utf-8"))
    expected_anchor = f'{current["authority_epoch"]}:{current["authority_head"]}'
    authority.close()

    checkpoint.rename(checkpoint.with_name("epoch-two-checkpoint.json"))
    authority_path.rename(authority_path.with_name("epoch-two-journal.json"))
    authority_head_path.rename(authority_head_path.with_name("epoch-two-head.json"))
    old_checkpoint.rename(checkpoint)
    old_journal.rename(authority_path)
    old_head.rename(authority_head_path)

    script = """
import os
from evolving_loop.retrieval_agent.evolution import (
    RetrievalCheckpointError,
    _open_retrieval_checkpoint_authority_for_operator,
)

epoch, head = os.environ["EXPECTED_AUTHORITY_ANCHOR"].split(":", 1)
try:
    authority = _open_retrieval_checkpoint_authority_for_operator(
        os.environ["CHECKPOINT_PATH"],
        os.environ["AUTHORITY_PATH"],
        os.environ["AUTHORITY_HEAD_PATH"],
        authentication_key=os.environ["OPERATOR_AUTHORITY_KEY"].encode("utf-8"),
        expected_authority_anchor=(int(epoch), head),
    )
except RetrievalCheckpointError:
    raise SystemExit(0)
else:
    authority.close()
    raise SystemExit("complete authority rollback was accepted")
"""
    environment = os.environ.copy()
    environment.update(
        {
            "CHECKPOINT_PATH": str(checkpoint),
            "AUTHORITY_PATH": str(authority_path),
            "AUTHORITY_HEAD_PATH": str(authority_head_path),
            "OPERATOR_AUTHORITY_KEY": _OPERATOR_AUTHORITY_KEY.decode("utf-8"),
            "EXPECTED_AUTHORITY_ANCHOR": expected_anchor,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert _OPERATOR_AUTHORITY_KEY.decode("utf-8") not in (
        completed.stdout + completed.stderr
    )


def test_checkpoint_authority_binds_staged_inode_in_commit_and_reconciliation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    encoded = b"same trusted checkpoint bytes\n"
    intended, intended_identity = _staged_checkpoint(
        checkpoint, encoded, suffix="intended"
    )
    replacement, replacement_identity = _staged_checkpoint(
        checkpoint, encoded, suffix="replacement"
    )
    assert replacement_identity != intended_identity
    token = authority.prepare(
        hashlib.sha256(encoded).hexdigest(),
        checkpoint_identity=intended_identity,
    )
    replacement.rename(checkpoint)

    with pytest.raises(RetrievalCheckpointError, match="identity|inode|replacement"):
        authority.commit(token)
    authority.close()

    pending = json.loads(authority_path.read_text(encoding="utf-8"))
    assert pending["pending"]["checkpoint_identity"] == {
        "st_dev": intended_identity[0],
        "st_ino": intended_identity[1],
    }
    with pytest.raises(RetrievalCheckpointError, match="identity|inode|replacement"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority_path,
            authority_head_path,
            authentication_key=_OPERATOR_AUTHORITY_KEY,
            expected_authority_anchor=(
                pending["authority_epoch"], pending["authority_head"]
            ),
        )

    assert intended.is_file()
    assert checkpoint.stat().st_ino == replacement_identity[1]


def test_operator_checkpoint_authority_never_persists_secret_key(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    encoded = b"trusted checkpoint\n"
    staged, staged_identity = _staged_checkpoint(
        checkpoint, encoded, suffix="secret-redaction"
    )
    token = authority.prepare(
        hashlib.sha256(encoded).hexdigest(),
        checkpoint_identity=staged_identity,
    )
    staged.rename(checkpoint)
    authority.commit(token)
    authority.close()

    persisted = authority_path.read_bytes() + authority_head_path.read_bytes()
    assert _OPERATOR_AUTHORITY_KEY not in persisted


def test_operator_checkpoint_authority_does_not_overwrite_replaced_journal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    first = b"first trusted checkpoint\n"
    staged, staged_identity = _staged_checkpoint(
        checkpoint, first, suffix="journal-first"
    )
    token = authority.prepare(
        hashlib.sha256(first).hexdigest(),
        checkpoint_identity=staged_identity,
    )
    staged.rename(checkpoint)
    authority.commit(token)
    displaced = authority_directory / "displaced-owned-journal.json"
    authority_path.rename(displaced)
    foreign = b"foreign journal replacement must survive\n"
    authority_path.write_bytes(foreign)

    second = b"second checkpoint\n"
    _second_staged, second_identity = _staged_checkpoint(
        checkpoint, second, suffix="journal-second"
    )
    with pytest.raises(RetrievalCheckpointError, match="changed|identity|replacement"):
        authority.prepare(
            hashlib.sha256(second).hexdigest(),
            checkpoint_identity=second_identity,
        )
    authority.close()

    assert authority_path.read_bytes() == foreign
    assert displaced.read_bytes() != foreign


def test_operator_checkpoint_authority_does_not_overwrite_replaced_head(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    first = b"first trusted checkpoint\n"
    first_staged, first_identity = _staged_checkpoint(
        checkpoint, first, suffix="head-first"
    )
    token = authority.prepare(
        hashlib.sha256(first).hexdigest(),
        checkpoint_identity=first_identity,
    )
    first_staged.rename(checkpoint)
    authority.commit(token)
    second = b"second trusted checkpoint\n"
    second_staged, second_identity = _staged_checkpoint(
        checkpoint, second, suffix="head-second"
    )
    token = authority.prepare(
        hashlib.sha256(second).hexdigest(),
        checkpoint_identity=second_identity,
    )
    checkpoint.rename(checkpoint.with_name("head-displaced-first.json"))
    second_staged.rename(checkpoint)
    displaced = authority_directory / "displaced-owned-head.json"
    authority_head_path.rename(displaced)
    foreign = b"foreign head replacement must survive\n"
    authority_head_path.write_bytes(foreign)

    with pytest.raises(RetrievalCheckpointError, match="changed|identity|replacement"):
        authority.commit(token)
    authority.close()

    assert authority_head_path.read_bytes() == foreign
    assert displaced.read_bytes() != foreign


def test_operator_checkpoint_authority_rejects_same_bytes_new_checkpoint_inode(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    first = b"first trusted checkpoint\n"
    staged, staged_identity = _staged_checkpoint(
        checkpoint, first, suffix="inode-first"
    )
    token = authority.prepare(
        hashlib.sha256(first).hexdigest(),
        checkpoint_identity=staged_identity,
    )
    staged.rename(checkpoint)
    authority.commit(token)
    displaced = checkpoint.with_name("displaced-checkpoint.json")
    checkpoint.rename(displaced)
    checkpoint.write_bytes(first)

    second = b"second checkpoint\n"
    _second_staged, second_identity = _staged_checkpoint(
        checkpoint, second, suffix="inode-second"
    )
    with pytest.raises(RetrievalCheckpointError, match="changed|identity|replacement"):
        authority.prepare(
            hashlib.sha256(second).hexdigest(),
            checkpoint_identity=second_identity,
        )
    authority.close()

    assert checkpoint.read_bytes() == first
    assert checkpoint.stat().st_ino != displaced.stat().st_ino


def test_operator_checkpoint_authority_rejects_record_replacement_after_preflight(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    checkpoint.parent.mkdir()
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "retrieval.json"
    authority_head_path = authority_directory / "retrieval.head.json"
    authority = evolution_module._open_retrieval_checkpoint_authority_for_operator(
        checkpoint,
        authority_path,
        authority_head_path,
        authentication_key=_OPERATOR_AUTHORITY_KEY,
    )
    encoded = b"trusted checkpoint\n"
    staged, staged_identity = _staged_checkpoint(
        checkpoint, encoded, suffix="preflight"
    )
    token = authority.prepare(
        hashlib.sha256(encoded).hexdigest(),
        checkpoint_identity=staged_identity,
    )
    staged.rename(checkpoint)
    authority.commit(token)
    authority.close()
    expected_checkpoint_identity = (
        checkpoint.stat().st_dev,
        checkpoint.stat().st_ino,
    )
    expected_authority_identity = (
        authority_path.stat().st_dev,
        authority_path.stat().st_ino,
    )
    expected_head_identity = (
        authority_head_path.stat().st_dev,
        authority_head_path.stat().st_ino,
    )
    displaced = authority_directory / "displaced-authority.json"
    authority_path.rename(displaced)
    authority_path.write_bytes(displaced.read_bytes())
    authority_path.chmod(0o600)

    with pytest.raises(RetrievalCheckpointError, match="changed|identity|replacement"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority_path,
            authority_head_path,
            authentication_key=_OPERATOR_AUTHORITY_KEY,
            expected_checkpoint_identity=expected_checkpoint_identity,
            expected_authority_identity=expected_authority_identity,
            expected_head_identity=expected_head_identity,
        )

    assert authority_path.read_bytes() == displaced.read_bytes()
    assert authority_path.stat().st_ino != displaced.stat().st_ino


def test_checkpoint_writer_quarantines_then_rejects_a_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    engine, _llm = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    parent = RetrievalGenome.seed()
    engine._original_parent = parent
    engine._current_parent = parent
    engine._scientific_inputs = {"frozen": True}
    engine._save_checkpoint(status="running", result=None)
    owned = checkpoint.read_bytes()
    displaced = checkpoint.with_name("displaced-owned-checkpoint.json")
    foreign = b"foreign checkpoint replacement must survive\n"
    real_move = evolution_module._move_artifact_entry_to_quarantine
    replacement_installed = False

    def replace_immediately_before_quarantine(parent_descriptor, name):
        nonlocal replacement_installed
        if name == checkpoint.name and not replacement_installed:
            replacement_installed = True
            os.rename(
                name,
                displaced.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(foreign)
        return real_move(parent_descriptor, name)

    monkeypatch.setattr(
        evolution_module,
        "_move_artifact_entry_to_quarantine",
        replace_immediately_before_quarantine,
    )

    with pytest.raises(RetrievalCheckpointError, match="changed|identity|replacement"):
        engine._save_checkpoint(status="running", result=None)

    assert replacement_installed
    assert checkpoint.read_bytes() == foreign
    assert displaced.read_bytes() == owned


def test_caller_authored_adjacent_checkpoint_sidecar_cannot_activate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    authority = tmp_path / "checkpoint.authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": "1" * 64,
                "authority_epoch": 1,
                "pending": None,
            }
        ),
        encoding="utf-8",
    )
    authority.chmod(0o600)

    with pytest.raises(RetrievalCheckpointError, match="adjacent|independent|authority"):
        evolution_module._open_retrieval_checkpoint_authority_for_operator(
            checkpoint,
            authority,
            tmp_path / "checkpoint.authority.head.json",
            authentication_key=_OPERATOR_AUTHORITY_KEY,
        )


def test_dev_acceptance_enforces_every_pareto_tail_recall_and_safety_gate() -> None:
    tasks = _tasks("dev", 20, entity_offset=100)
    parent = _evaluation("v000", tasks, error=1.0)
    accepted = _evaluation(
        "v001",
        tasks,
        error=0.99,
        mean_contextual_oracle_smae=0.98,
        mean_contextual_oracle_srmse=1.0,
    )
    assert accepted.dev_accepts(parent, tolerance=1e-12)

    failures = (
        {"task_count": 19},
        {"mean_contextual_oracle_smae": 1.01},
        {"mean_contextual_oracle_smae": 1.0, "mean_contextual_oracle_srmse": 1.0},
        {"mean_final_smae": 1.01},
        {"mean_final_srmse": 1.01},
        {"p90_smae": 1.01},
        {"p95_smae": 1.01},
        {"supporting_recall": 0.879},
        {"distractor_avoidance": 0.879},
        {"exact_quote_validity": 0.999},
        {"catastrophic_count": 1},
        {"invalid_count": 1},
    )
    for changes in failures:
        assert not replace(accepted, **changes).dev_accepts(parent, tolerance=1e-12)


def test_combined_tail_metrics_require_complete_finite_task_traces_and_linear_quantiles() -> None:
    tasks = _tasks("train", 4)
    left = _evaluation(
        "v001",
        tasks[:2],
        p90_smae=99.0,
        p95_smae=99.0,
        task_traces=(
            {**_evaluation("v001", tasks[:1]).task_traces[0], "final_smae": 0.0},
            {**_evaluation("v001", tasks[1:2]).task_traces[0], "final_smae": 1.0},
        ),
    )
    right = _evaluation(
        "v001",
        tasks[2:],
        p90_smae=77.0,
        p95_smae=77.0,
        task_traces=(
            {**_evaluation("v001", tasks[2:3]).task_traces[0], "final_smae": 2.0},
            {**_evaluation("v001", tasks[3:]).task_traces[0], "final_smae": 10.0},
        ),
    )

    combined = combine_retrieval_evaluations("v001", (left, right))

    assert combined.p90_smae == pytest.approx(7.6)
    assert combined.p95_smae == pytest.approx(8.8)
    with pytest.raises(Exception, match="trace|final_smae|finite|task_count"):
        combine_retrieval_evaluations(
            "v001",
            (replace(left, task_traces=left.task_traces[:1]), right),
        )
    with pytest.raises(Exception, match="trace|final_smae|finite"):
        combine_retrieval_evaluations(
            "v001",
            (
                replace(
                    left,
                    task_traces=(
                        {**left.task_traces[0], "final_smae": float("nan")},
                        left.task_traces[1],
                    ),
                ),
                right,
            ),
        )
    with pytest.raises(Exception, match="trace|task|coverage|duplicate"):
        combine_retrieval_evaluations(
            "v001",
            (
                left,
                replace(
                    right,
                    task_traces=tuple(
                        {**trace, "task_id": left.task_traces[0]["task_id"]}
                        for trace in right.task_traces
                    ),
                ),
            ),
        )


def test_screen_tail_gates_are_derived_from_complete_task_traces() -> None:
    class MisreportedScreenTailEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, **kwargs):
            result = super().evaluate(genome, tasks, **kwargs)
            if genome.version == "v001" and kwargs["stage"].endswith(
                "child_A_screen_train"
            ):
                return replace(
                    result,
                    p90_smae=0.9,
                    p95_smae=0.9,
                    task_traces=tuple(
                        {**trace, "final_smae": 9.0}
                        for trace in result.task_traces
                    ),
                )
            return result

    evaluator = MisreportedScreenTailEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 1.2, "v003": 1.3}
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.generations[0].rejection_reasons["v001"].startswith(
        "screen_gate:"
    )
    assert not any(
        call.version == "v001" and "child_train_fold" in call.stage
        for call in evaluator.calls
    )


def test_dev_tail_gates_are_derived_from_complete_task_traces() -> None:
    class MisreportedDevTailEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, **kwargs):
            result = super().evaluate(genome, tasks, **kwargs)
            if genome.version == "v001" and kwargs["stage"] == "child_dev":
                return replace(
                    result,
                    p90_smae=0.9,
                    p95_smae=0.9,
                    task_traces=tuple(
                        {**trace, "final_smae": 9.0}
                        for trace in result.task_traces
                    ),
                )
            return result

    evaluator = MisreportedDevTailEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 1.2, "v003": 1.3}
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.accepted is False
    assert result.child_dev is not None
    assert result.child_dev.p90_smae == 9.0
    assert result.child_dev.p95_smae == 9.0
    assert {"p90_smae", "p95_smae"}.issubset(result.rejection_reasons)


def test_complete_candidate_scope_validation_owns_only_a_b_or_c_fields(tmp_path) -> None:
    parent = RetrievalGenome.seed()
    library = RetrievalSkillLibrary(
        tmp_path / "skills.json",
        (
            RetrievalSkill(
                skill_id="round1_skill",
                version=1,
                parent_version=None,
                stage="round1",
                status="candidate",
                name="round1_skill",
                description="Round one.",
                applicability=RetrievalApplicability(),
                query_steps=("Search.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Check the opposite.",
                failure_conditions=("Wrong entity.",),
            ),
            RetrievalSkill(
                skill_id="both_skill",
                version=1,
                parent_version=None,
                stage="both",
                status="candidate",
                name="both_skill",
                description="Both rounds.",
                applicability=RetrievalApplicability(),
                query_steps=("Compose.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Check the opposite.",
                failure_conditions=("Wrong entity.",),
            ),
            RetrievalSkill(
                skill_id="round2_skill",
                version=1,
                parent_version=None,
                stage="round2",
                status="candidate",
                name="round2_skill",
                description="Round two.",
                applicability=RetrievalApplicability(),
                query_steps=("Fill the gap.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Check the opposite.",
                failure_conditions=("Wrong entity.",),
            ),
        ),
        persist=False,
    )

    assert parse_scoped_child(parent, _proposal(parent, "v001", "A"), scope="A")
    assert parse_scoped_child(parent, _proposal(parent, "v002", "B"), scope="B")
    assert parse_scoped_child(parent, _proposal(parent, "v003", "C"), scope="C")

    cross_scope = _proposal(parent, "v001", "A")
    cross_scope["round2_strategy"] = "gap_first"
    assert parse_scoped_child(parent, cross_scope, scope="A") is None
    incomplete = _proposal(parent, "v001", "A")
    incomplete.pop("round2_prompt")
    assert parse_scoped_child(parent, incomplete, scope="A") is None
    stale_parent = replace(parent, active_skill_ids=("round1_skill",))
    retains_untrusted_skill = _proposal(stale_parent, "v001", "A")
    assert (
        parse_scoped_child(
            stale_parent,
            retains_untrusted_skill,
            scope="A",
            skill_library=library,
        )
        is None
    )

    for scope, skill_id in (
        ("A", "round1_skill"),
        ("B", "both_skill"),
        ("C", "round2_skill"),
    ):
        payload = _proposal(parent, f"v{10 + len(skill_id) + ord(scope):03d}", scope)
        payload["active_skill_ids"] = [skill_id]
        child = parse_scoped_child(
            parent, payload, scope=scope, skill_library=library
        )
        assert child is None

    def active_payload(skill_id: str, stage: str, *, constrained: bool = False):
        return {
            "skill_id": skill_id,
            "version": 1,
            "parent_version": None,
            "stage": stage,
            "status": "accepted",
            "name": skill_id,
            "description": "Trusted prompt-active Skill.",
            "applicability": {
                "assumption_kinds": ["future_event"] if constrained else [],
                "gap_types": [],
                "temporal_relations": [],
            },
            "query_steps": ["Search."],
            "required_chain_fields": ["entity"],
            "counterevidence_rule": "Check the opposite.",
            "failure_conditions": ["Wrong entity."],
            "validated_task_ids": ["t1", "t2", "t3"],
            "validated_entities": ["e1", "e2"],
            "validation_smae_gain": 0.1,
            "validation_srmse_gain": 0.1,
            "merged_from_skill_ids": [],
            "quarantine_reason": None,
        }

    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        replace(
            parent,
            version="v900",
            parent=parent.version,
            active_skill_ids=(
                "both_skill",
                "constrained_round1",
                "round1_skill",
                "round2_skill",
            ),
        ),
        skills=(
            active_payload("round1_skill", "round1"),
            active_payload("both_skill", "both"),
            active_payload("round2_skill", "round2"),
            active_payload("constrained_round1", "round1", constrained=True),
        ),
        audit={
            "state": "accepted",
            "train_dev_split_sha256": "1" * 64,
            "verifier_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "metric_sha256": "4" * 64,
            "metric_cap": 5.0,
            "train_summary": {"task_count": 80},
            "dev_summary": {"task_count": 20},
            "acceptance_reason": "all gates passed",
        },
    )
    active_library = RetrievalSkillLibrary.from_release(release.path)
    for scope, skill_id, expected in (
        ("A", "round1_skill", True),
        ("A", "round2_skill", False),
        ("B", "both_skill", True),
        ("B", "round1_skill", False),
        ("C", "round2_skill", True),
        ("C", "both_skill", False),
        ("A", "constrained_round1", True),
    ):
        payload = _proposal(parent, f"v{100 + len(skill_id) + ord(scope):03d}", scope)
        payload["active_skill_ids"] = [skill_id]
        child = parse_scoped_child(
            parent, payload, scope=scope, skill_library=active_library
        )
        assert (child is not None) is expected

    scoped_a = _proposal(parent, "v001", "A")
    scoped_a["active_skill_ids"] = ["round1_skill"]
    engine, _llm = _engine(
        _FakeEvaluator(),
        [
            json.dumps(scoped_a),
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
    )
    engine.skill_library = active_library
    engine._original_parent = parent

    _children, rejections = engine._children_for_generation(
        0,
        parent,
        parent_library=library,
    )

    assert rejections["v001"] == "invalid_scoped_candidate"


def test_mutation_prompt_contains_only_the_sanitized_parent_skill_catalog(
    tmp_path,
) -> None:
    parent = RetrievalGenome.seed()
    release = _write_accepted_retrieval_release(
        tmp_path / "releases",
        replace(
            parent,
            version="v900",
            parent=parent.version,
            active_skill_ids=("future_only",),
        ),
        skills=(
            {
                "skill_id": "future_only",
                "version": 1,
                "parent_version": None,
                "stage": "round1",
                "status": "accepted",
                "name": "future_only",
                "description": "A validated future-event strategy.",
                "applicability": {
                    "assumption_kinds": ["future_event"],
                    "gap_types": [],
                    "temporal_relations": ["during"],
                },
                "query_steps": ["Search future events."],
                "required_chain_fields": ["entity"],
                "counterevidence_rule": "Search for cancellation.",
                "failure_conditions": ["No target match."],
                "validated_task_ids": ["private_train_task"],
                "validated_entities": ["private_entity"],
                "validation_smae_gain": 0.2,
                "validation_srmse_gain": 0.1,
                "merged_from_skill_ids": [],
                "quarantine_reason": None,
            },
        ),
        audit={
            "state": "accepted",
            "train_dev_split_sha256": "1" * 64,
            "verifier_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "metric_sha256": "4" * 64,
            "metric_cap": 5.0,
            "train_summary": {"task_count": 80},
            "dev_summary": {"task_count": 20},
            "acceptance_reason": "all gates passed",
        },
    )
    library = RetrievalSkillLibrary.from_release(release.path)
    engine, llm = _engine(_FakeEvaluator(), _responses(1))
    engine.skill_library = library

    engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    prompt = json.loads(llm.calls[0]["messages"][0]["content"])
    assert prompt["active_skill_catalog"] == [
        {
            "skill_id": "future_only",
            "stage": "round1",
            "applicability": {
                "assumption_kinds": ["future_event"],
                "gap_types": [],
                "temporal_relations": ["during"],
            },
        }
    ]
    encoded = json.dumps(prompt, sort_keys=True)
    for forbidden in (
        "private_train_task",
        "private_entity",
        "validation_smae_gain",
        "validation_srmse_gain",
        "validated_task_ids",
        "validated_entities",
    ):
        assert forbidden not in encoded


def test_invalid_scope_proposal_still_leaves_exactly_three_a_b_c_child_slots() -> None:
    parent = RetrievalGenome.seed()
    invalid_a = _proposal(parent, "v001", "A")
    invalid_a["round2_strategy"] = "gap_first"
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v002": 0.90, "v003": 0.95}
    )
    engine, llm = _engine(
        evaluator,
        [
            json.dumps(invalid_a),
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
    )

    result = engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    generation = result.generations[0]
    assert len(llm.calls) == 3
    assert generation.child_versions == ("v001", "v002", "v003")
    assert generation.child_scopes == ("A", "B", "C")
    assert len(generation.child_fingerprints) == 3
    assert generation.rejection_reasons["v001"] == "invalid_scoped_candidate"


def test_malformed_mutation_response_is_a_rejected_slot_not_an_aborted_run() -> None:
    parent = RetrievalGenome.seed()
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v002": 0.9, "v003": 0.95})
    engine, llm = _engine(
        evaluator,
        [
            "not json",
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
    )

    result = engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert len(llm.calls) == 3
    assert result.generations[0].child_versions == ("v001", "v002", "v003")
    assert result.generations[0].rejection_reasons["v001"] == (
        "invalid_scoped_candidate"
    )


def test_three_malformed_responses_keep_distinct_scoped_fingerprints() -> None:
    evaluator = _FakeEvaluator(errors={"v000": 1.0})
    engine, _llm = _engine(evaluator, ["not json", "not json", "not json"])

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    generation = result.generations[0]
    assert generation.child_scopes == ("A", "B", "C")
    assert len(set(generation.child_fingerprints)) == 3
    assert set(generation.rejection_reasons) == {"v001", "v002", "v003"}


def test_child_versions_continue_after_nonseed_parent_version() -> None:
    parent = replace(RetrievalGenome.seed(), version="v010", parent="v009")
    evaluator = _FakeEvaluator(
        errors={"v010": 1.0, "v011": 0.9, "v012": 0.95, "v013": 1.2}
    )
    responses = [
        json.dumps(_proposal(parent, "v011", scope))
        for scope in ("A", "B", "C")
    ]
    # Give each fixed scope its own consecutive host-owned version.
    responses[1] = json.dumps(_proposal(parent, "v012", "B"))
    responses[2] = json.dumps(_proposal(parent, "v013", "C"))
    engine, _llm = _engine(evaluator, responses)

    result = engine.evolve(
        parent,
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.generations[0].child_versions == ("v011", "v012", "v013")
    assert result.train_winner.version == "v011"


def test_train_only_successive_halving_and_final_read_only_dev_schedule() -> None:
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(
        errors={
            "v000": 1.0,
            "v001": 0.90,
            "v002": 0.95,
            "v003": 1.20,
            "v004": 0.80,
            "v005": 0.85,
            "v006": 1.10,
        }
    )
    engine, _llm = _engine(evaluator, _responses(2), generations=2)

    result = engine.evolve(RetrievalGenome.seed(), train, dev)

    stages = [call.stage for call in evaluator.calls]
    assert stages.count("parent_dev") == 1
    assert stages.count("child_dev") == 1
    assert stages.index("parent_dev") > max(
        index for index, stage in enumerate(stages) if "train" in stage
    )
    screen_calls = [call for call in evaluator.calls if "screen_train" in call.stage]
    assert screen_calls
    assert {call.task_ids for call in screen_calls} == {
        result.generations[0].screen_task_ids
    }
    for generation in result.generations:
        assert len(generation.child_fingerprints) == 3
        assert len(generation.promoted_fingerprints) <= 2
        version_by_fingerprint = dict(
            zip(
                generation.child_fingerprints,
                generation.child_versions,
                strict=True,
            )
        )
        completion = next(
            event
            for event in result.trace
            if event["kind"] == "generation_completed"
            and event["generation"] == generation.generation
        )
        assert completion["promoted"] == [
            version_by_fingerprint[fingerprint]
            for fingerprint in generation.promoted_fingerprints
        ]

    for generation in range(2):
        prefix = f"g{generation}_parent_train_fold_"
        fold_calls = [call for call in evaluator.calls if call.stage.startswith(prefix)]
        seen_entities: set[str] = set()
        for call in fold_calls:
            assert seen_entities.isdisjoint(call.entities)
            seen_entities.update(call.entities)
        assert {
            task_id for call in fold_calls for task_id in call.task_ids
        } == {
            task.numeric.task_id
            for task in train
            if task.numeric.task_id not in result.generations[generation].screen_task_ids
        }

    promoted_full_versions = {
        call.version
        for call in evaluator.calls
        if "child_train_fold" in call.stage
    }
    assert len(promoted_full_versions.intersection({"v001", "v002", "v003"})) <= 2
    assert len(promoted_full_versions.intersection({"v004", "v005", "v006"})) <= 2
    dev_calls = [call for call in evaluator.calls if call.stage in {"parent_dev", "child_dev"}]
    assert [call.version for call in dev_calls] == ["v000", "v004"]
    assert all(
        not call.persist and not call.writers_enabled and not call.evolver_enabled
        for call in dev_calls
    )
    assert result.train_winner.version == "v004"


def test_screen_and_remaining_folds_partition_all_train_entities_before_selection() -> None:
    grouped = _tasks("train", 80)
    train = tuple(
        grouped[entity_index * 8 + within_entity]
        for within_entity in range(8)
        for entity_index in range(10)
    )
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        train,
        _tasks("dev", 20, entity_offset=100),
    )

    generation = result.generations[0]
    screen_ids = set(generation.screen_task_ids)
    screen_entities = {
        task.numeric.entity_name for task in train if task.numeric.task_id in screen_ids
    }
    fold_entities = [set(entities) for entities in generation.fold_entities]
    assert len(screen_ids) == 8
    assert len(screen_entities) == 1
    assert len(fold_entities) == 4
    assert all(fold_entities)
    assert all(screen_entities.isdisjoint(entities) for entities in fold_entities)
    assert len(set.union(screen_entities, *fold_entities)) == 10


def test_train_partition_never_reduces_configured_entity_fold_count() -> None:
    one_entity = tuple(
        replace(task, numeric=replace(task.numeric, entity_name="single"))
        for task in _tasks("train", 80)
    )
    engine, _llm = _engine(_FakeEvaluator(), _responses(1))

    with pytest.raises(Exception, match="entit|fold|divers"):
        engine.evolve(
            RetrievalGenome.seed(),
            one_entity,
            _tasks("dev", 20, entity_offset=100),
        )


def test_every_evaluator_call_receives_the_complete_write_disabled_contract() -> None:
    calls: list[tuple[bool, bool, bool]] = []

    class ExactEvaluator:
        evaluator_hash = "exact-evaluator-v1"
        verifier_hash = "exact-verifier-v1"

        def evaluate(
            self,
            genome: RetrievalGenome,
            tasks: tuple[ContextTask, ...],
            *,
            stage: str,
            skill_library: RetrievalSkillLibrary | None,
            harness_factory: object,
            persist: bool,
            writers_enabled: bool,
            evolver_enabled: bool,
            cache_keys: tuple[object, ...],
            metric_cap: float,
        ) -> RetrievalEvaluation:
            del stage, skill_library, harness_factory, cache_keys, metric_cap
            calls.append((persist, writers_enabled, evolver_enabled))
            return _evaluation(genome.version, tasks, error=0.9)

    engine, _llm = _engine(ExactEvaluator(), _responses(1))  # type: ignore[arg-type]
    engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert calls
    assert set(calls) == {(False, False, False)}


def test_evaluator_that_cannot_accept_all_control_flags_fails_closed() -> None:
    class FlagDroppingEvaluator:
        def evaluate(
            self,
            genome: RetrievalGenome,
            tasks: tuple[ContextTask, ...],
            *,
            stage: str,
            persist: bool,
            evolver_enabled: bool,
            cache_keys: tuple[object, ...],
        ) -> RetrievalEvaluation:
            del stage, persist, evolver_enabled, cache_keys
            return _evaluation(genome.version, tasks)

    engine, _llm = _engine(FlagDroppingEvaluator(), _responses(1))  # type: ignore[arg-type]

    with pytest.raises(Exception, match="writers_enabled|contract|evaluator"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )


def test_engine_supplies_immutable_candidate_library_snapshots(tmp_path) -> None:
    library = RetrievalSkillLibrary(tmp_path / "skills.json", persist=False)
    attempted = RetrievalSkill(
        skill_id="evaluator_write",
        version=1,
        parent_version=None,
        stage="round1",
        status="candidate",
        name="evaluator_write",
        description="Must never be installed by evaluation.",
        applicability=RetrievalApplicability(),
        query_steps=("Write.",),
        required_chain_fields=("entity",),
        counterevidence_rule="Check conflicts.",
        failure_conditions=("Any write.",),
    )

    class MutatingEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, *, skill_library, **kwargs):
            assert skill_library is not None
            skill_library.add(attempted)
            return super().evaluate(
                genome,
                tasks,
                skill_library=skill_library,
                **kwargs,
            )

    evaluator = MutatingEvaluator()
    base, llm = _engine(evaluator, _responses(1))
    engine = RetrievalEvolutionEngine(
        llm,
        evaluator,
        base.config,
        skill_library=library,
    )

    with pytest.raises(RetrievalSkillError, match="read.only|immutable|mutation"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )
    assert library.get_by_id("evaluator_write") is None


def test_mutation_prompt_never_contains_tasks_labels_metrics_or_failure_examples() -> None:
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, llm = _engine(evaluator, _responses(1))

    engine.evolve(
        RetrievalGenome.seed(),
        _tasks("PRIVATE_TRAIN_TASK", 80),
        _tasks("PRIVATE_DEV_TASK", 20, entity_offset=100),
    )

    encoded = json.dumps(llm.calls, sort_keys=True)
    for forbidden in (
        "PRIVATE_TRAIN_TASK",
        "PRIVATE_DEV_TASK",
        "PRIVATE_GT_EVIDENCE",
        "PRIVATE_RELEVANCE_LABEL",
        "TRAIN_OR_DEV_EVALUATOR_SECRET",
        "mean_final_smae",
        "task_traces",
        "failure_example",
    ):
        assert forbidden not in encoded


def test_cache_key_binds_full_trusted_task_and_every_scientific_input(tmp_path) -> None:
    task = _tasks("train", 1)[0]
    genome = RetrievalGenome.seed()
    first_library = RetrievalSkillLibrary(tmp_path / "first.json", persist=False)
    second_library = RetrievalSkillLibrary(
        tmp_path / "second.json",
        (
            RetrievalSkill(
                skill_id="candidate",
                version=1,
                parent_version=None,
                stage="round1",
                status="candidate",
                name="candidate",
                description="Candidate.",
                applicability=RetrievalApplicability(),
                query_steps=("Search.",),
                required_chain_fields=("entity",),
                counterevidence_rule="Search for conflicts.",
                failure_conditions=("Wrong entity.",),
            ),
        ),
        persist=False,
    )
    base = build_inference_cache_key(
        task,
        genome,
        first_library,
        verifier_hash="verifier-a",
        evaluator_hash="evaluator-a",
        metric_hash="metric-a",
        metric_cap=5.0,
        harness_hash="harness-a",
        scientific_inputs_hash="science-a",
    )
    variants = (
        build_inference_cache_key(
            replace(task, numeric=replace(task.numeric, task_id="other")),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(task, numeric=replace(task.numeric, history_values=(9.0,))),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(
                task,
                documents=(replace(task.documents[0], content="changed document"),),
            ),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(task, future_timestamps=("2099-01-01", "2099-01-02")),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            replace(task, gt_evidence=("changed trusted label",)),
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            replace(genome, round1_prompt="changed"),
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            second_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-b",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-b",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-b",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        ),
        build_inference_cache_key(
            task,
            genome,
            first_library,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=6.0,
            harness_hash="harness-b",
            scientific_inputs_hash="science-b",
        ),
    )
    assert len({base.digest(), *(item.digest() for item in variants)}) == 12


def test_cache_key_revalidates_the_exact_current_skill_authority_epoch(tmp_path) -> None:
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            [
                {
                    "skill_id": "historical_skill",
                    "name": "historical_skill",
                    "description": "Operator-approved historical strategy.",
                    "applicability": "scheduled event",
                    "query_strategy": "Find the event window.",
                    "verification_rule": "Require an exact quote.",
                    "created_from_task": "train_1",
                    "validation_smae": 0.1,
                    "validation_srmse": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    library = _migrate_legacy_for_operator(path)
    stale_snapshot = library.clone(persist=False)
    task = _tasks("train", 1)[0]

    def key(candidate: RetrievalSkillLibrary):
        return build_inference_cache_key(
            task,
            RetrievalGenome.seed(),
            candidate,
            verifier_hash="verifier-a",
            evaluator_hash="evaluator-a",
            metric_hash="metric-a",
            metric_cap=5.0,
            harness_hash="harness-a",
            scientific_inputs_hash="science-a",
        )

    accepted_key = key(stale_snapshot)
    library.apply_operations(
        (RetrievalSkillOperation.quarantine("historical_skill", "unsafe"),)
    )

    with pytest.raises(RetrievalSkillError, match="current|authority|epoch"):
        key(stale_snapshot)
    assert key(library).digest() != accepted_key.digest()


def test_checkpoint_resume_binds_science_completion_and_child_fingerprints(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)

    first = engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["scientific_inputs"]["dataset_split_hash"] == "split-v1"
    assert payload["scientific_inputs"]["verifier_hash"] == "verifier-v1"
    assert payload["scientific_inputs"]["evaluator_hash"] == "evaluator-v1"
    assert payload["scientific_inputs"]["metric_cap"] == 5.0
    assert payload["scientific_inputs"]["random_seed"] == 17
    assert payload["original_parent_fingerprint"] == RetrievalGenome.seed().fingerprint()
    assert len(payload["child_fingerprints"]) == 3
    assert payload["task_completion"]

    resumed_evaluator = _FakeEvaluator()
    resumed, resumed_llm = _engine(
        resumed_evaluator, [], checkpoint_path=checkpoint
    )
    second = resumed.evolve(RetrievalGenome.seed(), train, dev)
    assert second == first
    assert resumed_evaluator.calls == []
    assert resumed_llm.calls == []

    mismatched, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
        dataset_split_hash="different-split",
    )
    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
        mismatched.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_task_completion_binds_cached_evaluation_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    first_cache_key = next(iter(payload["evaluation_cache"]))
    payload["evaluation_cache"][first_cache_key]["evaluation"][
        "mean_final_smae"
    ] = 4.5
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="evaluation|completion|digest|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_scientific_binding_rejects_json_type_drift(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["scientific_inputs"]["random_seed"] = 17.0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="scientific inputs|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_binds_explicit_harness_implementation_and_config_digest(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)

    def configured_factory(value):
        def factory(*_args, **_kwargs):
            return value

        return factory

    factory_a = configured_factory("behavior-a")
    factory_b = configured_factory("behavior-b")
    assert factory_a.__qualname__ == factory_b.__qualname__

    first = RetrievalEvolutionEngine(
        llm,
        evaluator,
        replace(engine.config, harness_hash="harness-implementation-config-a"),
        harness_factory=factory_a,
    )
    first.evolve(RetrievalGenome.seed(), train, dev)
    resumed = RetrievalEvolutionEngine(
        FakeLLMClient([]),
        _FakeEvaluator(),
        replace(engine.config, harness_hash="harness-implementation-config-b"),
        harness_factory=factory_b,
    )

    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_harness_factory_without_explicit_frozen_digest_fails_construction() -> None:
    def factory(*_args, **_kwargs):
        return None

    with pytest.raises(
        RetrievalEvolutionError, match="harness.*digest|harness_hash"
    ):
        RetrievalEvolutionEngine(
            FakeLLMClient([]),
            _FakeEvaluator(),
            RetrievalEvolutionConfig(),
            harness_factory=factory,
        )


def test_checkpoint_binds_transient_retry_science_policy(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    first, _ = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    first.evolve(RetrievalGenome.seed(), train, dev)
    resumed, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
        transient_retries=1,
    )

    with pytest.raises(RetrievalCheckpointError, match="scientific inputs"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_writer_ignores_attacker_fixed_temp_symlink(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch", encoding="utf-8")
    fixed_temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    fixed_temporary.symlink_to(victim)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )

    engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert victim.read_text(encoding="utf-8") == "do-not-touch"
    assert fixed_temporary.is_symlink()


def test_checkpoint_failure_retains_a_replacement_at_the_unique_temporary_name(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint.json"
    engine, _llm = _engine(
        _FakeEvaluator(),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    parent = RetrievalGenome.seed()
    engine._original_parent = parent
    engine._current_parent = parent
    engine._scientific_inputs = {}
    real_publish = evolution_module._rename_artifact_entry_noreplace
    temporary_name: str | None = None
    displaced_name = ".owned-checkpoint-displaced.tmp"
    foreign_bytes = b"foreign replacement must survive\n"

    def replace_temporary_then_fail(
        parent_descriptor, source, destination
    ):
        nonlocal temporary_name
        if destination == checkpoint.name:
            temporary_name = str(source)
            evolution_module.os.rename(
                source,
                displaced_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            descriptor = evolution_module.os.open(
                source,
                evolution_module.os.O_WRONLY
                | evolution_module.os.O_CREAT
                | evolution_module.os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            with evolution_module.os.fdopen(descriptor, "wb") as handle:
                handle.write(foreign_bytes)
            raise OSError("checkpoint publication failed after replacement")
        return real_publish(parent_descriptor, source, destination)

    monkeypatch.setattr(
        evolution_module,
        "_rename_artifact_entry_noreplace",
        replace_temporary_then_fail,
    )

    with pytest.raises(RetrievalCheckpointError, match="publication"):
        engine._save_checkpoint(status="running", result=None)

    assert temporary_name is not None
    assert (checkpoint.parent / temporary_name).read_bytes() == foreign_bytes
    assert (checkpoint.parent / displaced_name).is_file()


def test_checkpoint_read_rejects_parent_replacement_at_file_open(
    tmp_path, monkeypatch
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    checkpoint = run_directory / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _ = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 1.2, "v003": 1.3}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    engine.evolve(RetrievalGenome.seed(), train, dev)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / checkpoint.name).write_bytes(checkpoint.read_bytes())
    displaced = tmp_path / "displaced"
    real_open = evolution_module.os.open
    swapped = False

    def swap_before_file_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        is_checkpoint_open = (
            dir_fd is None and Path(path) == checkpoint
        ) or (dir_fd is not None and path == checkpoint.name)
        if not swapped and is_checkpoint_open:
            swapped = True
            run_directory.rename(displaced)
            replacement.rename(run_directory)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evolution_module.os, "open", swap_before_file_open)
    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)

    with pytest.raises(
        RetrievalCheckpointError,
        match="path|parent|directory|changed",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)
    assert swapped


@pytest.mark.parametrize("existing", (False, True))
def test_checkpoint_commit_rejects_parent_replacement_at_entry_operation(
    tmp_path, monkeypatch, existing
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    checkpoint = run_directory / "checkpoint.json"
    engine, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
    )
    engine._scientific_inputs = {"science": "frozen"}
    engine._original_parent = RetrievalGenome.seed()
    engine._current_parent = RetrievalGenome.seed()
    if existing:
        engine._save_checkpoint(status="running", result=None)

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    if existing:
        (replacement / checkpoint.name).write_text(
            "do-not-touch", encoding="utf-8"
        )
    displaced = tmp_path / "displaced"
    real_operation = evolution_module._rename_artifact_entry_noreplace
    swapped = False

    def swap_before_entry_operation(
        parent_descriptor, source, destination
    ):
        nonlocal swapped
        if not swapped:
            swapped = True
            run_directory.rename(displaced)
            replacement.rename(run_directory)
        return real_operation(parent_descriptor, source, destination)

    monkeypatch.setattr(
        evolution_module,
        "_rename_artifact_entry_noreplace",
        swap_before_entry_operation,
    )
    engine._trace.append({"kind": "force_distinct_checkpoint_bytes"})

    with pytest.raises(
        RetrievalCheckpointError,
        match="path|parent|directory|changed",
    ):
        engine._save_checkpoint(status="running", result=None)
    assert swapped
    if existing:
        assert checkpoint.read_text(encoding="utf-8") == "do-not-touch"
    else:
        assert not checkpoint.exists()


@pytest.mark.parametrize("symlink_kind", ("path", "parent"))
def test_checkpoint_rejects_symlink_paths_and_parents(tmp_path, symlink_kind) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    if symlink_kind == "path":
        victim = real_directory / "victim.json"
        victim.write_text("do-not-touch", encoding="utf-8")
        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.symlink_to(victim)
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(real_directory, target_is_directory=True)
        checkpoint = linked_parent / "checkpoint.json"
    config = RetrievalEvolutionConfig(
        generations=1,
        train_folds=4,
        checkpoint_path=checkpoint,
        resume=False,
    )
    engine = RetrievalEvolutionEngine(
        FakeLLMClient(_responses(1)),
        _FakeEvaluator(),
        config,
    )

    with pytest.raises(RetrievalCheckpointError, match="symlink|unsafe|path"):
        engine.evolve(
            RetrievalGenome.seed(),
            _tasks("train", 80),
            _tasks("dev", 20, entity_offset=100),
        )
    if symlink_kind == "path":
        assert victim.read_text(encoding="utf-8") == "do-not-touch"


def test_completed_checkpoint_rejects_boolean_type_drift(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["result"]["accepted"] = "true"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="result|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_authority_rejects_a_fully_rewritten_release_decision(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        dev_overrides={"trace_final_smae": 1.01},
    )
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    assert result.accepted is False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["result"].update(
        {
            "accepted": True,
            "acceptance_reasons": ["all_dev_gates_passed"],
            "rejection_reasons": [],
            "selected_genome": payload["result"]["train_winner"],
            "release_genome": payload["result"]["train_winner"],
        }
    )
    checkpoint.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(RetrievalCheckpointError, match="authority|authenticate|digest"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_fresh_operator_activation_requires_an_independently_trusted_digest_and_epoch(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
            dev_overrides={"trace_final_smae": 4.0},
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    assert result.accepted is False
    trusted_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trusted_epoch = engine._checkpoint_authority_epoch
    assert type(trusted_epoch) is int

    forged = json.loads(checkpoint.read_text(encoding="utf-8"))
    forged["result"].update(
        {
            "accepted": True,
            "acceptance_reasons": ["all_dev_gates_passed"],
            "rejection_reasons": [],
            "selected_genome": forged["result"]["train_winner"],
            "release_genome": forged["result"]["train_winner"],
        }
    )
    checkpoint.write_text(
        json.dumps(forged, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    script = f"""
from pathlib import Path
from evolving_loop.retrieval_agent.evolution import (
    RetrievalCheckpointError,
    _authorize_retrieval_evolution_checkpoint_for_operator,
)
try:
    _authorize_retrieval_evolution_checkpoint_for_operator(
        Path({str(checkpoint)!r}),
        expected_sha256={trusted_sha256!r},
        expected_epoch={trusted_epoch!r},
    )
except RetrievalCheckpointError:
    raise SystemExit(0)
raise SystemExit("forged checkpoint was activated")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_authenticated_running_checkpoint_replays_completed_train_winner(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(2),
        checkpoint_path=checkpoint,
        generations=2,
    )
    parent = RetrievalGenome.seed()
    engine._validate_inputs(parent, train, dev)
    screen, remaining_folds = engine._partition_train(train)
    engine._scientific_inputs = engine._science_signature(parent, train, dev)
    assert engine._load_checkpoint(
        parent, train, dev, screen, remaining_folds
    ) is None
    engine._run_generation(0, screen, remaining_folds)
    genuine = engine._generations[0]
    forged_winner = RetrievalGenome.from_payload(genuine.child_proposals[2])
    engine._generations[0] = replace(
        genuine,
        train_winner_version=forged_winner.version,
        train_winner_fingerprint=forged_winner.fingerprint(),
    )
    engine._current_parent = forged_winner
    for event in engine._trace:
        if event.get("kind") == "generation_completed":
            event["train_winner"] = forged_winner.version
    engine._save_checkpoint(status="running", result=None)

    resumed_evaluator = _FakeEvaluator()
    resumed, _ = _engine(
        resumed_evaluator,
        [
            json.dumps(_proposal(forged_winner, "v004", "A")),
            json.dumps(_proposal(forged_winner, "v005", "B")),
            json.dumps(_proposal(forged_winner, "v006", "C")),
        ],
        checkpoint_path=checkpoint,
        generations=2,
    )

    with pytest.raises(
        RetrievalCheckpointError,
        match="replay|selection|winner|generation|Train",
    ):
        resumed.evolve(parent, train, dev)
    assert resumed_evaluator.calls == []


def test_running_checkpoint_rejects_out_of_order_partial_evaluation_coverage(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    parent = RetrievalGenome.seed()
    engine, _llm = _engine(
        _FakeEvaluator(),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    engine._validate_inputs(parent, train, dev)
    screen, remaining_folds = engine._partition_train(train)
    engine._scientific_inputs = engine._science_signature(parent, train, dev)
    assert engine._load_checkpoint(
        parent, train, dev, screen, remaining_folds
    ) is None
    children, _rejections = engine._children_for_generation(
        0,
        parent,
        parent_library=engine._library_for(parent),
    )
    child_b = dict(children)["B"]
    engine._evaluate_batch(
        child_b,
        screen,
        stage="g0_child_B_screen_train",
        readonly=False,
        library=engine._library_for(child_b),
    )

    resumed_evaluator = _FakeEvaluator()
    resumed, _ = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
    )

    with pytest.raises(
        RetrievalCheckpointError,
        match="stage|cursor|coverage|schedule|outcome",
    ):
        resumed.evolve(parent, train, dev)
    assert resumed_evaluator.calls == []


@pytest.mark.parametrize(
    ("cursor", "resumed_dev_stages", "accepted"),
    (
        ("none", ("parent_dev", "child_dev"), True),
        ("parent", ("child_dev",), True),
        ("parent_child", (), True),
        ("parent_child_terminal", (), False),
    ),
)
def test_running_checkpoint_resumes_each_legal_final_dev_prefix_exactly_once(
    tmp_path, cursor, resumed_dev_stages, accepted
) -> None:
    class FinalDevEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, **kwargs):
            result = super().evaluate(genome, tasks, **kwargs)
            if cursor == "parent_child_terminal" and kwargs["stage"] == "child_dev":
                raise RuntimeError("terminal Child Dev forecast")
            return result

    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    parent = RetrievalGenome.seed()
    interrupted_evaluator = FinalDevEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    interrupted, _ = _engine(
        interrupted_evaluator,
        _responses(1),
        checkpoint_path=checkpoint,
    )
    _complete_train_checkpoint(interrupted, parent, train, dev)
    assert interrupted._current_parent is not None
    winner = interrupted._current_parent

    if cursor != "none":
        interrupted._evaluate_batch(
            parent,
            dev,
            stage="parent_dev",
            readonly=True,
            library=interrupted._readonly_library(parent),
        )
    if cursor in {"parent_child", "parent_child_terminal"}:
        if cursor == "parent_child_terminal":
            with pytest.raises(
                RetrievalForecastingFailure,
                match="EvaluatorExecutionFailure:trusted evaluator execution failed",
            ):
                interrupted._evaluate_batch(
                    winner,
                    dev,
                    stage="child_dev",
                    readonly=True,
                    library=interrupted._readonly_library(winner),
                )
        else:
            interrupted._evaluate_batch(
                winner,
                dev,
                stage="child_dev",
                readonly=True,
                library=interrupted._readonly_library(winner),
            )

    resumed_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    resumed, resumed_llm = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
    )

    result = resumed.evolve(parent, train, dev)

    assert resumed_llm.calls == []
    assert tuple(
        call.stage
        for call in resumed_evaluator.calls
        if call.stage in {"parent_dev", "child_dev"}
    ) == resumed_dev_stages
    all_dev_calls = [
        call
        for call in (*interrupted_evaluator.calls, *resumed_evaluator.calls)
        if call.stage in {"parent_dev", "child_dev"}
    ]
    assert [call.stage for call in all_dev_calls] == ["parent_dev", "child_dev"]
    assert all(
        (call.persist, call.writers_enabled, call.evolver_enabled)
        == (False, False, False)
        for call in all_dev_calls
    )
    assert result.accepted is accepted
    if cursor == "parent_child_terminal":
        assert result.child_dev is None
        assert result.rejection_reasons == (
            "child_dev_failure:EvaluatorExecutionFailure:"
            "trusted evaluator execution failed",
        )
    else:
        assert result.parent_dev is not None
        assert result.child_dev is not None


@pytest.mark.parametrize(
    "cursor",
    (
        "child_without_parent",
        "parent_wrong_genome",
        "child_wrong_genome",
        "duplicate_parent_stage",
        "parent_before_train_complete",
        "release_audit_in_running",
    ),
)
def test_running_checkpoint_rejects_each_illegal_final_dev_cursor(
    tmp_path, cursor
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    parent = RetrievalGenome.seed()
    generations = 2 if cursor == "parent_before_train_complete" else 1
    interrupted, _ = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(generations),
        checkpoint_path=checkpoint,
        generations=generations,
    )
    _complete_train_checkpoint(
        interrupted,
        parent,
        train,
        dev,
        generation_count=1,
    )
    assert interrupted._current_parent is not None
    winner = interrupted._current_parent
    assert winner.fingerprint() != parent.fingerprint()

    if cursor == "child_without_parent":
        interrupted._evaluate_batch(
            winner,
            dev,
            stage="child_dev",
            readonly=True,
            library=interrupted._readonly_library(winner),
        )
    elif cursor == "parent_wrong_genome":
        interrupted._evaluate_batch(
            winner,
            dev,
            stage="parent_dev",
            readonly=True,
            library=interrupted._readonly_library(winner),
        )
    elif cursor == "child_wrong_genome":
        interrupted._evaluate_batch(
            parent,
            dev,
            stage="parent_dev",
            readonly=True,
            library=interrupted._readonly_library(parent),
        )
        interrupted._evaluate_batch(
            parent,
            dev,
            stage="child_dev",
            readonly=True,
            library=interrupted._readonly_library(parent),
        )
    elif cursor == "duplicate_parent_stage":
        interrupted._evaluate_batch(
            parent,
            dev,
            stage="parent_dev",
            readonly=True,
            library=interrupted._readonly_library(parent),
        )
        interrupted._evaluate_batch(
            winner,
            dev,
            stage="parent_dev",
            readonly=True,
            library=interrupted._readonly_library(winner),
        )
    elif cursor == "parent_before_train_complete":
        interrupted._evaluate_batch(
            parent,
            dev,
            stage="parent_dev",
            readonly=True,
            library=interrupted._readonly_library(parent),
        )
    else:
        interrupted._trace.append(
            {
                "kind": "dev_completed",
                "original_parent": parent.version,
                "train_winner": winner.version,
                "accepted": False,
                "rejection_reasons": ["forged"],
            }
        )
        interrupted._save_checkpoint(status="running", result=None)

    resumed_evaluator = _FakeEvaluator()
    resumed, resumed_llm = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
        generations=generations,
    )

    with pytest.raises(
        RetrievalCheckpointError,
        match="stage|cursor|coverage|schedule|audit|outcome",
    ):
        resumed.evolve(parent, train, dev)
    assert resumed_evaluator.calls == []
    assert resumed_llm.calls == []


def test_authenticated_checkpoint_rederives_rejected_dev_acceptance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
            dev_overrides={"trace_final_smae": 1.01},
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    forged = replace(
        result,
        accepted=True,
        acceptance_reasons=("all_dev_gates_passed",),
        rejection_reasons=(),
        selected_genome=result.train_winner,
        release_genome=result.train_winner,
    )
    engine._save_checkpoint(status="complete", result=forged)

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(RetrievalCheckpointError, match="accept|Dev|gate|result"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


@pytest.mark.parametrize("mutation", ("cache_key", "coverage"))
def test_authenticated_checkpoint_recomputes_cache_keys_and_coverage(
    tmp_path,
    mutation,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    cache_key, record = next(iter(engine._cache_records.items()))
    if mutation == "cache_key":
        record["task_cache_keys"][0] = "f" * 64
    else:
        record["task_ids"][0] = "invented-task"
    engine._cache_records[cache_key] = record
    engine._save_checkpoint(status="complete", result=result)

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(RetrievalCheckpointError, match="cache|coverage|task"):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


@pytest.mark.parametrize("mutation", ("generation_trace", "child_dev_trace"))
def test_authenticated_checkpoint_replays_selection_and_child_dev_trace(
    tmp_path,
    mutation,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    engine, _llm = _engine(
        _FakeEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    result = engine.evolve(RetrievalGenome.seed(), train, dev)
    if mutation == "generation_trace":
        forged_generation = replace(
            result.generations[0],
            rejection_reasons={},
        )
        engine._generations[0] = forged_generation
        forged_result = replace(result, generations=(forged_generation,))
    else:
        assert result.child_dev is not None
        forged_child_dev = replace(
            result.child_dev,
            p95_smae=4.0,
            task_traces=tuple(
                {**trace, "final_smae": 4.0}
                for trace in result.child_dev.task_traces
            ),
        )
        forged_result = replace(result, child_dev=forged_child_dev)
    engine._save_checkpoint(status="complete", result=forged_result)

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="generation|selection|trace|Dev|result|evaluation",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_rejects_a_rewritten_complete_child_vector(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    generation = payload["generations"][0]
    for field_name in ("child_versions", "child_fingerprints", "child_scopes"):
        generation[field_name].pop()
    payload["child_fingerprints"].pop()
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError, match="Child|generation|result|checkpoint"
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


def test_checkpoint_generation_cursor_must_match_completed_generation_records(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    engine, _llm = _engine(evaluator, _responses(1), checkpoint_path=checkpoint)
    engine.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["next_generation"] = 0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    resumed, _ = _engine(_FakeEvaluator(), [], checkpoint_path=checkpoint)
    with pytest.raises(
        RetrievalCheckpointError,
        match="generation|authority|authentication",
    ):
        resumed.evolve(RetrievalGenome.seed(), train, dev)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("generation", True),
        ("parent_version", 7),
        ("child_versions", [1, "v002", "v003"]),
        ("screen_task_ids", [False]),
        ("promoted_fingerprints", "not-a-list"),
        ("rejection_reasons", {"v001": False}),
    ),
)
def test_generation_checkpoint_parser_rejects_exact_json_type_drift(
    field_name,
    replacement,
) -> None:
    trace = RetrievalGenerationTrace(
        generation=0,
        parent_version="v000",
        parent_fingerprint="0" * 64,
        child_versions=("v001", "v002", "v003"),
        child_fingerprints=("1" * 64, "2" * 64, "3" * 64),
        child_scopes=("A", "B", "C"),
        child_proposals=({}, {}, {}),
        screen_task_ids=("train_001",),
        fold_entities=(("entity_a",), ("entity_b",)),
        promoted_fingerprints=("1" * 64,),
        train_winner_version="v001",
        train_winner_fingerprint="1" * 64,
        rejection_reasons={"v002": "screen_rank:not_promoted"},
        screen_summaries={},
        train_summaries={},
    )
    payload = trace.to_payload()
    payload[field_name] = replacement

    with pytest.raises(RetrievalCheckpointError, match="generation|trace|checkpoint"):
        RetrievalGenerationTrace.from_payload(payload)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("acceptance_reasons", [False]),
        ("rejection_reasons", [7]),
        ("generations", {}),
        ("trace", ["not-an-event"]),
    ),
)
def test_completed_result_parser_never_coerces_json_types(
    field_name,
    replacement,
) -> None:
    parent = RetrievalGenome.seed()
    result = RetrievalEvolutionResult(
        original_parent=parent,
        train_winner=parent,
        selected_genome=parent,
        accepted=False,
        acceptance_reasons=(),
        rejection_reasons=("no_gain",),
        parent_dev=None,
        child_dev=None,
        generations=(),
        trace=(),
        release_genome=None,
        release_published=False,
    )
    payload = result.to_payload()
    payload[field_name] = replacement

    with pytest.raises(RetrievalCheckpointError, match="result|checkpoint"):
        RetrievalEvolutionResult.from_payload(payload)


@pytest.mark.parametrize("mutation", ("summary_integer", "trace_integer"))
def test_checkpoint_evaluation_parser_normalizes_finite_integer_metrics(
    mutation,
) -> None:
    payload = _evaluation("v001", _tasks("train", 1)).to_payload()
    if mutation == "summary_integer":
        payload["mean_final_smae"] = 1
    else:
        payload["task_traces"][0]["final_smae"] = 1

    parsed = RetrievalEvaluation.from_payload(payload)

    assert type(parsed.mean_final_smae) is float
    assert type(parsed.task_traces[0]["final_smae"]) is float


@pytest.mark.parametrize("mutation", ("summary_boolean", "trace_boolean"))
def test_checkpoint_evaluation_parser_rejects_boolean_metrics(mutation) -> None:
    payload = _evaluation("v001", _tasks("train", 1)).to_payload()
    if mutation == "summary_boolean":
        payload["mean_final_smae"] = True
    else:
        payload["task_traces"][0]["final_smae"] = False

    with pytest.raises(RetrievalCheckpointError, match="evaluation|metric|trace|finite"):
        RetrievalEvaluation.from_payload(payload)


def test_evaluator_integer_trace_metrics_persist_and_resume_as_floats(tmp_path) -> None:
    class IntegerTraceEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, **kwargs):
            result = super().evaluate(genome, tasks, **kwargs)
            return replace(
                result,
                task_traces=tuple(
                    {**trace, "final_smae": 1}
                    for trace in result.task_traces
                ),
            )

    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    first, _ = _engine(
        IntegerTraceEvaluator(
            errors={"v000": 1.0, "v001": 0.9, "v002": 1.2, "v003": 1.3}
        ),
        _responses(1),
        checkpoint_path=checkpoint,
    )
    expected = first.evolve(RetrievalGenome.seed(), train, dev)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert all(
        type(trace["final_smae"]) is float
        for record in payload["evaluation_cache"].values()
        for trace in record["evaluation"]["task_traces"]
    )

    resumed_evaluator = _FakeEvaluator()
    resumed, _ = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
    )

    assert resumed.evolve(RetrievalGenome.seed(), train, dev) == expected
    assert resumed_evaluator.calls == []


def test_duplicate_task_trace_is_a_forecasting_failure_not_full_coverage() -> None:
    class DuplicateTraceEvaluator(_FakeEvaluator):
        def evaluate(self, genome, tasks, **kwargs):
            result = super().evaluate(genome, tasks, **kwargs)
            if genome.version == "v001" and "screen_train" in kwargs["stage"]:
                duplicate = dict(result.task_traces[0])
                return replace(
                    result,
                    task_traces=tuple(duplicate for _task in tasks),
                )
            return result

    evaluator = DuplicateTraceEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.generations[0].rejection_reasons["v001"].startswith(
        "forecasting_failure:"
    )


def test_transient_evaluator_retry_is_not_a_forecasting_failure() -> None:
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        transient_stage="g0_child_A_screen_train",
        failing_version="v002",
    )
    engine, _llm = _engine(evaluator, _responses(1), transient_retries=1)

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    transient_calls = [
        call for call in evaluator.calls if call.stage == "g0_child_A_screen_train"
    ]
    forecasting_calls = [
        call
        for call in evaluator.calls
        if call.version == "v002" and call.stage == "g0_child_B_screen_train"
    ]
    assert len(transient_calls) == 2
    assert len(forecasting_calls) == 1
    assert any(
        event["kind"] == "transient_retry" for event in result.trace
    )
    assert result.generations[0].rejection_reasons["v002"].startswith(
        "forecasting_failure:"
    )


def test_resume_reuses_checkpointed_children_after_transient_exhaustion(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    interrupted_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        transient_stage="g0_parent_screen_train",
    )
    interrupted, first_llm = _engine(
        interrupted_evaluator,
        _responses(1),
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    with pytest.raises(TransientLLMError):
        interrupted.evolve(RetrievalGenome.seed(), train, dev)
    assert len(first_llm.calls) == 3

    resumed_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2}
    )
    resumed, second_llm = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )

    result = resumed.evolve(RetrievalGenome.seed(), train, dev)

    assert second_llm.calls == []
    assert result.generations[0].child_versions == ("v001", "v002", "v003")
    assert len(result.generations[0].child_fingerprints) == 3


@pytest.mark.parametrize("mutation", ("reorder", "invalid_fingerprint"))
def test_pending_checkpoint_requires_canonical_scoped_rows_and_recomputed_fingerprints(
    tmp_path,
    mutation,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    parent = RetrievalGenome.seed()
    interrupted_evaluator = _FakeEvaluator(
        transient_stage="g0_parent_screen_train"
    )
    interrupted, _llm = _engine(
        interrupted_evaluator,
        [
            "not json",
            json.dumps(_proposal(parent, "v002", "B")),
            json.dumps(_proposal(parent, "v003", "C")),
        ],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    with pytest.raises(TransientLLMError):
        interrupted.evolve(parent, train, dev)
    assert interrupted._pending_children is not None
    rows = interrupted._pending_children["children"]
    assert isinstance(rows, list) and len(rows) == 3
    if mutation == "reorder":
        rows[:] = (rows[1], rows[0], rows[2])
        interrupted._all_child_fingerprints[:] = [
            row["fingerprint"] for row in rows
        ]
    else:
        rows[0]["fingerprint"] = "f" * 64
        interrupted._all_child_fingerprints[0] = "f" * 64
    interrupted._save_checkpoint(status="running", result=None)

    resumed, _ = _engine(
        _FakeEvaluator(),
        [],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    with pytest.raises(RetrievalCheckpointError, match="pending|scope|fingerprint"):
        resumed.evolve(parent, train, dev)


def test_resume_never_retries_a_checkpointed_terminal_forecasting_failure(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    train = _tasks("train", 80)
    dev = _tasks("dev", 20, entity_offset=100)
    interrupted_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        transient_stage="g0_child_B_screen_train",
        failing_version="v001",
    )
    interrupted, _first_llm = _engine(
        interrupted_evaluator,
        _responses(1),
        checkpoint_path=checkpoint,
        transient_retries=0,
    )
    with pytest.raises(TransientLLMError):
        interrupted.evolve(RetrievalGenome.seed(), train, dev)
    assert len(
        [
            call
            for call in interrupted_evaluator.calls
            if call.version == "v001"
            and call.stage == "g0_child_A_screen_train"
        ]
    ) == 1

    resumed_evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.1, "v002": 0.95, "v003": 1.2}
    )
    resumed, resumed_llm = _engine(
        resumed_evaluator,
        [],
        checkpoint_path=checkpoint,
        transient_retries=0,
    )

    result = resumed.evolve(RetrievalGenome.seed(), train, dev)

    assert resumed_llm.calls == []
    assert not any(
        call.version == "v001" and call.stage == "g0_child_A_screen_train"
        for call in resumed_evaluator.calls
    )
    assert result.generations[0].rejection_reasons["v001"] == (
        "forecasting_failure:EvaluatorExecutionFailure:"
        "trusted evaluator execution failed"
    )


def test_rejected_dev_gate_has_complete_trace_and_never_publishes_release() -> None:
    evaluator = _FakeEvaluator(
        errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2},
        dev_overrides={"trace_final_smae": 1.01},
    )
    engine, _llm = _engine(evaluator, _responses(1))

    result = engine.evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert result.accepted is False
    assert result.selected_genome == RetrievalGenome.seed()
    assert result.release_genome is None
    assert result.release_published is False
    assert "p95_smae" in result.rejection_reasons
    assert result.parent_dev is not None
    assert result.child_dev is not None
    assert len(result.generations[0].child_fingerprints) == 3
    assert {event["kind"] for event in result.trace} >= {
        "generation_started",
        "child_proposed",
        "screen_completed",
        "generation_completed",
        "dev_completed",
        "release_rejected",
    }


class _TransientMutationClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.raised = False

    def complete(self, *, system: str, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        if not self.raised:
            self.raised = True
            raise TransientLLMError("temporary mutation outage")
        return LLMResponse(self.responses.pop(0))


def test_transient_mutation_is_retried_without_creating_a_failed_child() -> None:
    evaluator = _FakeEvaluator(errors={"v000": 1.0, "v001": 0.9, "v002": 0.95, "v003": 1.2})
    llm = _TransientMutationClient(_responses(1))
    config = RetrievalEvolutionConfig(
        generations=1,
        screen_tasks=8,
        promote=2,
        train_folds=4,
        transient_retries=1,
        dataset_split_hash="split-v1",
        verifier_hash="verifier-v1",
        evaluator_hash="evaluator-v1",
        metric_hash="metric-v1",
    )

    result = RetrievalEvolutionEngine(llm, evaluator, config).evolve(
        RetrievalGenome.seed(),
        _tasks("train", 80),
        _tasks("dev", 20, entity_offset=100),
    )

    assert len(llm.calls) == 4
    assert len(result.generations[0].child_fingerprints) == 3
    assert "v001" not in result.generations[0].rejection_reasons
