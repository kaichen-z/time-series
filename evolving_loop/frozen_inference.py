"""Label-free inference and Dr-CiK submission export for frozen harnesses."""
from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from common.metrics import aggregate_drcik_point_metrics
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import ContextTask, Document
from evolving_loop.evaluation import score_after_resolution
from evolving_loop.harness import EvolvingForecastHarness
from evolving_loop.retrieval_agent.evolution import (
    _open_checkpoint_parent,
    _read_checkpoint_entry_snapshot,
    _revalidate_checkpoint_parent,
    _safe_checkpoint_path,
    _unique_checkpoint_temporary,
)
from evolving_loop.retrieval_agent.skill_library import (
    _move_artifact_entry_to_quarantine,
    _rename_artifact_entry_noreplace,
)

HarnessFactory = Callable[[HarnessPolicy], EvolvingForecastHarness]

_FROZEN_OUTPUT_NAMES = (
    "forecasts.jsonl",
    "deep_research.jsonl",
    "run_report.jsonl",
    "summary.json",
)


@dataclass(frozen=True)
class FrozenOutputTarget:
    """Preflight identity for one fixed, descriptor-published output bundle."""

    path: Path
    output_root: Path
    directory_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    root_identity: tuple[int, int]
    member_identities: tuple[tuple[str, tuple[int, int] | None], ...]


def _directory_identity(descriptor: int, label: str) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"frozen output {label} is not a directory")
    return metadata.st_dev, metadata.st_ino


def _frozen_output_entry_identity(
    directory_descriptor: int, name: str
) -> tuple[int, int] | None:
    try:
        metadata = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            f"frozen output member {name} must be a regular file"
        )
    return metadata.st_dev, metadata.st_ino


def _revalidate_frozen_output_target(
    target: FrozenOutputTarget,
    output_descriptor: int,
    parent_descriptor: int,
    root_descriptor: int,
) -> None:
    try:
        if _directory_identity(
            output_descriptor, "directory"
        ) != target.directory_identity:
            raise ValueError("frozen output directory identity changed")
        if _directory_identity(
            parent_descriptor, "parent"
        ) != target.parent_identity:
            raise ValueError("frozen output parent identity changed")
        if _directory_identity(
            root_descriptor, "root"
        ) != target.root_identity:
            raise ValueError("frozen output root identity changed")
        _revalidate_checkpoint_parent(
            target.path / ".frozen-output-entry", output_descriptor
        )
        _revalidate_checkpoint_parent(target.path, parent_descriptor)
        _revalidate_checkpoint_parent(
            target.output_root / ".frozen-output-root-entry",
            root_descriptor,
        )
        visible = os.stat(
            target.path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(visible.st_mode)
            or (visible.st_dev, visible.st_ino) != target.directory_identity
        ):
            raise ValueError("frozen output directory was replaced")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "frozen output directory identity changed after preflight"
        ) from error


def prepare_frozen_output_target(
    output_dir: str | Path,
    *,
    output_root: str | Path | None = None,
) -> FrozenOutputTarget:
    """Create/pin an output directory and snapshot every fixed bundle member."""
    destination = _safe_checkpoint_path(Path(output_dir))
    root = _safe_checkpoint_path(
        Path(output_root) if output_root is not None else destination
    )
    if destination != root and root not in destination.parents:
        raise ValueError("frozen output escapes the approved output root")

    descriptors: list[int] = []
    try:
        _, output_descriptor = _open_checkpoint_parent(
            destination / ".frozen-output-entry", create=True
        )
        descriptors.append(output_descriptor)
        _, parent_descriptor = _open_checkpoint_parent(
            destination, create=False
        )
        descriptors.append(parent_descriptor)
        _, root_descriptor = _open_checkpoint_parent(
            root / ".frozen-output-root-entry", create=False
        )
        descriptors.append(root_descriptor)
        target = FrozenOutputTarget(
            path=destination,
            output_root=root,
            directory_identity=_directory_identity(
                output_descriptor, "directory"
            ),
            parent_identity=_directory_identity(
                parent_descriptor, "parent"
            ),
            root_identity=_directory_identity(root_descriptor, "root"),
            member_identities=tuple(
                (
                    name,
                    _frozen_output_entry_identity(output_descriptor, name),
                )
                for name in _FROZEN_OUTPUT_NAMES
            ),
        )
        _revalidate_frozen_output_target(
            target,
            output_descriptor,
            parent_descriptor,
            root_descriptor,
        )
        return target
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("cannot establish a safe frozen output directory") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _open_frozen_output_target(
    target: FrozenOutputTarget,
) -> tuple[int, int, int]:
    descriptors: list[int] = []
    try:
        _, output_descriptor = _open_checkpoint_parent(
            target.path / ".frozen-output-entry", create=False
        )
        descriptors.append(output_descriptor)
        _, parent_descriptor = _open_checkpoint_parent(
            target.path, create=False
        )
        descriptors.append(parent_descriptor)
        _, root_descriptor = _open_checkpoint_parent(
            target.output_root / ".frozen-output-root-entry", create=False
        )
        descriptors.append(root_descriptor)
        _revalidate_frozen_output_target(
            target,
            output_descriptor,
            parent_descriptor,
            root_descriptor,
        )
        for name, expected_identity in target.member_identities:
            if (
                _frozen_output_entry_identity(output_descriptor, name)
                != expected_identity
            ):
                raise ValueError(
                    f"frozen output member {name} identity changed after preflight"
                )
        return output_descriptor, parent_descriptor, root_descriptor
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _publish_frozen_output_entry(
    output_descriptor: int,
    name: str,
    encoded: bytes,
    *,
    expected_identity: tuple[int, int] | None,
) -> tuple[int, int]:
    """Publish one staged entry relative to the held output descriptor."""
    if _frozen_output_entry_identity(output_descriptor, name) != expected_identity:
        raise ValueError(
            f"frozen output member {name} changed before publication"
        )
    temporary = _unique_checkpoint_temporary(
        output_descriptor, name, encoded
    )
    staged_identity, staged = _read_checkpoint_entry_snapshot(
        output_descriptor, temporary
    )
    if staged != encoded:
        raise ValueError("frozen output staged bytes changed before publication")

    if _frozen_output_entry_identity(output_descriptor, name) != expected_identity:
        raise ValueError(
            f"frozen output member {name} changed during publication"
        )
    if expected_identity is not None:
        quarantine = _move_artifact_entry_to_quarantine(
            output_descriptor, name
        )
        if quarantine is None:
            raise ValueError(
                f"frozen output member {name} disappeared during publication"
            )
        if (
            _frozen_output_entry_identity(output_descriptor, quarantine)
            != expected_identity
        ):
            raise ValueError(
                f"frozen output member {name} quarantine identity changed"
            )
    try:
        _rename_artifact_entry_noreplace(
            output_descriptor, temporary, name
        )
        visible_identity, visible = _read_checkpoint_entry_snapshot(
            output_descriptor, name
        )
        if visible_identity != staged_identity or visible != encoded:
            raise ValueError(
                f"frozen output member {name} changed after publication"
            )
        os.fsync(output_descriptor)
        return visible_identity
    except Exception as error:
        if name == "summary.json":
            try:
                _retire_exact_frozen_summary(
                    output_descriptor, staged_identity
                )
            except Exception as retirement_error:
                raise ValueError(
                    "frozen output summary publication became uncertain and "
                    "the exact completion marker could not be retired safely"
                ) from retirement_error
        raise ValueError(
            f"frozen output member {name} could not be published without replacement"
        ) from error


def _retire_exact_frozen_summary(
    output_descriptor: int,
    expected_identity: tuple[int, int],
) -> None:
    """Hide only the exact summary inode published by this transaction."""
    if (
        _frozen_output_entry_identity(output_descriptor, "summary.json")
        != expected_identity
    ):
        return
    quarantine = _move_artifact_entry_to_quarantine(
        output_descriptor, "summary.json"
    )
    if quarantine is None:
        if _frozen_output_entry_identity(output_descriptor, "summary.json") is None:
            return
        raise ValueError("frozen output summary disappeared ambiguously")
    if (
        _frozen_output_entry_identity(output_descriptor, quarantine)
        != expected_identity
    ):
        raise ValueError("frozen output summary retirement changed identity")
    os.fsync(output_descriptor)


def _public_score_payload(outcome: object | None) -> dict[str, object] | None:
    """Expose point-score fields without evaluator GT/document-label diagnostics."""
    if outcome is None:
        return None
    fields = (
        "task_id",
        "final_smae",
        "final_srmse",
        "coding_oracle_smae",
        "coding_oracle_srmse",
        "contextual_oracle_smae",
        "contextual_oracle_srmse",
        "decision_selection_smae_regret",
        "decision_selection_srmse_regret",
        "candidate_count",
    )
    return {field: getattr(outcome, field, None) for field in fields}


def inference_view(task: ContextTask) -> ContextTask:
    """Strip every evaluator-only field before any mutable harness code runs."""
    return replace(
        task,
        numeric=replace(task.numeric, future_values=()),
        documents=tuple(
            Document(document.document_id, document.content)
            for document in task.documents
        ),
        gt_evidence=(),
        labels_public=False,
    )


def run_frozen_inference(
    policy: HarnessPolicy,
    tasks: Sequence[ContextTask],
    harness_factory: HarnessFactory,
    *,
    output_dir: str | Path | FrozenOutputTarget,
    samples: int = 100,
    score_public: bool = False,
    artifact_kind: str = "genome",
) -> dict:
    """Run an immutable policy without outcome learning and export submissions."""
    if not tasks:
        raise ValueError("frozen inference needs at least one task")
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if score_public and any(not task.labels_public for task in tasks):
        raise ValueError("--score-public is forbidden when any selected task is hidden")

    output_target = (
        output_dir
        if isinstance(output_dir, FrozenOutputTarget)
        else prepare_frozen_output_target(output_dir)
    )
    destination = output_target.path
    forecast_path = destination / "forecasts.jsonl"
    research_path = destination / "deep_research.jsonl"
    report_path = destination / "run_report.jsonl"
    seen_ids: set[str] = set()
    outcomes = []
    forecast_rows: list[str] = []
    research_rows: list[str] = []
    report_rows: list[str] = []

    for task in tasks:
        task_id = task.numeric.task_id
        if task_id in seen_ids:
            raise ValueError(f"duplicate benchmark_id in inference input: {task_id}")
        seen_ids.add(task_id)

        # One isolated harness per task prevents accidental online skill writes
        # from changing later hidden predictions.
        result = harness_factory(policy).run(
            inference_view(task), allow_skill_writes=False
        )
        values = tuple(float(value) for value in result.forecast)
        if len(values) != task.numeric.prediction_length:
            raise ValueError(
                f"{task_id}: forecast length {len(values)} != "
                f"prediction_length {task.numeric.prediction_length}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{task_id}: forecast contains non-finite values")

        # The current harness is point-valued. Repeated trajectories are a valid,
        # deliberately degenerate empirical distribution and make that limitation explicit.
        forecast_rows.append(
            json.dumps(
                {
                    "benchmark_id": task_id,
                    "samples": [list(values) for _ in range(samples)],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        research_rows.append(
            json.dumps(
                {
                    "benchmark_id": task_id,
                    "cited_document_ids": list(result.retrieval.selected_document_ids),
                    "evidence": [item.claim for item in result.retrieval.evidence],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        outcome = score_after_resolution(task, result) if score_public else None
        if outcome is not None:
            outcomes.append(outcome)
        card = getattr(result, "retrieval_card", None)
        if card is not None:
            retrieval_payload = card.to_payload()
            assumption_stances = [
                {
                    "chain_id": chain.chain_id,
                    "assumption_ids": list(chain.addressed_assumption_ids),
                    "stance": chain.stance,
                }
                for chain in card.chains
            ]
            used_skill_ids = list(
                dict.fromkeys(
                    skill_id
                    for chain in card.chains
                    for skill_id in chain.used_skill_ids
                )
            )
        else:
            # Keep the report schema stable for legacy single-pass artifacts.
            retrieval_payload = {
                "round1": None,
                "round2": None,
                "chains": [],
                "selected_document_ids": list(
                    result.retrieval.selected_document_ids
                ),
                "rejected": list(result.retrieval.rejected),
                "unresolved_contradictions": [],
                "complete": bool(result.retrieval.sufficient),
                "gaps": [],
            }
            assumption_stances = []
            used_skill_ids = []
        report_rows.append(
            json.dumps(
                {
                    "benchmark_id": task_id,
                    "artifact_kind": artifact_kind,
                    "policy_version": policy.version,
                    "release_sha256": policy.retrieval_release_sha256,
                    "labels_accessed": score_public,
                    "selected_candidate_id": result.decision.selected.candidate_id,
                    "host_default_id": result.decision.host_default_id,
                    "forecast": list(values),
                    "retrieved_document_ids": list(result.retrieval.selected_document_ids),
                    "evidence": [asdict(item) for item in result.retrieval.evidence],
                    "impacts": [asdict(item) for item in result.retrieval.impacts],
                    "retrieval_rejections": list(result.retrieval.rejected),
                    "retrieval": retrieval_payload,
                    "assumption_stances": assumption_stances,
                    "used_skill_ids": used_skill_ids,
                    "decision": {
                        "rationale": result.decision.rationale,
                        "supporting_document_ids": list(
                            result.decision.supporting_document_ids
                        ),
                        "rejection_reason": result.decision.rejection_reason,
                        "llm_override_accepted": result.decision.llm_override_accepted,
                    },
                    "coding_candidates": [
                        {
                            "candidate_id": item.candidate_id,
                            "assumption": item.assumption,
                            "failure_condition": item.failure_condition,
                            "hindcast_smae": item.hindcast_smae,
                            "hindcast_srmse": item.hindcast_srmse,
                        }
                        for item in result.candidates
                    ],
                    "outcome": _public_score_payload(outcome),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    aggregate = (
        aggregate_drcik_point_metrics(
            [
                {"smae": item.final_smae, "srmse": item.final_srmse}
                for item in outcomes
            ]
        )
        if outcomes
        else None
    )
    summary = {
        "artifact_kind": artifact_kind,
        "policy_version": policy.version,
        "release_sha256": policy.retrieval_release_sha256,
        "num_tasks": len(tasks),
        "labels_accessed": score_public,
        "samples_per_task": samples,
        "probabilistic_note": (
            "Point forecasts are exported as repeated trajectories; uncertainty is not yet modeled."
        ),
        "forecasts_path": str(forecast_path),
        "deep_research_path": str(research_path),
        "run_report_path": str(report_path),
        "mean_smae": aggregate["smae"] if aggregate else None,
        "mean_srmse": aggregate["srmse"] if aggregate else None,
    }
    payloads = {
        "forecasts.jsonl": "".join(forecast_rows).encode("utf-8"),
        "deep_research.jsonl": "".join(research_rows).encode("utf-8"),
        "run_report.jsonl": "".join(report_rows).encode("utf-8"),
        # Publish the completion marker last.
        "summary.json": json.dumps(
            summary, indent=2, ensure_ascii=False
        ).encode("utf-8"),
    }
    output_descriptor, parent_descriptor, root_descriptor = (
        _open_frozen_output_target(output_target)
    )
    try:
        expected = dict(output_target.member_identities)
        published_summary_identity: tuple[int, int] | None = None
        prior_summary_identity = expected["summary.json"]
        if prior_summary_identity is not None:
            _revalidate_frozen_output_target(
                output_target,
                output_descriptor,
                parent_descriptor,
                root_descriptor,
            )
            summary_quarantine = _move_artifact_entry_to_quarantine(
                output_descriptor, "summary.json"
            )
            if (
                summary_quarantine is None
                or _frozen_output_entry_identity(
                    output_descriptor, summary_quarantine
                )
                != prior_summary_identity
            ):
                raise ValueError(
                    "frozen output completion marker changed during retirement"
                )
            os.fsync(output_descriptor)
            expected["summary.json"] = None
            _revalidate_frozen_output_target(
                output_target,
                output_descriptor,
                parent_descriptor,
                root_descriptor,
            )
        for name in _FROZEN_OUTPUT_NAMES:
            _revalidate_frozen_output_target(
                output_target,
                output_descriptor,
                parent_descriptor,
                root_descriptor,
            )
            published_identity = _publish_frozen_output_entry(
                output_descriptor,
                name,
                payloads[name],
                expected_identity=expected[name],
            )
            if name == "summary.json":
                published_summary_identity = published_identity
            _revalidate_frozen_output_target(
                output_target,
                output_descriptor,
                parent_descriptor,
                root_descriptor,
            )
    except Exception:
        if published_summary_identity is not None:
            try:
                _retire_exact_frozen_summary(
                    output_descriptor, published_summary_identity
                )
            except Exception as retirement_error:
                raise ValueError(
                    "frozen output failed after summary publication and "
                    "the exact completion marker could not be retired safely"
                ) from retirement_error
        raise
    finally:
        os.close(root_descriptor)
        os.close(parent_descriptor)
        os.close(output_descriptor)
    return summary
