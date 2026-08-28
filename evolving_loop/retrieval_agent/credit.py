"""Trusted, post-resolution Retrieval diagnostics and marginal credit."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Sequence

from common.metrics import drcik_point_metrics
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkillLibrary,
    _commit_evaluator_records,
    _record_digest,
)
from evolving_loop.retrieval_agent.schemas import EvidenceChain
from evolving_loop.retrieval_agent.verifier import (
    _verified_quote_spans,
    verify_round_result,
)

if TYPE_CHECKING:
    from evolving_loop.harness import CandidatePoolSnapshot, HarnessResult


@dataclass(frozen=True)
class SkillCredit:
    skill_id: str
    marginal_smae_gain: float
    marginal_srmse_gain: float


@dataclass(frozen=True)
class SkillNecessity:
    skill_id: str
    necessary: bool
    omitted_smae_regret: float = 0.0
    omitted_srmse_regret: float = 0.0


@dataclass(frozen=True)
class EvidenceChainCredit:
    chain_id: str
    numeric_eligible: bool
    marginal_smae_gain: float
    marginal_srmse_gain: float
    used_skill_ids: tuple[str, ...] = ()
    skill_credit: tuple[SkillCredit, ...] = ()


@dataclass(frozen=True)
class RetrievalTaskDiagnostics:
    supporting_recall: float
    gt_evidence_recall: float
    distractor_avoidance: float
    exact_quote_validity: float
    complete_chain_rate: float
    contextual_oracle_smae_gain: float
    contextual_oracle_srmse_gain: float
    invalid_count: int
    catastrophic_count: int
    chain_credit: tuple[EvidenceChainCredit, ...]


@dataclass(frozen=True)
class RetrievalCreditReport:
    coding_oracle_smae: float
    coding_oracle_srmse: float
    contextual_oracle_smae: float
    contextual_oracle_srmse: float
    final_smae: float
    final_srmse: float
    decision_smae_regret: float
    decision_srmse_regret: float
    chains: tuple[EvidenceChainCredit, ...]
    diagnostics: RetrievalTaskDiagnostics


@dataclass(frozen=True)
class RetrievalSkillTaskEvidence:
    """One pre-label leave-one-out replay scored on a resolved Train task."""

    skill_id: str
    task_id: str
    entity_name: str
    split: str
    exact_quote_validity: float
    without_skill_smae: float
    without_skill_srmse: float
    with_skill_smae: float
    with_skill_srmse: float
    added_catastrophic_count: int
    necessary: bool
    def __post_init__(self) -> None:
        if not self.skill_id or not self.task_id or not self.entity_name:
            raise ValueError("Skill evidence requires Skill, task, and entity IDs")
        if self.split not in {"train", "dev"}:
            raise ValueError("Skill evidence split must be train or dev")
        metrics = (
            self.exact_quote_validity,
            self.without_skill_smae,
            self.without_skill_srmse,
            self.with_skill_smae,
            self.with_skill_srmse,
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics):
            raise ValueError("Skill evidence metrics must be finite")
        if not 0.0 <= self.exact_quote_validity <= 1.0:
            raise ValueError("exact quote validity must be in [0, 1]")
        if not all(0.0 <= value <= 5.0 for value in metrics[1:]):
            raise ValueError("Skill evidence sMAE/sRMSE metrics must be in [0, 5]")
        if isinstance(self.added_catastrophic_count, bool) or self.added_catastrophic_count < 0:
            raise ValueError("added catastrophic count must be non-negative")
        if not isinstance(self.necessary, bool):
            raise ValueError("Skill necessity must be boolean")


@dataclass(frozen=True)
class RetrievalSkillReplayArtifact:
    """Canonical label-free pools retained to recompute one promotion row."""

    skill_id: str
    task_id: str
    entity_name: str
    split: str
    chain_id: str
    exact_quote_validity: float
    baseline_candidates: tuple[tuple[str, tuple[float, ...]], ...]
    with_skill_candidates: tuple[tuple[str, tuple[float, ...]], ...]
    without_skill_candidates: tuple[tuple[str, tuple[float, ...]], ...]
    primary_final_candidates: tuple[tuple[str, tuple[float, ...]], ...]
    with_skill_chains: tuple[EvidenceChain, ...]
    without_skill_chains: tuple[EvidenceChain, ...]
    primary_chains: tuple[EvidenceChain, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.skill_id,
                self.task_id,
                self.entity_name,
                self.chain_id,
            )
        ):
            raise ValueError("Skill replay requires stable provenance IDs")
        if self.split not in {"train", "dev"}:
            raise ValueError("Skill replay split must be train or dev")
        if isinstance(self.exact_quote_validity, bool) or not isinstance(
            self.exact_quote_validity, (int, float)
        ):
            raise ValueError("Skill replay quote validity must be numeric")
        quote_validity = float(self.exact_quote_validity)
        if not math.isfinite(quote_validity) or not 0.0 <= quote_validity <= 1.0:
            raise ValueError("Skill replay quote validity must be in [0, 1]")
        object.__setattr__(self, "exact_quote_validity", quote_validity)
        for field_name in (
            "baseline_candidates",
            "with_skill_candidates",
            "without_skill_candidates",
            "primary_final_candidates",
        ):
            raw = getattr(self, field_name)
            if not isinstance(raw, tuple) or not raw:
                raise ValueError("Skill replay candidate pools cannot be empty")
            normalized: list[tuple[str, tuple[float, ...]]] = []
            for item in raw:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("Skill replay candidates must be typed pairs")
                candidate_id, forecast = item
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise ValueError("Skill replay candidate IDs must be non-empty")
                if not isinstance(forecast, tuple) or not forecast:
                    raise ValueError("Skill replay forecasts cannot be empty")
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in forecast
                ):
                    raise ValueError("Skill replay forecasts must be numeric")
                values = tuple(float(value) for value in forecast)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Skill replay forecasts must be finite")
                normalized.append((candidate_id, values))
            identifiers = tuple(candidate_id for candidate_id, _ in normalized)
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("Skill replay candidate IDs cannot repeat")
            object.__setattr__(self, field_name, tuple(normalized))
        for field_name in (
            "with_skill_chains",
            "without_skill_chains",
            "primary_chains",
        ):
            chains = getattr(self, field_name)
            if not isinstance(chains, tuple) or any(
                not isinstance(chain, EvidenceChain) or not chain.numeric_eligible
                for chain in chains
            ):
                raise ValueError("Skill replay sources require numeric evidence chains")

    def to_payload(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "task_id": self.task_id,
            "entity_name": self.entity_name,
            "split": self.split,
            "chain_id": self.chain_id,
            "exact_quote_validity": self.exact_quote_validity,
            "baseline_candidates": [
                [candidate_id, list(forecast)]
                for candidate_id, forecast in self.baseline_candidates
            ],
            "with_skill_candidates": [
                [candidate_id, list(forecast)]
                for candidate_id, forecast in self.with_skill_candidates
            ],
            "without_skill_candidates": [
                [candidate_id, list(forecast)]
                for candidate_id, forecast in self.without_skill_candidates
            ],
            "primary_final_candidates": [
                [candidate_id, list(forecast)]
                for candidate_id, forecast in self.primary_final_candidates
            ],
            "with_skill_chains": [
                chain.to_payload() for chain in self.with_skill_chains
            ],
            "without_skill_chains": [
                chain.to_payload() for chain in self.without_skill_chains
            ],
            "primary_chains": [
                chain.to_payload() for chain in self.primary_chains
            ],
        }

    @classmethod
    def from_payload(cls, raw: object) -> "RetrievalSkillReplayArtifact":
        fields = {
            "skill_id",
            "task_id",
            "entity_name",
            "split",
            "chain_id",
            "exact_quote_validity",
            "baseline_candidates",
            "with_skill_candidates",
            "without_skill_candidates",
            "primary_final_candidates",
            "with_skill_chains",
            "without_skill_chains",
            "primary_chains",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError("invalid Skill replay artifact")

        def candidates(value: object) -> tuple[tuple[str, tuple[float, ...]], ...]:
            if not isinstance(value, list):
                raise ValueError("invalid Skill replay candidate pool")
            parsed = []
            for item in value:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[1], list)
                ):
                    raise ValueError("invalid Skill replay candidate")
                parsed.append((item[0], tuple(item[1])))
            return tuple(parsed)

        return cls(
            skill_id=raw["skill_id"],  # type: ignore[arg-type]
            task_id=raw["task_id"],  # type: ignore[arg-type]
            entity_name=raw["entity_name"],  # type: ignore[arg-type]
            split=raw["split"],  # type: ignore[arg-type]
            chain_id=raw["chain_id"],  # type: ignore[arg-type]
            exact_quote_validity=raw["exact_quote_validity"],  # type: ignore[arg-type]
            baseline_candidates=candidates(raw["baseline_candidates"]),
            with_skill_candidates=candidates(raw["with_skill_candidates"]),
            without_skill_candidates=candidates(raw["without_skill_candidates"]),
            primary_final_candidates=candidates(raw["primary_final_candidates"]),
            with_skill_chains=_replay_chains(raw["with_skill_chains"]),
            without_skill_chains=_replay_chains(raw["without_skill_chains"]),
            primary_chains=_replay_chains(raw["primary_chains"]),
        )


def _replay_chains(value: object) -> tuple[EvidenceChain, ...]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("invalid Skill replay evidence chains")
    return tuple(EvidenceChain.from_payload(item) for item in value)


def _score_pool(
    truth: Sequence[float], snapshot: "CandidatePoolSnapshot"
) -> tuple[dict[str, float | bool], int]:
    scored: list[tuple[str, dict[str, float | bool]]] = []
    invalid = 0
    for candidate in snapshot.candidates:
        try:
            score = drcik_point_metrics(truth, candidate.forecast)
        except (TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        scored.append((candidate.candidate_id, score))
    if not scored:
        return _invalid_drcik_score(), invalid
    _candidate_id, oracle = min(
        scored,
        key=lambda item: (
            float(item[1]["srmse"]),
            float(item[1]["smae"]),
            item[0],
        ),
    )
    return oracle, invalid


def _invalid_drcik_score() -> dict[str, float | bool]:
    return {
        "smae": 5.0,
        "srmse": 5.0,
        "smae_clipped": True,
        "srmse_clipped": True,
    }


def _score_forecast_drcik(
    truth: Sequence[float], forecast: Sequence[float]
) -> tuple[dict[str, float | bool], int]:
    try:
        return drcik_point_metrics(truth, forecast), 0
    except (TypeError, ValueError, OverflowError):
        return _invalid_drcik_score(), 1


def _entry_tuple(candidates: Iterable[object]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    return tuple(
        (str(candidate.candidate_id), tuple(float(value) for value in candidate.forecast))
        for candidate in candidates
    )


def candidate_pool_sha256(candidates: Iterable[object]) -> str:
    """Digest one executed candidate pool in canonical order."""
    return _candidate_entries_sha256(_entry_tuple(candidates))


def _candidate_entries_sha256(
    candidates: tuple[tuple[str, tuple[float, ...]], ...],
) -> str:
    payload = [
        [candidate_id, list(forecast)] for candidate_id, forecast in candidates
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _coding_entry_tuple(result: "HarnessResult") -> tuple[tuple[str, tuple[float, ...]], ...]:
    entries: list[tuple[str, tuple[float, ...]]] = []
    seen: set[str] = set()
    for candidate in getattr(result.coding, "candidates", ()):
        candidate_id = str(candidate.program.name)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        entries.append((candidate_id, tuple(float(value) for value in candidate.forecast)))
    return tuple(entries)


def _contextual_candidate_entries(
    result: "HarnessResult",
) -> tuple[
    dict[int, tuple[str, tuple[float, ...]]],
    tuple[tuple[str, tuple[float, ...]], ...],
]:
    evidence: dict[int, tuple[str, tuple[float, ...]]] = {}
    history: list[tuple[str, tuple[float, ...]]] = []
    for candidate in getattr(result, "candidates", ()):
        tags = tuple(getattr(candidate, "tags", ()))
        if "evidence_adjusted" in tags:
            _prefix, separator, raw_index = str(candidate.candidate_id).rpartition(
                "__evidence_"
            )
            if not separator or not raw_index.isdigit():
                raise ValueError("legacy evidence candidate lacks an executed stage index")
            index = int(raw_index)
            if index in evidence:
                raise ValueError("legacy evidence stage indexes must be unique")
            evidence[index] = (
                str(candidate.candidate_id),
                tuple(float(value) for value in candidate.forecast),
            )
        if "history_cleaned" in tags:
            history.append(
                (
                    str(candidate.candidate_id),
                    tuple(float(value) for value in candidate.forecast),
                )
            )
    return evidence, tuple(history)


def _expected_snapshot_stages(
    result: "HarnessResult",
) -> tuple[tuple[str, tuple[str, tuple[float, ...]] | None], ...]:
    evidence, history = _contextual_candidate_entries(result)
    card = getattr(result, "retrieval_card", None)
    if card is not None:
        numeric_chain_ids = tuple(
            str(chain.chain_id) for chain in card.chains if chain.numeric_eligible
        )
        return tuple(
            (chain_id, evidence.get(index))
            for index, chain_id in enumerate(numeric_chain_ids)
        )
    evidence_stages = tuple(
        (f"legacy_evidence_{index}", evidence.get(index))
        for index in range(max(evidence, default=-1) + 1)
    )
    history_stages = tuple(
        (f"legacy_history_clean_{index}", entry)
        for index, entry in enumerate(history)
    )
    return evidence_stages + history_stages


def _selected_candidate_matches_pool(result: "HarnessResult") -> bool:
    try:
        selected = result.decision.selected
        selected_id = str(selected.candidate_id)
        selected_forecast = tuple(float(value) for value in selected.forecast)
        final_forecast = tuple(float(value) for value in result.forecast)
        executed = dict(_entry_tuple(result.candidates))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return (
        selected_id in executed
        and selected_forecast == executed[selected_id]
        and final_forecast == selected_forecast
    )


def _validate_candidate_snapshots(
    result: "HarnessResult", snapshots: tuple["CandidatePoolSnapshot", ...]
) -> None:
    if not snapshots:
        raise ValueError("candidate-pool snapshots are required")
    if snapshots[0].after_chain_id is not None:
        raise ValueError("candidate snapshots must start with the numeric-only pool")

    executed = _entry_tuple(getattr(result, "candidates", ()))
    coding = _coding_entry_tuple(result)
    for name, entries in (("executed", executed), ("coding", coding)):
        identifiers = tuple(candidate_id for candidate_id, _forecast in entries)
        if not entries or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} candidate pool contains duplicate or missing IDs")
    frozen = tuple(_entry_tuple(snapshot.candidates) for snapshot in snapshots)
    if frozen[0] != coding:
        raise ValueError("baseline snapshot is not the executed numeric candidate pool")
    if frozen[-1] != executed:
        raise ValueError("final snapshot is not the executed cumulative candidate pool")

    expected_stages = _expected_snapshot_stages(result)
    expected_ids = tuple(stage_id for stage_id, _addition in expected_stages)
    actual_ids = tuple(snapshot.after_chain_id for snapshot in snapshots[1:])
    if actual_ids != expected_ids:
        raise ValueError(
            "candidate-pool snapshots are missing, extra, or out of executed legacy/chain order"
        )
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("candidate-pool snapshot chain IDs must be unique")

    for (stage_id, expected_addition), previous, current in zip(
        expected_stages, frozen, frozen[1:]
    ):
        if len(current) < len(previous):
            raise ValueError("candidate-pool snapshots are non-monotonic")
        if len(current) > len(previous) + 1:
            raise ValueError("candidate-pool snapshots are missing an executed cumulative step")
        for index, prior in enumerate(previous):
            if index >= len(current):
                raise ValueError("candidate-pool snapshots are non-prefix")
            if current[index][0] != prior[0]:
                raise ValueError("candidate-pool snapshots are non-prefix")
            if current[index][1] != prior[1]:
                raise ValueError("candidate-pool snapshot changed an existing forecast")
        addition = current[len(previous):]
        if expected_addition is None and addition:
            raise ValueError(
                f"snapshot stage {stage_id} added a candidate when execution created none"
            )
        if expected_addition is not None and addition != (expected_addition,):
            raise ValueError(
                f"snapshot stage {stage_id} does not add its exact executed candidate"
            )


def _chain_ledger(result: "HarnessResult") -> tuple[object, ...]:
    card = getattr(result, "retrieval_card", None)
    return tuple(card.chains) if card is not None else ()


def _retrieved_document_ids(result: "HarnessResult") -> tuple[str, ...]:
    card = getattr(result, "retrieval_card", None)
    if card is not None:
        return tuple(card.selected_document_ids)
    retrieval = getattr(result, "retrieval", None)
    return tuple(getattr(retrieval, "selected_document_ids", ()))


def _normalized_tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


def _retrieval_quality(
    task: ContextTask,
    result: "HarnessResult",
    chains: tuple[object, ...],
) -> tuple[float, float, float, float, float, int]:
    retrieved = set(_retrieved_document_ids(result))
    supporting = {document.document_id for document in task.documents if document.role == "supporting"}
    distractors = {document.document_id for document in task.documents if document.role == "distractor"}
    supporting_recall = len(retrieved & supporting) / len(supporting) if supporting else 1.0
    distractor_avoidance = (
        1.0 - len(retrieved & distractors) / len(distractors) if distractors else 1.0
    )

    recovered_text = " ".join(
        value
        for chain in chains
        for value in (
            str(getattr(chain, "claim", "")),
            *(citation.exact_quote for citation in getattr(chain, "citations", ())),
        )
    )
    recovered_tokens = _normalized_tokens(recovered_text)
    gt_evidence_recall = (
        sum(
            1
            for evidence in task.gt_evidence
            if (tokens := _normalized_tokens(evidence)) and tokens.issubset(recovered_tokens)
        )
        / len(task.gt_evidence)
        if task.gt_evidence
        else 1.0
    )

    documents = {document.document_id: document.content for document in task.documents}
    citations = tuple(
        citation for chain in chains for citation in getattr(chain, "citations", ())
    )
    valid_quotes = sum(
        bool(_verified_quote_spans(citation.exact_quote, documents.get(citation.document_id, "")))
        for citation in citations
    )
    card = getattr(result, "retrieval_card", None)
    rejections = tuple(getattr(card, "rejected", ())) if card is not None else tuple(
        getattr(getattr(result, "retrieval", None), "rejected", ())
    )
    audit_rounds = (
        tuple(item for item in (card.round1, card.round2) if item is not None)
        if card is not None
        else ()
    )
    audited_attempts = sum(item.quote_attempt_count for item in audit_rounds)
    audited_valid = sum(item.valid_quote_count for item in audit_rounds)
    if audited_attempts:
        exact_quote_validity = audited_valid / audited_attempts
    else:
        ungrounded = sum(str(reason).startswith("ungrounded_quote:") for reason in rejections)
        quote_attempts = len(citations) + ungrounded
        exact_quote_validity = valid_quotes / quote_attempts if quote_attempts else 1.0

    evidence_chains = chains
    if card is not None:
        evidence_chains = tuple(card.round1.chains) + (
            tuple(card.round2.chains) if card.round2 is not None else ()
        )
    complete_chain_rate = (
        sum(bool(getattr(chain, "numeric_eligible", False)) for chain in evidence_chains)
        / len(evidence_chains)
        if evidence_chains
        else 0.0
    )
    return (
        supporting_recall,
        gt_evidence_recall,
        distractor_avoidance,
        exact_quote_validity,
        complete_chain_rate,
        len(rejections),
    )


def assign_chain_credit(
    task: ContextTask, result: "HarnessResult"
) -> RetrievalCreditReport:
    """Assign contextual-oracle pool gain only after public labels resolve."""
    if not task.labels_public or not task.numeric.future_values:
        raise ValueError("Retrieval credit requires resolved public labels")
    snapshots = tuple(getattr(result, "candidate_pool_snapshots", ()))
    _validate_candidate_snapshots(result, snapshots)
    truth = task.numeric.future_values
    scored_snapshots = [_score_pool(truth, snapshot) for snapshot in snapshots]
    coding_score = scored_snapshots[0][0]
    contextual_score = scored_snapshots[-1][0]
    selection_valid = _selected_candidate_matches_pool(result)
    final_score, _ = (
        _score_forecast_drcik(truth, result.forecast)
        if selection_valid
        else (_invalid_drcik_score(), 1)
    )
    chains = _chain_ledger(result)
    snapshot_index = 1
    chain_credit: list[EvidenceChainCredit] = []
    previous = coding_score
    for chain in chains:
        marginal_smae = 0.0
        marginal_srmse = 0.0
        if chain.numeric_eligible:
            if snapshot_index >= len(snapshots):
                raise ValueError("missing candidate-pool snapshot for numeric chain")
            snapshot = snapshots[snapshot_index]
            if snapshot.after_chain_id != chain.chain_id:
                raise ValueError("candidate-pool snapshots do not follow verified chain order")
            current = scored_snapshots[snapshot_index][0]
            marginal_smae = float(previous["smae"]) - float(current["smae"])
            marginal_srmse = float(previous["srmse"]) - float(current["srmse"])
            previous = current
            snapshot_index += 1
        used_skill_ids = tuple(chain.used_skill_ids)
        direct_skill_credit = (
            (
                SkillCredit(used_skill_ids[0], marginal_smae, marginal_srmse),
            )
            if len(used_skill_ids) == 1 and chain.numeric_eligible
            else ()
        )
        chain_credit.append(
            EvidenceChainCredit(
                chain_id=chain.chain_id,
                numeric_eligible=bool(chain.numeric_eligible),
                marginal_smae_gain=marginal_smae,
                marginal_srmse_gain=marginal_srmse,
                used_skill_ids=used_skill_ids,
                skill_credit=direct_skill_credit,
            )
        )
    if chains and snapshot_index != len(snapshots):
        raise ValueError("candidate-pool snapshots contain unverified chain additions")

    quality = _retrieval_quality(task, result, chains)
    final_snapshot = snapshots[-1]
    catastrophic_count = 0
    invalid_candidates = scored_snapshots[-1][1]
    for candidate in final_snapshot.candidates:
        try:
            score = drcik_point_metrics(truth, candidate.forecast)
        except (TypeError, ValueError, OverflowError):
            continue
        if bool(score["smae_clipped"]) or bool(score["srmse_clipped"]):
            catastrophic_count += 1
    diagnostics = RetrievalTaskDiagnostics(
        supporting_recall=quality[0],
        gt_evidence_recall=quality[1],
        distractor_avoidance=quality[2],
        exact_quote_validity=quality[3],
        complete_chain_rate=quality[4],
        contextual_oracle_smae_gain=(
            float(coding_score["smae"]) - float(contextual_score["smae"])
        ),
        contextual_oracle_srmse_gain=(
            float(coding_score["srmse"]) - float(contextual_score["srmse"])
        ),
        invalid_count=(
            quality[5]
            + invalid_candidates
            + (0 if selection_valid else 1)
        ),
        catastrophic_count=catastrophic_count,
        chain_credit=tuple(chain_credit),
    )
    return RetrievalCreditReport(
        coding_oracle_smae=float(coding_score["smae"]),
        coding_oracle_srmse=float(coding_score["srmse"]),
        contextual_oracle_smae=float(contextual_score["smae"]),
        contextual_oracle_srmse=float(contextual_score["srmse"]),
        final_smae=float(final_score["smae"]),
        final_srmse=float(final_score["srmse"]),
        decision_smae_regret=float(final_score["smae"]) - float(contextual_score["smae"]),
        decision_srmse_regret=float(final_score["srmse"]) - float(contextual_score["srmse"]),
        chains=tuple(chain_credit),
        diagnostics=diagnostics,
    )


def validate_skill_necessity(
    task: ContextTask,
    result: "HarnessResult",
    chain_id: str,
    *,
    tolerance: float = 1e-12,
    eligible_skill_ids: Iterable[str] | None = None,
) -> tuple[SkillNecessity, ...]:
    """Score explicit pre-label leave-one-Skill-out replays, failing closed if absent."""
    if not task.labels_public or not task.numeric.future_values:
        raise ValueError("Skill necessity requires resolved public labels")
    snapshots = tuple(getattr(result, "candidate_pool_snapshots", ()))
    _validate_candidate_snapshots(result, snapshots)
    chain = next(
        (item for item in _chain_ledger(result) if item.chain_id == chain_id),
        None,
    )
    if chain is None:
        raise ValueError(f"unknown evidence chain: {chain_id}")
    full_index = next(
        (index for index, snapshot in enumerate(snapshots) if snapshot.after_chain_id == chain_id),
        None,
    )
    full = snapshots[full_index] if full_index is not None else None
    if full is None:
        raise ValueError("missing full candidate-pool snapshot for Skill replay")
    assert full_index is not None and full_index > 0
    previous = snapshots[full_index - 1]
    full_score = _score_pool(task.numeric.future_values, full)[0]
    replay_rows = tuple(getattr(result, "skill_leave_one_out_snapshots", ()))
    replay_keys = tuple((replay.chain_id, replay.skill_id) for replay in replay_rows)
    if len(replay_keys) != len(set(replay_keys)):
        raise ValueError("duplicate leave-one-out replay key")
    eligible = (
        None
        if eligible_skill_ids is None
        else frozenset(str(skill_id) for skill_id in eligible_skill_ids)
    )
    expected_keys = tuple(
        (str(ledger_chain.chain_id), str(skill_id))
        for ledger_chain in _chain_ledger(result)
        if ledger_chain.numeric_eligible
        for skill_id in ledger_chain.used_skill_ids
        if eligible is None or skill_id in eligible
    )
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("duplicate used Skill ID in verified chain ledger")
    if set(replay_keys) != set(expected_keys):
        raise ValueError("leave-one-out replay keys are missing, extra, or use the wrong chain")
    replays = {key: replay.snapshot for key, replay in zip(replay_keys, replay_rows)}
    main_by_chain = {
        snapshot.after_chain_id: snapshot for snapshot in snapshots[1:]
    }
    for (replay_chain_id, _skill_id), replay in replays.items():
        replay_full = main_by_chain.get(replay_chain_id)
        if replay_full is None or replay.after_chain_id != replay_chain_id:
            raise ValueError("leave-one-out replay has invalid chain provenance")
        replay_main_index = snapshots.index(replay_full)
        replay_previous = snapshots[replay_main_index - 1]
        prior_entries = _entry_tuple(replay_previous.candidates)
        replay_entries = _entry_tuple(replay.candidates)
        if replay_entries[:len(prior_entries)] != prior_entries:
            raise ValueError("leave-one-out replay changed an unrelated candidate")
        if len(replay_entries) != len({candidate_id for candidate_id, _ in replay_entries}):
            raise ValueError("leave-one-out replay contains duplicate candidate IDs")
    validated = []
    for skill_id in chain.used_skill_ids:
        if eligible is not None and skill_id not in eligible:
            continue
        omitted = replays.get((chain_id, skill_id))
        if omitted is None:
            validated.append(SkillNecessity(skill_id, False))
            continue
        omitted_score = _score_pool(task.numeric.future_values, omitted)[0]
        smae_regret = float(omitted_score["smae"]) - float(full_score["smae"])
        srmse_regret = float(omitted_score["srmse"]) - float(full_score["srmse"])
        necessary = (
            smae_regret >= -tolerance
            and srmse_regret >= -tolerance
            and (smae_regret > tolerance or srmse_regret > tolerance)
        )
        validated.append(
            SkillNecessity(skill_id, necessary, smae_regret, srmse_regret)
        )
    return tuple(validated)


def _catastrophic_count(
    truth: Sequence[float], snapshot: "CandidatePoolSnapshot"
) -> int:
    count = 0
    for candidate in snapshot.candidates:
        score, invalid = _score_forecast_drcik(truth, candidate.forecast)
        if not invalid and (bool(score["smae_clipped"]) or bool(score["srmse_clipped"])):
            count += 1
    return count


def _derive_retrieval_skill_attestation(
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
    eligible_skill_ids: Iterable[str] | None = None,
) -> tuple[
    tuple[RetrievalSkillTaskEvidence, ...],
    tuple[RetrievalSkillReplayArtifact, ...],
]:
    """Build diagnostics from resolved labels and frozen inference replays.

    Any incomplete or malformed frozen replay invalidates the batch.  These
    rows are descriptive only; promotion aggregation never accepts rows as an
    input boundary.
    """
    if split not in {"train", "dev"}:
        raise ValueError("Skill evidence split must be train or dev")
    derived: list[RetrievalSkillTaskEvidence] = []
    replay_artifacts: list[RetrievalSkillReplayArtifact] = []
    seen_task_skills: set[tuple[str, str]] = set()
    eligible = (
        None
        if eligible_skill_ids is None
        else frozenset(str(skill_id) for skill_id in eligible_skill_ids)
    )
    try:
        for task, result in task_results:
            report = assign_chain_credit(task, result)
            snapshots = tuple(result.candidate_pool_snapshots)
            snapshot_by_chain = {
                snapshot.after_chain_id: snapshot for snapshot in snapshots[1:]
            }
            replay_by_key = {
                (replay.chain_id, replay.skill_id): replay
                for replay in result.skill_leave_one_out_snapshots
            }
            primary_chains = tuple(
                chain for chain in _chain_ledger(result) if chain.numeric_eligible
            )
            with_skill_chains: list[EvidenceChain] = []
            for chain in _chain_ledger(result):
                if not chain.numeric_eligible:
                    continue
                with_skill_chains.append(chain)
                if not chain.used_skill_ids:
                    continue
                necessity = validate_skill_necessity(
                    task,
                    result,
                    chain.chain_id,
                    eligible_skill_ids=eligible,
                )
                full = snapshot_by_chain[chain.chain_id]
                full_score = _score_pool(task.numeric.future_values, full)[0]
                full_catastrophes = _catastrophic_count(task.numeric.future_values, full)
                for item in necessity:
                    task_skill_key = (task.numeric.task_id, item.skill_id)
                    if task_skill_key in seen_task_skills:
                        return (), ()
                    seen_task_skills.add(task_skill_key)
                    replay_source = replay_by_key[(chain.chain_id, item.skill_id)]
                    omitted = replay_source.snapshot
                    omitted_score = _score_pool(task.numeric.future_values, omitted)[0]
                    row = RetrievalSkillTaskEvidence(
                        skill_id=item.skill_id,
                        task_id=task.numeric.task_id,
                        entity_name=task.numeric.entity_name,
                        split=split,
                        exact_quote_validity=report.diagnostics.exact_quote_validity,
                        without_skill_smae=float(omitted_score["smae"]),
                        without_skill_srmse=float(omitted_score["srmse"]),
                        with_skill_smae=float(full_score["smae"]),
                        with_skill_srmse=float(full_score["srmse"]),
                        added_catastrophic_count=max(
                            0,
                            full_catastrophes
                            - _catastrophic_count(task.numeric.future_values, omitted),
                        ),
                        necessary=item.necessary,
                    )
                    derived.append(row)
                    replay_artifacts.append(
                        RetrievalSkillReplayArtifact(
                            skill_id=item.skill_id,
                            task_id=task.numeric.task_id,
                            entity_name=task.numeric.entity_name,
                            split=split,
                            chain_id=chain.chain_id,
                            exact_quote_validity=report.diagnostics.exact_quote_validity,
                            baseline_candidates=_entry_tuple(
                                snapshots[0].candidates
                            ),
                            with_skill_candidates=_entry_tuple(full.candidates),
                            without_skill_candidates=_entry_tuple(omitted.candidates),
                            primary_final_candidates=_entry_tuple(
                                snapshots[-1].candidates
                            ),
                            with_skill_chains=tuple(with_skill_chains),
                            without_skill_chains=replay_source.verified_chains,
                            primary_chains=primary_chains,
                        )
                    )
    except (AttributeError, KeyError, TypeError, ValueError):
        return (), ()
    return tuple(derived), tuple(replay_artifacts)


def _derive_retrieval_skill_evidence(
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
    eligible_skill_ids: Iterable[str] | None = None,
) -> tuple[RetrievalSkillTaskEvidence, ...]:
    return _derive_retrieval_skill_attestation(
        task_results,
        split=split,
        eligible_skill_ids=eligible_skill_ids,
    )[0]


def derive_retrieval_skill_attestation(
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
    eligible_skill_ids: Iterable[str] | None = None,
) -> tuple[
    tuple[RetrievalSkillTaskEvidence, ...],
    tuple[RetrievalSkillReplayArtifact, ...],
]:
    """Return canonical evidence rows and their executed pre-label pool sources."""
    return _derive_retrieval_skill_attestation(
        task_results,
        split=split,
        eligible_skill_ids=eligible_skill_ids,
    )


def _score_replay_pool(
    truth: Sequence[float],
    candidates: tuple[tuple[str, tuple[float, ...]], ...],
) -> tuple[dict[str, float | bool], int, int]:
    scored: list[tuple[str, dict[str, float | bool]]] = []
    invalid = 0
    catastrophic = 0
    for candidate_id, forecast in candidates:
        score, row_invalid = _score_forecast_drcik(truth, forecast)
        invalid += row_invalid
        if row_invalid:
            continue
        if bool(score["smae_clipped"]) or bool(score["srmse_clipped"]):
            catastrophic += 1
        scored.append((candidate_id, score))
    if not scored:
        return _invalid_drcik_score(), invalid, catastrophic
    _candidate_id, oracle = min(
        scored,
        key=lambda item: (
            float(item[1]["srmse"]),
            float(item[1]["smae"]),
            item[0],
        ),
    )
    return oracle, invalid, catastrophic


def _verified_replay_chains(
    task: ContextTask,
    chains: tuple[EvidenceChain, ...],
    allowed_skill_ids: frozenset[str],
) -> tuple[EvidenceChain, ...]:
    verified: list[EvidenceChain] = []
    seen: set[str] = set()
    for chain in chains:
        if chain.chain_id in seen:
            raise ValueError("Skill replay evidence chains cannot repeat")
        seen.add(chain.chain_id)
        if not set(chain.used_skill_ids).issubset(allowed_skill_ids):
            raise ValueError("Skill replay evidence chain used an unauthorized Skill")
        result = verify_round_result(
            task,
            {
                "evidence_chains": [chain.to_payload()],
                "counterevidence": [],
                "missing_information": [],
                "sufficient": True,
            },
            stage="round1",
            allowed_skill_ids=allowed_skill_ids,
            allowed_assumption_ids=chain.addressed_assumption_ids,
        )
        submitted_payload = chain.to_payload()
        submitted_payload.pop("chain_id")
        verified_payload = (
            {} if len(result.chains) != 1 else result.chains[0].to_payload()
        )
        verified_payload.pop("chain_id", None)
        if (
            len(result.chains) != 1
            or not result.chains[0].numeric_eligible
            or verified_payload != submitted_payload
        ):
            raise ValueError(
                "Skill replay evidence chain does not reverify against its task"
            )
        verified.append(result.chains[0])
    return tuple(verified)


def _validate_replay_projection(
    task: ContextTask,
    baseline: tuple[tuple[str, tuple[float, ...]], ...],
    pool: tuple[tuple[str, tuple[float, ...]], ...],
    chains: tuple[EvidenceChain, ...],
    allowed_skill_ids: frozenset[str],
) -> tuple[EvidenceChain, ...]:
    if pool[: len(baseline)] != baseline:
        raise ValueError("Skill replay changed its executed numeric baseline")
    baseline_by_id = dict(baseline)
    verified = _verified_replay_chains(task, chains, allowed_skill_ids)
    for candidate_id, forecast in pool[len(baseline) :]:
        base_id, separator, raw_index = candidate_id.rpartition("__evidence_")
        if not separator or not raw_index.isdigit() or base_id not in baseline_by_id:
            raise ValueError("Skill replay contains a non-projectable contextual candidate")
        index = int(raw_index)
        if index >= len(verified):
            raise ValueError("Skill replay contextual candidate lacks its verified chain")
        chain = verified[index]
        if chain.start_timestamp is None or chain.end_timestamp is None:
            raise ValueError("Skill replay contextual chain lacks its future window")
        affected = tuple(
            position
            for position, timestamp in enumerate(task.future_timestamps)
            if chain.start_timestamp <= timestamp <= chain.end_timestamp
        )
        if not affected or chain.magnitude_value is None:
            raise ValueError("Skill replay contextual chain does not affect the horizon")
        expected = list(baseline_by_id[base_id])
        sign = 1.0 if chain.direction == "up" else -1.0
        if chain.magnitude_kind == "absolute":
            for position in affected:
                expected[position] += sign * chain.magnitude_value
        elif chain.magnitude_kind == "relative":
            for position in affected:
                expected[position] *= 1.0 + sign * chain.magnitude_value
        elif chain.magnitude_kind == "multiplier":
            adjustment = chain.magnitude_value - 1.0
            for position in affected:
                expected[position] *= 1.0 + adjustment
        else:
            raise ValueError("Skill replay contextual chain has no numeric projection")
        if len(expected) != len(forecast) or any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(expected, forecast, strict=True)
        ):
            raise ValueError(
                "Skill replay forecast does not match its reverified evidence chain"
            )
    return verified


def _primary_trace_scores(
    task: ContextTask,
    replay: RetrievalSkillReplayArtifact,
    trace: Mapping[str, object],
    allowed_skill_ids: frozenset[str],
) -> None:
    _validate_replay_projection(
        task,
        replay.baseline_candidates,
        replay.primary_final_candidates,
        replay.primary_chains,
        allowed_skill_ids,
    )
    coding, _coding_invalid, _coding_catastrophes = _score_replay_pool(
        task.numeric.future_values,
        replay.baseline_candidates,
    )
    contextual, _contextual_invalid, _contextual_catastrophes = _score_replay_pool(
        task.numeric.future_values,
        replay.primary_final_candidates,
    )
    for field, candidates in (
        ("numeric_baseline_sha256", replay.baseline_candidates),
        ("contextual_pool_sha256", replay.primary_final_candidates),
    ):
        expected = trace.get(field)
        if (
            type(expected) is not str
            or expected != _candidate_entries_sha256(candidates)
        ):
            raise ValueError(
                "Skill replay candidate pool does not match its trusted task trace"
            )
    for field, actual in (
        ("coding_oracle_smae", coding["smae"]),
        ("coding_oracle_srmse", coding["srmse"]),
        ("contextual_oracle_smae", contextual["smae"]),
        ("contextual_oracle_srmse", contextual["srmse"]),
    ):
        expected = trace.get(field)
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(float(expected))
            or not math.isclose(
                float(expected),
                float(actual),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Skill replay primary pool does not match its trusted task trace"
            )


def recompute_retrieval_skill_evidence(
    tasks: Iterable[ContextTask],
    replays: Iterable[RetrievalSkillReplayArtifact],
    *,
    allowed_skill_ids: Iterable[str],
    task_traces: Iterable[Mapping[str, object]] | None = None,
) -> tuple[RetrievalSkillTaskEvidence, ...]:
    """Recompute promotion rows from immutable tasks and canonical replay pools."""
    scheduled_tasks = tuple(tasks)
    task_by_id = {task.numeric.task_id: task for task in scheduled_tasks}
    artifacts = tuple(replays)
    allowed = frozenset(str(skill_id) for skill_id in allowed_skill_ids)
    traces = None if task_traces is None else tuple(task_traces)
    trace_by_task = (
        {}
        if traces is None
        else {
            trace.get("task_id"): trace
            for trace in traces
            if type(trace.get("task_id")) is str
        }
    )
    if len(task_by_id) != len(scheduled_tasks):
        raise ValueError("Skill replay tasks must be unique")
    if traces is not None and len(trace_by_task) != len(traces):
        raise ValueError("Skill replay task traces must have unique string task IDs")
    seen: set[tuple[str, str]] = set()
    primary_by_task: dict[
        str,
        tuple[
            tuple[tuple[str, tuple[float, ...]], ...],
            tuple[tuple[str, tuple[float, ...]], ...],
            tuple[dict[str, object], ...],
        ],
    ] = {}
    rows: list[RetrievalSkillTaskEvidence] = []
    tolerance = 1e-12
    for replay in artifacts:
        key = (replay.task_id, replay.skill_id)
        task = task_by_id.get(replay.task_id)
        if key in seen or task is None or not task.labels_public:
            raise ValueError("Skill replay provenance does not match scheduled tasks")
        seen.add(key)
        if task.numeric.entity_name != replay.entity_name:
            raise ValueError("Skill replay entity does not match scheduled task")
        if replay.skill_id not in allowed:
            raise ValueError("Skill replay targets a Skill absent from its Genome")
        target = next(
            (
                chain
                for chain in replay.with_skill_chains
                if chain.chain_id == replay.chain_id
            ),
            None,
        )
        if target is None or replay.skill_id not in target.used_skill_ids:
            raise ValueError("Skill replay target is absent from its primary chain")
        _validate_replay_projection(
            task,
            replay.baseline_candidates,
            replay.with_skill_candidates,
            replay.with_skill_chains,
            allowed,
        )
        verified_without = _validate_replay_projection(
            task,
            replay.baseline_candidates,
            replay.without_skill_candidates,
            replay.without_skill_chains,
            allowed,
        )
        if any(replay.skill_id in chain.used_skill_ids for chain in verified_without):
            raise ValueError("Skill replay omitted run still used its target Skill")
        primary = (
            replay.baseline_candidates,
            replay.primary_final_candidates,
            tuple(chain.to_payload() for chain in replay.primary_chains),
        )
        prior_primary = primary_by_task.setdefault(replay.task_id, primary)
        if prior_primary != primary:
            raise ValueError("Skill replays disagree about their primary execution")
        if traces is not None:
            trace = trace_by_task.get(replay.task_id)
            if trace is None:
                raise ValueError("Skill replay lacks its trusted primary task trace")
            _primary_trace_scores(task, replay, trace, allowed)
        full, _full_invalid, full_catastrophes = _score_replay_pool(
            task.numeric.future_values,
            replay.with_skill_candidates,
        )
        omitted, _omitted_invalid, omitted_catastrophes = _score_replay_pool(
            task.numeric.future_values,
            replay.without_skill_candidates,
        )
        smae_regret = float(omitted["smae"]) - float(full["smae"])
        srmse_regret = float(omitted["srmse"]) - float(full["srmse"])
        rows.append(
            RetrievalSkillTaskEvidence(
                skill_id=replay.skill_id,
                task_id=replay.task_id,
                entity_name=replay.entity_name,
                split=replay.split,
                exact_quote_validity=replay.exact_quote_validity,
                without_skill_smae=float(omitted["smae"]),
                without_skill_srmse=float(omitted["srmse"]),
                with_skill_smae=float(full["smae"]),
                with_skill_srmse=float(full["srmse"]),
                added_catastrophic_count=max(
                    0, full_catastrophes - omitted_catastrophes
                ),
                necessary=(
                    smae_regret >= -tolerance
                    and srmse_regret >= -tolerance
                    and (smae_regret > tolerance or srmse_regret > tolerance)
                ),
            )
        )
    return tuple(rows)


def derive_retrieval_skill_evidence(
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
    eligible_skill_ids: Iterable[str] | None = None,
) -> tuple[RetrievalSkillTaskEvidence, ...]:
    """Return evaluator diagnostics with no authority to promote a Skill."""
    return _derive_retrieval_skill_evidence(
        task_results,
        split=split,
        eligible_skill_ids=eligible_skill_ids,
    )


def _accepted_retrieval_skill_from_evidence(
    current,
    evidence: Iterable[RetrievalSkillTaskEvidence],
):
    rows = tuple(evidence)
    tolerance = 1e-12
    task_ids = tuple(dict.fromkeys(row.task_id for row in rows))
    entities = tuple(sorted({row.entity_name for row in rows}))
    if (
        current is None
        or current.status != "candidate"
        or not rows
        or any(row.skill_id != current.skill_id for row in rows)
        or any(row.split != "train" for row in rows)
        or len(task_ids) < 3
        or len(entities) < 2
        or len(task_ids) != len(rows)
        or any(row.exact_quote_validity != 1.0 for row in rows)
        or any(not row.necessary for row in rows)
        or any(row.added_catastrophic_count != 0 for row in rows)
    ):
        return None
    without_smae = statistics.fmean(row.without_skill_smae for row in rows)
    without_srmse = statistics.fmean(row.without_skill_srmse for row in rows)
    with_smae = statistics.fmean(row.with_skill_smae for row in rows)
    with_srmse = statistics.fmean(row.with_skill_srmse for row in rows)
    smae_gain = without_smae - with_smae
    srmse_gain = without_srmse - with_srmse
    if not (
        smae_gain >= -tolerance
        and srmse_gain >= -tolerance
        and (smae_gain > tolerance or srmse_gain > tolerance)
    ):
        return None
    accepted = copy(current)
    for field, value in (
        ("version", current.version + 1),
        ("parent_version", current.version),
        ("status", "accepted"),
        ("validated_task_ids", task_ids),
        ("validated_entities", entities),
        ("validation_smae_gain", smae_gain),
        ("validation_srmse_gain", srmse_gain),
    ):
        object.__setattr__(accepted, field, value)
    return accepted


def _evaluate_and_promote_retrieval_skills(
    library: RetrievalSkillLibrary,
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
) -> tuple[str, ...]:
    """Derive and aggregate inside the only Skill-promotion entry point."""
    candidate_skill_ids = tuple(
        skill.skill_id for skill in library.all() if skill.status == "candidate"
    )
    evidence = _derive_retrieval_skill_evidence(
        task_results,
        split=split,
        eligible_skill_ids=candidate_skill_ids,
    )
    if split != "train":
        return ()
    grouped: dict[str, list[RetrievalSkillTaskEvidence]] = {}
    for row in evidence:
        grouped.setdefault(row.skill_id, []).append(row)
    accepted_records = []
    promoted = []
    for skill_id in sorted(grouped):
        current = library.get_by_id(skill_id)
        accepted = _accepted_retrieval_skill_from_evidence(
            current,
            grouped[skill_id],
        )
        if accepted is None:
            continue
        accepted_records.append(accepted)
        promoted.append(skill_id)
    if accepted_records:
        # The trusted evaluator commits the already-gated transition directly.
        # No public operation accepts active records, so there is no transferable
        # token or caller-constructible promotion request.
        proposed = {
            skill_id: tuple(history)
            for skill_id, history in library._skills.items()
        }
        for accepted in accepted_records:
            proposed[accepted.skill_id] = (
                *proposed[accepted.skill_id],
                accepted,
            )
        active_origins = dict(library._active_record_origins)
        active_origins.update(
            {_record_digest(record): "evaluator_promotion" for record in accepted_records}
        )
        records = library._all_from(proposed)
        for record in accepted_records:
            library._validate_active_origin(
                record, records, "evaluator_promotion"
            )
        proposed = library._validated_index(
            records,
            active_record_hashes=active_origins,
        )
        _commit_evaluator_records(library, proposed, active_origins)
    return tuple(promoted)


__all__ = [
    "EvidenceChainCredit",
    "RetrievalCreditReport",
    "RetrievalSkillReplayArtifact",
    "RetrievalSkillTaskEvidence",
    "RetrievalTaskDiagnostics",
    "SkillCredit",
    "SkillNecessity",
    "assign_chain_credit",
    "candidate_pool_sha256",
    "derive_retrieval_skill_attestation",
    "derive_retrieval_skill_evidence",
    "recompute_retrieval_skill_evidence",
    "validate_skill_necessity",
]
