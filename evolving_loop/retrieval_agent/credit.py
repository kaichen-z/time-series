"""Trusted, post-resolution Retrieval diagnostics and marginal credit."""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Sequence

from common.metrics import drcik_point_metrics
from evolving_loop.data import ContextTask
from evolving_loop.retrieval_agent.skill_library import (
    RetrievalSkillLibrary,
    RetrievalSkillOperation,
)
from evolving_loop.retrieval_agent.verifier import _verified_quote_spans

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


def _expected_snapshot_ids(result: "HarnessResult") -> tuple[str, ...]:
    card = getattr(result, "retrieval_card", None)
    if card is not None:
        return tuple(
            str(chain.chain_id) for chain in card.chains if chain.numeric_eligible
        )

    evidence_indexes: set[int] = set()
    history_count = 0
    for candidate in getattr(result, "candidates", ()):
        tags = tuple(getattr(candidate, "tags", ()))
        if "evidence_adjusted" in tags:
            _prefix, separator, raw_index = str(candidate.candidate_id).rpartition(
                "__evidence_"
            )
            if not separator or not raw_index.isdigit():
                raise ValueError("legacy evidence candidate lacks an executed stage index")
            index = int(raw_index)
            if index in evidence_indexes:
                raise ValueError("legacy evidence stage indexes must be unique")
            evidence_indexes.add(index)
        if "history_cleaned" in tags:
            history_count += 1
    evidence_ids = tuple(
        f"legacy_evidence_{index}"
        for index in range(max(evidence_indexes, default=-1) + 1)
    )
    history_ids = tuple(
        f"legacy_history_clean_{index}" for index in range(history_count)
    )
    return evidence_ids + history_ids


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

    expected_ids = _expected_snapshot_ids(result)
    actual_ids = tuple(snapshot.after_chain_id for snapshot in snapshots[1:])
    if actual_ids != expected_ids:
        raise ValueError(
            "candidate-pool snapshots are missing, extra, or out of executed legacy/chain order"
        )
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("candidate-pool snapshot chain IDs must be unique")

    for previous, current in zip(frozen, frozen[1:]):
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
    expected_keys = tuple(
        (str(ledger_chain.chain_id), str(skill_id))
        for ledger_chain in _chain_ledger(result)
        if ledger_chain.numeric_eligible
        for skill_id in ledger_chain.used_skill_ids
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
        full_entries = _entry_tuple(replay_full.candidates)
        if replay_entries[:len(prior_entries)] != prior_entries:
            raise ValueError("leave-one-out replay changed an unrelated candidate")
        if replay_entries not in (prior_entries, full_entries):
            raise ValueError("leave-one-out replay contains a non-executed candidate")
    validated = []
    for skill_id in chain.used_skill_ids:
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


def _derive_retrieval_skill_evidence(
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
) -> tuple[RetrievalSkillTaskEvidence, ...]:
    """Build diagnostics from resolved labels and frozen inference replays.

    Any incomplete or malformed frozen replay invalidates the batch.  These
    rows are descriptive only; promotion aggregation never accepts rows as an
    input boundary.
    """
    if split not in {"train", "dev"}:
        raise ValueError("Skill evidence split must be train or dev")
    derived: list[RetrievalSkillTaskEvidence] = []
    seen_task_skills: set[tuple[str, str]] = set()
    try:
        for task, result in task_results:
            report = assign_chain_credit(task, result)
            snapshots = tuple(result.candidate_pool_snapshots)
            snapshot_by_chain = {
                snapshot.after_chain_id: snapshot for snapshot in snapshots[1:]
            }
            replay_by_key = {
                (replay.chain_id, replay.skill_id): replay.snapshot
                for replay in result.skill_leave_one_out_snapshots
            }
            for chain in _chain_ledger(result):
                if not chain.numeric_eligible or not chain.used_skill_ids:
                    continue
                necessity = validate_skill_necessity(task, result, chain.chain_id)
                full = snapshot_by_chain[chain.chain_id]
                full_score = _score_pool(task.numeric.future_values, full)[0]
                full_catastrophes = _catastrophic_count(task.numeric.future_values, full)
                for item in necessity:
                    task_skill_key = (task.numeric.task_id, item.skill_id)
                    if task_skill_key in seen_task_skills:
                        return ()
                    seen_task_skills.add(task_skill_key)
                    omitted = replay_by_key[(chain.chain_id, item.skill_id)]
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
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    return tuple(derived)


def derive_retrieval_skill_evidence(
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
) -> tuple[RetrievalSkillTaskEvidence, ...]:
    """Return evaluator diagnostics with no authority to promote a Skill."""
    return _derive_retrieval_skill_evidence(task_results, split=split)


def evaluate_and_promote_retrieval_skills(
    library: RetrievalSkillLibrary,
    task_results: Iterable[tuple[ContextTask, "HarnessResult"]],
    *,
    split: str,
) -> tuple[str, ...]:
    """Derive and aggregate inside the only Skill-promotion entry point."""
    evidence = _derive_retrieval_skill_evidence(task_results, split=split)
    if split != "train":
        return ()
    tolerance = 1e-12
    grouped: dict[str, list[RetrievalSkillTaskEvidence]] = {}
    for row in evidence:
        grouped.setdefault(row.skill_id, []).append(row)
    operations = []
    promoted = []
    for skill_id in sorted(grouped):
        rows = grouped[skill_id]
        current = library.get_by_id(skill_id)
        task_ids = tuple(dict.fromkeys(row.task_id for row in rows))
        entities = tuple(sorted({row.entity_name for row in rows}))
        if (
            current is None
            or current.status != "candidate"
            or any(row.split != "train" for row in rows)
            or len(task_ids) < 3
            or len(entities) < 2
            or len(task_ids) != len(rows)
            or any(row.exact_quote_validity != 1.0 for row in rows)
            or any(not row.necessary for row in rows)
            or any(row.added_catastrophic_count != 0 for row in rows)
        ):
            continue
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
            continue
        accepted = replace(
            current,
            status="accepted",
            validated_task_ids=task_ids,
            validated_entities=entities,
            validation_smae_gain=smae_gain,
            validation_srmse_gain=srmse_gain,
        )
        operations.append(RetrievalSkillOperation.repair(skill_id, accepted))
        promoted.append(skill_id)
    if operations:
        library.apply_operations(operations)
    return tuple(promoted)


__all__ = [
    "EvidenceChainCredit",
    "RetrievalCreditReport",
    "RetrievalSkillTaskEvidence",
    "RetrievalTaskDiagnostics",
    "SkillCredit",
    "SkillNecessity",
    "assign_chain_credit",
    "derive_retrieval_skill_evidence",
    "evaluate_and_promote_retrieval_skills",
    "validate_skill_necessity",
]
