from __future__ import annotations

import re
from .agents import tokenize
from .models import (
    AgentBeliefState,
    ContextCompressionRecord,
    Diagnosis,
    Document,
    ForecastTask,
    QueryAction,
    RetrievalCandidateAssessment,
    RetrievedDocument,
)


CAUSAL_TERMS = {
    "cause", "caused", "because", "due", "result", "impact", "affect",
    "increase", "decrease", "spike", "drop", "shift", "restore", "resolved",
    "promotion", "discount", "policy", "outage", "bug", "patch", "forecast",
}
FORECAST_TERMS = {
    "future", "forecast", "expected", "projected", "scheduled", "baseline",
    "permanent", "temporary", "start", "end", "until", "through", "seasonal",
}
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|times?|x|units?|points?)?\b")


class RetrievalProcessRewardAgent:
    """Frozen, label-free proxy for a learned supplementary-retrieval PRM."""

    def rank(
        self,
        task: ForecastTask,
        action: QueryAction,
        candidates: list[RetrievedDocument],
        state: AgentBeliefState,
        top_k: int,
    ) -> tuple[list[RetrievedDocument], list[RetrievalCandidateAssessment]]:
        if not candidates:
            return [], []
        maximum_bm25 = max((max(item.score, 0.0) for item in candidates), default=1.0) or 1.0
        entity_terms = set(tokenize(task.entity_name))
        target_terms = set(tokenize(task.target_name + " " + task.target_description))
        query_terms = set(tokenize(action.query))
        task_years = set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(
            (*task.history_timestamps, *task.future_timestamps)
        )))
        prior_terms = set(tokenize(" ".join(item.claim for item in state.accepted_evidence)))
        scored: list[tuple[float, RetrievedDocument, RetrievalCandidateAssessment]] = []

        for item in candidates:
            terms = set(tokenize(item.document.text))
            exact_entity = task.entity_name.lower() in item.document.text.lower()
            entity_overlap = len(terms & entity_terms) / max(len(entity_terms), 1)
            target_overlap = len(terms & target_terms) / max(len(target_terms), 1)
            query_overlap = len(terms & query_terms) / max(len(query_terms), 1)
            relevance = min(
                1.0,
                0.45 * max(float(exact_entity), entity_overlap)
                + 0.35 * min(1.0, 3.0 * target_overlap)
                + 0.20 * min(1.0, 4.0 * query_overlap),
            )
            causal = min(1.0, len(terms & CAUSAL_TERMS) / 3.0)
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", item.document.text))
            temporal = 1.0 if years & task_years else 0.65 if not years else 0.1
            if not prior_terms:
                novelty = 1.0
            else:
                novelty = 1.0 - len(terms & prior_terms) / max(len(terms), 1)
            bm25 = max(item.score, 0.0) / maximum_bm25
            utility = (
                0.35 * bm25
                + 0.25 * relevance
                + 0.15 * causal
                + 0.15 * temporal
                + 0.10 * novelty
            )
            assessment = RetrievalCandidateAssessment(
                document_id=item.document.document_id,
                bm25_score=item.score,
                utility_score=utility,
                relevance_score=relevance,
                causal_score=causal,
                temporal_score=temporal,
                novelty_score=novelty,
                rationale=(
                    "Forecast-utility proxy combining retrieval relevance, causal signal, "
                    "temporal alignment, and novelty; no outcome labels are used."
                ),
            )
            scored.append((utility, item, assessment))

        scored.sort(key=lambda row: (-row[0], row[1].document.document_id))
        selected = [
            RetrievedDocument(item.document, utility, rank)
            for rank, (utility, item, _assessment) in enumerate(scored[:top_k], start=1)
        ]
        assessments = [row[2] for row in scored]
        return selected, assessments


class ImportanceAwareContextAgent:
    """Compress long documents while retaining forecast-relevant facts and chronology."""

    def __init__(self, total_character_budget: int = 12000, minimum_document_budget: int = 500) -> None:
        if total_character_budget <= 0 or minimum_document_budget <= 0:
            raise ValueError("context budgets must be positive")
        self.total_character_budget = total_character_budget
        self.minimum_document_budget = minimum_document_budget

    def _sentence_score(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        sentence: str,
    ) -> float:
        terms = set(tokenize(sentence))
        target_terms = set(tokenize(task.target_name + " " + task.target_description))
        entity_terms = set(tokenize(task.entity_name))
        score = 0.0
        score += 2.0 * float(task.entity_name.lower() in sentence.lower())
        score += 1.5 * min(1.0, len(terms & entity_terms) / max(len(entity_terms), 1))
        score += 2.0 * min(1.0, len(terms & target_terms) / max(len(target_terms), 1))
        score += 0.6 * min(3, len(terms & CAUSAL_TERMS))
        score += 0.5 * min(2, len(terms & FORECAST_TERMS))
        score += 1.0 * float(bool(DATE_RE.search(sentence)))
        score += 1.0 * float(bool(NUMBER_RE.search(sentence)))
        if diagnosis.seasonal_period and {"seasonal", "periodic", "cycle"} & terms:
            score += 0.75
        return score

    def compress(
        self,
        task: ForecastTask,
        diagnosis: Diagnosis,
        documents: list[RetrievedDocument],
    ) -> tuple[list[RetrievedDocument], list[ContextCompressionRecord]]:
        if not documents:
            return [], []
        total_original = sum(len(item.document.text) for item in documents)
        usable_budget = max(
            self.total_character_budget,
            self.minimum_document_budget * len(documents),
        )
        weights = [max(item.score, 0.05) for item in documents]
        remaining = max(0, usable_budget - self.minimum_document_budget * len(documents))
        weight_total = sum(weights)
        allocations = [
            self.minimum_document_budget + int(remaining * weight / weight_total)
            for weight in weights
        ]
        if total_original <= usable_budget:
            allocations = [len(item.document.text) for item in documents]

        compressed: list[RetrievedDocument] = []
        records: list[ContextCompressionRecord] = []
        for item, allocation in zip(documents, allocations):
            text = item.document.text
            sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
            if len(text) <= allocation:
                selected_text = text
                retained_count = len(sentences)
            else:
                ranked = sorted(
                    enumerate(sentences),
                    key=lambda row: (-self._sentence_score(task, diagnosis, row[1]), row[0]),
                )
                selected_indices: list[int] = []
                used = 0
                for index, sentence in ranked:
                    cost = len(sentence) + 1
                    if selected_indices and used + cost > allocation:
                        continue
                    selected_indices.append(index)
                    used += cost
                    if used >= allocation:
                        break
                selected_indices.sort()
                selected_text = "\n".join(sentences[index] for index in selected_indices)
                if len(selected_text) > allocation:
                    selected_text = selected_text[:allocation]
                retained_count = len(selected_indices)
            compressed.append(
                RetrievedDocument(
                    document=Document(item.document.document_id, selected_text),
                    score=item.score,
                    rank=item.rank,
                )
            )
            records.append(
                ContextCompressionRecord(
                    document_id=item.document.document_id,
                    utility_score=item.score,
                    original_characters=len(text),
                    retained_characters=len(selected_text),
                    allocated_characters=allocation,
                    retained_sentences=retained_count,
                )
            )
        return compressed, records
