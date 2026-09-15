"""Pure Train-only strategy artifacts for Numerical QD agent methods."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from ..contracts import require_sha256
from .contracts import (
    TRAIN_DIAGNOSTIC_CATEGORIES,
    MorphologyCellV2,
    MutationOperatorStatsV2,
    NumericalProposerPromptV2,
    TrainMutationFeedbackV2,
    _CanonicalContract,
    _nested,
    _nonnegative_int,
    _require_choice,
    _schema_version,
    _sequence,
    _sorted_strings,
    _text,
)
from .map_elites import NumericalQDArchive


_MAX_CONTEXT_RECORDS = 32
_MAX_SOURCE_BYTES = 1_048_576
_MAX_PROMPT_POPULATION = 16
_CURRICULUM_REASONS = frozenset({
    "unoccupied", "least_visited", "failure_matched",
})


@dataclass(frozen=True, slots=True)
class CurriculumTargetV2(_CanonicalContract):
    """One replayable target derived only from bounded Train aggregates."""

    schema_version: int
    cell: MorphologyCellV2
    reason: Literal["unoccupied", "least_visited", "failure_matched"]
    visit_count: int
    train_diagnostic_categories: tuple[str, ...]

    def __post_init__(self):
        _schema_version(self.schema_version)
        object.__setattr__(self, "cell", _nested(self.cell, MorphologyCellV2))
        _require_choice(self.reason, "curriculum reason", _CURRICULUM_REASONS)
        _nonnegative_int(self.visit_count, "visit_count")
        categories = _sorted_strings(
            self.train_diagnostic_categories,
            "train_diagnostic_categories",
            choices=TRAIN_DIAGNOSTIC_CATEGORIES,
        )
        if self.reason == "unoccupied" and self.visit_count != 0:
            raise ValueError("unoccupied curriculum target must have zero visits")
        if self.reason != "unoccupied" and self.visit_count == 0:
            raise ValueError("occupied curriculum target must have positive visits")
        if self.reason == "failure_matched" and not categories:
            raise ValueError("failure-matched curriculum target requires a Train category")
        if self.reason != "failure_matched" and categories:
            raise ValueError("only failure-matched targets may expose Train categories")
        object.__setattr__(self, "train_diagnostic_categories", categories)


@dataclass(frozen=True, slots=True)
class VerifiedReusableProgramV2(_CanonicalContract):
    """Host-verified source bytes bound to one feasible genome context."""

    schema_version: int
    member_id: str
    genome_sha256: str
    applicability_cells: tuple[str, ...]
    source_sha256: str
    source_text: str

    def __post_init__(self):
        _schema_version(self.schema_version)
        _text(self.member_id, "member_id")
        require_sha256(self.genome_sha256, "genome_sha256")
        object.__setattr__(self, "applicability_cells", _sorted_strings(
            self.applicability_cells, "applicability_cells", sha=True, nonempty=True,
        ))
        require_sha256(self.source_sha256, "source_sha256")
        _text(self.source_text, "source_text")
        encoded = self.source_text.encode("utf-8")
        if len(encoded) > _MAX_SOURCE_BYTES:
            raise ValueError("source_text exceeds 1048576 UTF-8 bytes")
        if hashlib.sha256(encoded).hexdigest() != self.source_sha256:
            raise ValueError("source SHA does not match exact source text")


@dataclass(frozen=True, slots=True)
class MutationPromptLineageV2(_CanonicalContract):
    """One bounded mutation prompt and its Host-recorded Train outcomes."""

    schema_version: int
    mutation_prompt: NumericalProposerPromptV2
    stats: MutationOperatorStatsV2

    def __post_init__(self):
        _schema_version(self.schema_version)
        prompt = _nested(self.mutation_prompt, NumericalProposerPromptV2)
        prompt = NumericalProposerPromptV2.from_payload(prompt.to_payload())
        if prompt.allowed_mutation_operators != ("policy_tune",):
            raise ValueError("mutation prompts may authorize only policy_tune")
        stats = _nested(self.stats, MutationOperatorStatsV2)
        stats = MutationOperatorStatsV2.from_payload(stats.to_payload())
        if any(value > stats.attempts for value in (
                stats.feasible, stats.promotions, stats.insertions)):
            raise ValueError("prompt outcomes cannot exceed attempts")
        if stats.promotions > stats.feasible or stats.insertions > stats.feasible:
            raise ValueError("prompt promotions/insertions require feasibility")
        if stats.credit != stats.insertions:
            raise ValueError("prompt credit must equal Host-recorded Train insertions")
        object.__setattr__(self, "mutation_prompt", prompt)
        object.__setattr__(self, "stats", stats)


@dataclass(frozen=True, slots=True)
class MutationPromptPopulationV2(_CanonicalContract):
    """A canonical bounded population; task prompts remain archive-owned."""

    schema_version: int
    capacity: int
    lineages: tuple[MutationPromptLineageV2, ...]

    def __post_init__(self):
        _schema_version(self.schema_version)
        _bounded_positive(self.capacity, "capacity", _MAX_PROMPT_POPULATION)
        lineages = tuple(
            MutationPromptLineageV2.from_payload(
                _nested(lineage, MutationPromptLineageV2).to_payload()
            )
            for lineage in _sequence(self.lineages, "lineages")
        )
        if not lineages:
            raise ValueError("prompt population must not be empty")
        if len(lineages) > self.capacity:
            raise ValueError("prompt population exceeds capacity")
        identities = tuple(lineage.mutation_prompt.fingerprint() for lineage in lineages)
        if len(identities) != len(set(identities)):
            raise ValueError("prompt population contains a duplicate identity")
        object.__setattr__(self, "lineages", tuple(sorted(
            lineages, key=lambda lineage: lineage.mutation_prompt.fingerprint(),
        )))


def _bounded_positive(value: object, field: str, maximum: int = _MAX_CONTEXT_RECORDS) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return value


def derive_curriculum_targets(
    declared_cells: tuple[MorphologyCellV2, ...],
    archive: NumericalQDArchive,
    feedback: TrainMutationFeedbackV2,
    *,
    maximum_targets: int,
) -> tuple[CurriculumTargetV2, ...]:
    """Prioritize unoccupied, least-visited, then failure-matched cells."""

    if type(declared_cells) is not tuple:
        raise TypeError("declared_cells must be an immutable tuple")
    if type(archive) is not NumericalQDArchive:
        raise TypeError("archive must be an immutable NumericalQDArchive")
    if type(feedback) is not TrainMutationFeedbackV2:
        raise TypeError("feedback must be sanitized TrainMutationFeedbackV2")
    maximum_targets = _bounded_positive(maximum_targets, "maximum_targets")
    if not declared_cells:
        raise ValueError("declared_cells must not be empty")
    if any(type(cell) is not MorphologyCellV2 for cell in declared_cells):
        raise TypeError("declared_cells must contain MorphologyCellV2 artifacts")
    cells = tuple(MorphologyCellV2.from_payload(cell.to_payload()) for cell in declared_cells)
    by_sha = {cell.fingerprint(): cell for cell in cells}
    if len(by_sha) != len(cells):
        raise ValueError("declared_cells must be unique")
    archive = NumericalQDArchive.from_payload(archive.to_payload())
    feedback = TrainMutationFeedbackV2.from_payload(feedback.to_payload())
    records = {record.cell.fingerprint(): record for record in archive.cells}
    if not set(records) <= set(by_sha):
        raise ValueError("archive contains an undeclared curriculum cell")

    def target(cell_sha, reason, categories=()):
        return CurriculumTargetV2(
            1, by_sha[cell_sha], reason,
            records[cell_sha].visit_count if cell_sha in records else 0,
            tuple(categories),
        )

    ordered = []
    emitted = set()
    for cell_sha in sorted(set(by_sha) - set(records)):
        ordered.append(target(cell_sha, "unoccupied"))
        emitted.add(cell_sha)

    occupied = tuple(records)
    if occupied:
        minimum = min(records[cell_sha].visit_count for cell_sha in occupied)
        for cell_sha in sorted(sha for sha in occupied
                               if records[sha].visit_count == minimum):
            ordered.append(target(cell_sha, "least_visited"))
            emitted.add(cell_sha)

    wanted = set(feedback.diagnostic_categories)
    matches = {}
    for entry in archive.entries.values():
        categories = wanted.intersection(entry.train_diagnostic_categories)
        if categories:
            cell_sha = entry.cell.fingerprint()
            matches.setdefault(cell_sha, set()).update(categories)
    for cell_sha in sorted(set(matches) - emitted):
        ordered.append(target(cell_sha, "failure_matched", tuple(sorted(matches[cell_sha]))))
    return tuple(ordered[:maximum_targets])


def select_reusable_programs(
    records: tuple[VerifiedReusableProgramV2, ...],
    *,
    eligible_genome_sha256s: tuple[str, ...],
    target_cell_sha256s: tuple[str, ...],
    maximum_records: int,
) -> tuple[VerifiedReusableProgramV2, ...]:
    """Select bounded verified source records for feasible target genomes."""

    if type(records) is not tuple:
        raise TypeError("records must be an immutable tuple")
    if type(eligible_genome_sha256s) is not tuple:
        raise TypeError("eligible_genome_sha256s must be an immutable tuple")
    if type(target_cell_sha256s) is not tuple:
        raise TypeError("target_cell_sha256s must be an immutable tuple")
    maximum_records = _bounded_positive(maximum_records, "maximum_records")
    eligible = set(_sorted_strings(
        eligible_genome_sha256s, "eligible_genome_sha256s", sha=True,
    ))
    targets = set(_sorted_strings(
        target_cell_sha256s, "target_cell_sha256s", sha=True, nonempty=True,
    ))
    verified = []
    for record in records:
        if type(record) is not VerifiedReusableProgramV2:
            raise TypeError("records must contain VerifiedReusableProgramV2 artifacts")
        record = VerifiedReusableProgramV2.from_payload(record.to_payload())
        matches = targets.intersection(record.applicability_cells)
        if record.genome_sha256 in eligible and matches:
            verified.append((record, len(matches)))
    ranked = sorted(
        verified,
        key=lambda item: (-item[1], item[0].source_sha256, item[0].fingerprint()),
    )
    unique = []
    seen_sources = set()
    for record, _ in ranked:
        if record.source_sha256 not in seen_sources:
            unique.append(record)
            seen_sources.add(record.source_sha256)
    return tuple(unique[:maximum_records])


def _validated_population(population: MutationPromptPopulationV2) -> MutationPromptPopulationV2:
    if type(population) is not MutationPromptPopulationV2:
        raise TypeError("population must be an immutable MutationPromptPopulationV2")
    return MutationPromptPopulationV2.from_payload(population.to_payload())


def _prompt_rank(lineage: MutationPromptLineageV2):
    stats = lineage.stats
    return (
        -stats.credit,
        -stats.insertions,
        -stats.promotions,
        -stats.feasible,
        stats.attempts,
        lineage.mutation_prompt.fingerprint(),
    )


def select_prompt_lineage(population: MutationPromptPopulationV2) -> MutationPromptLineageV2:
    """Select the strongest lineage, using its prompt SHA for exact ties."""

    population = _validated_population(population)
    return min(population.lineages, key=_prompt_rank)


def insert_prompt_child(
    population: MutationPromptPopulationV2,
    parent_prompt_sha256: str,
    child_prompt: NumericalProposerPromptV2,
) -> MutationPromptPopulationV2:
    """Insert a zero-credit child and evict the weakest incumbent if full."""

    population = _validated_population(population)
    require_sha256(parent_prompt_sha256, "parent_prompt_sha256")
    if type(child_prompt) is not NumericalProposerPromptV2:
        raise TypeError("child_prompt must be a NumericalProposerPromptV2 artifact")
    child_prompt = NumericalProposerPromptV2.from_payload(child_prompt.to_payload())
    by_sha = {
        lineage.mutation_prompt.fingerprint(): lineage for lineage in population.lineages
    }
    if parent_prompt_sha256 not in by_sha:
        raise ValueError("unknown parent prompt lineage")
    if child_prompt.parent_prompt_sha256 != parent_prompt_sha256:
        raise ValueError("child prompt must bind the selected parent prompt SHA")
    parent = by_sha[parent_prompt_sha256].mutation_prompt
    if (child_prompt.response_schema != parent.response_schema
            or child_prompt.max_response_bytes > parent.max_response_bytes
            or not set(child_prompt.allowed_mutation_operators)
            <= set(parent.allowed_mutation_operators)):
        raise ValueError("child prompt cannot expand parent response authority")
    child = MutationPromptLineageV2(
        1, child_prompt, MutationOperatorStatsV2(0, 0, 0, 0, 0),
    )
    child_sha = child_prompt.fingerprint()
    if child_sha in by_sha:
        return population
    lineages = list(population.lineages)
    if len(lineages) == population.capacity:
        weakest = max(lineages, key=_prompt_rank)
        lineages.remove(weakest)
    lineages.append(child)
    return MutationPromptPopulationV2(1, population.capacity, tuple(lineages))


def apply_prompt_train_credit(
    population: MutationPromptPopulationV2,
    prompt_sha256: str,
    feedback: TrainMutationFeedbackV2,
) -> MutationPromptPopulationV2:
    """Apply one Host-observed Train outcome only to the named lineage."""

    population = _validated_population(population)
    require_sha256(prompt_sha256, "prompt_sha256")
    if type(feedback) is not TrainMutationFeedbackV2:
        raise TypeError("feedback must be sanitized TrainMutationFeedbackV2")
    feedback = TrainMutationFeedbackV2.from_payload(feedback.to_payload())
    if feedback.operator != "policy_tune":
        raise ValueError("prompt credit requires policy_tune Train feedback")
    identities = {
        lineage.mutation_prompt.fingerprint() for lineage in population.lineages
    }
    if prompt_sha256 not in identities:
        raise ValueError("unknown prompt lineage")
    lineages = []
    for lineage in population.lineages:
        if lineage.mutation_prompt.fingerprint() != prompt_sha256:
            lineages.append(lineage)
            continue
        stats = lineage.stats
        lineages.append(MutationPromptLineageV2(
            1,
            lineage.mutation_prompt,
            MutationOperatorStatsV2(
                stats.attempts + 1,
                stats.feasible + int(feedback.feasible),
                stats.promotions + int(feedback.promoted),
                stats.insertions + int(feedback.inserted),
                stats.credit + int(feedback.inserted),
            ),
        ))
    return MutationPromptPopulationV2(1, population.capacity, tuple(lineages))
