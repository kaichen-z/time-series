"""Convert verified research catalogs into executable dictionary contracts."""

from __future__ import annotations

from typing import Mapping, Sequence, cast

from .contracts import DatasetRelease
from ..dictionary import MethodDefinition, MethodFamily, ToolDictionary


def tool_dictionary_from_payload(
    payload: Mapping[str, object],
    *,
    allowed_families: Sequence[str],
) -> ToolDictionary:
    """Load either a native ToolDictionary or a published DatasetRelease.

    A release describes more methods than the sandbox can execute, so callers pass the
    families they can materialize; the rest stay in the catalog until runtimes exist.
    """

    if "dictionary_id" in payload:
        return ToolDictionary.from_payload(payload)
    if "dataset_id" not in payload:
        raise ValueError("base methods must be a ToolDictionary or DatasetRelease")

    release = DatasetRelease.from_payload(payload)
    allowed = {str(family) for family in allowed_families}
    selected = tuple(
        card
        for card in release.methods
        if card.family in allowed and card.verification_status == "verified"
    )
    if not selected:
        raise ValueError("release contains no verified methods in the allowed families")

    selected_ids = {card.method_uid for card in selected}
    definitions = []
    for card in selected:
        raw_parents = card.lineage.get("parent_method_uids", ())
        parent_ids = (
            tuple(str(parent) for parent in raw_parents)
            if isinstance(raw_parents, Sequence)
            and not isinstance(raw_parents, (str, bytes))
            else ()
        )
        missing_parents = tuple(
            parent for parent in parent_ids if parent not in selected_ids
        )
        if missing_parents:
            raise ValueError(
                f"method {card.method_uid!r} requires excluded parent methods "
                f"{list(missing_parents)!r}; enable their families and runtimes"
            )
        definitions.append(
            MethodDefinition(
                method_id=card.method_uid,
                family=cast(MethodFamily, card.family),
                description=f"{card.canonical_name}: {card.description}",
                assumptions=card.assumptions,
                failure_conditions=card.failure_conditions,
                implementation_spec={
                    "canonical_name": card.canonical_name,
                    "category": card.category,
                    "applicability": dict(card.applicability),
                    "hyperparameters": list(card.hyperparameters),
                    "implementation_availability": card.implementation_availability,
                    "definition_source_ids": list(card.definition_source_ids),
                },
                dependencies=parent_ids,
            )
        )

    family_suffix = "-".join(sorted(allowed))
    return ToolDictionary(
        dictionary_id=f"{release.dataset_id}.{family_suffix}.v000",
        parent_dictionary_id=None,
        generation=0,
        methods=tuple(definitions),
    )
