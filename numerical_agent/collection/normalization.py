"""Conservative identity normalization and duplicate-candidate reporting."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .contracts import MethodCard


_APOSTROPHES = {"'", "’", "ʼ", "＇"}
_TRAILING_GENERIC_TOKENS = {"forecast", "forecasting", "method", "model"}


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    characters: list[str] = []
    for character in normalized:
        if character in _APOSTROPHES:
            continue
        if unicodedata.category(character).startswith(("P", "S")):
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


@dataclass(frozen=True)
class DuplicateCandidate:
    left_method_uid: str
    right_method_uid: str
    reasons: tuple[str, ...]
    requires_manual_review: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "left_method_uid": self.left_method_uid,
            "right_method_uid": self.right_method_uid,
            "reasons": list(self.reasons),
            "requires_manual_review": self.requires_manual_review,
        }


def _name_token_sets(card: MethodCard) -> tuple[frozenset[str], ...]:
    token_sets = []
    for value in (card.canonical_name, *card.aliases):
        tokens = normalize_name(value).split()
        if tokens and tokens[-1] in _TRAILING_GENERIC_TOKENS:
            tokens = tokens[:-1]
        token_sets.append(frozenset(tokens))
    return tuple(token_sets)


def _shared_source_claim(left: MethodCard, right: MethodCard) -> bool:
    shared_sources = set(left.definition_source_ids) & set(right.definition_source_ids)
    if not shared_sources:
        return False
    return any(
        left_tokens
        and right_tokens
        and (left_tokens <= right_tokens or right_tokens <= left_tokens)
        for left_tokens in _name_token_sets(left)
        for right_tokens in _name_token_sets(right)
    )


def _pair_reasons(left: MethodCard, right: MethodCard) -> tuple[str, ...]:
    left_canonical = normalize_name(left.canonical_name)
    right_canonical = normalize_name(right.canonical_name)
    left_names = {normalize_name(left.canonical_name), *(normalize_name(a) for a in left.aliases)}
    right_names = {
        normalize_name(right.canonical_name),
        *(normalize_name(alias) for alias in right.aliases),
    }
    reasons: list[str] = []
    if left_canonical == right_canonical:
        reasons.append("canonical_collision")
    elif left_names & right_names:
        reasons.append("alias_collision")
    if _shared_source_claim(left, right):
        reasons.append("shared_source_claim")
    return tuple(reasons)


def find_duplicate_candidates(
    methods: Sequence[MethodCard],
) -> tuple[DuplicateCandidate, ...]:
    ordered = sorted(methods, key=lambda method: method.method_uid)
    candidates = []
    for left, right in combinations(ordered, 2):
        reasons = _pair_reasons(left, right)
        if reasons:
            candidates.append(
                DuplicateCandidate(left.method_uid, right.method_uid, reasons)
            )
    return tuple(candidates)
