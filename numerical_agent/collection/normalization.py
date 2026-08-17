"""Conservative identity normalization and duplicate-candidate reporting."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .contracts import MethodCard


_APOSTROPHES = {"'", "’", "ʼ", "＇"}
_TOKEN_STOPWORDS = {"forecast", "forecasting", "method", "model"}


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


def _tokens(card: MethodCard) -> set[str]:
    values = (card.canonical_name, *card.aliases)
    tokens = {
        token
        for value in values
        for token in normalize_name(value).split()
        if token not in _TOKEN_STOPWORDS
    }
    return tokens


def _shared_source_claim(left: MethodCard, right: MethodCard) -> bool:
    shared_sources = set(left.definition_source_ids) & set(right.definition_source_ids)
    if not shared_sources:
        return False
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    return bool(left_tokens & right_tokens)


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
