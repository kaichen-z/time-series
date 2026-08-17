"""Trusted provenance and lineage verification for forecast-method registries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .contracts import MethodCard, SourceRecord


AUTHORITATIVE_DEFINITION_TYPES = {
    "paper",
    "textbook",
    "official_docs",
    "model_card",
}


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    severity: Literal["error", "warning"]
    message: str
    method_uid: str = ""
    source_id: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "method_uid": self.method_uid,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class VerificationReport:
    issues: tuple[VerificationIssue, ...]
    source_count: int
    method_count: int

    @property
    def is_publishable(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(issue.code for issue in self.issues))

    def to_payload(self) -> dict[str, object]:
        return {
            "is_publishable": self.is_publishable,
            "source_count": self.source_count,
            "method_count": self.method_count,
            "issue_codes": list(self.issue_codes),
            "issues": [issue.to_payload() for issue in self.issues],
        }


def _parents(method: MethodCard) -> tuple[str, ...]:
    raw = method.lineage.get("parent_method_uids", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(str(item) for item in raw)


def _lineage_cycle(methods: Mapping[str, MethodCard]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(method_uid: str) -> tuple[str, ...]:
        if method_uid in visited:
            return ()
        if method_uid in visiting:
            start = stack.index(method_uid)
            return tuple(stack[start:] + [method_uid])
        visiting.add(method_uid)
        stack.append(method_uid)
        for parent_uid in _parents(methods[method_uid]):
            if parent_uid not in methods:
                continue
            cycle = visit(parent_uid)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(method_uid)
        visited.add(method_uid)
        return ()

    for method_uid in sorted(methods):
        cycle = visit(method_uid)
        if cycle:
            return cycle
    return ()


def verify_registry(
    sources: Sequence[SourceRecord], methods: Sequence[MethodCard]
) -> VerificationReport:
    source_by_id = {source.source_id: source for source in sources}
    method_by_id = {method.method_uid: method for method in methods}
    issues: list[VerificationIssue] = []

    for source in sorted(sources, key=lambda item: item.source_id):
        if source.review_status != "verified":
            issues.append(
                VerificationIssue(
                    "source_not_verified",
                    "warning",
                    "Source is retained for collection but is not release-verified.",
                    source_id=source.source_id,
                )
            )

    for method in sorted(methods, key=lambda item: item.method_uid):
        if method.verification_status != "verified":
            issues.append(
                VerificationIssue(
                    "method_not_verified",
                    "error",
                    "Method cannot enter a published dataset until it is verified.",
                    method_uid=method.method_uid,
                )
            )

        known_definition_sources = []
        for source_id in method.definition_source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                issues.append(
                    VerificationIssue(
                        "unknown_definition_source",
                        "error",
                        "Definition references a source absent from the registry.",
                        method_uid=method.method_uid,
                        source_id=source_id,
                    )
                )
                continue
            known_definition_sources.append(source)
            if source.review_status != "verified":
                issues.append(
                    VerificationIssue(
                        "unverified_definition_source",
                        "error",
                        "Definition source has not passed source review.",
                        method_uid=method.method_uid,
                        source_id=source_id,
                    )
                )

        authoritative = any(
            source.review_status == "verified"
            and source.primary
            and (
                source.source_type in AUTHORITATIVE_DEFINITION_TYPES
                or (
                    source.source_type == "official_repo"
                    and method.category == "api_service"
                )
            )
            for source in known_definition_sources
        )
        if not authoritative:
            issues.append(
                VerificationIssue(
                    "missing_authoritative_definition",
                    "error",
                    "Method needs a verified primary definition source.",
                    method_uid=method.method_uid,
                )
            )

        for source_id in method.implementation_source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                issues.append(
                    VerificationIssue(
                        "unknown_implementation_source",
                        "error",
                        "Implementation references a source absent from the registry.",
                        method_uid=method.method_uid,
                        source_id=source_id,
                    )
                )
            elif source.review_status != "verified":
                issues.append(
                    VerificationIssue(
                        "unverified_implementation_source",
                        "error",
                        "Implementation source has not passed source review.",
                        method_uid=method.method_uid,
                        source_id=source_id,
                    )
                )

        parents = _parents(method)
        if method.family == "combined" and len(set(parents)) < 2:
            issues.append(
                VerificationIssue(
                    "combined_requires_two_parents",
                    "error",
                    "Combined methods require at least two distinct parent method UIDs.",
                    method_uid=method.method_uid,
                )
            )
        for parent_uid in parents:
            if parent_uid == method.method_uid:
                issues.append(
                    VerificationIssue(
                        "self_parent",
                        "error",
                        "Method lineage cannot include the method itself.",
                        method_uid=method.method_uid,
                    )
                )
            elif parent_uid not in method_by_id:
                issues.append(
                    VerificationIssue(
                        "unknown_parent_method",
                        "error",
                        "Method lineage references an unknown parent.",
                        method_uid=method.method_uid,
                    )
                )

    cycle = _lineage_cycle(method_by_id)
    if cycle:
        issues.append(
            VerificationIssue(
                "cyclic_lineage",
                "error",
                "Method lineage contains a cycle: " + " -> ".join(cycle),
                method_uid=cycle[0],
            )
        )

    return VerificationReport(tuple(issues), len(sources), len(methods))
