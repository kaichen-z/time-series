"""What earlier generations already did, recovered from the evolution repository's commits.

The commit log is the loop's memory: `apply_operations` writes one summary per operation and
`commit_module` stores them in the commit body, so nothing extra has to be maintained.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Collection

# "generation 7: 3 operations"
_SUBJECT = re.compile(r"^generation (\d+):")
# "- rewrite regime_aware_ar: Rewrite to be more selective: only accept ..."
_OPERATION = re.compile(r"^- (add|delete|rewrite|merge) (.+?): (.*)$")
# "merge alpha, beta -> gamma"
_MERGE_NAMES = re.compile(r"^(.*?) -> (.+)$")

# Bookkeeping emitted when an earlier operation in the same batch already consumed the name.
_ALREADY_GONE = "already removed earlier in this batch"


@dataclass(frozen=True)
class Operation:
    """One applied operation, recovered from the commit that carried it."""

    generation: int
    op: str
    name: str
    reason: str
    sources: tuple[str, ...] = ()

    @property
    def removes(self) -> tuple[str, ...]:
        """The names this operation took out of the module."""
        if self.op == "delete":
            return (self.name,)
        return tuple(name for name in self.sources if name != self.name)


@dataclass(frozen=True)
class History:
    """Every operation the loop has applied, oldest first."""

    operations: tuple[Operation, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.operations)

    def for_method(self, name: str) -> tuple[Operation, ...]:
        """Everything done to one method, oldest first."""
        return tuple(op for op in self.operations if op.name == name)

    def live(self) -> frozenset[str]:
        """The added methods still standing, replayed from the operations.

        Seed methods never appear as an operation, so this undercounts a real module. Prefer
        passing the module's own names to `removed`.
        """
        names: set[str] = set()
        for operation in self.operations:
            names.difference_update(operation.removes)
            if operation.op in ("add", "merge"):
                names.add(operation.name)
        return frozenset(names)

    def removed(self, live: Collection[str] | None = None) -> tuple[Operation, ...]:
        """Every removal of a method not in `live`, newest first.

        A method removed, re-added and removed again is listed for both removals. That
        repetition is the evidence that it was tried a second time and failed a second time.
        """
        standing = self.live() if live is None else frozenset(live)
        buried = [
            operation
            for operation in self.operations
            if any(name not in standing for name in operation.removes)
        ]
        return tuple(reversed(buried))


def parse_history(log_text: str) -> History:
    """Parse `git log --format=%s%n%b` output into the operations it recorded."""
    commits: list[list[Operation]] = []
    generation = 0
    for line in log_text.splitlines():
        subject = _SUBJECT.match(line)
        if subject:
            generation = int(subject.group(1))
            commits.append([])
            continue
        match = _OPERATION.match(line)
        # The seed commit lists bare names with no operation, which carry no reasoning.
        if not match or not generation:
            continue
        op, name, reason = match.group(1), match.group(2).strip(), match.group(3).strip()
        if reason.startswith(_ALREADY_GONE):
            continue
        sources: tuple[str, ...] = ()
        if op == "merge":
            names = _MERGE_NAMES.match(name)
            if names:
                sources = tuple(part.strip() for part in names.group(1).split(",") if part.strip())
                name = names.group(2).strip()
        commits[-1].append(Operation(generation, op, name, reason, sources))
    # git logs commits newest first, but the operations inside one commit applied in order.
    return History(tuple(op for commit in reversed(commits) for op in commit))
