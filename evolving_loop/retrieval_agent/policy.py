"""Immutable retrieval genomes and verifiable, versioned release artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .schemas import RetrievalContractError


ROUND1_STRATEGIES = frozenset({"timeline_first", "entity_first", "contrastive"})
ROUND2_STRATEGIES = frozenset(
    {"counterevidence_first", "gap_first", "causal_chain_first"}
)
SECOND_ROUND_TRIGGERS = frozenset(
    {"on_named_gap", "on_incomplete_chain", "always", "never"}
)
BOUNDS = MappingProxyType(
    {
        "max_selected_documents": (1, 20),
        "max_evidence_chains": (1, 12),
        "max_citations_per_chain": (1, 8),
    }
)

_VERSION = re.compile(r"^v\d{3}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_GENOME_FIELDS = frozenset(
    {
        "schema_version",
        "version",
        "parent",
        "round1_prompt",
        "round2_prompt",
        "round1_strategy",
        "round2_strategy",
        "second_round_trigger",
        "max_selected_documents",
        "max_evidence_chains",
        "max_citations_per_chain",
        "require_counterevidence_search",
        "require_target_match",
        "require_temporal_overlap",
        "active_skill_ids",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "version",
        "parent",
        "genome_sha256",
        "round1_prompt_sha256",
        "round2_prompt_sha256",
        "skills_sha256",
        "state",
        "train_dev_split_sha256",
        "verifier_sha256",
        "evaluator_sha256",
        "metric_sha256",
        "metric_cap",
        "resource_budgets",
        "train_summary",
        "dev_summary",
        "acceptance_reason",
        "audit_sha256",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "state",
        "train_dev_split_sha256",
        "verifier_sha256",
        "evaluator_sha256",
        "metric_sha256",
        "metric_cap",
        "train_summary",
        "dev_summary",
        "acceptance_reason",
    }
)
_AUDIT_HASH_FIELDS = (
    "train_dev_split_sha256",
    "verifier_sha256",
    "evaluator_sha256",
    "metric_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SEED_ROUND1_PROMPT = """You are Retrieval Round 1 for contextual time-series forecasting.
Construct an evidence ledger from the provided documents before any decision context is considered.
Start with the target entity and its forecast window, then trace a chronological timeline of events.
Use exact quotes, retain only evidence that matches the target and overlaps or explains the relevant
time window, and explicitly search for counterevidence. Return only the strict round-one JSON
contract supplied by the host; never infer facts that are absent from named documents.
"""

_SEED_ROUND2_PROMPT = """You are Retrieval Round 2 for contextual time-series forecasting.
Use the verified Round 1 ledger, named gaps, and sanitized assumptions to seek counterevidence and
fill only unresolved causal links. Prefer evidence that can disprove an apparently plausible chain.
Every claim requires an exact quote from a named document. Return only the strict round-two JSON
contract supplied by the host; do not request, derive, or expose candidate scores or forecasts.
"""


class RetrievalPolicyError(RetrievalContractError):
    """Raised when a Genome or release artifact violates the immutable policy contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalPolicyError(f"invalid {field}")
    return value


def _version(value: object, field: str) -> str:
    result = _text(value, field)
    if not _VERSION.fullmatch(result):
        raise RetrievalPolicyError(f"invalid {field}")
    return result


def _bounded_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetrievalPolicyError(f"invalid {field}")
    lower, upper = BOUNDS[field]
    if not lower <= value <= upper:
        raise RetrievalPolicyError(f"{field} must be within [{lower}, {upper}]")
    return value


def _host_requirement(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise RetrievalPolicyError(f"invalid {field}")
    if not value:
        raise RetrievalPolicyError(f"cannot weaken host verification: {field}")
    return True


def _skill_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RetrievalPolicyError("invalid active_skill_ids")
    result = tuple(_text(item, "active_skill_ids") for item in value)
    if any(not _IDENTIFIER.fullmatch(item) for item in result):
        raise RetrievalPolicyError("invalid active_skill_ids")
    if len(result) != len(set(result)):
        raise RetrievalPolicyError("duplicate active_skill_ids")
    return result


def _exact_mapping(raw: object, fields: frozenset[str], context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise RetrievalPolicyError(f"{context} must be an object")
    keys = set(raw)
    if keys != fields:
        unknown = keys - fields
        missing = fields - keys
        if unknown:
            raise RetrievalPolicyError(f"forbidden {context} field: {sorted(unknown)[0]}")
        raise RetrievalPolicyError(f"missing {context} field: {sorted(missing)[0]}")
    return raw


def _resource_budgets(genome: "RetrievalGenome") -> dict[str, int]:
    return {
        "max_selected_documents": genome.max_selected_documents,
        "max_evidence_chains": genome.max_evidence_chains,
        "max_citations_per_chain": genome.max_citations_per_chain,
    }


def _audit_payload(
    genome: "RetrievalGenome", audit: Mapping[str, object] | None
) -> dict[str, object]:
    if audit is None:
        audit = {
            "state": "seed",
            "train_dev_split_sha256": None,
            "verifier_sha256": None,
            "evaluator_sha256": None,
            "metric_sha256": None,
            "metric_cap": None,
            "train_summary": None,
            "dev_summary": None,
            "acceptance_reason": "not_evaluated_seed",
        }
    value = _exact_mapping(audit, _AUDIT_FIELDS, "release audit")
    return dict(value)


def _audit_binding(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        field: manifest[field]
        for field in _AUDIT_FIELDS | {"resource_budgets"}
    }


def _validate_release_audit(manifest: Mapping[str, object], genome: "RetrievalGenome") -> None:
    state = manifest["state"]
    if state not in {"seed", "accepted"}:
        raise RetrievalPolicyError("invalid release state")
    budgets = _exact_mapping(manifest["resource_budgets"], frozenset(BOUNDS), "resource_budgets")
    if dict(budgets) != _resource_budgets(genome):
        raise RetrievalPolicyError("resource_budgets must match genome")
    if state == "seed":
        if genome.version != "v000" or genome.parent is not None:
            raise RetrievalPolicyError("only v000 without a parent may be a seed release")
        if any(manifest[field] is not None for field in _AUDIT_HASH_FIELDS):
            raise RetrievalPolicyError("seed release cannot invent evaluation hashes")
        if manifest["metric_cap"] is not None:
            raise RetrievalPolicyError("seed release cannot set metric_cap")
        if manifest["train_summary"] is not None or manifest["dev_summary"] is not None:
            raise RetrievalPolicyError("seed release cannot include evaluation summaries")
        if manifest["acceptance_reason"] != "not_evaluated_seed":
            raise RetrievalPolicyError("seed release requires not_evaluated_seed reason")
        if manifest["audit_sha256"] is not None:
            raise RetrievalPolicyError("seed release cannot invent audit hash")
        return

    if genome.version == "v000":
        raise RetrievalPolicyError("v000 must remain a seed release")
    for field in _AUDIT_HASH_FIELDS:
        value = manifest[field]
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise RetrievalPolicyError(f"accepted release requires {field}")
    metric_cap = manifest["metric_cap"]
    if isinstance(metric_cap, bool) or not isinstance(metric_cap, (int, float)) or not math.isfinite(metric_cap):
        raise RetrievalPolicyError("accepted release requires finite metric_cap")
    for field in ("train_summary", "dev_summary"):
        summary = manifest[field]
        if not isinstance(summary, Mapping) or not summary:
            raise RetrievalPolicyError(f"accepted release requires nonempty {field}")
    _text(manifest["acceptance_reason"], "acceptance_reason")
    expected_audit_hash = _digest(_canonical_json(_audit_binding(manifest)))
    if manifest["audit_sha256"] != expected_audit_hash:
        raise RetrievalPolicyError("accepted release audit hash mismatch")


@dataclass(frozen=True)
class RetrievalGenome:
    schema_version: int
    version: str
    parent: str | None
    round1_prompt: str
    round2_prompt: str
    round1_strategy: str
    round2_strategy: str
    second_round_trigger: str
    max_selected_documents: int
    max_evidence_chains: int
    max_citations_per_chain: int
    require_counterevidence_search: bool
    require_target_match: bool
    require_temporal_overlap: bool
    active_skill_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise RetrievalPolicyError("invalid genome schema_version")
        if self.schema_version != 1:
            raise RetrievalPolicyError("unsupported genome schema_version")
        version = _version(self.version, "version")
        parent = None if self.parent is None else _version(self.parent, "parent")
        if parent == version:
            raise RetrievalPolicyError("genome parent cannot equal version")
        round1_prompt = _text(self.round1_prompt, "round1_prompt")
        round2_prompt = _text(self.round2_prompt, "round2_prompt")
        if self.round1_strategy not in ROUND1_STRATEGIES:
            raise RetrievalPolicyError(f"invalid round1_strategy: {self.round1_strategy}")
        if self.round2_strategy not in ROUND2_STRATEGIES:
            raise RetrievalPolicyError(f"invalid round2_strategy: {self.round2_strategy}")
        if self.second_round_trigger not in SECOND_ROUND_TRIGGERS:
            raise RetrievalPolicyError(
                f"invalid second_round_trigger: {self.second_round_trigger}"
            )
        selected = _bounded_integer(self.max_selected_documents, "max_selected_documents")
        chains = _bounded_integer(self.max_evidence_chains, "max_evidence_chains")
        citations = _bounded_integer(self.max_citations_per_chain, "max_citations_per_chain")
        skill_ids = _skill_ids(self.active_skill_ids)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "round1_prompt", round1_prompt)
        object.__setattr__(self, "round2_prompt", round2_prompt)
        object.__setattr__(self, "max_selected_documents", selected)
        object.__setattr__(self, "max_evidence_chains", chains)
        object.__setattr__(self, "max_citations_per_chain", citations)
        object.__setattr__(
            self,
            "require_counterevidence_search",
            _host_requirement(
                self.require_counterevidence_search, "require_counterevidence_search"
            ),
        )
        object.__setattr__(
            self, "require_target_match", _host_requirement(self.require_target_match, "require_target_match")
        )
        object.__setattr__(
            self,
            "require_temporal_overlap",
            _host_requirement(self.require_temporal_overlap, "require_temporal_overlap"),
        )
        object.__setattr__(self, "active_skill_ids", skill_ids)

    @classmethod
    def seed(cls) -> "RetrievalGenome":
        return cls(
            schema_version=1,
            version="v000",
            parent=None,
            round1_prompt=_SEED_ROUND1_PROMPT,
            round2_prompt=_SEED_ROUND2_PROMPT,
            round1_strategy="timeline_first",
            round2_strategy="counterevidence_first",
            second_round_trigger="on_named_gap",
            max_selected_documents=8,
            max_evidence_chains=4,
            max_citations_per_chain=4,
            require_counterevidence_search=True,
            require_target_match=True,
            require_temporal_overlap=True,
            active_skill_ids=(),
        )

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "RetrievalGenome":
        value = _exact_mapping(raw, _GENOME_FIELDS, "genome")
        parent = value["parent"]
        if parent is not None and not isinstance(parent, str):
            raise RetrievalPolicyError("invalid parent")
        return cls(
            schema_version=value["schema_version"],
            version=value["version"],
            parent=parent,
            round1_prompt=value["round1_prompt"],
            round2_prompt=value["round2_prompt"],
            round1_strategy=value["round1_strategy"],
            round2_strategy=value["round2_strategy"],
            second_round_trigger=value["second_round_trigger"],
            max_selected_documents=value["max_selected_documents"],
            max_evidence_chains=value["max_evidence_chains"],
            max_citations_per_chain=value["max_citations_per_chain"],
            require_counterevidence_search=value["require_counterevidence_search"],
            require_target_match=value["require_target_match"],
            require_temporal_overlap=value["require_temporal_overlap"],
            active_skill_ids=_skill_ids(value["active_skill_ids"]),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "parent": self.parent,
            "round1_prompt": self.round1_prompt,
            "round2_prompt": self.round2_prompt,
            "round1_strategy": self.round1_strategy,
            "round2_strategy": self.round2_strategy,
            "second_round_trigger": self.second_round_trigger,
            "max_selected_documents": self.max_selected_documents,
            "max_evidence_chains": self.max_evidence_chains,
            "max_citations_per_chain": self.max_citations_per_chain,
            "require_counterevidence_search": self.require_counterevidence_search,
            "require_target_match": self.require_target_match,
            "require_temporal_overlap": self.require_temporal_overlap,
            "active_skill_ids": list(self.active_skill_ids),
        }

    def fingerprint(self) -> str:
        return _digest(_canonical_json(self.to_payload()))


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _skills_from_payload(raw: object) -> tuple[object, ...]:
    if not isinstance(raw, list):
        raise RetrievalPolicyError("skills.json must contain an array")
    return tuple(_freeze_json(item) for item in raw)


def _skills_payload(skills: Sequence[object]) -> list[object]:
    if isinstance(skills, (str, bytes)):
        raise RetrievalPolicyError("skills must be a sequence")
    try:
        payload = json.loads(json.dumps(list(skills), ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise RetrievalPolicyError("skills must be JSON serializable") from error
    if not isinstance(payload, list):  # Defensive: list() above makes this unreachable.
        raise RetrievalPolicyError("skills must be an array")
    return payload


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalPolicyError(f"invalid {label}: {path}") from error


def _reject_git_path(path: Path) -> None:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise RetrievalPolicyError(f"cannot resolve release path: {path}") from error
    if ".git" in path.parts or ".git" in resolved.parts:
        raise RetrievalPolicyError("release paths cannot contain .git")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RetrievalRelease:
    path: Path
    genome: RetrievalGenome
    round1_prompt: str
    round2_prompt: str
    skills: tuple[object, ...]
    manifest: Mapping[str, object]

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalRelease":
        release_path = Path(path)
        _reject_git_path(release_path)
        if not release_path.is_dir():
            raise RetrievalPolicyError(f"release directory does not exist: {release_path}")
        genome = RetrievalGenome.from_payload(_read_json(release_path / "genome.json", "genome.json"))
        try:
            round1_prompt = (release_path / "round1_prompt.md").read_text(encoding="utf-8")
            round2_prompt = (release_path / "round2_prompt.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise RetrievalPolicyError(f"invalid release prompt: {release_path}") from error
        skills_payload = _read_json(release_path / "skills.json", "skills.json")
        skills = _skills_from_payload(skills_payload)
        manifest_raw = _read_json(release_path / "manifest.json", "manifest.json")
        manifest = _exact_mapping(manifest_raw, _MANIFEST_FIELDS, "manifest")
        if (
            isinstance(manifest["schema_version"], bool)
            or not isinstance(manifest["schema_version"], int)
            or manifest["schema_version"] != 1
        ):
            raise RetrievalPolicyError("invalid manifest schema_version")
        expected = {
            "schema_version": 1,
            "version": genome.version,
            "parent": genome.parent,
            "genome_sha256": genome.fingerprint(),
            "round1_prompt_sha256": _digest(round1_prompt.encode("utf-8")),
            "round2_prompt_sha256": _digest(round2_prompt.encode("utf-8")),
            "skills_sha256": _digest(_canonical_json(skills_payload)),
        }
        for field, digest in expected.items():
            if manifest.get(field) != digest:
                label = field.removesuffix("_sha256").replace("_", " ")
                raise RetrievalPolicyError(f"release {label} hash or binding mismatch")
        _validate_release_audit(manifest, genome)
        if round1_prompt != genome.round1_prompt or round2_prompt != genome.round2_prompt:
            raise RetrievalPolicyError("release prompt does not match genome")
        return cls(
            path=release_path,
            genome=genome,
            round1_prompt=round1_prompt,
            round2_prompt=round2_prompt,
            skills=skills,
            manifest=MappingProxyType(dict(manifest)),
        )


def write_retrieval_release(
    releases_path: str | Path,
    genome: RetrievalGenome,
    *,
    skills: Sequence[object] = (),
    audit: Mapping[str, object] | None = None,
) -> RetrievalRelease:
    """Atomically publish ``genome`` and its prompt/skill artifacts under ``releases/vNNN``."""
    releases = Path(releases_path)
    _reject_git_path(releases)
    validated = RetrievalGenome.from_payload(genome.to_payload())
    destination = releases / validated.version
    _reject_git_path(destination)
    if _lexists(destination):
        raise RetrievalPolicyError(f"release destination already exists: {destination}")
    skills_payload = _skills_payload(skills)
    audit_payload = _audit_payload(validated, audit)
    manifest_audit = {
        **audit_payload,
        "resource_budgets": _resource_budgets(validated),
    }
    if manifest_audit["state"] == "accepted":
        manifest_audit["audit_sha256"] = _digest(_canonical_json(manifest_audit))
    else:
        manifest_audit["audit_sha256"] = None
    _validate_release_audit(manifest_audit, validated)
    if _lexists(releases) and not releases.is_dir():
        raise RetrievalPolicyError(f"release root is not a directory: {releases}")
    releases.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{validated.version}.", dir=releases))
    try:
        _write_json(temporary / "genome.json", validated.to_payload())
        (temporary / "round1_prompt.md").write_text(validated.round1_prompt, encoding="utf-8")
        (temporary / "round2_prompt.md").write_text(validated.round2_prompt, encoding="utf-8")
        _write_json(temporary / "skills.json", skills_payload)
        manifest = {
            "schema_version": 1,
            "version": validated.version,
            "parent": validated.parent,
            "genome_sha256": validated.fingerprint(),
            "round1_prompt_sha256": _digest(validated.round1_prompt.encode("utf-8")),
            "round2_prompt_sha256": _digest(validated.round2_prompt.encode("utf-8")),
            "skills_sha256": _digest(_canonical_json(skills_payload)),
            **manifest_audit,
        }
        _write_json(temporary / "manifest.json", manifest)
        if _lexists(destination):  # Reject dangling symlinks without replacing them.
            raise RetrievalPolicyError(f"release destination already exists: {destination}")
        os.rename(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return RetrievalRelease.load(destination)


__all__ = [
    "BOUNDS",
    "ROUND1_STRATEGIES",
    "ROUND2_STRATEGIES",
    "SECOND_ROUND_TRIGGERS",
    "RetrievalGenome",
    "RetrievalPolicyError",
    "RetrievalRelease",
    "write_retrieval_release",
]
