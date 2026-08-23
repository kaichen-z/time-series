"""Resumable construction of the standalone forecasting-method Git repository."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.payload import canonical_json_bytes, read_json_object
from common.sandbox import check_code
from common.tracing import TraceEvent, emit

from . import MODULE_NAME, _has_commit, git, init_repo
from .module import (
    EVOLUTION_DUNDERS,
    EVOLUTION_IMPORTS,
    Method,
    MethodModule,
    ModuleError,
    parse_method,
    write_module,
)
from .prompts import BOOTSTRAP_SYSTEM, render_bootstrap_user
from .analysis_skills import DEFAULT_SKILLS_PATH, validate_skill_source
from .portfolio import PolicyPortfolio, render_policy_source


class BootstrapError(RuntimeError):
    """Raised when a method repository cannot be safely bootstrapped or resumed."""


@dataclass(frozen=True)
class BootstrapResult:
    """Auditable outcome of one complete catalog bootstrap."""

    total: int
    succeeded: int
    failed: int
    resumed: int
    commit: str


def bootstrap_repository(
    repo: str | Path,
    definitions: Sequence[Mapping[str, object]],
    excluded: Sequence[Mapping[str, object]],
    llm: LLMClient,
    *,
    attempts_per_method: int = 2,
) -> BootstrapResult:
    """Generate every definition, checkpoint each valid method, and create one seed commit.

    The work directory under ``.bootstrap`` is intentionally outside the seed commit. If an
    LLM or transport call aborts the command, the next invocation verifies the catalog digest
    and resumes from the first missing method without regenerating completed work.
    """
    if attempts_per_method < 1:
        raise ValueError("attempts_per_method must be at least 1")
    if not definitions:
        raise BootstrapError("no method definitions were selected")

    root = Path(repo)
    if (root / MODULE_NAME).exists():
        raise BootstrapError(f"{root} is already seeded; refusing to overwrite {MODULE_NAME}")
    root = init_repo(root)
    if _has_commit(root):
        raise BootstrapError(f"{root} already has a commit; refusing to bootstrap over it")

    work = root / ".bootstrap"
    methods_dir = work / "methods"
    transcripts_dir = work / "transcripts"
    manifest_path = work / "manifest.json"
    progress_path = work / "progress.json"
    methods_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    normalized_definitions = [_normalize_definition(item) for item in definitions]
    definition_hash = hashlib.sha256(
        canonical_json_bytes({"definitions": normalized_definitions})
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "definition_hash": definition_hash,
        "method_names": [item["name"] for item in normalized_definitions],
    }
    if manifest_path.exists():
        previous = read_json_object(manifest_path)
        if previous != manifest:
            raise BootstrapError("resume definition set does not match the existing bootstrap manifest")
    else:
        _atomic_json(manifest_path, manifest)

    progress = _load_progress(progress_path, definition_hash)
    resumed = 0
    methods: dict[str, Method] = {}
    for definition in normalized_definitions:
        name = str(definition["name"])
        checkpoint = methods_dir / f"{name}.py"
        if checkpoint.exists():
            try:
                methods[name] = _validated_method(checkpoint.read_text(encoding="utf-8"), name)
            except (ValueError, ModuleError) as exc:
                raise BootstrapError(f"invalid bootstrap checkpoint for {name}: {exc}") from exc
            resumed += 1
            continue

        entry = _progress_entry(progress, name)
        last_error = str(entry.get("error", ""))
        used_attempts = int(entry.get("attempts", 0))
        for attempt in range(used_attempts + 1, attempts_per_method + 1):
            emit(
                TraceEvent(
                    name,
                    "evolution",
                    "method_start",
                    {"stage": "bootstrap", "attempt": attempt},
                )
            )
            user = _bootstrap_request(definition, attempt, last_error)
            response = llm.complete(
                system=BOOTSTRAP_SYSTEM,
                messages=[{"role": "user", "content": user}],
                temperature=0.0,
            )
            error = ""
            try:
                payload = parse_json_object(response.text)
                code = payload.get("code")
                if not isinstance(code, str) or not code.strip():
                    raise ValueError("response contained no non-empty 'code' string")
                method = _validated_method(code, name)
            except (ValueError, ModuleError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                last_error = error
                entry.update({"status": "failed", "attempts": attempt, "error": error})
                _write_transcript(transcripts_dir, name, attempt, user, response.text, error)
                _atomic_json(progress_path, progress)
                emit(
                    TraceEvent(
                        name,
                        "evolution",
                        "method_end",
                        {"stage": "bootstrap", "ok": False, "attempt": attempt, "error": error},
                    )
                )
                continue

            _atomic_text(checkpoint, method.source.rstrip() + "\n")
            methods[name] = method
            entry.update({"status": "succeeded", "attempts": attempt, "error": ""})
            _write_transcript(transcripts_dir, name, attempt, user, response.text, "")
            _atomic_json(progress_path, progress)
            emit(
                TraceEvent(
                    name,
                    "evolution",
                    "method_end",
                    {"stage": "bootstrap", "ok": True, "attempt": attempt},
                )
            )
            break

    ordered_methods = tuple(
        methods[str(definition["name"])]
        for definition in normalized_definitions
        if str(definition["name"]) in methods
    )
    if not ordered_methods:
        raise BootstrapError("bootstrap produced no valid methods")

    failures = [
        {
            "name": str(definition["name"]),
            "category": str(definition.get("category", "")),
            "attempts": int(_progress_entry(progress, str(definition["name"])).get("attempts", 0)),
            "error": str(_progress_entry(progress, str(definition["name"])).get("error", "")),
        }
        for definition in normalized_definitions
        if str(definition["name"]) not in methods
    ]
    module = MethodModule(ordered_methods)
    write_module(root / MODULE_NAME, module)
    skill_source = DEFAULT_SKILLS_PATH.read_text(encoding="utf-8")
    validate_skill_source(skill_source)
    _atomic_text(root / "skills.py", skill_source)
    _atomic_text(root / "policies.py", render_policy_source(PolicyPortfolio.flagship5()))
    summary = {
        "schema_version": 1,
        "definition_hash": definition_hash,
        "total": len(normalized_definitions),
        "succeeded": len(ordered_methods),
        "failed": len(failures),
        "resumed": resumed,
        "failures": failures,
    }
    _atomic_json(root / "bootstrap_summary.json", summary)
    _atomic_json(root / "excluded_methods.json", {"methods": [dict(item) for item in excluded]})
    _atomic_text(root / ".gitignore", ".bootstrap/\ncodex-cache/\n__pycache__/\n")

    git(
        root,
        "add",
        MODULE_NAME,
        "skills.py",
        "policies.py",
        "bootstrap_summary.json",
        "excluded_methods.json",
        ".gitignore",
    )
    git(root, "commit", "--quiet", "-m", f"seed {len(ordered_methods)} forecasting methods")
    commit = git(root, "rev-parse", "--short", "HEAD")
    emit(
        TraceEvent(
            "module",
            "evolution",
            "generation",
            {
                "generation": 0,
                "methods": len(ordered_methods),
                "failed": len(failures),
                "commit": commit,
            },
        )
    )
    return BootstrapResult(
        total=len(normalized_definitions),
        succeeded=len(ordered_methods),
        failed=len(failures),
        resumed=resumed,
        commit=commit,
    )


def _normalize_definition(definition: Mapping[str, object]) -> dict[str, object]:
    name = str(definition.get("name", "")).strip()
    if not name or not name.isidentifier():
        raise BootstrapError(f"invalid method name {name!r}")
    return {
        "name": name,
        "category": str(definition.get("category", "")),
        "description": str(definition.get("description", "")),
        "assumptions": [str(item) for item in definition.get("assumptions", ())],
        "failure_conditions": [
            str(item) for item in definition.get("failure_conditions", ())
        ],
    }


def _bootstrap_request(definition: Mapping[str, object], attempt: int, error: str) -> str:
    payload = json.loads(
        render_bootstrap_user(
            name=str(definition["name"]),
            description=str(definition.get("description", "")),
            assumptions=[str(item) for item in definition.get("assumptions", ())],
            failure_conditions=[
                str(item) for item in definition.get("failure_conditions", ())
            ],
        )
    )
    payload["attempt"] = attempt
    if error:
        payload["previous_validation_error"] = error
        payload["revision_instruction"] = (
            "Return a corrected implementation that fixes this exact validation error."
        )
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _validated_method(code: str, name: str) -> Method:
    method = parse_method(code, expected_name=name)
    check_code(MethodModule((method,)).render(), EVOLUTION_IMPORTS, EVOLUTION_DUNDERS)
    return method


def _load_progress(path: Path, definition_hash: str) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": 1, "definition_hash": definition_hash, "methods": {}}
    progress = read_json_object(path)
    if progress.get("definition_hash") != definition_hash:
        raise BootstrapError("progress definition hash does not match the bootstrap manifest")
    if not isinstance(progress.get("methods"), dict):
        raise BootstrapError("bootstrap progress has no methods object")
    return progress


def _progress_entry(progress: dict[str, object], name: str) -> dict[str, object]:
    entries = progress["methods"]
    assert isinstance(entries, dict)
    entry = entries.setdefault(name, {"status": "pending", "attempts": 0, "error": ""})
    if not isinstance(entry, dict):
        raise BootstrapError(f"invalid progress entry for {name}")
    return entry


def _write_transcript(
    directory: Path,
    name: str,
    attempt: int,
    request: str,
    response: str,
    error: str,
) -> None:
    _atomic_json(
        directory / f"{name}_attempt_{attempt:02d}.json",
        {"method": name, "attempt": attempt, "request": request, "response": response, "error": error},
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(path, canonical_json_bytes(payload))


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)
