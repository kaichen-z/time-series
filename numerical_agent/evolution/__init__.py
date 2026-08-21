"""Self-evolving forecasting-method module: the LLM adds, deletes, and merges its functions."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.payload import write_json
from common.tracing import TraceEvent, emit

from .execution import Task, report_payload, run_module
from .module import MethodModule, ModuleError, apply_operations, parse_method, read_module, write_module
from .prompts import (
    BOOTSTRAP_SYSTEM,
    EVOLVE_SYSTEM,
    render_bootstrap_user,
    render_evolve_user,
    render_retry_user,
)

MODULE_NAME = "methods.py"


@dataclass(frozen=True)
class Generation:
    """What one evolution generation did to the module."""

    number: int
    applied: tuple[str, ...]
    method_count: int
    commit: str
    rejected: str = ""
    retried: bool = False
    converged: bool = False


def git(repo: Path, *args: str) -> str:
    """Run one git command inside the evolution repository."""
    completed = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def init_repo(repo: str | Path) -> Path:
    """Create the evolution repository if it does not exist yet."""
    root = Path(repo)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        git(root, "init", "--quiet")
        git(root, "config", "user.name", "numerical-agent")
        git(root, "config", "user.email", "numerical-agent@localhost")
    return root


def commit_module(repo: Path, subject: str, body: Sequence[str]) -> str:
    """Commit the current module and return the new commit hash."""
    git(repo, "add", MODULE_NAME)
    if not git(repo, "status", "--porcelain", "--", MODULE_NAME) and _has_commit(repo):
        return git(repo, "rev-parse", "--short", "HEAD")
    message = subject if not body else subject + "\n\n" + "\n".join(f"- {line}" for line in body)
    git(repo, "commit", "--quiet", "-m", message)
    return git(repo, "rev-parse", "--short", "HEAD")


def _has_commit(repo: Path) -> bool:
    try:
        git(repo, "rev-parse", "HEAD")
    except RuntimeError:
        return False
    return True


def bootstrap(
    repo: str | Path,
    definitions: Sequence[Mapping[str, object]],
    llm: LLMClient,
) -> MethodModule:
    """Write one function per catalog definition and commit the starting module."""
    root = init_repo(repo)
    methods = []
    for index, definition in enumerate(definitions, start=1):
        name = str(definition["name"])
        emit(TraceEvent(name, "evolution", "method_start", {"stage": "bootstrap", "index": index}))
        user = render_bootstrap_user(
            name=name,
            description=str(definition.get("description", "")),
            assumptions=[str(item) for item in definition.get("assumptions", ())],
            failure_conditions=[str(item) for item in definition.get("failure_conditions", ())],
        )
        try:
            code = _request_code(llm, BOOTSTRAP_SYSTEM, user)
            methods.append(parse_method(code, expected_name=name))
        except (ValueError, ModuleError) as exc:
            # One bad definition must not abort a 111-method bootstrap.
            emit(TraceEvent(name, "evolution", "method_end",
                            {"stage": "bootstrap", "ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            continue
        emit(TraceEvent(name, "evolution", "method_end", {"stage": "bootstrap", "ok": True}))

    if not methods:
        raise ModuleError("bootstrap produced no valid methods")
    module = MethodModule(tuple(methods))
    write_module(root / MODULE_NAME, module)
    commit = commit_module(
        root, f"seed {len(methods)} forecasting methods", [f"{m.name}" for m in methods[:20]]
    )
    emit(TraceEvent("module", "evolution", "generation", {"generation": 0, "methods": len(methods), "commit": commit}))
    return module


def evolve_once(
    repo: str | Path,
    tasks: Sequence[Task],
    llm: LLMClient,
    generation: int,
) -> Generation:
    """Measure the module, ask for operations, apply them, and commit the result."""
    root = Path(repo)
    module_path = root / MODULE_NAME
    module = read_module(module_path)

    _, reports = run_module(module_path, tasks)
    payload = report_payload(reports)
    write_json(root / f"generation_{generation:03d}_metrics.json", {"reports": payload})

    user = render_evolve_user(
        module_source=module_path.read_text(encoding="utf-8"),
        reports=payload,
        generation=generation,
        task_count=len(tasks),
    )
    messages = [{"role": "user", "content": user}]
    updated: MethodModule | None = None
    summaries: tuple[str, ...] = ()
    failure = ""

    # One retry: a malformed operation is usually a formatting slip in a single field, and the
    # rest of the batch is worth recovering rather than discarding along with it.
    for attempt in range(2):
        response = llm.complete(
            system=EVOLVE_SYSTEM, messages=messages,
            temperature=0.7, # This might get chanfed but better than 0.0
        )
        _write_transcript(root, generation, EVOLVE_SYSTEM, messages, response.text, attempt)

        operations = _parse_operations(response.text)
        if not operations:
            if attempt:
                break
            return Generation(
                generation, (), len(module.names()), _head(root),
                "no operations proposed", converged=True,
            )

        try:
            updated, summaries = apply_operations(module, operations)
            write_module(module_path, updated)
            failure = ""
            break
        except (ModuleError, ValueError) as exc:
            # A rejected generation leaves the previous commit standing, unmodified.
            write_module(module_path, module)
            updated, failure = None, str(exc)
            emit(TraceEvent("module", "evolution", "generation",
                            {"generation": generation, "ok": False,
                             "attempt": attempt + 1, "error": failure}))
            messages = messages + [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": render_retry_user(failure)},
            ]

    if updated is None:
        return Generation(
            generation, (), len(module.names()), _head(root),
            failure or "no operations proposed", retried=True,
        )

    retried = len(messages) > 1
    commit = commit_module(root, f"generation {generation}: {len(summaries)} operations", summaries)
    emit(TraceEvent("module", "evolution", "generation",
                    {"generation": generation, "methods": len(updated.names()),
                     "operations": len(summaries), "commit": commit, "retried": retried}))
    return Generation(generation, summaries, len(updated.names()), commit, retried=retried)


def run_evolution(
    repo: str | Path,
    tasks: Sequence[Task],
    llm: LLMClient,
    generations: int,
) -> tuple[Generation, ...]:
    """Run consecutive generations, stopping only when the model proposes nothing.

    A rejected generation leaves the module untouched, so the next generation re-measures and
    tries again rather than ending the run.
    """
    results = []
    for number in range(1, generations + 1):
        outcome = evolve_once(repo, tasks, llm, number)
        results.append(outcome)
        if outcome.converged:
            break
    return tuple(results)


def _request_code(llm: LLMClient, system: str, user: str) -> str:
    response = llm.complete(
        system=system, messages=[{"role": "user", "content": user}], temperature=0.0
    )
    payload = parse_json_object(response.text)
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("response contained no non-empty 'code' string")
    return code


def _parse_operations(text: str) -> list[dict]:
    payload = parse_json_object(text)
    operations = payload.get("operations", [])
    if not isinstance(operations, list):
        raise ModuleError("'operations' must be a list")
    return [item for item in operations if isinstance(item, dict)]


def _head(repo: Path) -> str:
    return git(repo, "rev-parse", "--short", "HEAD") if _has_commit(repo) else ""


def _write_transcript(
    repo: Path,
    generation: int,
    system: str,
    messages: Sequence[Mapping[str, str]],
    response: str,
    attempt: int = 0,
) -> None:
    """Write one attempt's transcript; a retry gets its own file rather than overwriting."""
    suffix = "" if not attempt else f"_retry{attempt}"
    destination = repo / "transcripts" / f"generation_{generation:03d}{suffix}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    turns = "\n\n".join(f"## {m['role']}\n\n{m['content']}" for m in messages)
    destination.write_text(
        f"# system\n\n{system}\n\n# user\n\n{turns}\n\n# response\n\n{response}\n",
        encoding="utf-8",
    )
