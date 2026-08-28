"""Self-evolving forecasting-method module: the LLM adds, deletes, and merges its functions."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.payload import read_json_object, write_json
from common.tracing import TraceEvent, emit

from .execution import Task, reports_as_json, run_module
from .history import parse_history
from .memory import record_generation
from .module import (
    Method,
    MethodModule,
    ModuleError,
    apply_operations,
    parse_method,
    read_module,
    write_module,
)
from .prompts import (
    WRITE_METHOD_PROMPT,
    IMPROVE_METHODS_PROMPT,
    build_write_request,
    build_improve_request,
    build_retry_request,
)

METHODS_FILENAME = "methods.py"


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
    val_best_smae: float | None = None


def run_git(repo: Path, *args: str) -> str:
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
        run_git(root, "init", "--quiet")
        run_git(root, "config", "user.name", "numerical-agent")
        run_git(root, "config", "user.email", "numerical-agent@localhost")
    return root


def commit_module(repo: Path, subject: str, body: Sequence[str]) -> str:
    """Commit the current module and return the new commit hash."""
    run_git(repo, "add", METHODS_FILENAME)
    if not run_git(repo, "status", "--porcelain", "--", METHODS_FILENAME) and _has_commit(repo):
        return run_git(repo, "rev-parse", "--short", "HEAD")
    message = subject if not body else subject + "\n\n" + "\n".join(f"- {line}" for line in body)
    run_git(repo, "commit", "--quiet", "-m", message)
    return run_git(repo, "rev-parse", "--short", "HEAD")


def _has_commit(repo: Path) -> bool:
    try:
        run_git(repo, "rev-parse", "HEAD")
    except RuntimeError:
        return False
    return True


def bootstrap(
    repo: str | Path,
    definitions: Sequence[Mapping[str, object]],
    llm: LLMClient,
    preset: Sequence[Method] = (),
) -> MethodModule:
    """Write one function per catalog definition and commit the starting module.

    preset methods are seeded verbatim rather than written by the model: the foundation-model
    wrappers depend on each package's exact calling convention, which a model can only guess at,
    and a wrong guess arrives as a crashed method rather than as the mistake it is.
    """
    root = init_repo(repo)
    methods = list(preset)
    for index, definition in enumerate(definitions, start=1):
        name = str(definition["name"])
        emit(TraceEvent(name, "evolution", "method_start", {"stage": "bootstrap", "index": index}))
        user = build_write_request(
            name=name,
            description=str(definition.get("description", "")),
            assumptions=[str(item) for item in definition.get("assumptions", ())],
            failure_conditions=[str(item) for item in definition.get("failure_conditions", ())],
        )
        try:
            code = _request_code(llm, WRITE_METHOD_PROMPT, user)
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
    write_module(root / METHODS_FILENAME, module)
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
    val_tasks: Sequence[Task] = (),
    memory_llm: LLMClient | None = None,
) -> Generation:
    """Measure the module, ask for operations, apply them, and commit the result."""
    root = Path(repo)
    module_path = root / METHODS_FILENAME
    module = read_module(module_path)
    _preload(llm)

    # Validation is measured on the same module as train, so the two are directly comparable,
    # and is written to disk only. It never reaches the prompt: a score the model can read is a
    # score it will optimize against, which would cost us the one held-out signal we have.
    payload, val_payload = _measure(root, module_path, tasks, val_tasks)
    write_json(
        root / f"generation_{generation:03d}_metrics.json",
        {"reports": payload, "val_reports": val_payload},
    )
    val_smae = _best_mean_smae(val_payload)
    if val_payload:
        emit(TraceEvent("module", "evolution", "validation",
                        {"generation": generation, "tasks": len(val_tasks),
                         "methods": len(val_payload), "best_mean_smae": val_smae}))

    user = build_improve_request(
        module_source=module_path.read_text(encoding="utf-8"),
        reports=payload,
        generation=generation,
        task_count=len(tasks),
        history=parse_history(run_git(root, "log", "--format=%s%n%b")),
        live=module.names(),
    )
    messages = [{"role": "user", "content": user}]
    updated: MethodModule | None = None
    summaries: tuple[str, ...] = ()
    failure = ""

    # One retry: a malformed operation is usually a formatting slip in a single field, and the
    # rest of the batch is worth recovering rather than discarding along with it.
    for attempt in range(2):
        response = llm.complete(
            system=IMPROVE_METHODS_PROMPT, messages=messages,
            temperature=0.2,
        )
        _write_transcript(root, generation, IMPROVE_METHODS_PROMPT, messages, response.text, attempt)

        try:
            # An empty or truncated response fails to parse as JSON at all, which is the same
            # kind of transient slip as a malformed operation -- worth a retry, not a crash.
            operations = _parse_operations(response.text)
            if not operations:
                if attempt:
                    break
                record_generation(
                    root, memory_llm, generation=generation, applied=(),
                    method_count=len(module.names()), val_best_smae=val_smae,
                    reasoning=response.text,
                )
                return Generation(
                    generation, (), len(module.names()), _head(root),
                    "no operations proposed", converged=True, val_best_smae=val_smae,
                )
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
                {"role": "user", "content": build_retry_request(failure)},
            ]

    if updated is None:
        record_generation(
            root, memory_llm, generation=generation, applied=(),
            method_count=len(module.names()), val_best_smae=val_smae,
            reasoning=f"The batch was rejected and nothing was applied: {failure}",
        )
        return Generation(
            generation, (), len(module.names()), _head(root),
            failure or "no operations proposed", retried=True, val_best_smae=val_smae,
        )

    retried = len(messages) > 1
    commit = commit_module(root, f"generation {generation}: {len(summaries)} operations", summaries)
    # After the commit: a failed summary must not discard work that already succeeded.
    record_generation(
        root, memory_llm, generation=generation, applied=summaries,
        method_count=len(updated.names()), val_best_smae=val_smae, reasoning=response.text,
    )
    emit(TraceEvent("module", "evolution", "generation",
                    {"generation": generation, "methods": len(updated.names()),
                     "operations": len(summaries), "commit": commit, "retried": retried}))
    return Generation(
        generation, summaries, len(updated.names()), commit,
        retried=retried, val_best_smae=val_smae,
    )


def _preload(llm: LLMClient) -> None:
    """Load a local model's weights before measuring; a no-op for clients that call out."""
    loader = getattr(llm, "preload", None)
    if loader is None:
        return
    loader()
    emit(TraceEvent("module", "evolution", "llm_preloaded", {"model": getattr(llm, "model_id", "")}))


MEASUREMENTS_DIRNAME = "measurements"


def _measure(
    root: Path,
    module_path: Path,
    tasks: Sequence[Task],
    val_tasks: Sequence[Task],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Measure train and val, reusing a cached result for a module already measured.

    Keyed on the module's content, not the generation number: a crash after measurement (an
    OOM in the LLM call is the usual one) re-runs the generation without paying for the same
    hour of measurement twice, and a rejected generation leaves the module unchanged so the
    next generation reads the same entry rather than remeasuring it.
    """
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    cached = root / MEASUREMENTS_DIRNAME / f"{digest}.json"
    if cached.exists():
        payload = read_json_object(cached)
        emit(TraceEvent("module", "evolution", "measurement_reused", {"digest": digest[:12]}))
        return list(payload["reports"]), list(payload["val_reports"])  # type: ignore[arg-type]

    reports = reports_as_json(run_module(module_path, tasks)[1])
    val_reports = reports_as_json(run_module(module_path, val_tasks)[1]) if val_tasks else []
    write_json(cached, {"reports": reports, "val_reports": val_reports})
    return reports, val_reports


def _best_mean_smae(reports: Sequence[Mapping[str, object]]) -> float | None:
    """Lowest mean sMAE any method reached, or None when nothing scored."""
    scored = [r["mean_smae"] for r in reports if r.get("mean_smae") is not None]
    return min(float(value) for value in scored) if scored else None


def run_evolution(
    repo: str | Path,
    tasks: Sequence[Task],
    llm: LLMClient,
    generations: int,
    val_tasks: Sequence[Task] = (),
    memory_llm: LLMClient | None = None,
) -> tuple[Generation, ...]:
    """Run consecutive generations, stopping only when the model proposes nothing.

    A rejected generation leaves the module untouched, so the next generation re-measures and
    tries again rather than ending the run.
    """
    results = []
    for number in range(1, generations + 1):
        outcome = evolve_once(
            repo, tasks, llm, number, val_tasks=val_tasks, memory_llm=memory_llm
        )
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
    return run_git(repo, "rev-parse", "--short", "HEAD") if _has_commit(repo) else ""


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
