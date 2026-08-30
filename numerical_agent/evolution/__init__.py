"""Self-evolving forecasting-method module: the LLM adds, deletes, and merges its functions."""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.llm import LLMClient, parse_json_object
from common.payload import write_json
from common.tracing import TraceEvent, emit

from .execution import Outcome, Task, oracle_scaled_summary, report_payload, run_module
from .identity import IdentityError, identity_contract, validate_repair
from .module import MethodModule, ModuleError, apply_operations, parse_method, read_module, write_module
from .prompts import (
    BOOTSTRAP_SYSTEM,
    EVOLVE_SYSTEM,
    MUTATE_SYSTEM,
    SELECT_SYSTEM,
    render_bootstrap_user,
    render_evolve_user,
    render_mutate_user,
    render_select_user,
)
from .morphology import (
    AssumptionGrounding,
    MorphologyCard,
    MorphologyError,
    MorphologyInputError,
    MorphologyObservation,
    MorphologyReasoner,
    MorphologyToolCall,
)
from .numerical_loop import run_numerical_loop
from .numerical_package import NumericalForecastPackage, RankedNumericalForecast

MODULE_NAME = "methods.py"


@dataclass(frozen=True)
class Generation:
    """What one evolution generation did to the module."""

    number: int
    applied: tuple[str, ...]
    method_count: int
    commit: str
    rejected: str = ""


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
    *,
    selector_llm: LLMClient | None = None,
    isolate_methods: bool = False,
    validation_tasks: Sequence[Task] = (),
) -> Generation:
    """Measure the module, ask for operations, apply them, and commit the result."""
    root = Path(repo)
    module_path = root / MODULE_NAME
    module = read_module(module_path)

    _, reports = run_module(module_path, tasks, isolated=isolate_methods)
    payload = report_payload(reports)
    write_json(root / f"generation_{generation:03d}_metrics.json", {"reports": payload})

    if selector_llm is None:
        system = EVOLVE_SYSTEM
        user = render_evolve_user(
            module_source=module_path.read_text(encoding="utf-8"),
            reports=payload,
            generation=generation,
            task_count=len(tasks),
        )
        response = llm.complete(
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.7,
        )
        _write_transcript(root, generation, system, user, response.text)
        selected_names: set[str] | None = None
    else:
        inventory = []
        for method in module.methods:
            contract = identity_contract(method.name, method.source)
            inventory.append({
                "name": method.name,
                "docstring": method.docstring,
                **contract.payload(),
            })
        select_user = render_select_user(
            reports=payload,
            method_inventory=inventory,
            generation=generation,
            task_count=len(tasks),
        )
        selection = selector_llm.complete(
            system=SELECT_SYSTEM,
            messages=[{"role": "user", "content": select_user}],
            temperature=0.0,
        )
        _write_transcript(
            root, generation, SELECT_SYSTEM, select_user, selection.text, stage="selector"
        )
        try:
            selected = _parse_targets(selection.text, set(module.names()))
        except (ValueError, ModuleError) as exc:
            emit(
                TraceEvent(
                    "module",
                    "evolution",
                    "generation",
                    {
                        "generation": generation,
                        "stage": "selector",
                        "ok": False,
                        "error": str(exc),
                    },
                )
            )
            return Generation(generation, (), len(module.names()), _head(root), str(exc))
        if not selected:
            return Generation(generation, (), len(module.names()), _head(root), "no targets selected")
        selected_actions = {target["name"]: target["action"] for target in selected}
        selected_names = set(selected_actions)
        selected_reports = [
            report for report in payload if str(report["method"]) in selected_names
        ]
        selected_source = "\n\n".join(
            module.get(name).source for name in module.names() if name in selected_names
        )
        user = render_mutate_user(
            reports=selected_reports,
            selected=selected,
            selected_source=selected_source,
            all_method_names=module.names(),
            identity_contracts=[
                identity_contract(name, module.get(name).source).payload()
                for name in module.names()
                if name in selected_names
            ],
            generation=generation,
            task_count=len(tasks),
        )
        response = llm.complete(
            system=MUTATE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
        )
        _write_transcript(
            root, generation, MUTATE_SYSTEM, user, response.text, stage="mutator"
        )

    try:
        operations = _parse_operations(response.text)
        if selected_names is not None:
            _validate_selected_operations(operations, selected_actions)
    except (ValueError, ModuleError) as exc:
        emit(
            TraceEvent(
                "module",
                "evolution",
                "generation",
                {
                    "generation": generation,
                    "stage": "mutator" if selected_names is not None else "evolver",
                    "ok": False,
                    "error": str(exc),
                },
            )
        )
        return Generation(generation, (), len(module.names()), _head(root), str(exc))
    if not operations:
        return Generation(generation, (), len(module.names()), _head(root), "no operations proposed")

    try:
        updated, summaries = apply_operations(module, operations)
        if selected_names is not None:
            _validate_evolved_module(module, updated, operations, payload)
    except (ModuleError, ValueError) as exc:
        # A rejected generation leaves the previous commit standing, unmodified.
        write_module(module_path, module)
        emit(TraceEvent("module", "evolution", "generation",
                        {"generation": generation, "ok": False, "error": str(exc)}))
        return Generation(generation, (), len(module.names()), _head(root), str(exc))

    if validation_tasks:
        accepted, validation = _validate_candidate(
            root,
            generation,
            module_path,
            updated,
            validation_tasks,
            isolate_methods=isolate_methods,
        )
        if not accepted:
            return Generation(
                generation,
                (),
                len(module.names()),
                _head(root),
                (
                    "validation scaled pair failed Pareto acceptance: "
                    f"parent=({validation['parent_mean_smae']:.6g}, "
                    f"{validation['parent_mean_srmse']:.6g}), "
                    f"child=({validation['child_mean_smae']:.6g}, "
                    f"{validation['child_mean_srmse']:.6g})"
                ),
            )

    write_module(module_path, updated)

    commit = commit_module(root, f"generation {generation}: {len(summaries)} operations", summaries)
    emit(TraceEvent("module", "evolution", "generation",
                    {"generation": generation, "methods": len(updated.names()),
                     "operations": len(summaries), "commit": commit}))
    return Generation(generation, summaries, len(updated.names()), commit)


def run_evolution(
    repo: str | Path,
    tasks: Sequence[Task],
    llm: LLMClient,
    generations: int,
    *,
    selector_llm: LLMClient | None = None,
    isolate_methods: bool = False,
    validation_tasks: Sequence[Task] = (),
) -> tuple[Generation, ...]:
    """Run consecutive generations, stopping early when a generation changes nothing."""
    results = []
    for number in range(1, generations + 1):
        outcome = evolve_once(
            repo,
            tasks,
            llm,
            number,
            selector_llm=selector_llm,
            isolate_methods=isolate_methods,
            validation_tasks=validation_tasks,
        )
        results.append(outcome)
        if not outcome.applied:
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


def _parse_targets(text: str, existing_names: set[str]) -> list[dict[str, str]]:
    payload = parse_json_object(text)
    raw = payload.get("targets", [])
    if not isinstance(raw, list):
        raise ModuleError("'targets' must be a list")
    targets: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in raw[:10]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        action = str(item.get("action", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if name not in existing_names or action not in {"delete", "repair", "fork", "merge"} or not reason:
            continue
        if name in seen_names:
            raise ModuleError(f"duplicate selector target {name!r}")
        seen_names.add(name)
        targets.append({"name": name, "action": action, "reason": reason})
    return targets


def _validate_selected_operations(
    operations: Sequence[Mapping[str, object]], selected_actions: Mapping[str, str]
) -> None:
    selected_names = set(selected_actions)
    for operation in operations:
        op = str(operation.get("op", ""))
        if op in {"delete", "repair"}:
            touched = {str(operation.get("name", ""))}
        elif op == "fork":
            touched = {str(operation.get("from", ""))}
        elif op == "merge":
            raw = operation.get("names", ())
            names = (
                [str(name) for name in raw]
                if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
                else []
            )
            touched = set(names)
            if len(touched) < 2:
                raise ModuleError("merge requires at least two distinct selected names")
            into = str(operation.get("into", ""))
            if into not in touched:
                raise ModuleError("merge 'into' must be one of its selected names")
        else:
            raise ModuleError(f"two-stage evolution does not allow operation {op!r}")
        if not touched or not touched.issubset(selected_names):
            raise ModuleError("operation touches methods outside selected targets")
        if any(selected_actions[name] != op for name in touched):
            raise ModuleError(
                f"operation {op!r} does not match selector action for "
                f"{', '.join(sorted(touched))}"
            )


def _validate_evolved_module(
    parent: MethodModule,
    child: MethodModule,
    operations: Sequence[Mapping[str, object]],
    reports: Sequence[Mapping[str, object]],
) -> None:
    """Validate semantic identity and evidence against the final candidate state."""
    by_name = {str(report.get("method", "")): report for report in reports}
    parent_names = set(parent.names())
    child_names = set(child.names())

    for name in sorted(parent_names - child_names):
        report = by_name.get(name, {})
        total = int(report.get("total", 0) or 0)
        not_applicable = int(report.get("not_applicable", 0) or 0)
        coverage = float(report.get("coverage", 0.0) or 0.0)
        if (total and not_applicable == total) or coverage < 0.5:
            raise ModuleError(
                f"{name} is not sufficiently evaluated for deletion; "
                "NotApplicable or low coverage is not evidence of inferiority"
            )

    for name in sorted(parent_names & child_names):
        before = parent.get(name)
        after = child.get(name)
        if before is not None and after is not None and before.source != after.source:
            try:
                validate_repair(
                    name,
                    before.source,
                    after.source,
                    identity_contract(name, before.source),
                )
            except IdentityError as exc:
                raise ModuleError(str(exc)) from exc

    for operation in operations:
        if str(operation.get("op", "")) == "fork":
            source = str(operation.get("from", ""))
            if source not in child_names:
                raise ModuleError(f"fork parent {source} must remain in the candidate module")
            if not str(operation.get("new_identity", "")).strip():
                raise ModuleError("fork must state a non-empty new_identity")


def _validate_candidate(
    root: Path,
    generation: int,
    parent_path: Path,
    child: MethodModule,
    tasks: Sequence[Task],
    *,
    isolate_methods: bool,
) -> tuple[bool, dict[str, float]]:
    parent_outcomes, _ = run_module(parent_path, tasks, isolated=isolate_methods)
    with tempfile.TemporaryDirectory(prefix="method-evolution-child-") as directory:
        child_path = write_module(Path(directory) / MODULE_NAME, child)
        child_outcomes, _ = run_module(child_path, tasks, isolated=isolate_methods)
    parent_metrics = oracle_scaled_summary(parent_outcomes, tasks)
    child_metrics = oracle_scaled_summary(child_outcomes, tasks)
    payload = {
        **{f"parent_{name}": value for name, value in parent_metrics.items()},
        **{f"child_{name}": value for name, value in child_metrics.items()},
    }
    write_json(root / f"generation_{generation:03d}_validation.json", payload)
    accepted = _scaled_validation_accepts(payload)
    return accepted, payload


def _scaled_validation_accepts(metrics: Mapping[str, float]) -> bool:
    tolerance = 1e-12
    fields = ("mean_smae", "mean_srmse", "median_smae", "median_srmse")
    return all(
        math.isfinite(float(metrics[f"child_{field}"]))
        and float(metrics[f"child_{field}"])
        <= float(metrics[f"parent_{field}"]) + tolerance
        for field in fields
    ) and any(
        float(metrics[f"child_{field}"])
        < float(metrics[f"parent_{field}"]) - tolerance
        for field in fields
    )


def _head(repo: Path) -> str:
    return git(repo, "rev-parse", "--short", "HEAD") if _has_commit(repo) else ""


def _write_transcript(
    repo: Path,
    generation: int,
    system: str,
    user: str,
    response: str,
    *,
    stage: str | None = None,
) -> None:
    suffix = f"_{stage}" if stage else ""
    destination = repo / "transcripts" / f"generation_{generation:03d}{suffix}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"# system\n\n{system}\n\n# user\n\n{user}\n\n# response\n\n{response}\n",
        encoding="utf-8",
    )
