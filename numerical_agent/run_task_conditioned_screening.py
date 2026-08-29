"""Evolve and freeze task-conditioned screening on Train/Dev only."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from common.data import load_tasks_by_id
from common.llm import CodexCLIClient, CodexCLIConfig
from common.payload import read_json_object, write_json

from .evolution.cache import OutcomeCache
from .evolution.execution import Outcome, Task, require_unique_outcome_keys, require_unique_task_ids
from .evolution.filtering import build_filter_dictionary, parse_filter_source
from .evolution.module import read_module
from .evolution.portfolio import (
    PolicyOutcomeCache,
    _run_combined,
    read_policy_file,
    require_flagship_runtimes,
)
from .evolution.screening import (
    ScreeningConstraints,
    ScreeningPolicy,
    compare_screening,
    evaluate_screening,
    materialize_active_dictionary,
    profile_task,
)
from .evolution.screening_evolution import (
    complete_target_batches,
    evolve_screening_on_train_once,
    migrate_filter_dictionary,
    render_screening_source,
    select_refinement_targets,
)
from .main import _add_tsfm_runtime_options, _runtime_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="103-candidate method repository")
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--outcome-cache-dir", required=True)
    parser.add_argument("--policy-outcome-cache-dir", required=True)
    parser.add_argument("--target-batches-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=80)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument(
        "--seed-policy", choices=("all", "legacy"), default="all",
        help="start from the complete selectable Master Dictionary or the legacy global filter",
    )
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--codex-reasoning-effort", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--codex-cache-dir", default=None)
    parser.add_argument("--baseline-method", default="toto_2_0")
    parser.add_argument("--screen-min-candidates", type=int, default=12)
    parser.add_argument(
        "--screen-max-candidates",
        type=int,
        default=None,
        help="optional safety ceiling; defaults to the runtime candidate count",
    )
    parser.add_argument("--screen-min-unique-dictionaries", type=int, default=3)
    parser.add_argument("--screen-max-mean-jaccard", type=float, default=0.995)
    parser.add_argument("--screen-min-group-support", type=int, default=4)
    parser.add_argument("--screen-min-dev-oracle-retention", type=float, default=0.9)
    parser.add_argument("--screen-batch-size", type=int, default=8)
    parser.add_argument("--screen-refinement-generations", type=int, default=3)
    parser.add_argument("--screen-refinement-batch-size", type=int, default=24)
    _add_tsfm_runtime_options(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.screen_refinement_generations < 0:
        raise ValueError("screen refinement generations must not be negative")
    if not 1 <= args.screen_refinement_batch_size <= 24:
        raise ValueError("screen refinement batch size must be between 1 and 24")
    started = time.monotonic()
    repo = Path(args.repo).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, dev = load_frozen_partitions(
        args.split_file, args.tasks_file,
        train_limit=args.train_limit, dev_limit=args.dev_limit,
    )
    if not train or not dev:
        raise ValueError("screening evolution requires nonempty Train and Dev partitions")

    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    candidate_names = tuple(module.names()) + portfolio.names
    legacy_path = repo / "frozen_dictionary.py"
    if not legacy_path.is_file():
        legacy_path = repo / "dictionary.py"
    legacy_dictionary = parse_filter_source(legacy_path.read_text(encoding="utf-8"))
    seed_dictionary = (
        build_filter_dictionary(module, portfolio)
        if args.seed_policy == "all"
        else legacy_dictionary
    )
    parent = migrate_filter_dictionary(
        seed_dictionary,
        fallback_names=_fallback_names(module.names(), tuple(policy.name for policy in portfolio.tsfm)),
    )
    candidate_count = _validate_screening_parent(parent, candidate_names)
    max_candidates = (
        candidate_count
        if args.screen_max_candidates is None
        else args.screen_max_candidates
    )
    if not 1 <= max_candidates <= candidate_count:
        raise ValueError(
            "screen max candidates must be between 1 and "
            f"{candidate_count}, found {max_candidates}"
        )
    constraints = ScreeningConstraints(
        baseline_method=args.baseline_method,
        min_active_candidates=args.screen_min_candidates,
        max_active_candidates=max_candidates,
        min_unique_active_dictionaries=args.screen_min_unique_dictionaries,
        max_mean_pairwise_jaccard=args.screen_max_mean_jaccard,
        min_group_support=args.screen_min_group_support,
        min_dev_oracle_retention=args.screen_min_dev_oracle_retention,
    )
    if parent.get(constraints.baseline_method) is None:
        raise ValueError(f"unknown screening baseline {constraints.baseline_method!r}")

    train_outcomes, train_cache_summary = _training_outcomes(
        args, repo, module, portfolio, train
    )
    original_parent = parent
    batches_payload = read_json_object(args.target_batches_file)
    raw_batches = batches_payload.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("target batches file needs a nonempty batches list")
    batches = complete_target_batches(
        parent,
        tuple(tuple(str(name) for name in batch) for batch in raw_batches),
        batch_size=args.screen_batch_size,
    )
    if any(not 1 <= len(batch) <= 24 for batch in batches):
        raise ValueError("each screening target batch must contain 1 to 24 names")

    agent = CodexCLIClient(CodexCLIConfig(
        model=args.codex_model,
        reasoning_effort=args.codex_reasoning_effort,
        timeout_seconds=args.codex_timeout,
        cache_dir=args.codex_cache_dir or output / "agent-cache",
    ))
    generations = []
    for number, batch in enumerate(batches, 1):
        result = evolve_screening_on_train_once(
            parent, train, train_outcomes, agent,
            generation=number,
            required_targets=batch,
            transcript_dir=output / "transcripts",
            constraints=constraints,
            enforce_final_constraints=False,
        )
        child_source = render_screening_source(result.child)
        (output / f"generation_{number:03d}_child_screening_policy.py").write_text(
            child_source, encoding="utf-8"
        )
        generation_payload = {
            "generation": number,
            "phase": "review",
            "evaluation_scope": "train_only",
            "required_targets": list(batch),
            "accepted": result.accepted,
            "gate": asdict(result.gate),
            "parent_hash": result.parent.fingerprint(),
            "child_hash": result.child.fingerprint(),
            "train_parent": asdict(result.train_parent),
            "train_child": asdict(result.train_child),
            "oracle_shields": [asdict(shield) for shield in result.oracle_shields],
            "action_decisions": [
                asdict(decision) for decision in result.action_decisions
            ],
            "agent_calls": result.agent_calls,
        }
        write_json(output / f"generation_{number:03d}_screening_result.json", generation_payload)
        generations.append(generation_payload)
        if result.accepted:
            parent = result.child

    attempted_refinements: set[str] = set()
    for refinement in range(1, args.screen_refinement_generations + 1):
        train_score = evaluate_screening(parent, train, train_outcomes)
        if _train_constraints_met(train_score, constraints):
            break
        needed_families = tuple(
            family
            for family in constraints.required_conditioned_families
            if train_score.conditioned_entries_by_family.get(family, 0) < 1
        )
        batch = select_refinement_targets(
            parent,
            train,
            train_outcomes,
            constraints=constraints,
            excluded_names=frozenset(attempted_refinements),
            required_families=needed_families,
            limit=args.screen_refinement_batch_size,
        )
        if not batch:
            break
        attempted_refinements.update(batch)
        missing_families = tuple(
            family
            for family in needed_families
            if any(parent.get(name).family == family for name in batch)  # type: ignore[union-attr]
        )
        number = len(generations) + 1
        result = evolve_screening_on_train_once(
            parent,
            train,
            train_outcomes,
            agent,
            generation=number,
            required_targets=batch,
            transcript_dir=output / "transcripts",
            constraints=constraints,
            enforce_final_constraints=False,
            required_conditioning_families=missing_families,
        )
        child_source = render_screening_source(result.child)
        (output / f"generation_{number:03d}_child_screening_policy.py").write_text(
            child_source, encoding="utf-8"
        )
        generation_payload = {
            "generation": number,
            "phase": "refinement",
            "evaluation_scope": "train_only",
            "refinement": refinement,
            "required_conditioning_families": list(missing_families),
            "required_targets": list(batch),
            "accepted": result.accepted,
            "gate": asdict(result.gate),
            "parent_hash": result.parent.fingerprint(),
            "child_hash": result.child.fingerprint(),
            "train_parent": asdict(result.train_parent),
            "train_child": asdict(result.train_child),
            "oracle_shields": [asdict(shield) for shield in result.oracle_shields],
            "action_decisions": [
                asdict(decision) for decision in result.action_decisions
            ],
            "agent_calls": result.agent_calls,
        }
        write_json(
            output / f"generation_{number:03d}_screening_result.json",
            generation_payload,
        )
        generations.append(generation_payload)
        if result.accepted:
            parent = result.child

    # Freeze the Train winner before loading any Dev outcome.  Dev is a single,
    # read-only final gate and never influences a later mutation or refinement.
    train_score = evaluate_screening(parent, train, train_outcomes)
    dev_outcomes, dev_cache_summary = _training_outcomes(
        args, repo, module, portfolio, dev
    )
    dev_score = evaluate_screening(parent, dev, dev_outcomes)
    original_train_score = evaluate_screening(
        original_parent, train, train_outcomes
    )
    original_dev_score = evaluate_screening(original_parent, dev, dev_outcomes)
    final_gate = compare_screening(
        original_train_score,
        train_score,
        original_dev_score,
        dev_score,
        constraints=constraints,
        enforce_final_constraints=True,
    )
    final_constraints_met = final_gate.accepted and _constraints_met(
        train_score, dev_score, constraints
    )
    policy_artifacts = _write_policy_artifacts(
        output, parent, accepted=final_constraints_met
    )
    _write_active(output / "train_active_dictionaries.jsonl", parent, train)
    _write_active(output / "dev_active_dictionaries.jsonl", parent, dev)
    manifest = {
        "schema_version": 1,
        "phase": "task_conditioned_screening",
        "train_tasks": len(train),
        "dev_tasks": len(dev),
        "candidate_count": len(parent.entries),
        "reviewed_candidate_count": sum(len(batch) for batch in batches),
        "mutation_batch_size": args.screen_batch_size,
        "refinement_generation_limit": args.screen_refinement_generations,
        "refinement_batch_size": args.screen_refinement_batch_size,
        "constraints": asdict(constraints),
        "final_constraints_met": final_constraints_met,
        "seed_policy": args.seed_policy,
        **policy_artifacts,
        "source_hashes": {
            "methods.py": _sha256(repo / "methods.py"),
            "policies.py": _sha256(repo / "policies.py"),
            "legacy_dictionary.py": _sha256(legacy_path),
        },
        "cache": _merge_cache_summaries(train_cache_summary, dev_cache_summary),
        "train": asdict(train_score),
        "dev": asdict(dev_score),
        "final_dev_gate": asdict(final_gate),
        "dev_evaluations": 1,
        "generations": generations,
        "accepted_train_generations": [
            row["generation"] for row in generations if row["accepted"]
        ],
        "accepted_generations": (
            [row["generation"] for row in generations if row["accepted"]]
            if final_constraints_met
            else []
        ),
        "elapsed_seconds": time.monotonic() - started,
        "public_test_accessed": False,
    }
    write_json(output / "screening_manifest.json", manifest)
    (output / "SCREENING_REPORT.md").write_text(_report(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if final_constraints_met else 2


def load_frozen_partitions(
    split_file: str | Path,
    tasks_file: str | Path,
    *,
    train_limit: int,
    dev_limit: int,
) -> tuple[tuple[Task, ...], tuple[Task, ...]]:
    if train_limit < 1 or dev_limit < 0:
        raise ValueError("Train limit must be positive and Dev limit must be nonnegative")
    split = read_json_object(split_file)
    raw = split["partitions"]  # type: ignore[index]
    train_ids = tuple(str(value) for value in raw["train"]["task_ids"][:train_limit])
    dev_ids = tuple(str(value) for value in raw["dev"]["task_ids"][:dev_limit])
    catalog = {
        task.task_id: task
        for task in load_tasks_by_id(tasks_file, [*train_ids, *dev_ids])
    }

    def partition(name: str, limit: int) -> tuple[Task, ...]:
        ids = raw[name]["task_ids"][:limit]  # type: ignore[index]
        rows = []
        for task_id in ids:
            source = catalog.get(str(task_id))
            if source is None:
                raise ValueError(f"missing {name} task {task_id}")
            rows.append(Task(
                source.task_id, tuple(source.history_values), source.prediction_length,
                source.frequency, tuple(source.future_values),
            ))
        if len(rows) != limit:
            raise ValueError(f"requested {limit} {name} tasks, loaded {len(rows)}")
        return tuple(rows)

    return (
        partition("train", train_limit),
        partition("dev", dev_limit) if dev_limit else (),
    )


def _training_outcomes(args, repo, module, portfolio, tasks) -> tuple[tuple[Outcome, ...], dict]:
    require_unique_task_ids(tasks)
    portfolio.validate_namespace(module.names())
    method_cache = OutcomeCache(
        args.outcome_cache_dir,
        skills_path=repo / "skills.py" if (repo / "skills.py").is_file() else None,
    )
    policy_cache = PolicyOutcomeCache(args.policy_outcome_cache_dir)
    rows: list[Outcome] = []
    for method in module.methods:
        rows.extend(method_cache.evaluate_method(
            method, tasks, isolated=True, require_forecasts=True
        ))
    runtimes = _runtime_registry(args)
    try:
        require_flagship_runtimes(portfolio, runtimes)
        for policy in portfolio.tsfm:
            rows.extend(policy_cache.evaluate(policy, task, runtimes) for task in tasks)
    finally:
        runtimes.close()
    require_unique_outcome_keys(rows)
    by_key = {(row.method, row.task_id): row for row in rows}
    for policy in portfolio.combined:
        rows.extend(_run_combined(policy, task, by_key) for task in tasks)
    return tuple(rows), {
        "statistical_hits": method_cache.stats.hits,
        "statistical_misses": method_cache.stats.misses,
        "tsfm_hits": policy_cache.stats.hits,
        "tsfm_misses": policy_cache.stats.misses,
    }


def _validate_screening_parent(
    parent: ScreeningPolicy, candidate_names: Sequence[str]
) -> int:
    """Require the parent to index the complete parsed runtime namespace."""
    expected = set(candidate_names)
    names = tuple(entry.name for entry in parent.entries)
    actual = set(names)
    duplicates = tuple(sorted(name for name in actual if names.count(name) > 1))
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    if duplicates or missing or extra:
        details = []
        if duplicates:
            details.append("duplicate=" + ", ".join(duplicates))
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("extra=" + ", ".join(extra))
        raise ValueError("screening parent namespace mismatch: " + "; ".join(details))
    return len(candidate_names)


def _fallback_names(statistical: Sequence[str], tsfm: Sequence[str]) -> tuple[str, ...]:
    statistical_fallback = next(
        (name for name in ("naive_last", "seasonal_naive", "ses") if name in statistical),
        statistical[0],
    )
    tsfm_fallbacks = tuple(name for name in ("timesfm_2_5", "toto_2_0") if name in tsfm)
    names = (statistical_fallback, *tsfm_fallbacks)
    return names if len(names) >= 3 else (statistical_fallback, *tuple(tsfm)[:2])


def _write_active(path: Path, policy: ScreeningPolicy, tasks: Sequence[Task]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            active = materialize_active_dictionary(policy, profile_task(task))
            handle.write(json.dumps(asdict(active), sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_policy_artifacts(
    output: Path,
    policy: ScreeningPolicy,
    *,
    accepted: bool,
) -> dict[str, str | None]:
    """Always preserve the candidate, but publish a frozen policy only after its gate."""
    source = render_screening_source(policy)
    candidate = output / "candidate_screening_policy.py"
    candidate.write_text(source, encoding="utf-8")
    frozen = output / "frozen_screening_policy.py"
    if accepted:
        frozen.write_text(source, encoding="utf-8")
    elif frozen.exists():
        frozen.unlink()
    return {
        "candidate_screening_policy_sha256": _sha256(candidate),
        "candidate_screening_policy_fingerprint": policy.fingerprint(),
        "frozen_screening_policy_sha256": _sha256(frozen) if accepted else None,
        "frozen_screening_policy_fingerprint": policy.fingerprint() if accepted else None,
    }


def _merge_cache_summaries(*summaries: dict) -> dict:
    """Combine disjoint Train/Dev cache accounting without hiding either pass."""
    keys = {key for summary in summaries for key in summary}
    return {
        key: sum(int(summary.get(key, 0)) for summary in summaries)
        for key in sorted(keys)
    }


def _train_constraints_met(train_score, constraints: ScreeningConstraints) -> bool:
    """Check refinement goals using Train only; Dev is reserved for the final gate."""
    return all(
        (
            train_score.global_oracle_retention >= 1.0 - 1e-12,
            train_score.min_active_candidates >= constraints.min_active_candidates,
            train_score.max_active_candidates <= constraints.max_active_candidates,
            train_score.unique_active_dictionaries
            >= min(constraints.min_unique_active_dictionaries, train_score.task_count),
            train_score.mean_pairwise_jaccard
            <= constraints.max_mean_pairwise_jaccard + 1e-12,
            all(
                train_score.conditioned_entries_by_family.get(family, 0) >= 1
                for family in constraints.required_conditioned_families
            ),
        )
    )


def _constraints_met(
    train_score, dev_score, constraints: ScreeningConstraints
) -> bool:
    return all(
        (
            train_score.global_oracle_retention >= 1.0 - 1e-12,
            dev_score.global_oracle_retention
            >= constraints.min_dev_oracle_retention - 1e-12,
            train_score.min_active_candidates >= constraints.min_active_candidates,
            dev_score.min_active_candidates >= constraints.min_active_candidates,
            train_score.max_active_candidates <= constraints.max_active_candidates,
            dev_score.max_active_candidates <= constraints.max_active_candidates,
            train_score.unique_active_dictionaries
            >= min(constraints.min_unique_active_dictionaries, train_score.task_count),
            dev_score.unique_active_dictionaries
            >= min(constraints.min_unique_active_dictionaries, dev_score.task_count),
            dev_score.mean_pairwise_jaccard
            <= constraints.max_mean_pairwise_jaccard + 1e-12,
            all(
                dev_score.conditioned_entries_by_family.get(family, 0) >= 1
                for family in constraints.required_conditioned_families
            ),
        )
    )


def _report(manifest: dict) -> str:
    train = manifest["train"]
    dev = manifest["dev"]
    title = (
        "# Frozen Task-Conditioned Screening Report"
        if manifest["final_constraints_met"]
        else "# Rejected Task-Conditioned Screening Candidate Report"
    )
    train_families = train["conditioned_entries_by_family"]
    dev_families = dev["conditioned_entries_by_family"]
    return "\n".join((
        title,
        "",
        f"- Candidates: {manifest['candidate_count']}",
        f"- Train / Dev: {manifest['train_tasks']} / {manifest['dev_tasks']}",
        f"- Accepted generations: {manifest['accepted_generations']}",
        f"- Candidate SHA-256: `{manifest['candidate_screening_policy_sha256']}`",
        f"- Frozen SHA-256: `{manifest['frozen_screening_policy_sha256']}`",
        f"- Public Test accessed: `{manifest['public_test_accessed']}`",
        f"- Final constraints met: `{manifest['final_constraints_met']}`",
        f"- Dev evaluations: {manifest.get('dev_evaluations', 'not recorded')}",
        f"- Final Dev gate: {manifest.get('final_dev_gate', {}).get('reason', 'not recorded')}",
        "",
        "| Split | Coverage | Active success | Failure exposure | N/A exposure | Oracle retention | Mean regret | Compression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Train | {train['coverage']:.4f} | {train['active_success_rate']:.4f} | {train['failure_exposure']:.4f} | {train['not_applicable_exposure']:.4f} | {train['global_oracle_retention']:.4f} | {train['mean_active_oracle_regret']:.4f} | {train['compression']:.4f} |",
        f"| Dev | {dev['coverage']:.4f} | {dev['active_success_rate']:.4f} | {dev['failure_exposure']:.4f} | {dev['not_applicable_exposure']:.4f} | {dev['global_oracle_retention']:.4f} | {dev['mean_active_oracle_regret']:.4f} | {dev['compression']:.4f} |",
        "",
        "| Split | Mean active | Min / Max | Unique dictionaries | Pairwise Jaccard | Conditioned Statistical / TSFM / Combined |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Train | {train['mean_active_candidates']:.2f} | {train['min_active_candidates']} / {train['max_active_candidates']} | {train['unique_active_dictionaries']} | {train['mean_pairwise_jaccard']:.4f} | {train_families['statistical']} / {train_families['tsfm']} / {train_families['combined']} |",
        f"| Dev | {dev['mean_active_candidates']:.2f} | {dev['min_active_candidates']} / {dev['max_active_candidates']} | {dev['unique_active_dictionaries']} | {dev['mean_pairwise_jaccard']:.4f} | {dev_families['statistical']} / {dev_families['tsfm']} / {dev_families['combined']} |",
        "",
    ))


if __name__ == "__main__":
    raise SystemExit(main())
