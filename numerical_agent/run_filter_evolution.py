"""Run one cache-only generation of single-agent dictionary filtering."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from common.llm import CodexCLIClient, CodexCLIConfig
from common.payload import write_json

from .evolution import git
from .evolution.cache import OutcomeCache, SCALED_METRIC_CAP, SCALED_METRIC_SCHEMA
from .evolution.filtering import (
    build_filter_dictionary,
    evolve_filter_once,
    parse_filter_source,
    render_filter_source,
    require_cached_portfolio_outcomes,
)
from .evolution.module import read_module
from .evolution.portfolio import PolicyOutcomeCache, read_policy_file
from .run_evolution import _evolution_tasks


SCALED_METRIC_POLICY = {
    "scaled_metric_schema": SCALED_METRIC_SCHEMA,
    "scaled_metric_cap": SCALED_METRIC_CAP,
    "objective": "pareto_minimize_smae_srmse",
    "aggregation": "mean_capped_task_metrics",
    "ordering": "joint_scaled_error_smae_srmse_name",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--outcome-cache-dir", required=True)
    parser.add_argument("--policy-outcome-cache-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--validation-tail", type=int, default=2)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--codex-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--codex-cache-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    repo = Path(args.repo).resolve()
    module_path = repo / "methods.py"
    policy_path = repo / "policies.py"
    dictionary_path = repo / "dictionary.py"
    if not (repo / ".git").is_dir():
        raise ValueError(f"{repo} must be a Git repository")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("filter evolution repository has tracked modifications")
    module = read_module(module_path)
    portfolio = read_policy_file(policy_path)
    parent = (
        parse_filter_source(dictionary_path.read_text(encoding="utf-8"))
        if dictionary_path.is_file()
        else build_filter_dictionary(module, portfolio)
    )
    if not dictionary_path.is_file():
        dictionary_path.write_text(render_filter_source(parent), encoding="utf-8")
        git(repo, "add", "dictionary.py")
        git(repo, "commit", "--quiet", "-m", "seed unified 103-candidate filter dictionary")

    train, dev = _evolution_tasks(
        args.split_file,
        args.tasks_file,
        train_limit=args.train_limit,
        validation_tail=args.validation_tail,
    )
    if len(train) != args.train_limit or len(dev) != args.validation_tail:
        raise ValueError(
            f"requested {args.train_limit}+{args.validation_tail} tasks, loaded {len(train)}+{len(dev)}"
        )
    method_cache = OutcomeCache(
        args.outcome_cache_dir,
        skills_path=repo / "skills.py" if (repo / "skills.py").is_file() else None,
    )
    policy_cache = PolicyOutcomeCache(args.policy_outcome_cache_dir)
    outcomes = require_cached_portfolio_outcomes(
        module,
        portfolio,
        train + dev,
        outcome_cache=method_cache,
        policy_cache=policy_cache,
        isolated_methods=True,
    )
    client = CodexCLIClient(
        CodexCLIConfig(
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning_effort,
            timeout_seconds=args.codex_timeout,
            cache_dir=args.codex_cache_dir or repo / "filter-agent-cache",
        )
    )
    source_hashes = {
        "methods.py": _sha256(module_path),
        "policies.py": _sha256(policy_path),
    }
    result = evolve_filter_once(
        parent,
        train,
        dev,
        outcomes,
        client,
        generation=args.generation,
        transcript_dir=repo / "filter-transcripts",
    )
    child_path = repo / f"generation_{args.generation:03d}_child_dictionary.py"
    child_path.write_text(render_filter_source(result.child), encoding="utf-8")
    commit = git(repo, "rev-parse", "--short", "HEAD")
    if result.accepted:
        dictionary_path.write_text(render_filter_source(result.child), encoding="utf-8")
        git(repo, "add", "dictionary.py")
        git(repo, "commit", "--quiet", "-m", f"generation {args.generation}: filter dictionary")
        commit = git(repo, "rev-parse", "--short", "HEAD")
    if source_hashes != {"methods.py": _sha256(module_path), "policies.py": _sha256(policy_path)}:
        raise RuntimeError("filter generation modified executable forecasting sources")
    elapsed = time.monotonic() - started
    changes = [
        {
            "name": before.name,
            "from_status": before.status,
            "to_status": after.status,
            "applicability": list(after.applicability),
            "reason": after.reason,
        }
        for before, after in zip(result.parent.entries, result.child.entries, strict=True)
        if before != after
    ]
    payload = {
        "schema_version": 2,
        "metric_policy": SCALED_METRIC_POLICY,
        "diagnostic_only_metrics": ["mase", "mae", "smape"],
        "generation": args.generation,
        "accepted": result.accepted,
        "reason": result.reason,
        "commit": commit,
        "elapsed_seconds": elapsed,
        "agent": {
            "model": args.codex_model,
            "reasoning_effort": args.codex_reasoning_effort,
            "calls": result.agent_calls,
        },
        "tasks": {
            "train": [task.task_id for task in train],
            "dev": [task.task_id for task in dev],
        },
        "cache": {
            "method_hits": method_cache.stats.hits,
            "method_misses": method_cache.stats.misses,
            "policy_hits": policy_cache.stats.hits,
            "policy_misses": policy_cache.stats.misses,
        },
        "parent": {
            "train": asdict(result.train_parent),
            "dev": asdict(result.dev_parent),
            "status_counts": _status_counts(result.parent),
        },
        "child": {
            "train": asdict(result.train_child),
            "dev": asdict(result.dev_child),
            "status_counts": _status_counts(result.child),
        },
        "changes": changes,
        "source_hashes": source_hashes,
    }
    payload["manifest_sha256"] = _manifest_fingerprint(payload)
    report_path = repo / f"generation_{args.generation:03d}_filter_result.json"
    write_json(report_path, payload)
    (repo / f"generation_{args.generation:03d}_filter_report.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _status_counts(dictionary) -> dict[str, int]:
    return {
        status: sum(entry.status == status for entry in dictionary.entries)
        for status in ("keep", "specialized", "repair", "quarantine", "discard")
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _markdown(payload: dict[str, object]) -> str:
    parent = payload["parent"]
    child = payload["child"]
    assert isinstance(parent, dict) and isinstance(child, dict)
    lines = [
        "# Single-Agent Dictionary Filter Smoke Test",
        "",
        f"- Accepted: `{payload['accepted']}`",
        f"- Reason: {payload['reason']}",
        f"- Elapsed: {float(payload['elapsed_seconds']):.2f} seconds",
        f"- Changes: {len(payload['changes'])}",
        "",
        "| Split | Parent mean sMAE | Child mean sMAE | Parent mean sRMSE | Child mean sRMSE | Parent coverage | Child coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev"):
        left = parent[split]
        right = child[split]
        assert isinstance(left, dict) and isinstance(right, dict)
        lines.append(
            f"| {split} | {left['mean_smae']:.6f} | {right['mean_smae']:.6f} | "
            f"{left['mean_srmse']:.6f} | {right['mean_srmse']:.6f} | "
            f"{left['coverage']:.4f} | {right['coverage']:.4f} |"
        )
    lines.extend((
        "",
        "| Split | Side | Crash / invalid / missing / malformed |",
        "|---|---|---:|",
    ))
    for split in ("train", "dev"):
        for side, scores in (("Parent", parent[split]), ("Child", child[split])):
            assert isinstance(scores, dict)
            lines.append(
                f"| {split} | {side} | {scores['eligible_crashed']} / "
                f"{scores['eligible_invalid']} / {scores['eligible_missing']} / "
                f"{scores['eligible_malformed_success']} |"
            )
    lines.extend(("", "## Proposed changes", ""))
    for change in payload["changes"]:
        assert isinstance(change, dict)
        lines.append(
            f"- `{change['name']}`: `{change['from_status']}` → `{change['to_status']}`; "
            f"{change['reason']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
