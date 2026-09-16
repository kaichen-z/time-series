"""Closed, opt-in stopping policy and content-addressed trial accounting."""
from ..contracts import _require_exact_schema, require_sha256


def validate_policy(payload):
    row = _require_exact_schema(payload, (
        "minimum_effective_candidates", "generation_cap", "train_task_sha256s",
    ), field="effective trial policy")
    for name in ("minimum_effective_candidates", "generation_cap"):
        if type(row[name]) is not int or row[name] < 1:
            raise ValueError("effective trial targets and caps must be positive integers")
    if type(row["train_task_sha256s"]) is not dict or not row["train_task_sha256s"]:
        raise ValueError("effective trials require fixed Train identities")
    for name, sha in row["train_task_sha256s"].items():
        if type(name) is not str or not name:
            raise ValueError("effective trial task IDs must be text")
        require_sha256(sha, "effective Train task SHA")


def validate_evidence(rows):
    if type(rows) is not list:
        raise ValueError("effective trial evidence must be a list")
    for row in rows:
        _require_exact_schema(row, ("status", "forecast_sha256", "covered_task_ids",
            "child_sha256", "evaluation_sha256", "parent_registry_sha256"), field="effective trial evidence")
        if row["status"] not in {"effective", "duplicate", "unchanged", "infeasible",
                                 "missing_forecast", "invalid_forecast", "not_applicable"}:
            raise ValueError("unknown effective trial status")
        for name in ("child_sha256", "parent_registry_sha256"):
            require_sha256(row[name], name)
        for name in ("forecast_sha256", "evaluation_sha256"):
            if row[name] is not None:
                require_sha256(row[name], name)
        covered = row["covered_task_ids"]
        if (type(covered) is not list or any(type(name) is not str for name in covered)
                or covered != sorted(set(covered))):
            raise ValueError("effective coverage must be sorted unique task IDs")
        if row["status"] in {"effective", "duplicate", "unchanged"} and (
                not covered or row["forecast_sha256"] is None or row["evaluation_sha256"] is None):
            raise ValueError("effective behavior requires execution evidence")


def effective_summary(policy, generation, seen):
    count = len(seen)
    return {
        "attempted_generations": generation,
        "effective_count": count,
        "minimum_effective_candidates": policy["minimum_effective_candidates"],
        "generation_cap": policy["generation_cap"],
        "status": ("target_reached" if count >= policy["minimum_effective_candidates"] else
                   "attempt_cap_exhausted" if generation >= policy["generation_cap"] else "resource_limit"),
    }
