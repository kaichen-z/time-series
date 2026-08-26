import json
import math
from pathlib import Path

from numerical_agent.evolution.numerical_selector import (
    CandidateDiagnostics,
    DecisionPolicy,
    HindcastFold,
)
from numerical_agent.evolution.selector_evolution import DecisionCase, evaluate_decision
from numerical_agent.run_task_conditioned_screening import load_frozen_partitions


train, dev = load_frozen_partitions(
    "splits/drcik_public_80_20_99_v1.json",
    "external/Dr-CiK/full-download/Dr-CiK_public/tasks",
    train_limit=80,
    dev_limit=20,
)
tasks = {task.task_id: task for task in train + dev}


def diagnostics(payload):
    folds = []
    for raw in payload["folds"]:
        row = dict(raw)
        row["forecast"] = tuple(row["forecast"])
        row["truth"] = tuple(row["truth"])
        folds.append(HindcastFold(**row))
    row = dict(payload)
    row["folds"] = tuple(folds)
    for field in (
        "median_mase", "recent_mase", "worst_mase", "mase_mad",
        "median_mae", "median_smape", "median_rmsse", "normalized_bias",
        "slope_error",
    ):
        if row[field] is None:
            row[field] = math.inf
    row["fold_forecasts"] = tuple(tuple(values) for values in row["fold_forecasts"])
    row["fold_truths"] = tuple(tuple(values) for values in row["fold_truths"])
    return CandidateDiagnostics(**row)


def load_cases(path):
    cases = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        cases[payload["task_id"]] = DecisionCase(
            tasks[payload["task_id"]],
            tuple(payload["active_names"]),
            {name: diagnostics(value) for name, value in payload["diagnostics"].items()},
            {name: tuple(value) for name, value in payload["final_forecasts"].items()},
            payload["families"],
            tuple(payload.get("conditioned_names", ())),
        )
    return cases


root = Path("runs/numerical_selector")
part1_train = load_cases(root / "smae_train_only_combined_v3_80_20_20260825/train_decision_cases.jsonl")
part1_dev = load_cases(root / "smae_train_only_combined_v3_80_20_20260825/dev_decision_cases.jsonl")
part2_train = load_cases(root / "part2_v11_compiled_specialists_80_20_20260825/train_decision_cases.jsonl")
part2_dev = load_cases(root / "part2_v11_compiled_specialists_80_20_20260825/dev_decision_cases.jsonl")


def rank(value):
    return (
        value.recent_mase,
        value.median_mase,
        value.worst_mase,
        value.mase_mad,
        abs(value.normalized_bias),
        value.name,
    )


def safety_core(k):
    names = set()
    for case in part1_train.values():
        eligible = [
            value for name, value in case.diagnostics.items()
            if name in case.forecasts
            and value.eligible
            and value.successful_folds >= 3
            and not value.explosion
            and value.worst_mase <= 10.0
        ]
        names.update(value.name for value in sorted(eligible, key=rank)[:k])
    return names


def extend(base, source, core):
    result = []
    for task_id, current in base.items():
        reference = source[task_id]
        names = tuple(dict.fromkeys((
            *current.active_names,
            *(name for name in reference.active_names if name in core),
        )))
        result.append(DecisionCase(
            current.task,
            names,
            {**current.diagnostics, **{
                name: reference.diagnostics[name]
                for name in names if name in reference.diagnostics
            }},
            {**current.forecasts, **{
                name: reference.forecasts[name]
                for name in names if name in reference.forecasts
            }},
            {**current.families, **{
                name: reference.families[name]
                for name in names if name in reference.families
            }},
            current.conditioned_names,
        ))
    return tuple(result)


policy = DecisionPolicy()
base_train = evaluate_decision(policy, tuple(part2_train.values()))
base_dev = evaluate_decision(policy, tuple(part2_dev.values()))
print("base", base_train.mean_smae, base_train.mean_srmse, base_dev.mean_smae, base_dev.mean_srmse)
rows = []
for k in range(1, 11):
    core = safety_core(k)
    score = evaluate_decision(policy, extend(part2_train, part1_train, core))
    dev_probe = evaluate_decision(policy, extend(part2_dev, part1_dev, core))
    rows.append((score.mean_smae, score.mean_srmse, k, len(core), core))
    print(
        "train_dev_probe", k, len(core), score.mean_smae, score.mean_srmse,
        dev_probe.mean_smae, dev_probe.mean_srmse, score.mean_mase, score.ensemble_rate,
    )
safe = [row for row in rows if row[1] <= base_train.mean_srmse + 1e-12]
best = min(safe or rows)
_, _, selected_k, core_size, selected_core = best
dev_score = evaluate_decision(policy, extend(part2_dev, part1_dev, selected_core))
print("chosen", selected_k, core_size, sorted(selected_core))
print(
    "dev", dev_score.mean_smae, dev_score.mean_srmse, dev_score.mean_mase,
    dev_score.mean_mae, dev_score.ensemble_rate, dev_score.method_diversity,
)
