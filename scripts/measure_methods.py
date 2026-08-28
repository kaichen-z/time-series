"""Measure one methods.py module against a task partition and print/save a table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_tasks
from common.payload import read_json_object
from numerical_agent.evolution.execution import Task, reports_as_json, run_module


def load_partition(split_file: str, tasks_file: str, partition: str) -> tuple[Task, ...]:
    payload = read_json_object(split_file)
    wanted = set(payload["partitions"][partition]["task_ids"])
    catalog = {task.task_id: task for task in load_tasks(tasks_file)}
    return tuple(
        Task(task.task_id, tuple(task.history_values), task.prediction_length,
             task.frequency, tuple(task.future_values))
        for task_id, task in catalog.items()
        if task_id in wanted
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods_py")
    parser.add_argument("--split-file", default="splits/drcik_public_80_20_99_v1.json")
    parser.add_argument(
        "--tasks-file", default="/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK/data/tasks/train.jsonl"
    )
    parser.add_argument("--partition", default="train")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--torch-device",
        default=None,
        help="e.g. cuda:0; sets torch's default device so a method that never says .to(...) "
             "still runs on the GPU instead of silently defaulting to CPU",
    )
    args = parser.parse_args()

    if args.torch_device:
        import torch

        torch.set_default_device(args.torch_device)

    tasks = load_partition(args.split_file, args.tasks_file, args.partition)
    print(f"{args.partition} tasks: {len(tasks)}", flush=True)
    _, reports = run_module(args.methods_py, tasks)
    payload = reports_as_json(reports)
    payload.sort(key=lambda r: (r["mean_smae"] is None, r["mean_smae"]))

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    header = ["method", "sMAE", "sRMSE", "shape", "var", "coverage"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in payload:
        smae = r["mean_smae"]
        srmse = r["mean_srmse"]
        shape = r["mean_shape_correlation"]
        var = r["mean_variance_ratio"]
        cov = r["coverage"]
        print(
            f"| {r['method']} | {smae if smae is None else round(smae,3)} | "
            f"{srmse if srmse is None else round(srmse,3)} | "
            f"{shape if shape is None else round(shape,3)} | "
            f"{var if var is None else round(var,3)} | {round(cov,3)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
