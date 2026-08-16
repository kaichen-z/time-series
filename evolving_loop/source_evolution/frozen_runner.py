"""Internal frozen runner used inside an accepted source-evolution worktree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evolving_loop.cli import _components, _factory, _select_manifest_split, _task_subset
from evolving_loop.co_evolution import HarnessPolicy
from evolving_loop.data import load_context_tasks, load_huggingface_context_tasks
from evolving_loop.frozen_inference import run_frozen_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    args = SimpleNamespace(**config["runtime"])
    if config["data_source"] == "hidden_test":
        tasks = load_huggingface_context_tasks(labels_public=False)
    elif config["data_source"] == "public_dev":
        tasks = load_huggingface_context_tasks(labels_public=True)
    else:
        tasks = load_context_tasks(config["tasks_file"], include_unlabeled=True)
    if config.get("manifest_path") and config.get("split_name", "all") != "all":
        tasks = _select_manifest_split(tasks, config["manifest_path"], config["split_name"])
    if config.get("task_ids"):
        requested = set(config["task_ids"])
        tasks = [task for task in tasks if task.numeric.task_id in requested]
        missing = requested - {task.numeric.task_id for task in tasks}
        if missing:
            raise ValueError("unknown task IDs: " + ", ".join(sorted(missing)))
    tasks = _task_subset(tasks, args.seed, args.limit)
    llm, library, retrieval_library, decision_library, tsfm = _components(args)
    policy = HarnessPolicy.load(config["policy_path"]) if config.get("policy_path") else HarnessPolicy()
    summary = run_frozen_inference(
        policy,
        tasks,
        _factory(
            args,
            llm,
            library,
            retrieval_library,
            decision_library,
            tsfm,
            isolate_library=True,
        ),
        output_dir=config["output_dir"],
        samples=config["samples"],
        score_public=config["score_public"],
        artifact_kind="source",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
