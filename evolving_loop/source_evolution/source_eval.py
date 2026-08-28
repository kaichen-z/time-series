"""Internal evaluator executed inside one source-evolution worktree."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from evolving_loop.cli import _components, _factory
from evolving_loop.co_evolution import HarnessPolicy, evaluate_policy
from evolving_loop.data import load_context_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    tasks = load_context_tasks(config["tasks_file"])
    by_id = {task.numeric.task_id: task for task in tasks}
    train = [by_id[value] for value in config["train_ids"]]
    dev = [by_id[value] for value in config["dev_ids"]]
    with tempfile.TemporaryDirectory(prefix="source-eval-skills-") as directory:
        runtime = dict(config["runtime"])
        runtime.update(
            {
                "library_path": str(Path(directory) / "coding.json"),
                "retrieval_library_path": str(Path(directory) / "retrieval.json"),
                "decision_library_path": str(Path(directory) / "decision.json"),
            }
        )
        args = SimpleNamespace(**runtime)
        llm, library, retrieval_library, decision_library, tsfm = _components(args)
        policy = HarnessPolicy(
            coding_initial_programs=args.coding_initial_programs,
            coding_mutations=args.coding_mutations,
            coding_validation_folds=args.coding_validation_folds,
        )
        factory = _factory(
            args,
            llm,
            library,
            retrieval_library,
            decision_library,
            tsfm,
            isolate_library=True,
        )
        harness = factory(policy)
        # Source candidates cannot use outcome-driven skill writes during acceptance;
        # this keeps all resolved-label handling in the immutable evaluator.
        train_result = evaluate_policy(
            policy, train, factory, learn_skills=False, harness=harness
        )
        val_result = evaluate_policy(
            policy, dev, factory, learn_skills=False, harness=harness
        )
    print(
        json.dumps(
            {
                "train": asdict(train_result),
                "val": asdict(val_result),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
