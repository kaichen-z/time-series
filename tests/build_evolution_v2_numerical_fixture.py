"""Build canonical, deterministic Train80/Dev20 inputs for Numerical V2 smoke."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import Task
from evolving_loop.package_numerical_supply import NumericalSupplyRelease
from evolving_loop.v2.cli import _NUMERICAL_SOURCE_SHA256
from evolving_loop.v2.contracts import canonical_v2_bytes
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
)
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest


def _anchor_release() -> ChampionRelease:
    assumption = EvolutionAssumption(
        assumption_id="toto_2_0_history",
        candidate_name="toto_2_0",
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="History length supports the reviewed seed anchor.",
        failure_condition="History length no longer supports the seed anchor.",
    )
    recipe = ChampionRecipe(
        name="select_toto_2_0",
        kind="select",
        parents=("toto_2_0",),
        fallback_parent="toto_2_0",
        assumptions=(assumption,),
    )
    return ChampionRelease(
        policy=FittedChampionPolicy(
            recipe=recipe,
            thresholds=((assumption.assumption_id, 0.0),),
        ),
        source_hashes=(("reviewed_seed_anchor", hashlib.sha256(b"toto-2.0-v1").hexdigest()),),
        metric_policy_fingerprint=hashlib.sha256(b"drcik-point-metrics-v1").hexdigest(),
        lineage=("toto_2_0",),
    )


def _seed_supply() -> NumericalSupplyRelease:
    return NumericalSupplyRelease(
        schema_version=1,
        version="n000",
        parent_sha256=None,
        anchor_release_payload=_anchor_release().to_payload(),
        alternatives=(),
        atlas_release_sha256=None,
        source_fingerprints={"dictionary": _NUMERICAL_SOURCE_SHA256},
        runtime_fingerprints={
            "materializer": hashlib.sha256(b"numerical-cli-materializer-v1").hexdigest()
        },
    )


def _task(index: int) -> tuple[Task, dict[str, object]]:
    if index < 40:
        history = tuple(float(index) for _ in range(20))
    else:
        # Preserve the variable-length strata exercised by the established
        # Numerical QD runner fixture.  The train folds therefore fit their
        # thresholds independently instead of all collapsing to one policy.
        history = tuple(float(index + offset) for offset in range(8 + index))
    future = (history[-1] + 1.0, history[-1] + 1.0)
    # These stable identities deliberately mirror the established runner
    # fixture: fold allocation is identity-bound, so changing only the names
    # could collapse the independently fit Build policies.
    task_id = f"build_case_{index:03d}" if index < 80 else f"evolution_{index:03d}"
    task = Task(
        task_id=task_id,
        history_values=history,
        future_values=future,
        prediction_length=2,
        frequency="D",
        seasonal_period="7",
        entity_name=f"entity-{index // 2:03d}",
    )
    payload = {
        "task_id": task_id,
        "entity_name": task.entity_name,
        "history_values": list(history),
        "future_values": list(future),
        "prediction_length": 2,
        "frequency": "D",
        "seasonal_period": "7",
        "target_name": "demand",
        "target_description": "deterministic daily demand",
        "history_timestamps": [f"h{offset:02d}" for offset in range(len(history))],
        "future_timestamps": ["f00", "f01"],
        "documents": [{"document_id": f"doc-{index:03d}", "content": "history context"}],
        "gt_evidence": [],
    }
    return task, payload


def build(output: Path) -> None:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("fixture output must be an empty directory")
    else:
        output.mkdir(parents=True)
    values = tuple(_task(index) for index in range(100))
    train_tasks = tuple(task for task, _payload in values[:80])
    manifest = {
        "schema_version": 1,
        "fold_manifest": build_group_fold_manifest(
            train_tasks, seed=20260903
        ).to_payload(),
        "train": [payload for _task_value, payload in values[:80]],
        "dev": [payload for _task_value, payload in values[80:]],
    }
    (output / "seed_supply.json").write_bytes(
        canonical_v2_bytes(_seed_supply().to_payload())
    )
    (output / "task_manifest.json").write_bytes(canonical_v2_bytes(manifest))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        build(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Numerical V2 fixture: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
