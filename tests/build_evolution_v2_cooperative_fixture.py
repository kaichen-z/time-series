"""Build the canonical deterministic 4 Train / 1 Dev cooperative inputs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolving_loop.package_numerical_supply import (
    NumericalAlternativeSpec,
    NumericalSupplyRelease,
)
from evolving_loop.retrieval_agent.policy import RetrievalGenome
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.cooperative.contracts import DecisionModuleV2, RetrievalModuleV2
from numerical_agent.evolution.champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
)


DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "evolution_v2_cooperative"


def _policy(candidate_id: str, threshold: float = 0.0) -> FittedChampionPolicy:
    assumption = EvolutionAssumption(
        assumption_id=f"{candidate_id}_history",
        candidate_name=candidate_id,
        feature="history_length",
        direction="above",
        horizon_region="full",
        operator="select",
        rationale="The deterministic smoke history supports this candidate.",
        failure_condition="The smoke history is no longer long enough.",
    )
    return FittedChampionPolicy(
        recipe=ChampionRecipe(
            name=f"select_{candidate_id}",
            kind="select",
            parents=(candidate_id,),
            fallback_parent=candidate_id,
            assumptions=(assumption,),
        ),
        thresholds=((assumption.assumption_id, threshold),),
    )


def _seed_supply() -> NumericalSupplyRelease:
    anchor_policy = _policy("safe_anchor")
    anchor = ChampionRelease(
        policy=anchor_policy,
        source_hashes=(
            ("dictionary", hashlib.sha256(b"cooperative-safe-anchor").hexdigest()),
        ),
        metric_policy_fingerprint=hashlib.sha256(
            b"cooperative-smoke-metrics-v1"
        ).hexdigest(),
        lineage=("safe_anchor",),
    )
    alternative_policy = _policy("seasonal_naive")
    alternative = NumericalAlternativeSpec(
        candidate_id="seasonal_naive",
        family="statistical",
        materializer_kind="dictionary",
        recipe_payload=alternative_policy.recipe.to_payload(),
        full_build_policy_payload=alternative_policy.to_payload(),
        build_fold_policy_payloads=tuple(
            (fold, _policy("seasonal_naive", float(fold)).to_payload())
            for fold in range(5)
        ),
        assumption_ids=("seasonal_naive_history",),
        failure_conditions=("The smoke history is no longer long enough.",),
    )
    return NumericalSupplyRelease(
        schema_version=1,
        version="n001",
        parent_sha256=hashlib.sha256(b"cooperative-supply-parent").hexdigest(),
        anchor_release_payload=anchor.to_payload(),
        alternatives=(alternative,),
        atlas_release_sha256=None,
        source_fingerprints={
            "dictionary": hashlib.sha256(b"cooperative-dictionary-v1").hexdigest()
        },
        runtime_fingerprints={
            "materializer": hashlib.sha256(
                b"cooperative-smoke-materializer-v1"
            ).hexdigest()
        },
    )


def _task(index: int) -> dict[str, object]:
    history = tuple(float(index + offset + 1) for offset in range(8))
    future = (history[-1] + 1.0, history[-1] + 1.0)
    return {
        "task_id": f"cooperative-{index:02d}",
        "entity_name": f"Cooperative Entity {index:02d}",
        "history_values": list(history),
        "future_values": list(future),
        "prediction_length": 2,
        "frequency": "D",
        "seasonal_period": "7",
        "target_name": "demand",
        "target_description": "deterministic daily demand",
        "history_timestamps": [f"h{offset:02d}" for offset in range(len(history))],
        "future_timestamps": ["f00", "f01"],
        "documents": [
            {
                "document_id": f"context-{index:02d}",
                "content": "Deterministic context with no future labels.",
            }
        ],
        "gt_evidence": [],
    }


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    expected = {
        "seed_supply.json",
        "tasks_4_1.json",
        "retrieval_release.json",
        "decision_policy.json",
    }
    if not output.is_dir() or any(
        path.name not in expected for path in output.iterdir()
    ):
        raise ValueError("fixture output may contain only cooperative fixture files")
    tasks = [_task(index) for index in range(5)]
    retrieval = RetrievalModuleV2(
        1,
        hashlib.sha256(b"cooperative-retrieval-seed-v1").hexdigest(),
        RetrievalGenome.seed().to_payload(),
        (),
    )
    decision = DecisionModuleV2(
        1,
        "Select the safest valid numerical candidate using only retrieved context.",
        (),
        True,
        2,
        "last",
    )
    payloads = {
        "seed_supply.json": _seed_supply().to_payload(),
        "tasks_4_1.json": {
            "schema_version": 1,
            "train": tasks[:4],
            "dev": tasks[4:],
        },
        "retrieval_release.json": retrieval.to_payload(),
        "decision_policy.json": decision.to_payload(),
    }
    for name, payload in payloads.items():
        (output / name).write_bytes(canonical_v2_bytes(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        build(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Cooperative Evolution V2 fixture: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
