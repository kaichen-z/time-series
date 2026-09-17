"""Evaluate the big-run champion vs seed on the 99 public_test tasks.

Champion differs from seed only in retrieval+decision (numerical identical), so
the numerical forecasts cancel in the comparison. We materialize the (seed)
numerical registry for the 99 test tasks from the champion supply release +
cached forecasts, then run the two-stage pipeline with champion vs seed
retrieval/decision and compare joint error (sMAE+sRMSE)/2.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host
from evolving_loop.v2.real.bridges import load_sealed_bundle_closure
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.run_package_coevolution import _build_registry
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest

ROOT = Path(".").resolve()
RUN = ROOT / "runs/evolution_v2/real-30m-haiku-big-20260916-r1"
MANIFEST = ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"
SPLIT = ROOT / "splits/drcik_public_80_20_99_v3.json"
TASKS_DIR = ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SEED_SUPPLY = RUN / "prepared/p2/seed_supply.json"

SEED_RETRIEVAL = "2c51c4bf3b129ccd49020da2a789d41100d10b07e4617e0c48dba6f589e0b456"
SEED_DECISION = "c2fc1c5081889779a4a19c0d128bf5024d26b48b39c0c4c7f83ac5b34215c276"

print("== building host ==", flush=True)
manifest = RealEvolutionManifestV2.from_payload(_read_canonical(MANIFEST))
import tempfile
host = build_real_host(
    manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
    projection_train_size=8, projection_dev_size=3, projection_fold_count=2,
)

print("== loading 99 public_test tasks ==", flush=True)
split = json.loads(SPLIT.read_text())
test_ids = tuple(split["partitions"]["public_test"]["task_ids"])
test_tasks = load_context_tasks_by_ids(str(TASKS_DIR), test_ids)
assert len(test_tasks) == 99, len(test_tasks)

print("== materializing seed numerical registry over test tasks ==", flush=True)
supply_payload = json.loads(SEED_SUPPLY.read_text())
release = parse_numerical_supply_release(supply_payload)
portfolio = read_policy_file(str(host.source_repo / "policies.py"))
# Materializer requires the exact 80-task/5-fold Train manifest. Test tasks are
# absent from its task_fold_map -> task_fold=None -> production-mode forecasts,
# which is exactly what a held-out test evaluation wants.
folds = build_group_fold_manifest(tuple(t.numeric for t in host.train_tasks), seed=20260906, fold_count=5)
materializer = NumericalPackageMaterializer(
    forecast_store=host.forecast_store,
    screening_policy=host.screening_policy,
    fold_manifest=folds,
    original_tasks=host.tasks,
    source_fingerprints=supply_payload["source_fingerprints"],
    runtime_fingerprints=supply_payload["runtime_fingerprints"],
    combined_policies=portfolio.combined,
    atlas_release=None,
)
registry = _build_registry(test_tasks, release, materializer)
print("   registry task_ids:", len(registry.task_ids), flush=True)

print("== loading P3 closure (champion + seed retrieval/decision) ==", flush=True)
closure = load_sealed_bundle_closure(RUN / "p3", tasks=tuple(host.tasks), host=host)
catalog = closure.catalog
champ_retrieval, champ_decision = closure.retrieval, closure.decision
seed_retrieval = catalog.resolve_retrieval(SEED_RETRIEVAL)
seed_decision = catalog.resolve_decision(SEED_DECISION)

# Build a pipeline adapter to reuse its _skills_for + host factory wiring.
from evolving_loop.v2.cooperative.adapters import CooperativePipelineAdapter
adapter = CooperativePipelineAdapter(
    catalog, host.retrieval_factory, host.decision_factory,
    metric_cap=closure.metric_cap,
    retrieval_skill_library=host.retrieval_skill_library,
    empty_skill_path=RUN / "prepared/p4-empty-skills.json",
)


def evaluate(label, retrieval, decision):
    skills = adapter._skills_for(retrieval)
    ev = PackagePipelineEvaluator._evaluate_components(
        candidate_sha256="0" * 64,
        registry=registry,
        tasks=test_tasks,
        stage="dev",
        retrieval_factory=lambda: adapter.retrieval_factory(retrieval.genome, skills),
        decision_factory=lambda: adapter.decision_factory(decision),
        metric_cap=closure.metric_cap,
        expected_retrieval_sha256=retrieval.genome.fingerprint(),
        expected_decision_prompt_sha256=hashlib.sha256(decision.prompt.encode("utf-8")).hexdigest(),
    )
    joint = (ev.mean_smae + ev.mean_srmse) / 2.0
    print(f"   [{label}] sMAE={ev.mean_smae:.5f} sRMSE={ev.mean_srmse:.5f} "
          f"joint={joint:.5f} coverage={ev.coverage} invalid={ev.invalid_count} "
          f"catastrophic={ev.catastrophic_count}", flush=True)
    return joint


print("== evaluating on 99 public_test ==", flush=True)
seed_joint = evaluate("SEED   ", seed_retrieval, seed_decision)
champ_joint = evaluate("CHAMPION", champ_retrieval, champ_decision)
gain = (seed_joint - champ_joint) / max(abs(seed_joint), 1e-12)
print("\n== RESULT on 99 public_test ==", flush=True)
print(f"   seed joint      = {seed_joint:.5f}")
print(f"   champion joint  = {champ_joint:.5f}")
print(f"   relative gain   = {gain:+.4f}  ({'champion better' if gain > 0 else 'champion worse/equal'})")
host.close()
