"""Diagnostic: how much ACTIONABLE grounded evidence do tasks actually contain,
and can applying it (post_adjust) move the error vs toto?

Runs the champion pipeline over the 20 dev tasks with a trace capturing each task's
retrieval evidence card, then: projects effects, counts actionable ones, and scores
toto vs reference_future_event_adjust(toto, evidence) vs truth."""
from __future__ import annotations
import hashlib, json, statistics, tempfile
from pathlib import Path
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from evolving_loop.v2.real.bridges import load_sealed_bundle_closure
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.run_package_coevolution import _build_registry
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.v2.cooperative.adapters import CooperativePipelineAdapter
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment import project_evidence, reference_future_event_adjust, identity_adjust

ROOT = Path(".").resolve()
RUN = ROOT / "runs/evolution_v2/real-30m-haiku-big-20260916-r1"
MANIFEST = ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"
SPLIT = ROOT / "splits/drcik_public_80_20_99_v3.json"
TASKS_DIR = ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SEED_SUPPLY = RUN / "prepared/p2/seed_supply.json"
IDENTITY = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CAP = 5.0

manifest = RealEvolutionManifestV2.from_payload(_read_canonical(MANIFEST))
host = build_real_host(manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
                       projection_train_size=8, projection_dev_size=3, projection_fold_count=2)
dev_ids = tuple(json.loads(SPLIT.read_text())["partitions"]["dev"]["task_ids"])
tasks = load_context_tasks_by_ids(str(TASKS_DIR), dev_ids)[:8]
print("dev tasks:", len(tasks), flush=True)

mroot = host.source_repo
port = read_policy_file(str(mroot / "policies.py"))
scr = _load_screening_policy(str(mroot / "dictionary.py"))
toto_fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                        mroot / "methods.py", mroot / "skills.py", port, None,
                        screening_hash=scr.fingerprint(), runtime_identity={}, cache_only=True,
                        identity_hash_override=IDENTITY)
toto = {t.numeric.task_id: tuple(toto_fs.forecast("toto_2_0", tuple(t.numeric.history_values),
        t.numeric.prediction_length, t.numeric.frequency)) for t in tasks}

supply = json.loads(SEED_SUPPLY.read_text())
release = parse_numerical_supply_release(supply)
folds = build_group_fold_manifest(tuple(t.numeric for t in host.train_tasks), seed=20260906, fold_count=5)
mat = NumericalPackageMaterializer(forecast_store=host.forecast_store, screening_policy=host.screening_policy,
        fold_manifest=folds, original_tasks=host.tasks, source_fingerprints=supply["source_fingerprints"],
        runtime_fingerprints=supply["runtime_fingerprints"], combined_policies=port.combined, atlas_release=None)
registry = _build_registry(tasks, release, mat)

closure = load_sealed_bundle_closure(RUN / "p3", tasks=tuple(host.tasks), host=host)
adapter = CooperativePipelineAdapter(closure.catalog, host.retrieval_factory, host.decision_factory,
        metric_cap=closure.metric_cap, retrieval_skill_library=host.retrieval_skill_library,
        empty_skill_path=RUN / "prepared/p4-empty-skills.json")
retr, dec = closure.retrieval, closure.decision
skills = adapter._skills_for(retr)

cards = {}
def sink(task, result):
    cards[task.numeric.task_id] = result.retrieval_card

PackagePipelineEvaluator._evaluate_components(
    candidate_sha256="0" * 64, registry=registry, tasks=tasks, stage="dev",
    retrieval_factory=lambda: adapter.retrieval_factory(retr.genome, skills),
    decision_factory=lambda: adapter.decision_factory(dec),
    metric_cap=closure.metric_cap,
    expected_retrieval_sha256=retr.genome.fingerprint(),
    expected_decision_prompt_sha256=hashlib.sha256(dec.prompt.encode("utf-8")).hexdigest(),
    trace_sink=sink,
)

def joint(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0

n_actionable_tasks = tot_effects = tot_actionable = 0
toto_j, ident_j, ref_j, deviated = [], [], [], 0
by_task = {t.numeric.task_id: t for t in tasks}
for tid, card in cards.items():
    t = by_task[tid]; n = t.numeric
    effects = project_evidence(card)
    act = [e for e in effects if e.actionable]
    tot_effects += len(effects); tot_actionable += len(act)
    if act:
        n_actionable_tasks += 1
    base = toto[tid]; truth = n.future_values; fts = t.future_timestamps
    ref = reference_future_event_adjust(base, effects, fts)
    toto_j.append(joint(base, truth))
    ident_j.append(joint(identity_adjust(base, effects, fts), truth))
    ref_j.append(joint(ref, truth))
    if tuple(ref) != tuple(base):
        deviated += 1

print(f"\n== evidence headroom on {len(cards)} dev tasks ==", flush=True)
print(f"  total chains: {tot_effects} | actionable(grounded+numeric+entity/target+dir+mag+window): {tot_actionable}")
print(f"  tasks with >=1 actionable effect: {n_actionable_tasks}/{len(cards)}")
print(f"  reference adjuster deviated from toto on: {deviated}/{len(cards)} tasks")
print(f"  mean joint  toto={statistics.mean(toto_j):.5f}  identity={statistics.mean(ident_j):.5f}  reference={statistics.mean(ref_j):.5f}")

# Per-criterion breakdown over ALL chains: which requirement drops them.
crit = {"grounded": 0, "numeric_eligible": 0, "entity_match": 0, "target_match": 0,
        "direction(up/down)": 0, "has_magnitude": 0, "has_window": 0}
dump = {}
import math as _m
for tid, card in cards.items():
    effs = project_evidence(card)
    dump[tid] = [e.__dict__ for e in effs]
    for e in effs:
        crit["grounded"] += e.grounded
        crit["numeric_eligible"] += e.numeric_eligible
        crit["entity_match"] += e.entity_match
        crit["target_match"] += e.target_match
        crit["direction(up/down)"] += e.direction in {"increase", "decrease"}
        crit["has_magnitude"] += e.magnitude_value is not None and _m.isfinite(e.magnitude_value or _m.nan)
        crit["has_window"] += e.start_timestamp is not None and e.end_timestamp is not None
print(f"\n  per-criterion pass counts (of {tot_effects} chains):")
for k, v in crit.items():
    print(f"    {k:22s}: {v}")
Path(".scratch/dev_cards.json").write_text(json.dumps(dump, default=str, indent=0))
print("  cached chains -> .scratch/dev_cards.json")
host.close(); toto_fs.close()
