"""(A) Generate retrieval effect cards for MORE tasks, so the evolution has enough
signal-bearing practice tasks to give a real generalization verdict.

Runs the champion two-stage pipeline over each task with a trace that captures the
retrieval evidence card, projects it to EvidenceEffect dicts, and appends to
.scratch/effect_cards.json INCREMENTALLY (one dump per task, so a slow/interrupted run
keeps everything already done). This DOES call the retrieval LLM, so it is slow; run it
in the background once and the evolution stays offline afterwards.

Run:  TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/gen_effect_cards.py
"""
from __future__ import annotations
import hashlib, json, tempfile, traceback
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host
from evolving_loop.v2.real.bridges import load_sealed_bundle_closure
from evolving_loop.package_numerical_supply import parse_numerical_supply_release
from evolving_loop.package_numerical_evolution import NumericalPackageMaterializer
from evolving_loop.run_package_coevolution import _build_registry
from evolving_loop.v2.cooperative.adapters import CooperativePipelineAdapter
from numerical_agent.evolution.task_local_evolution import build_group_fold_manifest
from evolving_loop.package_pipeline_evaluator import PackagePipelineEvaluator
from evolving_loop.adjustment import project_evidence

ROOT = Path(".").resolve()
RUN = ROOT / "runs/evolution_v2/real-30m-haiku-big-20260916-r1"
MANIFEST = ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"
SPLIT = ROOT / "splits/drcik_public_80_20_99_v3.json"
TASKS_DIR = ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SEED_SUPPLY = RUN / "prepared/p2/seed_supply.json"
OUT = ROOT / ".scratch/effect_cards.json"

# Which tasks to generate for: all 20 dev (extend to train later if this proves fast).
parts = json.loads(SPLIT.read_text())["partitions"]
WANT = list(parts["dev"]["task_ids"])

manifest = RealEvolutionManifestV2.from_payload(_read_canonical(MANIFEST))
host = build_real_host(manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
                       projection_train_size=8, projection_dev_size=3, projection_fold_count=2)
supply = json.loads(SEED_SUPPLY.read_text())
release = parse_numerical_supply_release(supply)
folds = build_group_fold_manifest(tuple(t.numeric for t in host.train_tasks), seed=20260906, fold_count=5)
# Match the probe's construction exactly.
from numerical_agent.evolution.portfolio import read_policy_file
port = read_policy_file(str(host.source_repo / "policies.py"))
mat = NumericalPackageMaterializer(
    forecast_store=host.forecast_store, screening_policy=host.screening_policy,
    fold_manifest=folds, original_tasks=host.tasks,
    source_fingerprints=supply["source_fingerprints"],
    runtime_fingerprints=supply["runtime_fingerprints"],
    combined_policies=port.combined, atlas_release=None)

closure = load_sealed_bundle_closure(RUN / "p3", tasks=tuple(host.tasks), host=host)
adapter = CooperativePipelineAdapter(
    closure.catalog, host.retrieval_factory, host.decision_factory,
    metric_cap=closure.metric_cap, retrieval_skill_library=host.retrieval_skill_library,
    empty_skill_path=RUN / "prepared/p4-empty-skills.json")
retr, dec = closure.retrieval, closure.decision
skills = adapter._skills_for(retr)

out = {}
if OUT.exists():
    try:
        out = json.loads(OUT.read_text())
    except Exception:
        out = {}

todo = [tid for tid in WANT if tid not in out]
print(f"generating cards for {len(todo)} tasks (already have {len(out)}) -> {OUT}", flush=True)

for i, tid in enumerate(todo, 1):
    try:
        tasks = load_context_tasks_by_ids(str(TASKS_DIR), (tid,))
        if not tasks:
            print(f"[{i}/{len(todo)}] {tid}: not found", flush=True); continue
        registry = _build_registry(tasks, release, mat)
        cards = {}
        PackagePipelineEvaluator._evaluate_components(
            candidate_sha256="0" * 64, registry=registry, tasks=tasks, stage="dev",
            retrieval_factory=lambda: adapter.retrieval_factory(retr.genome, skills),
            decision_factory=lambda: adapter.decision_factory(dec),
            metric_cap=closure.metric_cap,
            expected_retrieval_sha256=retr.genome.fingerprint(),
            expected_decision_prompt_sha256=hashlib.sha256(dec.prompt.encode("utf-8")).hexdigest(),
            trace_sink=lambda task, result: cards.__setitem__(task.numeric.task_id, result.retrieval_card),
        )
        card = cards.get(tid)
        effs = project_evidence(card) if card is not None else ()
        out[tid] = [e.__dict__ for e in effs]
        OUT.write_text(json.dumps(out, default=str, indent=0))
        print(f"[{i}/{len(todo)}] {tid}: {len(out[tid])} chains  (saved)", flush=True)
    except Exception as exc:
        print(f"[{i}/{len(todo)}] {tid}: ERROR {exc!r}", flush=True)
        traceback.print_exc()

print(f"done. total tasks with cards: {len(out)}", flush=True)
host.close()
