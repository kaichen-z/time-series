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
from dataclasses import replace
from pathlib import Path
from evolving_loop.adjustment.coevolve import PerceptionConfig

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

# perception config: default = full numeric context (the improvement). Cards are keyed
# by the config fingerprint so different configs coexist for co-evolution.
PERCEPTION = PerceptionConfig()
OUT = ROOT / f".scratch/effect_cards_{PERCEPTION.fingerprint()}.json"
print(f"perception={PERCEPTION}  -> {OUT.name}", flush=True)

# --- runtime injection (no file edits, so the manifest identity check still passes) ---
# Add the evolvable numeric `series_context` to the retrieval payload, and a calibration
# instruction to the prompt, so the LLM can size magnitudes and sanity-check direction.
import evolving_loop.retrieval_agent.two_stage_agent as _tsa

_orig_r1, _orig_r2 = _tsa.build_round1_payload, _tsa.build_round2_payload


def _inject(task, p):
    ctx = PERCEPTION.build_context(task.numeric.history_values)
    return {**p, "series_context": ctx} if ctx else p


_tsa.build_round1_payload = lambda task, **kw: _inject(task, _orig_r1(task, **kw))
_tsa.build_round2_payload = lambda task, *a, **kw: _inject(task, _orig_r2(task, *a, **kw))

_CALIB = (
    "\n\nCALIBRATION (data-grounded): the request may include `series_context` (recent "
    "target values, summary stats, typical_scale). If present, treat it as the ground "
    "truth of scale and units. An `add` adjustment_value must be comparable to "
    "typical_scale and within the recent range -- never emit a value orders of magnitude "
    "larger (e.g. do not output 90 when recent values are ~2-3); an off-scale document "
    "number refers to something else, so mark it irrelevant, not an edit. Use `multiply` "
    "for percentages or 'N times'. Do not assert a `direction` that contradicts the recent "
    "observed trend without strong, explicit, quoted evidence; otherwise use `unknown`. "
    "Prefer preserve/none when the evidence cannot be reconciled with this scale."
)
_orig_complete = _tsa.TwoStageRetrievalAgent._complete
_tsa.TwoStageRetrievalAgent._complete = (
    lambda self, prompt, payload, *, stage: _orig_complete(self, prompt + _CALIB, payload, stage=stage)
)

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
