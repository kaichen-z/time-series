"""Self-consistency probe: run retrieval N times (temperature>0, cache broken per sample)
on a few key signal tasks, to see if multi-sampling makes the direction/magnitude more
STABLE and more ACCURATE (or just consistently wrong). If it stabilizes into the right
direction + sane magnitude, self-consistency can supply the accurate window scaling the
oracle headroom needs; if it's consistently off-scale, this path is also blocked.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/self_consistency_probe.py
"""
from __future__ import annotations
import hashlib, json, tempfile
from collections import Counter
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
from numerical_agent.evolution.portfolio import read_policy_file
from evolving_loop.adjustment import project_evidence

ROOT = Path(".").resolve()
RUN = ROOT / "runs/evolution_v2/real-30m-haiku-big-20260916-r1"
MANIFEST = ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"
TASKS_DIR = ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SEED_SUPPLY = RUN / "prepared/p2/seed_supply.json"
KEY_TASKS = ["task_152", "task_46", "task_206"]   # windowed signal tasks
N = 5
PERCEPTION = PerceptionConfig()
SAMPLE_IDX = [0]

import evolving_loop.retrieval_agent.two_stage_agent as _tsa
_orig_r1, _orig_r2 = _tsa.build_round1_payload, _tsa.build_round2_payload


def _inject(task, p):
    ctx = PERCEPTION.build_context(task.numeric.history_values)
    q = {**p, "series_context": ctx} if ctx else dict(p)
    q["_sc_nonce"] = SAMPLE_IDX[0]          # different per sample -> breaks the LLM cache
    return q


_tsa.build_round1_payload = lambda task, **kw: _inject(task, _orig_r1(task, **kw))
_tsa.build_round2_payload = lambda task, *a, **kw: _inject(task, _orig_r2(task, *a, **kw))
_CALIB = ("\n\nCALIBRATION: if `series_context` is present, an additive change must be "
          "comparable to typical_scale (never orders of magnitude larger); use multiply for "
          "percentages; don't assert a direction contradicting the recent trend without strong "
          "quoted evidence, else use unknown.")
_orig_complete = _tsa.TwoStageRetrievalAgent._complete
_tsa.TwoStageRetrievalAgent._complete = (
    lambda self, prompt, payload, *, stage: _orig_complete(self, prompt + _CALIB, payload, stage=stage))

manifest = RealEvolutionManifestV2.from_payload(_read_canonical(MANIFEST))
host = build_real_host(manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
                       projection_train_size=8, projection_dev_size=3, projection_fold_count=2)

# force temperature > 0 so repeated samples actually differ
_orig_llm = host.llm_client.complete
host.llm_client.complete = (
    lambda *, system, messages, temperature=0.0: _orig_llm(system=system, messages=messages, temperature=0.8))

supply = json.loads(SEED_SUPPLY.read_text())
release = parse_numerical_supply_release(supply)
folds = build_group_fold_manifest(tuple(t.numeric for t in host.train_tasks), seed=20260906, fold_count=5)
port = read_policy_file(str(host.source_repo / "policies.py"))
mat = NumericalPackageMaterializer(
    forecast_store=host.forecast_store, screening_policy=host.screening_policy,
    fold_manifest=folds, original_tasks=host.tasks,
    source_fingerprints=supply["source_fingerprints"],
    runtime_fingerprints=supply["runtime_fingerprints"], combined_policies=port.combined, atlas_release=None)
closure = load_sealed_bundle_closure(RUN / "p3", tasks=tuple(host.tasks), host=host)
adapter = CooperativePipelineAdapter(
    closure.catalog, host.retrieval_factory, host.decision_factory,
    metric_cap=closure.metric_cap, retrieval_skill_library=host.retrieval_skill_library,
    empty_skill_path=RUN / "prepared/p4-empty-skills.json")
retr, dec = closure.retrieval, closure.decision
skills = adapter._skills_for(retr)

results = {}
for tid in KEY_TASKS:
    tasks = load_context_tasks_by_ids(str(TASKS_DIR), (tid,))
    if not tasks:
        continue
    registry = _build_registry(tasks, release, mat)
    samples = []
    for s in range(N):
        SAMPLE_IDX[0] = s
        cards = {}
        try:
            PackagePipelineEvaluator._evaluate_components(
                candidate_sha256="0" * 64, registry=registry, tasks=tasks, stage="dev",
                retrieval_factory=lambda: adapter.retrieval_factory(retr.genome, skills),
                decision_factory=lambda: adapter.decision_factory(dec),
                metric_cap=closure.metric_cap,
                expected_retrieval_sha256=retr.genome.fingerprint(),
                expected_decision_prompt_sha256=hashlib.sha256(dec.prompt.encode("utf-8")).hexdigest(),
                trace_sink=lambda task, result: cards.__setitem__(task.numeric.task_id, result.retrieval_card))
            effs = project_evidence(cards.get(tid)) if cards.get(tid) is not None else ()
            summ = [(e.direction, e.magnitude_kind, e.magnitude_value) for e in effs]
        except Exception as exc:
            summ = [("ERROR", str(exc)[:40], None)]
        samples.append(summ)
        Path(".scratch/self_consistency.json").write_text(json.dumps(results | {tid: samples}, default=str))
        print(f"{tid} sample {s+1}/{N}: {summ}", flush=True)
    results[tid] = samples

print("\n== self-consistency summary ==", flush=True)
for tid, samples in results.items():
    dirs = [s[0][0] if s else "empty" for s in samples]
    mags = [s[0][2] for s in samples if s and s[0][2] is not None]
    print(f"{tid}: first-chain directions {dict(Counter(dirs))} | magnitudes {mags}")
host.close()
