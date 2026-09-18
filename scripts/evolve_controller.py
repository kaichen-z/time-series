"""Evolve the decision CONTROLLER (instruction sequence) with the CVaR fitness + LOO.

The orchestration is discovered, not hand-designed: seeds = SEED_CONTROLLERS, evolution
mutates/recombines the instruction sequence, scored by the robust CVaR objective.

Cards file: CARDS_FILE env overrides; else the file with the MOST tasks (so it auto-picks
80-train once ready, else the 20-dev improved cards). Run:
  TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/evolve_controller.py"""
from __future__ import annotations
import json, os, statistics, tempfile
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import EvidenceEffect, _canon_direction
from evolving_loop.adjustment.fitness import cvar_downside_fitness, worst_case
from evolving_loop.adjustment.controller import (
    SEED_CONTROLLERS, IDENTITY_CONTROLLER, run_controller, controller_to_text,
    run_controller_evolution, build_event_effect_pool,
)

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"


def _load(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


if os.environ.get("CARDS_FILE"):
    CARDS = ROOT / ".scratch" / os.environ["CARDS_FILE"]
else:
    cands = list((ROOT / ".scratch").glob("effect_cards*.json"))
    CARDS = max(cands, key=lambda p: len(_load(p))) if cands else ROOT / ".scratch/effect_cards_f8d9d5862942.json"
raw = _load(CARDS)
REFS = _load(ROOT / ".scratch/semantic_refs.json")   # cached LLM semantic mappings (per task)
print(f"cards: {CARDS.name} ({len(raw)} tasks) | semantic refs: {len(REFS)}", flush=True)

manifest = RealEvolutionManifestV2.from_payload(
    _read_canonical(ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"))
host = build_real_host(manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
                       projection_train_size=8, projection_dev_size=3, projection_fold_count=2)
mr = host.source_repo
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)

# tasks: union of all split ids so any cards file resolves
split = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
all_ids = tuple({tid for part in split.values() for tid in part["task_ids"]})
tasks = {t.numeric.task_id: t for t in
         load_context_tasks_by_ids("external/Dr-CiK/full-download/Dr-CiK_public/tasks", all_ids)}
FIELDS = list(EvidenceEffect.__dataclass_fields__.keys())


def effects_for(tid):
    out = []
    for d in raw.get(tid, []):
        kw = {k: d.get(k) for k in FIELDS}
        kw["direction"] = _canon_direction(kw.get("direction"))
        out.append(EvidenceEffect(**kw))
    return out


def joint(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


DATA = []
for tid in raw:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    DATA.append(dict(tid=tid, cands={"toto_2_0": base}, truth=n.future_values,
                     fts=[str(x) for x in t.future_timestamps], effs=effects_for(tid),
                     hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                     semantic_ref=REFS.get(tid, "none"),
                     toto_j=joint(base, n.future_values)))
print(f"tasks with data: {len(DATA)}", flush=True)


def _pool(data):
    return build_event_effect_pool([(d["hv"], d["hts"]) for d in data])


def eval_ctrl(controller, data, pool):
    per, deltas = {}, []
    for d in data:
        out, _ = run_controller(controller, d["cands"], d["effs"], d["hv"], d["hts"], d["fts"],
                                semantic_ref=d["semantic_ref"], effect_pool=pool)
        jp = joint(out, d["truth"]); per[d["tid"]] = jp
        deltas.append(d["toto_j"] - jp)
    return cvar_downside_fitness(deltas), per, deltas


POOL_ALL = _pool(DATA)                       # in-sample pool (all tasks)
fitness = lambda c: eval_ctrl(c, DATA, POOL_ALL)[0]

print("\n== evolving controller (CVaR fitness) ==", flush=True)
best, best_fit, hist = run_controller_evolution(SEED_CONTROLLERS, fitness, generations=30, pop_size=48, elite=10)
print("gen best:", ", ".join(f"g{g}:{fv:.4f}" for g, fv in hist[::5]))
print("\n-- evolved champion controller --")
print(controller_to_text(best))

_, champ_per, champ_deltas = eval_ctrl(best, DATA, POOL_ALL)
toto_mean = statistics.mean([d["toto_j"] for d in DATA])
our_mean = statistics.mean(champ_per.values())
regress = [d["tid"] for d in DATA if champ_per[d["tid"]] > d["toto_j"] + 1e-6]
print(f"\nin-sample: toto mean {toto_mean:.5f} -> champion {our_mean:.5f} | "
      f"worst delta {worst_case(champ_deltas):+.4f} | regressions: {regress or 'NONE'}")

# leave-one-task-out generalization
print("\n== leave-one-task-out ==", flush=True)
lt, le = [], []
for held in DATA:
    train = [x for x in DATA if x["tid"] != held["tid"]]
    pool_tr = _pool(train)                    # LOO-safe: pool excludes the held-out task
    b, _, _ = run_controller_evolution(SEED_CONTROLLERS, lambda c: eval_ctrl(c, train, pool_tr)[0],
                                       generations=20, pop_size=32, seed=20260918)
    jp = eval_ctrl(b, [held], pool_tr)[1][held["tid"]]
    lt.append(held["toto_j"]); le.append(jp)
print(f"MEAN held-out: toto {statistics.mean(lt):.5f} -> controller {statistics.mean(le):.5f}")
host.close(); fs.close()
