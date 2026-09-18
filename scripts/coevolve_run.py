"""(B) End-to-end CO-EVOLUTION of the 3-genome interaction on cached dev data.

Run from repo root:  TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/coevolve_run.py

Uses .scratch/effect_cards.json if present (produced by scripts/gen_effect_cards.py),
else falls back to .scratch/dev_cards.json. Co-evolves (focus, qualify, integrate)
jointly under a stratified, downside-protected fitness, prints the auditable champion
interaction + a replayable protocol trace + a leave-one-task-out honesty check."""
from __future__ import annotations
import json, statistics, tempfile
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import EvidenceEffect, _canon_direction
from evolving_loop.adjustment.fitness import cvar_downside_fitness
from evolving_loop.adjustment.coevolve import (
    Interaction, FocusPolicy, QualifyPolicy, IntegratePolicy,
    run_pipeline, interaction_to_text, mutate_interaction, run_coevolution,
)
import random

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
REG_PENALTY, COMPLEXITY, EPS = 5.0, 0.0, 1e-6

def _load(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


# CARDS_FILE env overrides; else use whichever cards file currently has MORE tasks
import os
if os.environ.get("CARDS_FILE"):
    CARDS = ROOT / ".scratch" / os.environ["CARDS_FILE"]
else:
    _cands = [ROOT / ".scratch/effect_cards.json", ROOT / ".scratch/dev_cards.json"]
    CARDS = max(_cands, key=lambda p: len(_load(p)))
raw = _load(CARDS)
print(f"using cards: {CARDS.name}  ({len(raw)} tasks)", flush=True)

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

dev_ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
                ["partitions"]["dev"]["task_ids"])
tasks = {t.numeric.task_id: t for t in
         load_context_tasks_by_ids("external/Dr-CiK/full-download/Dr-CiK_public/tasks", dev_ids)}
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
    DATA.append(dict(tid=tid, base=base, truth=n.future_values,
                     fts=[str(x) for x in t.future_timestamps],
                     hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                     effs=effects_for(tid), toto_j=joint(base, n.future_values)))
print(f"tasks with data: {len(DATA)}", flush=True)


def eval_inter(inter, data=DATA):
    per, fired, deltas = {}, set(), []
    for d in data:
        out, state = run_pipeline(inter, d["base"], d["effs"], d["fts"], d["hv"], d["hts"])
        jp = joint(out, d["truth"]); per[d["tid"]] = jp
        if state.fired:
            fired.add(d["tid"])
        deltas.append(d["toto_j"] - jp)
    return cvar_downside_fitness(deltas), per, fired   # robust (CVaR) objective


fitness = lambda it: eval_inter(it)[0]

# seeds: default + a few structural variants (different magnitude source / integrate op)
rng = random.Random(0)
SEEDS = [Interaction(),
         Interaction(qualify=QualifyPolicy(magnitude_source="calibrated")),
         Interaction(qualify=QualifyPolicy(magnitude_source="doc")),
         Interaction(integrate=IntegratePolicy(op="shift"))]

print("\n== co-evolving (focus, qualify, integrate) ==", flush=True)
best, best_fit, hist = run_coevolution(SEEDS, fitness, generations=30, pop_size=48, elite=10)
print("gen best fitness:", ", ".join(f"g{g}:{fv:.4f}" for g, fv in hist[::5]))

print("\n-- evolved champion interaction (auditable) --")
print(interaction_to_text(best))
_, champ_per, champ_fired = eval_inter(best)

print("\n-- dev joint: toto vs default-interaction vs evolved champion --")
_, def_per, _ = eval_inter(Interaction())
w = 16
print(f"{'task':10s}{'toto':>{w}s}{'default':>{w}s}{'EVOLVED':>{w}s}")
for d in DATA:
    print(f"{d['tid']:10s}{d['toto_j']:{w}.5f}{def_per[d['tid']]:{w}.5f}{champ_per[d['tid']]:{w}.5f}")
print("-" * (10 + w * 3))
print(f"{'MEAN':10s}{statistics.mean([d['toto_j'] for d in DATA]):{w}.5f}"
      f"{statistics.mean(def_per.values()):{w}.5f}{statistics.mean(champ_per.values()):{w}.5f}")

# replayable protocol trace on a fired task (protocol clarity)
for d in DATA:
    if d["tid"] in champ_fired:
        _, state = run_pipeline(best, d["base"], d["effs"], d["fts"], d["hv"], d["hts"])
        print(f"\n-- protocol trace on {d['tid']} --\n{state.to_text()}")
        break

# leave-one-task-out generalization
print("\n== leave-one-task-out generalization (honest) ==")
lt, le = [], []
for held in DATA:
    train = [x for x in DATA if x["tid"] != held["tid"]]
    b, _, _ = run_coevolution(SEEDS, lambda it: eval_inter(it, train)[0],
                              generations=20, pop_size=32, seed=20260918)
    jp = eval_inter(b, [held])[1][held["tid"]]
    lt.append(held["toto_j"]); le.append(jp)
    tag = "identity" if abs(jp - held["toto_j"]) < EPS else ("improved" if jp < held["toto_j"] else "REGRESSED")
    print(f"  held {held['tid']}: toto={held['toto_j']:.5f}  loo={jp:.5f}  {tag}")
print(f"  MEAN held-out: toto={statistics.mean(lt):.5f}  co-evolved={statistics.mean(le):.5f}")
host.close(); fs.close()
