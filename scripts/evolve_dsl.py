"""End-to-end evolution of a Level-0 DSL interaction policy on cached dev data.

Run from repo root:  PYTHONPATH=$PWD .venv/bin/python scripts/evolve_dsl.py

Pipeline (all offline, no LLM):
  1. load per-task (toto base, truth, timestamps, history, cached retrieval effects);
  2. define a stratified, downside-protected fitness (reward event-task improvement,
     5x-penalize ANY regression, small parsimony penalty);
  3. run (mu+lambda) evolution over rules-as-data policies (evolve.run_evolution);
  4. print the evolved champion as English (auditable) + its dev table vs toto/seeds;
  5. leave-one-signal-task-out honesty check on the champion's *structure*.

NOTE on data: only task_152 in this dev slice carries actionable signal, so the
absolute numbers are a concept validation, not a generalization claim. The point is
that the search FINDS an auditable, zero-regression champion automatically. Real
acceptance needs more event-driven tasks (see the docs)."""
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
from evolving_loop.adjustment.dsl import (
    IDENTITY, GROUNDED_EVENT, GROUNDED_EVENT_CAL_WEEKEND, GROUNDED_EVENT_CAL_CASCADE,
    apply_policy, policy_to_text,
)
from evolving_loop.adjustment.evolve import run_evolution

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
REG_PENALTY = 5.0       # regressions hurt 5x more than improvements help
COMPLEXITY = 0.002      # mild parsimony pressure per rule
EPS = 1e-6

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
tasks = load_context_tasks_by_ids("external/Dr-CiK/full-download/Dr-CiK_public/tasks", dev_ids)[:8]
raw = json.loads(Path(".scratch/dev_cards.json").read_text())
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


# precompute per-task bundle
DATA = []
for t in tasks:
    n = t.numeric
    base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    DATA.append(dict(
        tid=n.task_id, base=base, truth=n.future_values,
        fts=[str(x) for x in t.future_timestamps],
        hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
        effs=effects_for(n.task_id), toto_j=joint(base, n.future_values),
    ))


def eval_policy(policy, data=DATA):
    """Return (fitness, per_task_joint dict, fired_task set)."""
    total, per, fired = 0.0, {}, set()
    for d in data:
        out, f = apply_policy(policy, d["base"], d["effs"], d["fts"],
                              history_values=d["hv"], history_timestamps=d["hts"])
        jp = joint(out, d["truth"])
        per[d["tid"]] = jp
        if f:
            fired.add(d["tid"])
        delta = d["toto_j"] - jp                 # + = improvement
        total += delta if delta >= 0 else REG_PENALTY * delta
    total -= COMPLEXITY * len(policy.rules)
    return total, per, fired


def fitness(policy):
    return eval_policy(policy)[0]


SEEDS = [IDENTITY, GROUNDED_EVENT, GROUNDED_EVENT_CAL_WEEKEND, GROUNDED_EVENT_CAL_CASCADE]

print("== evolving Level-0 DSL policy on dev (offline, no LLM) ==", flush=True)
best, best_fit, hist = run_evolution(SEEDS, fitness, generations=30, pop_size=48, elite=10)
print("generation best fitness:", ", ".join(f"g{g}:{fv:.4f}" for g, fv, _ in hist[::5]))

print("\n-- evolved champion (auditable) --")
print(policy_to_text(best))
_, champ_per, champ_fired = eval_policy(best)

print("\n-- dev joint: toto vs seeds vs evolved champion --")
seed_pers = {name: eval_policy(p)[1] for name, p in
             [("grounded_fixed", GROUNDED_EVENT), ("cal[weekend]", GROUNDED_EVENT_CAL_WEEKEND),
              ("cal[cascade]", GROUNDED_EVENT_CAL_CASCADE)]}
cols = ["toto", "grounded_fixed", "cal[weekend]", "cal[cascade]", "EVOLVED"]
w = 15
print(f"{'task':10s}" + "".join(f"{c:>{w}s}" for c in cols))
for d in DATA:
    tid = d["tid"]
    vals = [d["toto_j"], seed_pers["grounded_fixed"][tid], seed_pers["cal[weekend]"][tid],
            seed_pers["cal[cascade]"][tid], champ_per[tid]]
    print(f"{tid:10s}" + "".join(f"{v:{w}.5f}" for v in vals))
print("-" * (10 + w * len(cols)))
mean_row = [statistics.mean([d["toto_j"] for d in DATA])]
for name in ["grounded_fixed", "cal[weekend]", "cal[cascade]"]:
    mean_row.append(statistics.mean(seed_pers[name].values()))
mean_row.append(statistics.mean(champ_per.values()))
print(f"{'MEAN':10s}" + "".join(f"{v:{w}.5f}" for v in mean_row))
print(f"\nchampion fires on: {sorted(champ_fired)}  (fitness={best_fit:.4f}, rules={len(best.rules)})")

# ---- honesty check 1: does the in-sample champion regress anywhere? ----
regress = [d["tid"] for d in DATA if champ_per[d["tid"]] > d["toto_j"] + EPS]
print(f"in-sample regressions vs toto: {regress or 'NONE (kernel + fitness held the floor)'}")

# ---- honesty check 2: leave-one-task-out generalization ----
# Re-evolve on the other 7 tasks, then score the held-out one. A policy that only
# generalizes should improve the held-out task ONLY when it carries real signal
# (task_152); "improving" a held-out no-signal task is overfitting leaking through.
print("\n== leave-one-task-out generalization (honest) ==")
loo_toto, loo_evo = [], []
for held in DATA:
    train = [d for d in DATA if d["tid"] != held["tid"]]
    b, _, _ = run_evolution(SEEDS, lambda p: eval_policy(p, train)[0],
                            generations=20, pop_size=32, seed=20260918)
    jp = eval_policy(b, [held])[1][held["tid"]]
    loo_toto.append(held["toto_j"]); loo_evo.append(jp)
    if abs(jp - held["toto_j"]) < EPS:
        tag = "held (identity)"
    elif jp < held["toto_j"]:
        tag = "improved" + ("  <- real signal" if held["tid"] == "task_152" else "  <- OVERFIT leak")
    else:
        tag = "REGRESSED  <- overfit hurt"
    print(f"  held {held['tid']}: toto={held['toto_j']:.5f}  loo-evolved={jp:.5f}   {tag}")
print(f"  MEAN held-out: toto={statistics.mean(loo_toto):.5f}  evolved={statistics.mean(loo_evo):.5f}")
print(f"  in-sample MEAN evolved was {statistics.mean(champ_per.values()):.5f} "
      f"-> the gap to held-out is the overfit.")
host.close(); fs.close()
