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
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, horizon_window_mask, effects_from_cordp)
from evolving_loop.adjustment.fitness import cvar_downside_fitness, worst_case
from evolving_loop.adjustment.controller import (
    SEED_CONTROLLERS, IDENTITY_CONTROLLER, run_controller, controller_to_text,
    run_controller_evolution, build_event_effect_pool,
)
from evolving_loop.adjustment.math_menu import build_math_menu
from evolving_loop.adjustment.skill_memory import SkillMemory

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
# CorDP cards: cached per-task document-anchored corrections for the evolvable CorDPAdjust
if os.environ.get("CORDP_CARDS"):
    CORDP_FILE = ROOT / ".scratch" / os.environ["CORDP_CARDS"]
else:
    _cc = [ROOT / ".scratch/cordp_cards_train.json", ROOT / ".scratch/cordp_cards.json"]
    CORDP_FILE = next((p for p in _cc if p.exists()), _cc[0])
CORDP = _load(CORDP_FILE)
print(f"cards: {CARDS.name} ({len(raw)} tasks) | semantic refs: {len(REFS)} | "
      f"cordp: {CORDP_FILE.name} ({len(CORDP)} tasks)", flush=True)


def cordp_for(tid):
    c = CORDP.get(tid) or {}
    corrs = tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3)
    return float(c.get("confidence") or 0.0), corrs

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


def roi_mask(ccorrs, fts):
    """Region of interest = union of the CorDP correction windows (GATE-INDEPENDENT, so every
    controller is scored on the same steps -- the event windows where text acts)."""
    m = [False] * len(fts)
    for (start, end, _mult) in ccorrs:
        wm = horizon_window_mask(fts, str(start), str(end))
        m = [a or b for a, b in zip(m, wm)]
    return m


def windowed_joint(fc, truth, roi):
    idx = [i for i, r in enumerate(roi) if r]
    if not idx:
        return None
    sub_t = [truth[i] for i in idx]
    sub_f = [fc[i] for i in idx]
    m = drcik_point_metrics(sub_t, sub_f, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


DATA = []
_task_ids = sorted(set(CORDP) if os.environ.get("CORDP_ONLY") else set(raw) | set(CORDP))
for tid in _task_ids:                           # CORDP_ONLY -> evolve on the CorDP task set alone
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    cands = {"toto_2_0": base}
    try:                                              # numerical also exposes a statistical candidate
        cands["seasonal_naive"] = tuple(fs.forecast("seasonal_naive", tuple(n.history_values),
                                                    n.prediction_length, n.frequency))
    except Exception:
        pass
    cconf, ccorrs = cordp_for(tid)
    fts = [str(x) for x in t.future_timestamps]
    roi = roi_mask(ccorrs, fts)
    merged_effs = effects_from_cordp(ccorrs) or effects_for(tid)   # one evidence source (CorDP), all splits
    DATA.append(dict(tid=tid, cands=cands, truth=n.future_values,
                     fts=fts, effs=merged_effs,
                     hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                     semantic_ref=REFS.get(tid, "none"),
                     cordp_conf=cconf, cordp_corrections=ccorrs, roi=roi,
                     menu=build_math_menu(list(n.history_values),
                                          [str(x) for x in t.history_timestamps], fts),
                     toto_j=joint(base, n.future_values),
                     toto_roi=windowed_joint(list(base), list(n.future_values), roi)))
print(f"tasks with data: {len(DATA)}", flush=True)


def _pool(data):
    return build_event_effect_pool([(d["hv"], d["hts"]) for d in data])


from dataclasses import replace as _dc_replace
_OBJ = os.environ.get("FITNESS", "cvar")     # "cvar" (default) or "mean" (diagnostic)
_BETA = float(os.environ.get("BETA", "2.0"))  # CVaR downside weight (lower = more tolerant)
_MAX_CAP = float(os.environ["MAX_CAP"]) if os.environ.get("MAX_CAP") else None  # small-step cap


def _clamp_caps(c):
    if _MAX_CAP is None:
        return c
    from evolving_loop.adjustment.controller import Controller
    steps = tuple(_dc_replace(s, cap=min(s.cap, _MAX_CAP)) if hasattr(s, "cap") else s
                  for s in c.steps)
    return Controller(steps=steps, name=c.name)


def _score(deltas):
    if _OBJ == "mean":
        return statistics.mean(deltas) if deltas else 0.0
    return cvar_downside_fitness(deltas, beta=_BETA)


_WINDOWED = bool(os.environ.get("WINDOWED"))     # score on ROI (event windows) not whole horizon
_CALL_COST = float(os.environ.get("CALL_COST", "0.002"))  # efficiency: charge per CallAgent call


def eval_ctrl(controller, data, pool):
    controller = _clamp_caps(controller)     # small-step: cap all adjustments at MAX_CAP
    per, deltas = {}, []
    for d in data:
        out, trace = run_controller(controller, d["cands"], d["effs"], d["hv"], d["hts"], d["fts"],
                                    semantic_ref=d["semantic_ref"], effect_pool=pool,
                                    cordp_conf=d["cordp_conf"], cordp_corrections=d["cordp_corrections"],
                                    menu=d.get("menu", ()))
        ncalls = sum(1 for t in trace if t.startswith("call:"))   # agent-as-tool calls
        cost = _CALL_COST * ncalls                                # efficiency penalty
        if _WINDOWED:
            if d.get("toto_roi") is None:        # no ROI -> task carries no signal, skip
                continue
            jp = windowed_joint(list(out), list(d["truth"]), d["roi"])
            per[d["tid"]] = jp
            deltas.append(d["toto_roi"] - jp - cost)
        else:
            jp = joint(out, d["truth"]); per[d["tid"]] = jp
            deltas.append(d["toto_j"] - jp - cost)
    return _score(deltas), per, deltas


POOL_ALL = _pool(DATA)                       # in-sample pool (all tasks)
fitness = lambda c: eval_ctrl(c, DATA, POOL_ALL)[0]
print(f"objective: {_OBJ} (beta={_BETA})", flush=True)

# skill memory: accumulate good controllers across runs (extra seeds now, save champion after)
MEM = SkillMemory(ROOT / ".scratch/skill_memory.json")
seeds = tuple(SEED_CONTROLLERS) + tuple(MEM.seeds())
print(f"\n== evolving controller (CVaR fitness) | seeds: {len(SEED_CONTROLLERS)} built + "
      f"{len(MEM.seeds())} from skill memory ==", flush=True)
best, best_fit, hist = run_controller_evolution(seeds, fitness, generations=30, pop_size=48, elite=10)
best = _clamp_caps(best)          # report/store the actually-evaluated (capped) controller
print("gen best:", ", ".join(f"g{g}:{fv:.4f}" for g, fv in hist[::5]))
print("\n-- evolved champion controller --")
print(controller_to_text(best))

_, champ_per, champ_deltas = eval_ctrl(best, DATA, POOL_ALL)
_scored = [d for d in DATA if d["tid"] in champ_per]      # WINDOWED scores only ROI tasks
_tref = "toto_roi" if _WINDOWED else "toto_j"
toto_mean = statistics.mean([d[_tref] for d in _scored]) if _scored else 0.0
our_mean = statistics.mean(champ_per.values()) if champ_per else 0.0
regress = [d["tid"] for d in _scored if champ_per[d["tid"]] > d[_tref] + 1e-6]
wins = [d["tid"] for d in _scored if champ_per[d["tid"]] < d[_tref] - 1e-6]
print(f"\nin-sample ({'ROI' if _WINDOWED else 'full'}, {len(_scored)} scored): "
      f"toto mean {toto_mean:.5f} -> champion {our_mean:.5f} | "
      f"wins {len(wins)} | worst delta {worst_case(champ_deltas):+.4f} | regressions: {regress or 'NONE'}")

# sediment the champion into skill memory for future runs
MEM.add(best, best_fit, meta={"cards": CARDS.name, "tasks": len(DATA)})
MEM.save()
print(f"skill memory: {len(MEM.entries)} controllers stored -> .scratch/skill_memory.json")

# leave-one-task-out generalization (skippable for a quick diagnostic)
if os.environ.get("SKIP_LOO"):
    print("\n(SKIP_LOO set -- skipping leave-one-task-out)")
    host.close(); fs.close(); raise SystemExit(0)
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
