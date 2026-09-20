"""The recall lever: a learned HELP-vs-HARM separator to safely raise CorDP recall.

oracle_gap.py showed our conf>=0.8 gate is 100% precision / 44% recall -- 0 regressions but it leaves
22 of 39 helpable tasks untouched (+2.2% on the table). The residual is 100% text-side: separate the
39 HELP tasks from the 16 HARM tasks BETTER than raw self-reported confidence does.

So we evolve an auditable linear gate over INFERENCE-TIME features (no truth): confidence, correction
magnitude, window fraction, #windows, direction consistency, history volatility. Fire the flat CorDP
correction where the gate scores > 0; everything else stays Toto; the kernel still clamps +/-50%.
Trained on train with grouped (leave-domain-out) CV + a significance gate vs the plain conf baseline,
reported honestly on test99 (gain, wins/reg, and recall of the HELP tasks).

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/recall_gate.py
"""
from __future__ import annotations
import json, math, os, random, shutil, statistics
from pathlib import Path

from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask

ROOT = Path(".").resolve(); CAP = 5.0
PROPOSER = os.environ.get("PROPOSER", "hand")   # hand | llm  (llm: let the LLM WRITE new clues)
GATE = os.environ.get("GATE", "binary")         # binary | lambda  (lambda: continuous partial trust)
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
# ---- CLUE REGISTRY: name -> fn(base, history, corrections, conf) -> float. LLM can ADD to this. ----
def _cl_conf(base, history, corrections, conf): return conf
def _cl_mag(base, history, corrections, conf):
    ms = [abs(c[2] - 1) for c in corrections]; return sum(ms) / len(ms) if ms else 0.0
def _cl_magmax(base, history, corrections, conf):
    return max((abs(c[2] - 1) for c in corrections), default=0.0)
def _cl_wfrac(base, history, corrections, conf):
    H = len(base) or 1; return max(((c[1] - c[0]) for c in corrections), default=0) / H
def _cl_nwin(base, history, corrections, conf): return min(1.0, len(corrections) / 3.0)
def _cl_dir(base, history, corrections, conf):
    dirs = [1 if c[2] > 1 else -1 for c in corrections]
    return 1.0 if dirs and len(set(dirs)) == 1 else 0.0
def _cl_histcv(base, history, corrections, conf):
    if len(history) < 2: return 0.0
    hm = statistics.mean(history); return min(1.0, statistics.pstdev(history) / (abs(hm) + 1e-9))

CLUE_FNS = {"conf": _cl_conf, "mag": _cl_mag, "magmax": _cl_magmax, "wfrac": _cl_wfrac,
            "nwin": _cl_nwin, "dir": _cl_dir, "histcv": _cl_histcv}


def fkeys():
    return list(CLUE_FNS) + ["bias"]


def clues(d):
    f = {k: fn(d["toto"], d["hist"], d["corr"], d["conf"]) for k, fn in CLUE_FNS.items()}
    f["bias"] = 1.0
    return f


_GRP: dict = {}


def group(tid):
    if tid not in _GRP:
        try:
            d = json.loads((ROOT / TASKS / f"{tid}.json").read_text())
            et = ((d.get("showcase") or {}).get("entity") or {}).get("type") or "unknown"
            _GRP[tid] = f"{et}|{(d.get('task_metadata') or {}).get('frequency') or ''}"
        except Exception:
            _GRP[tid] = "unknown"
    return _GRP[tid]


def load(part, keep_all=False):
    cards = json.loads((ROOT / f".scratch/cordp_cards_{part}.json").read_text())
    ids = tuple(t for t in SPLIT[part]["task_ids"] if t in cards)
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS, ids)}
    data = []
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        n = t.numeric; truth = list(n.future_values); H = len(truth)
        toto = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        if len(toto) != H:
            continue
        c = cards.get(tid) or {}
        fts = [str(x) for x in t.future_timestamps]
        corr, mults = [], []
        for r in (c.get("corrections") or []):
            if len(r) >= 3:
                on = [i for i, m in enumerate(horizon_window_mask(fts, str(r[0]), str(r[1]))) if m]
                if on:
                    corr.append((on[0], on[-1] + 1, float(r[2]))); mults.append(float(r[2]))
        if not corr and not keep_all:
            continue
        conf = float(c.get("confidence") or 0.0)
        data.append(dict(tid=tid, toto=list(toto), truth=truth, corr=corr,
                         hist=list(n.history_values), conf=conf,
                         wfrac=max(((e - s) for s, e, _ in corr), default=0) / H, group=group(tid)))
    return data


def corrected(d):
    out = list(d["toto"])
    for s, e, m in d["corr"]:
        for i in range(s, e):
            out[i] = d["toto"][i] * m
    return list(apply_bounded_delta(d["toto"], out))


def _j(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def helps(d):
    return _j(corrected(d), d["truth"]) < _j(d["toto"], d["truth"]) - 1e-9


def _score(w, d):
    f = clues(d)
    return sum(w.get(k, 0.0) * f.get(k, 0.0) for k in fkeys())


def strength(w, d):
    """Trust strength lambda in [0,1]. binary gate: {0,1}; lambda gate: sigmoid(score) -- a smooth,
    learned PARTIAL trust that can down-weight (not just drop) a shaky correction."""
    s = _score(w, d)
    if GATE == "lambda":
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))
    return 1.0 if s > 0 else 0.0


def apply_scaled(d, lam):
    """Apply the correction scaled by lambda: out = base*(1 + lam*(mult-1)), then the kernel."""
    if lam <= 1e-9 or not d["corr"]:
        return d["toto"]
    out = list(d["toto"])
    for s, e, m in d["corr"]:
        for i in range(s, min(e, len(out))):
            out[i] = d["toto"][i] * (1.0 + lam * (m - 1.0))
    return list(apply_bounded_delta(d["toto"], out))


def fires(w, d):
    return strength(w, d) > 0.5


def gain(w, data):
    ds = [_j(d["toto"], d["truth"]) - _j(apply_scaled(d, strength(w, d)), d["truth"]) for d in data]
    return statistics.mean(ds) - 0.5 * statistics.mean(min(0.0, x) for x in ds) if ds else 0.0


def evolve(data, rng, gens=40, pop=40, elite=8):
    K = fkeys()
    seeds = [{"conf": 1.0, "bias": -0.8}, {"conf": 1.0, "bias": -0.65},
             {k: 0.0 for k in K}, {"bias": 1.0}]
    pop_ = [dict(s) for s in seeds] + [{k: rng.gauss(0, 1.0) for k in K} for _ in range(pop)]
    best = max(pop_, key=lambda w: gain(w, data))
    for g in range(gens):
        parents = sorted(pop_, key=lambda w: gain(w, data), reverse=True)[:elite]
        s = 0.6 * (1 - g / gens) + 0.05
        pop_ = parents + [{k: rng.choice(parents).get(k, 0.0) + rng.gauss(0, s) for k in K}
                          for _ in range(pop - elite)]
        cur = max(pop_, key=lambda w: gain(w, data))
        if gain(cur, data) > gain(best, data):
            best = cur
    return best


def gfolds(data, K=3):
    groups = {}
    for d in data:
        groups.setdefault(d["group"], []).append(d)
    out = [[] for _ in range(K)]; load_ = [0] * K
    for _g, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        f = load_.index(min(load_)); out[f].extend(items); load_[f] += len(items)
    return out


def cv(data, rng, K=3):
    folds = gfolds(data, K); sc = []
    for f in range(K):
        val = folds[f]; tr = [d for g in range(K) if g != f for d in folds[g]]
        if val and tr:
            sc.append(gain(evolve(tr, rng, gens=25, pop=28), val))
    return (statistics.mean(sc) if sc else 0.0, statistics.pstdev(sc) if len(sc) > 1 else 0.0)


def report(w, data, tag):
    rows, wins, reg, fire_help, fire_harm = [], 0, 0, 0, 0
    HELP = [d for d in data if helps(d)]
    for d in data:
        f = fires(w, d)
        out = apply_scaled(d, strength(w, d))
        mb = drcik_point_metrics(d["truth"], d["toto"], cap=CAP)
        mo = drcik_point_metrics(d["truth"], out, cap=CAP)
        rows.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"]))
        if mo["smae"] + mo["srmse"] < mb["smae"] + mb["srmse"] - 1e-9: wins += 1
        elif mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9: reg += 1
        if f and helps(d): fire_help += 1
        elif f and not helps(d): fire_harm += 1
    bs = statistics.mean(r[0] for r in rows); os_ = statistics.mean(r[2] for r in rows)
    br = statistics.mean(r[1] for r in rows); or_ = statistics.mean(r[3] for r in rows)
    rec = fire_help / len(HELP) if HELP else 0.0
    print(f"  {tag:22s} sMAE {(bs-os_)/bs:+.2%} | sRMSE {(br-or_)/br:+.2%} | wins {wins} reg {reg} "
          f"| recall {rec:.0%} ({fire_help}/{len(HELP)} HELP, {fire_harm} HARM)")


TR, TE = load("train"), load("public_test")
rng = random.Random(20260919)
CONF = lambda d: d["conf"] >= 0.8 and d["wfrac"] <= 0.2      # our current baseline gate as weights-free fn
conf_w = {"conf": 1.0, "bias": -0.8}                          # linear-gate equivalent of conf>=0.8

print(f"== recall gate: learned HELP-vs-HARM separator vs plain conf gate (test99) ==")
print(f"train {len(TR)} | test {len(TE)} tasks with a correction | proposer={PROPOSER} | gate={GATE}\n")

if PROPOSER == "llm":
    # LLM WRITES new clues; keep each only if it improves grouped-CV gain on train (keep-if-CV-helps).
    from scripts.llm_design import propose_clues
    from common.llm import ClaudeCLIClient, ClaudeCLIConfig
    client = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                             cache_dir=str(ROOT / ".scratch/clue-design-cache")))
    try:
        proposed = propose_clues(client, set(CLUE_FNS), n=6)
    except Exception as e:
        print(f"  (LLM proposer failed: {type(e).__name__}: {e})"); proposed = {}
    print(f"LLM proposed {len(proposed)} clues that compiled: {list(proposed)}")
    base_cv = cv(TR, rng)[0]
    print(f"base {len(CLUE_FNS)}-clue grouped-CV gain = {base_cv:+.4f}\n")
    for name, fn in proposed.items():
        CLUE_FNS[name] = fn                                  # tentatively add the clue
        cvg = cv(TR, rng)[0]
        if cvg >= base_cv + 0.002:
            base_cv = cvg
            print(f"  + '{name}': CV gain -> {cvg:+.4f}   ✓ KEEP")
        else:
            del CLUE_FNS[name]
            print(f"  - '{name}': CV gain {cvg:+.4f}   ✗ drop")
    print(f"\ngrown clue set ({len(CLUE_FNS)}): {list(CLUE_FNS)}\n")

learned = evolve(TR, rng)
mu, sd = cv(TR, rng)
print(f"learned gate: grouped-CV gain {mu:+.4f} ± {sd:.4f}")
print(f"weights: {{{', '.join(f'{k}:{learned.get(k,0):+.2f}' for k in fkeys())}}}\n")

print("TEST99 (55 correction subset):")
report(conf_w, TE, "baseline conf>=0.8")
report(learned, TE, "learned recall gate")
TE99 = load("public_test", keep_all=True)
print(f"\nTEST99 (WHOLE {len(TE99)} tasks):")
report(conf_w, TE99, "baseline conf>=0.8")
report(learned, TE99, "learned recall gate")
fs.close()
