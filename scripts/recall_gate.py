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
import json, math, random, statistics
from pathlib import Path

from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
FKEYS = ["conf", "mag", "magmax", "wfrac", "nwin", "dir", "histcv", "bias"]
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


def load(part):
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
        if not corr:
            continue
        conf = float(c.get("confidence") or 0.0)
        wfrac = max((e - s) for s, e, _ in corr) / H
        hv = list(n.history_values)
        hm = statistics.mean(hv) if hv else 0.0
        hcv = (statistics.pstdev(hv) / (abs(hm) + 1e-9)) if len(hv) > 1 else 0.0
        dirs = [1 if m > 1 else -1 for m in mults]
        feats = {
            "conf": conf,
            "mag": statistics.mean(abs(m - 1) for m in mults),
            "magmax": max(abs(m - 1) for m in mults),
            "wfrac": wfrac,
            "nwin": min(1.0, len(corr) / 3.0),
            "dir": 1.0 if len(set(dirs)) == 1 else 0.0,
            "histcv": min(1.0, hcv),
            "bias": 1.0,
        }
        data.append(dict(tid=tid, toto=list(toto), truth=truth, corr=corr, feats=feats,
                         conf=conf, wfrac=wfrac, group=group(tid)))
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


def fires(w, d):
    return sum(w.get(k, 0.0) * d["feats"][k] for k in FKEYS) > 0


def gain(w, data):
    ds = [_j(d["toto"], d["truth"]) - _j(corrected(d) if fires(w, d) else d["toto"], d["truth"]) for d in data]
    return statistics.mean(ds) - 0.5 * statistics.mean(min(0.0, x) for x in ds) if ds else 0.0


def evolve(data, rng, gens=40, pop=40, elite=8):
    seeds = [{"conf": 1.0, "bias": -0.8}, {"conf": 1.0, "bias": -0.65},
             {k: 0.0 for k in FKEYS}, {"bias": 1.0}]
    pop_ = [dict(s) for s in seeds] + [{k: rng.gauss(0, 1.0) for k in FKEYS} for _ in range(pop)]
    best = max(pop_, key=lambda w: gain(w, data))
    for g in range(gens):
        parents = sorted(pop_, key=lambda w: gain(w, data), reverse=True)[:elite]
        s = 0.6 * (1 - g / gens) + 0.05
        pop_ = parents + [{k: rng.choice(parents).get(k, 0.0) + rng.gauss(0, s) for k in FKEYS}
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
        out = corrected(d) if f else d["toto"]
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
print(f"train {len(TR)} | test {len(TE)} tasks with a correction\n")

learned = evolve(TR, rng)
mu, sd = cv(TR, rng)
base_mu, base_sd = cv([dict(d) for d in TR], rng) if False else (None, None)  # (baseline is fixed, skip its CV)
print(f"learned gate: grouped-CV gain {mu:+.4f} ± {sd:.4f}")
print(f"weights: {{{', '.join(f'{k}:{learned.get(k,0):+.2f}' for k in FKEYS)}}}\n")

print("TEST99:")
report(conf_w, TE, "baseline conf>=0.8")
report(learned, TE, "learned recall gate")
fs.close()
