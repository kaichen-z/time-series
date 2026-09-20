"""Evolve the RETRIEVAL document-selection policy, supervised by Dr-CiK ground-truth roles
(role=="supporting" vs "distractor"). Dense signal (~37 docs/task) -> dodges the sparse
forecast ceiling. The policy is an auditable linear rule over deployable document FEATURES.

Refactored so the FEATURE SET is a REGISTRY (name -> fn(task, doc)->float): DSL auto-growth
(grow_retrieval_dsl.py) and two-loop co-evolution (coevolve_loops.py) import from here and can
ADD new features. Run standalone: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/evolve_retrieval.py
"""
from __future__ import annotations
import json, math, random, re, statistics
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids

ROOT = Path(".").resolve()
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]

_TOK = re.compile(r"[a-z0-9]+")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_FUTURE = re.compile(r"\b(will|forecast|predict|projected?|expected to|anticipat|outlook|"
                     r"going to|in the future|next (week|month|quarter|year)|guidance)\b")
_CAUSAL = re.compile(r"\b(due to|because|caused by|resulted in|led to|following the|as a result|"
                     r"owing to|triggered by|attributable to|in response to|driven by|impact of)\b")
_HEDGE = re.compile(r"\b(correlation|variance analysis|may be related|not directly|no direct|"
                    r"statistical association|no significant|unrelated|coincid|for reference)\b")


def toks(s):
    return set(_TOK.findall((s or "").lower()))


# ---- FEATURE REGISTRY: name -> fn(task, doc) -> float. DSL-growth appends to this. ----
def _f_exact_entity(task, doc):
    ent = (task.numeric.entity_name or "").strip().lower()
    return 1.0 if ent and ent in (doc.content or "").lower() else 0.0


def _f_entity_frac(task, doc):
    et = toks(task.numeric.entity_name)
    return len(toks(doc.content) & et) / len(et) if et else 0.0


def _f_target_frac(task, doc):
    tt = toks(task.target_name) | toks(task.target_description)
    return len(toks(doc.content) & tt) / len(tt) if tt else 0.0


def _f_time_overlap(task, doc):
    fut = {m.group(0) for m in _YEAR.finditer(" ".join(str(x) for x in task.future_timestamps))}
    dy = {m.group(0) for m in _YEAR.finditer(doc.content or "")}
    return 1.0 if (dy & fut) else (0.0 if dy else 0.5)


def _f_future_claim(task, doc):
    return 1.0 if _FUTURE.search((doc.content or "").lower()) else 0.0


def _f_numeric_density(task, doc):
    c = doc.content or ""
    return min(1.0, len(re.findall(r"\d", c)) / max(1, len(c)) * 20)


def _f_len_norm(task, doc):
    return min(1.0, len(doc.content or "") / 4000.0)


def _f_causal_relevance(task, doc):
    low = (doc.content or "").lower()
    return math.tanh(len(_CAUSAL.findall(low))) - math.tanh(len(_HEDGE.findall(low)))


FEATURE_FNS: dict = {
    "exact_entity": _f_exact_entity, "entity_frac": _f_entity_frac, "target_frac": _f_target_frac,
    "time_overlap": _f_time_overlap, "future_claim": _f_future_claim,
    "numeric_density": _f_numeric_density, "len_norm": _f_len_norm,
    "causal_relevance": _f_causal_relevance,
}


def feats():
    return list(FEATURE_FNS.keys())


def register_feature(name, fn):
    FEATURE_FNS[name] = fn


def doc_features(task, doc) -> dict:
    return {k: fn(task, doc) for k, fn in FEATURE_FNS.items()}


def load(part):
    ids = tuple(SPLIT[part]["task_ids"])
    tasks = load_context_tasks_by_ids(TASKS_DIR, ids)
    data = []
    for t in tasks:
        rows = [(doc_features(t, d), d.role == "supporting", d.subtype) for d in t.documents]
        if any(r[1] for r in rows):
            data.append(rows)
    return data


def score(feat, w):
    return sum(w.get(k, 0.0) * v for k, v in feat.items()) + w.get("bias", 0.0)


def f1_of(rows, w):
    sel = {i for i, (f, _s, _st) in enumerate(rows) if score(f, w) > 0}
    gt = {i for i, (_f, s, _st) in enumerate(rows) if s}
    if not gt:
        return 0.0
    tp = len(sel & gt); fp = len(sel - gt); fn = len(gt - sel)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def fitness(w, data):
    return statistics.mean(f1_of(rows, w) for rows in data) if data else 0.0


def evolve(train, rng, gens=60, pop=48, elite=8, sigma=0.6):
    keys = feats() + ["bias"]
    seeds = [{k: 0.0 for k in keys},
             {**{k: 0.0 for k in keys}, "exact_entity": 2.0, "bias": -1.0}]
    population = list(seeds)
    while len(population) < pop:
        population.append({k: rng.gauss(0, 1.0) for k in keys})
    best = max(population, key=lambda w: fitness(w, train)); best_fit = fitness(best, train)
    hist = []
    for g in range(gens):
        parents = sorted(population, key=lambda w: fitness(w, train), reverse=True)[:elite]
        s = sigma * (1 - g / gens) + 0.05
        children = [{k: rng.choice(parents)[k] + rng.gauss(0, s) for k in keys}
                    for _ in range(pop - elite)]
        population = parents + children
        cur = max(population, key=lambda w: fitness(w, train)); cf = fitness(cur, train)
        if cf > best_fit:
            best, best_fit = cur, cf
        hist.append(round(best_fit, 4))
    return best, best_fit, hist


def subtype_rejection(data, w):
    from collections import defaultdict
    rej = defaultdict(lambda: [0, 0])
    for rows in data:
        for f, s, st in rows:
            if not s:
                key = st or "other"; rej[key][1] += 1
                if not score(f, w) > 0:
                    rej[key][0] += 1
    return {k: (v[0], v[1]) for k, v in sorted(rej.items())}


if __name__ == "__main__":
    TR, DV, TE = load("train"), load("dev"), load("public_test")
    rng = random.Random(20260919)
    print(f"ROI: train {len(TR)} | dev {len(DV)} | test {len(TE)} tasks\n")
    ALL = {"bias": 1.0}; ENT = {"exact_entity": 2.0, "bias": -1.0}
    print("== baselines (train / dev / test F1) ==")
    for nm, w in [("select-all", ALL), ("exact-entity", ENT)]:
        print(f"  {nm:12s}: {fitness(w,TR):.3f} / {fitness(w,DV):.3f} / {fitness(w,TE):.3f}")
    print("  oracle      : 1.000 / 1.000 / 1.000")
    best, bf, hist = evolve(TR, rng)
    print(f"\n== EVOLVED policy | gen best F1: {hist[::10] + [hist[-1]]}")
    print("  weights:", {k: round(best[k], 2) for k in feats() + ["bias"]})
    print(f"\n== EVOLVED F1: train {fitness(best,TR):.3f} | dev {fitness(best,DV):.3f} | test {fitness(best,TE):.3f}")
    print("== per-subtype rejection (test) ==")
    for st, (rj, tot) in subtype_rejection(TE, best).items():
        print(f"    {st:11s} {rj}/{tot} = {100*rj/tot:.0f}%")
    Path(".scratch/retrieval_policy.json").write_text(json.dumps(best))
