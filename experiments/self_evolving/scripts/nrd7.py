"""Evidence-grounded, consensus extraction for Dr-CiK (2026-09-27).

Each of several GPT models extracts events with an evidence quote, the entity and the evidence date.
A program verifies every event (quote really occurs in the task documents, evidence date inside the
forecast window +- pad, entity matches the target); only verified events whose direction and window
agree across >= `need` models survive.  Magnitude = k[strength] * sigma(history) fitted on Train
(Numerical), Decision = confidence threshold + trust gate (as in nrd5).  Document role/subtype labels
are never read.  Dev once; test99 exploratory.
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd_coevolve as C
import nrd3 as R3
import nrd4 as N4
import nrd5 as N5
import nrd5_extract as X
from evolving_loop.adjustment.post_adjust import horizon_window_mask

INSTR = """You are the Retrieval agent of a forecasting system. A strong numeric model already produced a base
forecast. From the document excerpts, report only concrete, dated, forward-looking events that will move
THIS target series (the named entity and variable) away from the base forecast inside the forecast window.
Many documents are distractors: other entities, other time periods, generic background or methodology.
For each event give: window (timestamps inside the forecast window), direction (up/down), strength
(small/medium/large), confidence in [0,1], the entity it concerns, the date the document gives for the
event (YYYY-MM-DD), and an exact short quote (10-40 words) copied verbatim from the document that states
the event. If you cannot quote such text, do not report the event. Empty list if nothing qualifies."""

SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {
    "type": "object", "properties": {"tid": {"type": "string"}, "events": {"type": "array", "items": {
        "type": "object", "properties": {
            "start": {"type": "string"}, "end": {"type": "string"},
            "direction": {"type": "string", "enum": ["up", "down"]},
            "strength": {"type": "string", "enum": ["small", "medium", "large"]},
            "confidence": {"type": "number"}, "entity": {"type": "string"},
            "evidence_date": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["start", "end", "direction", "strength", "confidence", "entity", "evidence_date", "quote"],
        "additionalProperties": False}}}, "required": ["tid", "events"], "additionalProperties": False}}},
    "required": ["results"], "additionalProperties": False}
CACHE = Path(".scratch/self_evolving/extract_cache_v7"); CACHE.mkdir(parents=True, exist_ok=True)
SEL = {"top_k": 10, "chars": 2500, "pad_days": 3, "w": {"inwin": 3.0, "anydate": 0.3, "entity": 1.5, "event": 0.5, "base": -0.5}}


def call(prompt, model):
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"; op = Path(td) / "o.json"; sp.write_text(json.dumps(SCHEMA))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", model, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=prompt, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        return json.loads(op.read_text())


def extract(model, tids, bases, batch=6, workers=6):
    cf = CACHE / f"{model}.json"; cache = json.loads(cf.read_text()) if cf.exists() else {}
    todo = [t for t in tids if t not in cache]; groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    def run(g):
        tasks = [X.compact(t, X.load_raw(t), bases[t], SEL) for t in g]
        prompt = INSTR + "\nReturn exactly one result per task id, no prose.\nINPUT TASKS:\n" + json.dumps(tasks, ensure_ascii=False)
        for _ in range(2):
            try: return {x["tid"]: x["events"] for x in call(prompt, model)["results"] if x["tid"] in g}
            except Exception: continue
        return {}
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(run, groups): cache.update(r)
    cf.write_text(json.dumps(cache))
    return {t: cache.get(t, []) for t in tids}


def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def verify(tid, e, pad_days=3):
    raw = X.load_raw(tid); why = []
    q = norm(e.get("quote", ""))
    docs = [norm(d.get("content") or "") for d in raw["documents"]]
    qt = q.split()
    if len(qt) < 5: why.append("short_quote")
    elif not any(q in d for d in docs):
        # tolerate small copy differences: >= 90% of quote 5-grams must occur in one document
        grams = {" ".join(qt[i:i + 5]) for i in range(len(qt) - 4)}
        if not any(sum(g in d for g in grams) >= 0.9 * len(grams) for d in docs): why.append("quote_not_found")
    fut = [t for t in (X._ts(x) for x in raw["series"]["future_timestamps"]) if t]
    ed = X._ts(e.get("evidence_date", ""))
    if ed is None or not fut or not (min(fut) - timedelta(days=pad_days) <= ed <= max(fut) + timedelta(days=pad_days)):
        why.append("date_outside_window")
    ent = ((raw.get("showcase") or {}).get("entity") or {}).get("name") or ""
    et, ee = set(norm(ent).split()), set(norm(e.get("entity", "")).split())
    if et and len(et & ee) / len(et) < 0.5: why.append("entity_mismatch")
    return why


def consensus(per_model, need):
    """keep events of the first model that are matched (same direction, overlapping window) by >= need models."""
    out = []
    pool = [(m, ev) for m, evs in per_model.items() for ev in evs]
    used = set()
    for i, (m, ev) in enumerate(pool):
        if i in used: continue
        grp = [(i, m, ev)]
        for j, (m2, ev2) in enumerate(pool):
            if j <= i or j in used or m2 == m or m2 in [g[1] for g in grp]: continue
            if ev2["dir"] == ev["dir"] and ev2["s"] < ev["e"] and ev["s"] < ev2["e"]: grp.append((j, m2, ev2))
        if len(grp) >= need:
            for g in grp: used.add(g[0])
            st = sorted(x[2]["st"] for x in grp)[len(grp) // 2]
            out.append(dict(s=min(x[2]["s"] for x in grp), e=max(x[2]["e"] for x in grp), dir=ev["dir"], st=st,
                            conf=sum(x[2]["conf"] for x in grp) / len(grp), typ=ev["typ"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-6-luna,gpt-6-sol,gpt-5.6-terra"); ap.add_argument("--need", type=int, default=2)
    ap.add_argument("--no-verify", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="train,dev,public_test")
    a = ap.parse_args(); models = a.models.split(",")
    C.FOLD_MODE = "strat"
    TH = json.load(open(".scratch/self_evolving/toto_hindcast.json"))
    D = [R3.prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))]
    bases = {d["tid"]: d["fc"]["toto_2_0"] for d in D}
    parts = a.parts.split(","); DS = [d for d in D if d["part"] in parts]
    stats = {}
    raw_by_model = {m: extract(m, [d["tid"] for d in DS], bases) for m in models}
    for d in DS:
        per = {}
        for m in models:
            evs = []
            for e in raw_by_model[m][d["tid"]]:
                why = [] if a.no_verify else verify(d["tid"], e)
                for w in why or ["ok"]: stats[w] = stats.get(w, 0) + 1
                if not why: evs += N5.to_corr(d, [e])
            per[m] = evs
        d["ev"] = consensus(per, a.need) if a.need > 1 else [x for evs in per.values() for x in evs]
    print("verification:", stats, flush=True)
    train = [d for d in DS if d["part"] == "train"]
    # precision of surviving events on Train with the Train-fitted magnitude
    trust = N4.trust_profile(train); p = N5.fit_nd(train, trust)
    gs = [N5.corr_gain(d, c, p["k"][c["st"]] if p["k"][c["st"]] > 0 else 0.3) for d in train for c in d["ev"]]
    print(f"train events {len(gs)} helpful {sum(g > 0 for g in gs)} ({(sum(g > 0 for g in gs) / max(1, len(gs))):.0%})  nd {p}", flush=True)
    f, per_f, _ = N5.cv_score(train, C.gfolds(train, 3))
    print(f"train CV {[round(s['joint_gain'] * 100, 2) for s in per_f]} W/R {[(s['wins'], s['regressions']) for s in per_f]}", flush=True)
    res = dict(stats=stats, nd=p, cv=[s for s in per_f], train=N5.summarize(train, p, trust))
    for part in ("dev", "public_test"):
        ds = [d for d in DS if d["part"] == part]
        if ds:
            s = N5.summarize(ds, p, trust); res[part] = s
            print(f"{part}{' (exploratory)' if part == 'public_test' else ''}: sMAE {s['smae_gain']:+.2%} sRMSE {s['srmse_gain']:+.2%} W/R {s['wins']}/{s['regressions']} events {sum(len(d['ev']) for d in ds)}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
