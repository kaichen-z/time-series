"""Retrieval extraction program for the N/R/D engine (2026-09-27).

An extraction program = (document selector genome, GPT instruction text).  The selector ranks each
task's documents with interpretable, label-free features (dates inside the forecast window, entity
match, event/baseline vocabulary) and keeps the top-k excerpts.  GPT then reads only those excerpts
and returns events {start, end, direction, strength, confidence, type}; it never outputs a numeric
multiplier -- Numerical converts strength into magnitude from the history.  Document role/subtype
labels are evaluation labels and are never read here.
"""
from __future__ import annotations
import hashlib, json, re, subprocess, tempfile, statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

TASKS = Path("external/Dr-CiK/full-download/Dr-CiK_public/tasks")
CACHE = Path(".scratch/self_evolving/extract_cache"); CACHE.mkdir(parents=True, exist_ok=True)
EVENT = ["holiday", "closure", "closed", "outage", "strike", "promotion", "maintenance", "shutdown", "festival",
         "storm", "flood", "disrupt", "suspend", "repair", "scheduled", "effective", "cancel", "surge", "spike",
         "drop", "delay", "launch", "event", "expected", "forecast", "will"]
BASE = ["baseline", "methodology", "parameter", "calibration", "specification", "typical", "normal operating",
        "overview", "definition", "historical", "average", "profile"]
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}

INSTR0 = """You are the Retrieval agent of a forecasting system. A strong numeric model has already produced a
base forecast from the history. Read the document excerpts and decide whether they describe a concrete,
dated, forward-looking event that will move the target series away from the base forecast inside the
forecast window. Ignore generic background, methodology/definition text, historical descriptions, events
outside the window, and events about other entities. For each such event give the affected window
(timestamps inside the forecast window), the direction (up or down), a strength class (small, medium,
large) and your confidence in [0,1]. If nothing qualifies, return an empty event list."""


def load_raw(tid):
    return json.loads((TASKS / f"{tid}.json").read_text())


def _dates(text):
    out = []
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", text):
        try: out.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError: pass
    for m in re.finditer(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})", text, re.I):
        try: out.append(datetime(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))))
        except ValueError: pass
    return out


def _ts(s):
    s = str(s).replace("Z", "").replace("T", " ")[:19]
    try: return datetime.fromisoformat(s)
    except ValueError: return None


def select_docs(raw, g):
    """Retrieval genome part 1: label-free document ranking."""
    fut = [t for t in (_ts(x) for x in raw["series"]["future_timestamps"]) if t]
    lo, hi = (min(fut), max(fut)) if fut else (None, None)
    ent = ((raw.get("showcase") or {}).get("entity") or {}).get("name") or raw.get("domain", "")
    etoks = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", ent)]
    scored = []
    for d in raw["documents"]:
        txt = d.get("content") or ""; low = txt.lower()
        ds = _dates(txt); pad = timedelta(days=g["pad_days"])
        inwin = sum(1 for x in ds if lo and lo - pad <= x <= hi + pad)
        f = {"inwin": min(3, inwin), "anydate": min(3, len(ds)), "entity": sum(1 for w in etoks if w in low) / max(1, len(etoks)),
             "event": min(5, sum(low.count(w) for w in EVENT)), "base": min(5, sum(low.count(w) for w in BASE))}
        scored.append((sum(g["w"][k] * v for k, v in f.items()), d["document_id"], txt))
    scored.sort(key=lambda x: -x[0])
    return [(i, t[:g["chars"]]) for _s, i, t in scored[:g["top_k"]]]


def sel0():
    return {"top_k": 6, "chars": 1500, "pad_days": 1,
            "w": {"inwin": 2.0, "anydate": 0.3, "entity": 1.0, "event": 0.5, "base": -0.5}}


def compact(tid, raw, base, g):
    s = raw["series"]; h = s["history_values"]
    return {"tid": tid, "target": raw["task_metadata"].get("target_description", ""),
            "frequency": raw["task_metadata"].get("frequency"),
            "history_tail": [[a, round(b, 4)] for a, b in zip(s["history_timestamps"][-8:], h[-8:])],
            "forecast_window": [s["future_timestamps"][0], s["future_timestamps"][-1]],
            "base_forecast_head": [round(x, 4) for x in base[:6]],
            "documents": [{"id": i, "text": t} for i, t in select_docs(raw, g)]}


SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {
    "type": "object", "properties": {
        "tid": {"type": "string"},
        "events": {"type": "array", "items": {"type": "object", "properties": {
            "start": {"type": "string"}, "end": {"type": "string"},
            "direction": {"type": "string", "enum": ["up", "down"]},
            "strength": {"type": "string", "enum": ["small", "medium", "large"]},
            "confidence": {"type": "number"}, "type": {"type": "string"}},
            "required": ["start", "end", "direction", "strength", "confidence", "type"], "additionalProperties": False}}},
    "required": ["tid", "events"], "additionalProperties": False}}},
    "required": ["results"], "additionalProperties": False}


def _call(prompt, model):
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"; op = Path(td) / "o.json"; sp.write_text(json.dumps(SCHEMA))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", model, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=prompt, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        return json.loads(op.read_text())


def extract(program, tids, bases, model="gpt-6-luna", batch=8, workers=6):
    """Returns {tid: [event,...]} for the program; cached per (program, tid)."""
    key = hashlib.sha256(json.dumps([program, model], sort_keys=True).encode()).hexdigest()[:16]
    cf = CACHE / f"{key}.json"; cache = json.loads(cf.read_text()) if cf.exists() else {}
    todo = [t for t in tids if t not in cache]
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    def run(group):
        tasks = [compact(t, load_raw(t), bases[t], program["selector"]) for t in group]
        prompt = program["instructions"] + "\nReturn exactly one result per task id, no prose.\nINPUT TASKS:\n" + json.dumps(tasks, ensure_ascii=False)
        for attempt in range(2):
            try:
                r = _call(prompt, model); return {x["tid"]: x["events"] for x in r["results"] if x["tid"] in group}
            except Exception:
                continue
        return {}
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(run, groups): cache.update(res)
    cf.write_text(json.dumps(cache))
    return {t: cache.get(t, []) for t in tids}, key
