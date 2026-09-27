"""State-timeline v2 with gt_evidence-supervised Retrieval evolution (2026-09-27).

Schema v2 separates measurement errors / one-off disturbances (to repair) from real, lasting effects,
and supports periodic windows ("daily 08:00-14:00 from A to B").  Retrieval's instruction text is evolved
with GPT as mutator; fitness on a Train minibatch = evidence F1 against annotations.gt_evidence (Train
labels only -- never used on dev/test) + history-repair forecast gain.  Numerical = history repair
(same-phase median imputation) + Toto re-forecast; Decision = repair gate fitted on Train.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, statistics, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
import numpy as np

TASKS = Path("external/Dr-CiK/full-download/Dr-CiK_public/tasks")
CACHE = Path(".scratch/self_evolving/tl2_cache"); CACHE.mkdir(parents=True, exist_ok=True)

INSTR0 = """You help a numeric forecaster use documents. The series has a HISTORY period and a FORECAST period.
Read ALL documents; many are distractors (other entities, other periods, generic background). Build the
timeline of intervals that affect THIS series, each with a verbatim evidence quote:
- kind "measurement_error": readings were wrong (sensor fault, logging/telemetry corruption, offline).
- kind "one_off_disturbance": a real but temporary past event that will not repeat (outage, promotion,
  technical issue, maintenance that has stopped).
- kind "recurring_pattern": a periodic effect (e.g. maintenance every 14 days, daily downtime 08:00-14:00).
- kind "state": a real condition switching on/off at given times (weather, holiday, event) -- real
  effects, NOT errors; list every on-interval in history and forecast.
- kind "future_event": a stated change for the forecast period.
Give start/end (ISO timestamps), recurrence (none/daily/weekly/every_n_days) with daily_start/daily_end
(HH:MM) and every_n_days when periodic, recurs_in_future (true/false), direction (up/down/none) and quote.
Do not invent magnitudes. Empty list if nothing applies."""

SCHEMA = {"type": "object", "properties": {"intervals": {"type": "array", "items": {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["measurement_error", "one_off_disturbance", "recurring_pattern", "state", "future_event"]},
    "label": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"},
    "recurrence": {"type": "string", "enum": ["none", "daily", "weekly", "every_n_days"]},
    "daily_start": {"type": "string"}, "daily_end": {"type": "string"}, "every_n_days": {"type": "number"},
    "recurs_in_future": {"type": "boolean"}, "direction": {"type": "string", "enum": ["up", "down", "none"]},
    "quote": {"type": "string"}},
    "required": ["kind", "label", "start", "end", "recurrence", "daily_start", "daily_end", "every_n_days",
                 "recurs_in_future", "direction", "quote"], "additionalProperties": False}}},
    "required": ["intervals"], "additionalProperties": False}


def raw(tid): return json.loads((TASKS / f"{tid}.json").read_text())


def prompt(instr, tid):
    d = raw(tid); s = d["series"]; ent = ((d.get("showcase") or {}).get("entity") or {})
    p = {"entity": ent.get("name"), "entity_type": ent.get("type"), "target": d["task_metadata"].get("target_description"),
         "frequency": d["task_metadata"].get("frequency"),
         "history_period": [s["history_timestamps"][0], s["history_timestamps"][-1]],
         "forecast_period": [s["future_timestamps"][0], s["future_timestamps"][-1]],
         "documents": [{"id": x["document_id"], "text": (x.get("content") or "")[:2500]} for x in d["documents"]]}
    return instr + "\nTASK:\n" + json.dumps(p, ensure_ascii=False)


def codex(pr, schema, model, timeout=1500):
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"; op = Path(td) / "o.json"; sp.write_text(json.dumps(schema))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", model, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=pr, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return json.loads(op.read_text())


def extract(instr, tids, model="gpt-6-luna", workers=8):
    key = hashlib.sha256((instr + model).encode()).hexdigest()[:16]
    cf = CACHE / f"{key}.json"; cache = json.loads(cf.read_text()) if cf.exists() else {}
    todo = [t for t in tids if t not in cache]
    def run(t):
        for _ in range(2):
            try: return t, codex(prompt(instr, t), SCHEMA, model)["intervals"]
            except Exception: continue
        return t, []
    with ThreadPoolExecutor(workers) as ex:
        for t, r in ex.map(run, todo): cache[t] = r
    cf.write_text(json.dumps(cache)); return {t: cache[t] for t in tids}, key


# ----------------------------------------------------------------------------- evidence scoring (Train only)
def toks(s): return set(w for w in re.findall(r"[a-z0-9:]+", s.lower()) if len(w) > 2)


def evidence_f1(ivs, tid):
    gt = [e["evidence"] for e in raw(tid)["annotations"]["gt_evidence"]]
    act = [g for g in gt if re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|every|daily|periodic|will|expect", g.lower())]   # actionable items
    if not act: return None
    qs = [toks(i["quote"]) for i in ivs]
    def match(q, g): tg = toks(g); return len(q & tg) / max(1, min(len(q), len(tg))) >= 0.6
    rec = sum(1 for g in act if any(match(q, g) for q in qs)) / len(act)
    prec = (sum(1 for q in qs if any(match(q, g) for g in gt)) / len(qs)) if qs else 0.0
    return 2 * rec * prec / (rec + prec + 1e-9), rec, prec, [g for g in act if not any(match(q, g) for q in qs)][:3], \
        [i["quote"][:150] for i, q in zip(ivs, qs) if not any(match(q, g) for g in gt)][:3]


# ----------------------------------------------------------------------------- Numerical: masks + repair
def ts(x):
    s = str(x).replace("Z", "").replace("T", " ")[:19]
    for cand in (s, s[:16], s[:10]):
        try: return datetime.fromisoformat(cand)
        except ValueError: pass
    return None


def hm_time(x):
    try: h, m = str(x).split(":")[:2]; return dtime(int(h), int(m))
    except Exception: return None


def mask(stamps, iv):
    s, e = ts(iv["start"]), ts(iv["end"])
    if s is None or e is None: return np.zeros(len(stamps), bool)
    if e < s: s, e = e, s
    if e.hour == 0 and e.minute == 0 and len(str(iv["end"])) <= 10: e = e + timedelta(days=1) - timedelta(seconds=1)
    ds, de = hm_time(iv.get("daily_start")), hm_time(iv.get("daily_end"))
    out = []
    for t in stamps:
        ok = t is not None and s <= t <= e
        if ok and iv["recurrence"] == "daily" and ds and de: ok = ds <= t.time() <= de if ds <= de else (t.time() >= ds or t.time() <= de)
        if ok and iv["recurrence"] == "every_n_days" and iv.get("every_n_days"):
            n = int(iv["every_n_days"]) or 1; ok = ((t - s).days % n) == 0 or iv["kind"] != "recurring_pattern"
        out.append(ok)
    return np.array(out)


REPAIR_KINDS = {"measurement_error", "one_off_disturbance", "recurring_pattern"}


def period_of(freq):
    f = (freq or "").lower()
    if "5 min" in f: return 288
    if "min" in f: return 60
    if "hour" in f: return 24
    if "day" in f: return 7
    return 1


def repair(tid, ivs, kinds=REPAIR_KINDS):
    d = raw(tid); y = np.array(d["series"]["history_values"], float)
    stamps = [ts(x) for x in d["series"]["history_timestamps"]]; bad = np.zeros(len(y), bool)
    for iv in ivs:
        if iv["kind"] in kinds and not iv["recurs_in_future"]: bad |= mask(stamps, iv)
    if not bad.any() or bad.mean() > 0.95: return None, 0.0
    p = period_of(d["task_metadata"].get("frequency")); z = y.copy(); n = len(y); good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        same = [y[j] for j in range(i % p, n, p) if not bad[j]] if p > 1 else []
        z[i] = float(np.median(same)) if len(same) >= 2 else float(np.interp(i, good, y[good]))
    return z.tolist(), float(bad.mean())


# ----------------------------------------------------------------------------- GPT mutator
MUT = """You improve the instruction text for an LLM that extracts a timeline of anomalies/states from
documents for a forecasting system. Below: the CURRENT INSTRUCTIONS and TRAIN-ONLY FEEDBACK from several
tasks -- annotated evidence sentences the extractor MISSED, and quotes it extracted that are NOT part of
the annotated evidence (likely distractors), plus forecast outcome of repairing the history. Rewrite the
instructions to fix these systematic errors (e.g. how to recognise distractors, how to express periodic
windows, when an effect is an error vs a real lasting effect). Keep the kinds, fields and output format
unchanged, stay general (no task-specific entities or dates), under 350 words.
Return JSON {"instructions": "..."}."""


def mutate(instr, feedback, model="gpt-6-sol"):
    schema = {"type": "object", "properties": {"instructions": {"type": "string"}}, "required": ["instructions"], "additionalProperties": False}
    return codex(MUT + "\nCURRENT INSTRUCTIONS:\n" + instr + "\nFEEDBACK:\n" + json.dumps(feedback, ensure_ascii=False)[:12000], schema, model, 900)["instructions"]
