"""Precise-time re-extraction of Dr-CiK correction cards with GPT (2026-09-27).
One call per task; the model sees all documents (first 2000 chars each, no role/subtype labels), the
history tail, and every forecast step's timestamp with Toto's value, and must return corrections whose
start/end are exact forecast-step timestamps.  Output mirrors the old CorDP card format:
{tid: {confidence, corrections: [[start, end, multiplier], ...]}}."""
import json, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASKS = Path("external/Dr-CiK/full-download/Dr-CiK_public/tasks")
OUT = Path(".scratch/self_evolving/cards_v9.json")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-6-luna"
INSTR = """You correct a strong numeric forecast using documents. The base forecast gives a value for every
forecast step (timestamps listed). Many documents are distractors (other entities, other periods,
background, methodology). Report corrections only for concrete events that the documents tie to THIS
entity and variable and that fall inside the forecast steps. For each correction give the EXACT first and
last affected forecast-step timestamps (copy them from the list; affect only the steps the event really
covers, e.g. the closure hours, not the whole day), a multiplicative factor applied to the base on those
steps (e.g. 0.5 = halve, 1.3 = +30%), your confidence in [0,1], and a short verbatim quote. If nothing
qualifies, return no corrections."""
SCHEMA = {"type": "object", "properties": {
    "confidence": {"type": "number"},
    "corrections": {"type": "array", "items": {"type": "object", "properties": {
        "start": {"type": "string"}, "end": {"type": "string"}, "multiplier": {"type": "number"},
        "confidence": {"type": "number"}, "quote": {"type": "string"}},
        "required": ["start", "end", "multiplier", "confidence", "quote"], "additionalProperties": False}}},
    "required": ["confidence", "corrections"], "additionalProperties": False}


def task_prompt(tid, base):
    d = json.loads((TASKS / f"{tid}.json").read_text()); s = d["series"]
    ent = ((d.get("showcase") or {}).get("entity") or {})
    payload = {"entity": ent.get("name"), "entity_type": ent.get("type"),
               "target": d["task_metadata"].get("target_description"), "frequency": d["task_metadata"].get("frequency"),
               "history_tail": [[a, round(b, 4)] for a, b in zip(s["history_timestamps"][-12:], s["history_values"][-12:])],
               "forecast_steps": [[a, round(b, 4)] for a, b in zip(s["future_timestamps"], base)],
               "documents": [{"id": x["document_id"], "text": (x.get("content") or "")[:2000]} for x in d["documents"]]}
    return INSTR + "\nTASK:\n" + json.dumps(payload, ensure_ascii=False)


def call(prompt):
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"; op = Path(td) / "o.json"; sp.write_text(json.dumps(SCHEMA))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", MODEL, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=prompt, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
        return json.loads(op.read_text())


def main():
    D = json.load(open(".scratch/self_evolving/nrd_cache.json")); bases = {d["tid"]: d["fc"]["toto_2_0"] for d in D}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [t for t in bases if t not in out]
    def run(t):
        for _ in range(2):
            try:
                r = call(task_prompt(t, bases[t]))
                return t, {"confidence": r["confidence"], "corrections": [[c["start"], c["end"], c["multiplier"], c["confidence"], c["quote"][:200]] for c in r["corrections"]]}
            except Exception:
                continue
        return t, None
    with ThreadPoolExecutor(8) as ex:
        for i, (t, r) in enumerate(ex.map(run, todo), 1):
            if r is not None: out[t] = r
            if i % 20 == 0: OUT.write_text(json.dumps(out)); print(i, len(todo), flush=True)
    OUT.write_text(json.dumps(out)); print("done", len(out), "tasks;", sum(1 for v in out.values() if v["corrections"]), "with corrections", flush=True)


if __name__ == "__main__":
    main()
