"""State-timeline extraction for Dr-CiK (2026-09-27).
GPT reads all documents (no role/subtype labels) and returns a timeline of intervals over the HISTORY and
the FORECAST window:
  kind = "history_anomaly"  : something in the past distorted the readings (maintenance, outage, sensor
                              error, one-off promotion, grid issue) -- with recurs_in_future
  kind = "state"            : a recurring condition whose on/off timeline is given (weather clear/cloudy,
                              holiday, event, occupancy) -- occurs in history and/or forecast
  kind = "future_event"     : a stated future change without a past analogue (e.g. new permanent level)
Every interval: start, end (timestamps), label (short canonical name), direction (up/down/none relative
to normal), recurs_in_future (bool), quote.  Magnitudes are NOT asked: Numerical measures them from
the history.  Output: .scratch/self_evolving/timeline_<model>.json"""
import json, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASKS = Path("external/Dr-CiK/full-download/Dr-CiK_public/tasks")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-6-luna"
PARTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["train"]
OUT = Path(f".scratch/self_evolving/timeline_{MODEL}.json")
INSTR = """You help a numeric forecaster use documents. The series has a HISTORY period and a FORECAST period
(both timestamp ranges given). Read ALL documents (many are distractors about other entities, other
periods, or generic background) and build a precise timeline of intervals that affect THIS series:
- kind "history_anomaly": past periods where readings were distorted by something that should not be
  learned as normal (maintenance, outage, sensor error, one-off promotion, technical issue). Say whether
  it recurs in the forecast period (recurs_in_future) -- documents often state e.g. "will not be in
  maintenance in the future"; if it is periodic and nothing says it stops, it recurs.
- kind "state": a condition switching on/off at given times (e.g. weather clear vs cloudy, holiday, event,
  store open/closed). List EVERY on-interval you can find, in the history and in the forecast, with exact
  start/end timestamps at the series' resolution; use the same short label for the same condition.
- kind "future_event": a change stated for the forecast period without a past analogue.
For each interval give direction (up/down/none) = effect on the series relative to normal, and a short
verbatim quote. Do not invent magnitudes. Use ISO timestamps. If nothing applies, return an empty list."""
SCHEMA = {"type": "object", "properties": {"intervals": {"type": "array", "items": {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["history_anomaly", "state", "future_event"]},
    "label": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"},
    "direction": {"type": "string", "enum": ["up", "down", "none"]}, "recurs_in_future": {"type": "boolean"},
    "quote": {"type": "string"}},
    "required": ["kind", "label", "start", "end", "direction", "recurs_in_future", "quote"], "additionalProperties": False}}},
    "required": ["intervals"], "additionalProperties": False}


def prompt(tid):
    d = json.loads((TASKS / f"{tid}.json").read_text()); s = d["series"]
    ent = ((d.get("showcase") or {}).get("entity") or {})
    p = {"entity": ent.get("name"), "entity_type": ent.get("type"), "target": d["task_metadata"].get("target_description"),
         "frequency": d["task_metadata"].get("frequency"),
         "history_period": [s["history_timestamps"][0], s["history_timestamps"][-1]],
         "forecast_period": [s["future_timestamps"][0], s["future_timestamps"][-1]],
         "documents": [{"id": x["document_id"], "text": (x.get("content") or "")[:2500]} for x in d["documents"]]}
    return INSTR + "\nTASK:\n" + json.dumps(p, ensure_ascii=False)


def call(pr):
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"; op = Path(td) / "o.json"; sp.write_text(json.dumps(SCHEMA))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", MODEL, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=pr, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1500)
        return json.loads(op.read_text())


def main():
    split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
    tids = [t for p in PARTS for t in split[p]["task_ids"]]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [t for t in tids if t not in out]
    def run(t):
        for _ in range(2):
            try: return t, call(prompt(t))["intervals"]
            except Exception: continue
        return t, None
    with ThreadPoolExecutor(8) as ex:
        for i, (t, r) in enumerate(ex.map(run, todo), 1):
            if r is not None: out[t] = r
            if i % 20 == 0: OUT.write_text(json.dumps(out)); print(i, len(todo), flush=True)
    OUT.write_text(json.dumps(out)); print("done", len(out), flush=True)


if __name__ == "__main__":
    main()
