"""CorDP perception on TimesX tasks (adapted from cordp_tmmd.py): the LLM sees Toto's median forecast + leak-free documents and
proposes window multipliers. Two prompt variants:
  drcik   : the exact SYSTEM prompt used on Dr-CiK (scripts/cordp_train.py) -> pure transfer
  outlook : adapted to macro reports -- allows a horizon-wide level/trend correction when the
            documents give an explicit forward-looking direction for the target.
Writes timemmd/cards_<variant>.json {tid: {relevant, confidence, corrections:[[s,e,mult,rationale]]}}.
Usage: PYTHONPATH=<repo> <repo>/.venv/bin/python cordp_tmmd.py <variant> <parts,comma> [workers]
"""
import json, shutil, statistics, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError

VAR = sys.argv[1]; PARTS = sys.argv[2].split(","); W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ROOT = Path(__file__).resolve().parent / "timesx"
TASKS = {t["tid"]: t for t in json.loads((ROOT / "tasks.json").read_text())}
TOTO = json.loads((ROOT / "toto.json").read_text())
SUB = json.loads((ROOT / "extract_ids2.json").read_text())
llm = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                      timeout_seconds=600, cache_dir=str(ROOT / f"llm-cache-{VAR}")))
DRCIK = (
    "You are a forecasting-correction module. A strong statistical model has ALREADY produced "
    "a base forecast (median values with timestamps). Your ONLY job: decide whether the "
    "documented real-world events imply the TRUE values in specific sub-windows will differ "
    "from this base, and if so by what MULTIPLICATIVE factor.\n"
    "Rules:\n"
    "1. Anchor to the shown base scale. A correction multiplier is typically within [0.3, 3.0]; "
    "never produce values orders of magnitude off the base.\n"
    "2. Correct ONLY sub-windows where a document explicitly implies a change (holiday closure, "
    "outage, strike, promotion, weather event...). Give the timestamps and a one-line rationale "
    "quoting the document. Do NOT rescale the whole horizon on vague 'stabilized/steady' wording "
    "-- the base already has the level.\n"
    "3. If the documents do NOT clearly imply any deviation from the base, return "
    "\"relevant\": false with an empty corrections list. Do not guess.\n"
    "4. multiplier < 1 => lower than base, > 1 => higher.\n"
    'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, '
    '"end_timestamp": iso, "multiplier": number, "rationale": str}], "confidence": number}.')
OUTLOOK = (
    "You are a forecasting-correction module. A strong statistical model has ALREADY produced a base "
    "forecast (median values with timestamps) using ONLY the past numbers. You also get recent "
    "reports and news published BEFORE the forecast start (facts + forward-looking outlooks). Your ONLY "
    "job: decide whether this text implies the TRUE future values will be systematically higher or "
    "lower than the base in some window (possibly the whole horizon), and by what MULTIPLICATIVE factor.\n"
    "Rules:\n"
    "1. Anchor to the base. Multipliers are usually within [0.8, 1.25]; never beyond [0.5, 2.0].\n"
    "2. Only correct when a document gives a concrete, target-relevant direction (e.g. 'prices expected "
    "to rise in the next 1-3 months', 'cases increasing', 'drought worsening'). Quote it in the rationale. "
    "Remember the base already extrapolates the recent trend -- correct only for information the "
    "numbers alone could not know.\n"
    "3. If the text is generic, historical, contradictory or not about this target, return "
    "\"relevant\": false with no corrections. Do not guess.\n"
    "4. multiplier < 1 => lower than base, > 1 => higher. Windows use the base's timestamps.\n"
    'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, '
    '"end_timestamp": iso, "multiplier": number, "rationale": str}], "confidence": number}.')
SYSTEM = {"drcik": DRCIK, "outlook": OUTLOOK}[VAR]

def user(t, base):
    hv = t["history"]; fts = [x[:10] for x in t["future_ts"]]
    hist = {"typical_scale": round(statistics.median([abs(x) for x in hv]) or 1.0, 4),
            "overall_mean": round(statistics.mean(hv), 4), "last_values": [round(b, 4) for b in hv[-12:]]}
    docs = [{"document_id": k, "content": t[k][:3500]} for k in ("scenario", "holiday", "covariates") if t.get(k)]
    return json.dumps({"target_name": t["var"], "target_description": t["background"], "frequency": t["freq"],
                       "history_summary": hist, "documents": docs,
                       "base_forecast_median": [[fts[i], round(base[i], 4)] for i in range(len(fts))]}, ensure_ascii=False)

out_p = ROOT / f"cards_{VAR}.json"
cards = {k: v for k, v in (json.loads(out_p.read_text()) if out_p.exists() else {}).items() if "error" not in v}
lock = threading.Lock()
todo = [tid for p in PARTS for tid in SUB[p] if tid not in cards]

def run(tid):
    t = TASKS[tid]
    try:
        resp = llm.complete(system=SYSTEM, messages=[{"role": "user", "content": user(t, TOTO[tid])}], temperature=0.0)
        o = parse_json_object(resp.text)
    except (JsonExtractionError, Exception) as e:  # noqa: BLE001
        return tid, {"error": str(e)[:200]}
    try: conf = float(o.get("confidence"))
    except (TypeError, ValueError): conf = 0.0
    cs = []
    for c in (o.get("corrections") or []):
        try: cs.append([c.get("start_timestamp"), c.get("end_timestamp"), float(c.get("multiplier")), str(c.get("rationale", ""))[:300]])
        except (TypeError, ValueError): continue
    return tid, {"relevant": bool(o.get("relevant")), "confidence": conf, "corrections": cs}

done = 0
with ThreadPoolExecutor(W) as ex:
    for tid, card in ex.map(run, todo):
        with lock:
            cards[tid] = card; done += 1
            if done % 20 == 0 or done == len(todo):
                out_p.write_text(json.dumps(cards)); print(f"{VAR} {done}/{len(todo)}", flush=True)
out_p.write_text(json.dumps(cards))
