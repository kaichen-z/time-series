"""Perception for ResidualAdjust: text -> a SHAPED, GROUNDED residual (the advisor's
'mathematical formulation of the residue'). For each task the LLM sees the Toto base numbers
+ documents and returns residual specs: a WINDOW, a SHAPE (level/ramp/bump/decay) describing
the event's temporal profile, a fractional AMPLITUDE anchored on the base scale, and a
GROUNDING tag per magnitude (quote | regime | guess). The principle: the text supplies the
STRUCTURE and a POINTER to where the number lives; a magnitude tagged `guess` is dropped
(grounded=False) rather than invented.

Writes .scratch/residual_cards_{PARTITION}.json for ResidualAdjust to consume.
Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD PARTITION=train .venv/bin/python scripts/gen_residual_cards.py [LIMIT]
"""
from __future__ import annotations
import json, os, shutil, statistics, sys
from pathlib import Path

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import _parse
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore

ROOT = Path(".").resolve()
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS_DIR = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
PART = os.environ.get("PARTITION", "train")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
_SHAPES = {"level", "step", "ramp", "bump", "decay"}

ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
            ["partitions"][PART]["task_ids"])
if LIMIT:
    ids = ids[:LIMIT]
tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS_DIR, ids)}

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
llm = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                      timeout_seconds=900, cache_dir=str(ROOT / ".scratch/resid-cache")))

SYSTEM = (
    "You convert documented events into a MATHEMATICAL RESIDUAL on top of an existing base "
    "forecast (median values with timestamps). You describe HOW the true values depart from the "
    "base over a window: the window, the temporal SHAPE, and the size -- but you never invent a "
    "size that the evidence does not support.\n"
    "For each affected sub-window output one residual:\n"
    "  - start_timestamp / end_timestamp: the window the event acts on (keep it LOCAL, not the "
    "whole horizon; the base already carries the overall level).\n"
    "  - shape: the event's temporal profile -- 'level' (flat offset), 'ramp' (builds linearly), "
    "'bump' (rises to a peak mid-window then fades), 'decay' (jumps then relaxes back).\n"
    "  - amplitude: the PEAK fractional change vs the base scale in that window (signed; e.g. "
    "-0.4 = 40% below base at the peak). Anchor to the shown base numbers; keep |amplitude|<=1 "
    "unless a document states a larger figure.\n"
    "  - amplitude_grounding: 'quote' if a document states or directly implies the size; 'regime' "
    "if the size comes from a comparable historical pattern; 'guess' if neither. BE HONEST -- a "
    "'guess' will be DISCARDED, so only use quote/regime when a real source exists.\n"
    "  - quote: the document text that fixes the size/shape (empty if grounding='regime').\n"
    "  - tau_frac: for shape='decay', the relaxation time as a fraction of the window (0.1-1.0).\n"
    "If no document implies a departure from the base, return relevant=false with []. Do not guess.\n"
    'Output STRICT JSON: {"relevant": bool, "confidence": number, "residuals": [{"start_timestamp": '
    'iso, "end_timestamp": iso, "shape": str, "amplitude": number, "amplitude_grounding": '
    '"quote"|"regime"|"guess", "quote": str, "tau_frac": number}]}.')


def wd(ts):
    p = _parse(ts); return p.weekday() if p else -1


def build_user(task, base):
    n = task.numeric
    hv, hts = list(n.history_values), [str(x) for x in task.history_timestamps]
    fts = [str(x) for x in task.future_timestamps]
    we = [x for x, t in zip(hv, hts) if wd(t) >= 5]
    wk = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]
    hist = {"typical_scale": round(statistics.median([abs(x) for x in hv]) or 1.0, 4),
            "overall_mean": round(statistics.mean(hv), 4),
            "weekday_mean": round(statistics.mean(wk), 4) if wk else None,
            "weekend_mean": round(statistics.mean(we), 4) if we else None,
            "last_values": [round(x, 4) for x in hv[-12:]]}
    docs = [{"document_id": d.document_id, "content": d.content[:3500]} for d in task.documents]
    return json.dumps({"target_name": task.target_name,
                       "target_description": task.target_description, "frequency": n.frequency,
                       "history_summary": hist, "documents": docs,
                       "base_forecast_median": [[fts[i], round(base[i], 4)] for i in range(len(fts))]},
                      ensure_ascii=False)


cards = {}
for k, tid in enumerate(ids, 1):
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    resp = llm.complete(system=SYSTEM, messages=[{"role": "user", "content": build_user(t, base)}],
                        temperature=0.0)
    try:
        out = parse_json_object(resp.text)
    except JsonExtractionError:
        print(f"[{k}/{len(ids)}] {tid}: parse fail"); continue
    try:
        conf = float(out.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    specs = []
    for r in (out.get("residuals") or []):
        try:
            amp = float(r.get("amplitude"))
        except (TypeError, ValueError):
            continue
        shape = str(r.get("shape") or "level").lower()
        if shape not in _SHAPES:
            shape = "level"
        grounded = str(r.get("amplitude_grounding") or "guess").lower() != "guess"
        try:
            tau = float(r.get("tau_frac"))
        except (TypeError, ValueError):
            tau = 0.5
        specs.append({"start": r.get("start_timestamp"), "end": r.get("end_timestamp"),
                      "shape": shape, "amplitude": amp, "grounded": grounded,
                      "tau_frac": tau, "grounding": r.get("amplitude_grounding"),
                      "quote": (r.get("quote") or "")[:120]})
    cards[tid] = {"confidence": conf, "residuals": specs}
    Path(f".scratch/residual_cards_{PART}.json").write_text(json.dumps(cards, default=str))
    shp = [(s["shape"], round(s["amplitude"], 2), s["grounding"]) for s in specs]
    print(f"[{k}/{len(ids)}] {tid}: conf={conf:.2f} n={len(specs)} grounded={sum(s['grounded'] for s in specs)} {shp}",
          flush=True)
fs.close()
ng = sum(1 for c in cards.values() if any(s["grounded"] for s in c["residuals"]))
print(f"\ndone. {len(cards)} cards | with>=1 grounded residual: {ng}")
