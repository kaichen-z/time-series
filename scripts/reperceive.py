"""Live LLM RE-PERCEPTION: produce better multipliers to attack the +22% ceiling.

The 16 HARM tasks are (11 wrong-direction / 5 wrong-magnitude). Two improvements over the cached
CorDP perception:
  1. SELF-CONSISTENCY: call the LLM K times (temp>0) and keep a step only if >=ceil(K/2) calls agree
     it should be corrected AND agree on direction; the multiplier is the median. Consensus doubles
     as a confounder filter -- a real event gets agreement across calls, a hallucination/confounder
     does not.
  2. MAGNITUDE PROVENANCE: instruct the LLM to use an explicit quoted number if the document states
     one, else a historical analog from the shown history, else a CONSERVATIVE multiplier near 1 --
     never guess a large magnitude ("under-correct beats over-correct").

Re-perceives the test tasks that currently have a correction (the comparison set), writes new cards,
and reports new-vs-old multipliers under trust-all + kernel on test99.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD [K=3] .venv/bin/python scripts/reperceive.py
"""
from __future__ import annotations
import json, os, shutil, statistics
from pathlib import Path

from common.llm import ClaudeCLIClient, ClaudeCLIConfig, parse_json_object, JsonExtractionError
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.adjustment.post_adjust import _parse, horizon_window_mask, apply_bounded_delta
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
K = int(os.environ.get("K", "3"))
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
llm = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                      timeout_seconds=900, cache_dir=str(ROOT / ".scratch/reperceive-cache")))

SYSTEM = (
    "You are a forecasting-correction module. A strong statistical model has ALREADY produced a base "
    "forecast (median values with timestamps). Decide whether the documented real-world events imply "
    "the TRUE values in specific sub-windows differ from this base, and by what MULTIPLICATIVE factor.\n"
    "Rules:\n"
    "1. Anchor to the shown base scale. Correct ONLY sub-windows where a document EXPLICITLY implies a "
    "change (holiday closure, outage, strike, promotion, weather...). Quote the document in a rationale. "
    "Do NOT rescale the whole horizon on vague 'steady/stabilized' wording.\n"
    "2. MAGNITUDE PROVENANCE -- where the number comes from, in priority order:\n"
    "   (a) if the document states an explicit numeric change (e.g. 'fell 30%'), use exactly that;\n"
    "   (b) else estimate the size from the HISTORY analog provided (compare the relevant regime's "
    "level, e.g. weekend vs weekday or the recent level, to the overall level);\n"
    "   (c) if NEITHER a quote NOR a clear historical analog gives the size, DO NOT guess a large "
    "factor -- return a conservative multiplier within [0.9, 1.1], or omit the correction. "
    "Under-correcting is much better than over-correcting.\n"
    "3. If the documents do not clearly imply any deviation, return \"relevant\": false, empty list.\n"
    "4. multiplier < 1 => lower than base, > 1 => higher.\n"
    'Output STRICT JSON: {"relevant": bool, "corrections": [{"start_timestamp": iso, "end_timestamp": '
    'iso, "multiplier": number, "provenance": "quote|analog|conservative", "rationale": str}], '
    '"confidence": number}.')


def wd(ts):
    p = _parse(ts); return p.weekday() if p else -1


def build_user(task, base):
    n = task.numeric; hv = list(n.history_values); hts = [str(x) for x in task.history_timestamps]
    fts = [str(x) for x in task.future_timestamps]
    we = [x for x, t in zip(hv, hts) if wd(t) >= 5]; wk = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]
    hist = {"typical_scale": round(statistics.median([abs(x) for x in hv]) or 1.0, 4),
            "overall_mean": round(statistics.mean(hv), 4),
            "weekday_mean": round(statistics.mean(wk), 4) if wk else None,
            "weekend_mean": round(statistics.mean(we), 4) if we else None,
            "last_values": [round(x, 4) for x in hv[-12:]]}
    docs = [{"document_id": d.document_id, "content": d.content[:3500]} for d in task.documents]
    return json.dumps({"target_name": task.target_name, "target_description": task.target_description,
                       "frequency": n.frequency, "history_summary": hist, "documents": docs,
                       "base_forecast_median": [[fts[i], round(base[i], 4)] for i in range(len(fts))]},
                      ensure_ascii=False)


def one_call(task, base, temp):
    resp = llm.complete(system=SYSTEM, messages=[{"role": "user", "content": build_user(task, base)}],
                        temperature=temp)
    try:
        out = parse_json_object(resp.text)
    except JsonExtractionError:
        return []
    fts = [str(x) for x in task.future_timestamps]; H = len(fts)
    step = [None] * H
    for c in (out.get("corrections") or []):
        try:
            m = float(c.get("multiplier"))
        except (TypeError, ValueError):
            continue
        for i, on in enumerate(horizon_window_mask(fts, str(c.get("start_timestamp")), str(c.get("end_timestamp")))):
            if on:
                step[i] = m
    return step   # per-step multiplier (or None)


def consensus(steps, H):
    """Keep step i only if >= ceil(K/2) calls corrected it AND agree on direction; mult = median."""
    need = (len(steps) + 1) // 2
    out = [1.0] * H
    for i in range(H):
        ms = [s[i] for s in steps if i < len(s) and s[i] is not None]
        if len(ms) < need:
            continue
        up = sum(1 for m in ms if m > 1); dn = sum(1 for m in ms if m < 1)
        if up and dn:                      # direction disagreement -> drop (confounder filter)
            continue
        out[i] = statistics.median(ms)
    return out


def steps_to_windows(step_mult, fts):
    wins = []; i = 0; H = len(step_mult)
    while i < H:
        if abs(step_mult[i] - 1.0) < 1e-9:
            i += 1; continue
        j = i
        while j < H and abs(step_mult[j] - step_mult[i]) < 1e-9:
            j += 1
        wins.append([fts[i], fts[j - 1], step_mult[i]]); i = j
    return wins


old_cards = json.loads((ROOT / ".scratch/cordp_cards_public_test.json").read_text())
ids = [t for t in SPLIT["public_test"]["task_ids"] if t in old_cards and (old_cards[t].get("corrections"))]
ONLY = os.environ.get("ONLY", "")
if ONLY:
    keep = set(ONLY.split(","))
    ids = [t for t in ids if t in keep]
IDS = os.environ.get("IDS", "")
if IDS:
    ids = IDS.split(",")                       # explicit override (e.g. the no-correction tasks)
OUTF = os.environ.get("OUT", ".scratch/cordp_cards_public_test_reperceived.json")
tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS, tuple(ids))}
new_cards = {}
print(f"== re-perceiving {len(ids)} test tasks, K={K} self-consistency ==", flush=True)
for k, tid in enumerate(ids, 1):
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    H = len(n.future_values)
    fts = [str(x) for x in t.future_timestamps]
    steps = [s for s in (one_call(t, base, 0.3 + 0.15 * r) for r in range(K)) if s]
    if not steps:
        new_cards[tid] = {"confidence": 0.0, "corrections": []}
        continue
    cm = consensus(steps, H)
    new_cards[tid] = {"confidence": 1.0, "corrections": steps_to_windows(cm, fts)}
    Path(OUTF).write_text(json.dumps(new_cards, default=str))
    print(f"[{k}/{len(ids)}] {tid}: {len(new_cards[tid]['corrections'])} consensus windows", flush=True)


def eval_cards(cardset):
    bs = br = os_ = or_ = 0.0; n = 0; wins = reg = 0
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        nn = t.numeric; truth = list(nn.future_values); H = len(truth)
        base = list(fs.forecast("toto_2_0", tuple(nn.history_values), nn.prediction_length, nn.frequency))
        if len(base) != H:
            continue
        fts = [str(x) for x in t.future_timestamps]
        out = list(base)
        for r in (cardset.get(tid, {}).get("corrections") or []):
            if len(r) >= 3:
                for i, on in enumerate(horizon_window_mask(fts, str(r[0]), str(r[1]))):
                    if on:
                        out[i] = base[i] * float(r[2])
        out = list(apply_bounded_delta(base, out))
        mb = drcik_point_metrics(truth, base, cap=CAP); mo = drcik_point_metrics(truth, out, cap=CAP)
        bs += mb["smae"]; br += mb["srmse"]; os_ += mo["smae"]; or_ += mo["srmse"]; n += 1
        if mo["smae"] + mo["srmse"] < mb["smae"] + mb["srmse"] - 1e-9: wins += 1
        elif mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9: reg += 1
    return (bs - os_) / bs, (br - or_) / br, wins, reg


print("\n== new (re-perceived) vs old (cached) multipliers, trust-all + kernel, on the 55 subset ==")
for name, cs in [("OLD cached", old_cards), ("NEW re-perceived", new_cards)]:
    sm, sr, wn, rg = eval_cards(cs)
    print(f"  {name:18s}: sMAE {sm:+.2%} | sRMSE {sr:+.2%} | wins {wn} reg {rg}")
fs.close()
