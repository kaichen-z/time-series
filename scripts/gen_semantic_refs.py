"""Generate per-task semantic references (LLM: event -> which historical regime it
resembles) for the tasks in a cards file, and MERGE them into .scratch/semantic_refs.json.

This is step ② of the pipeline: SemanticAdjust / PooledSemanticAdjust need a cached LLM
regime choice per task. The LLM only picks a regime name (a semantic call it is good at);
the magnitude is read from data downstream. Event text = the task's gt_evidence.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD CARDS_FILE=effect_cards_train_<fp>.json \
     .venv/bin/python scripts/gen_semantic_refs.py
"""
from __future__ import annotations
import json, os, statistics, tempfile
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.llm import parse_json_object
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, _parse, horizon_window_mask)

ROOT = Path(".").resolve()
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CARDS = ROOT / ".scratch" / os.environ.get("CARDS_FILE", "effect_cards_f8d9d5862942.json")
OUT = ROOT / ".scratch/semantic_refs.json"
raw = json.loads(CARDS.read_text())
existing = json.loads(OUT.read_text()) if OUT.exists() else {}
todo = [tid for tid in raw if tid not in existing]
print(f"cards {CARDS.name}: {len(raw)} tasks | already have refs: {len(existing)} | todo: {len(todo)}", flush=True)

manifest = RealEvolutionManifestV2.from_payload(
    _read_canonical(ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"))
host = build_real_host(manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
                       projection_train_size=8, projection_dev_size=3, projection_fold_count=2)
mr = host.source_repo
port = read_policy_file(str(mr / "policies.py"))
scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)
split = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
all_ids = tuple({tid for part in split.values() for tid in part["task_ids"]})
tasks = {t.numeric.task_id: t for t in
         load_context_tasks_by_ids("external/Dr-CiK/full-download/Dr-CiK_public/tasks", all_ids)}
FIELDS = list(EvidenceEffect.__dataclass_fields__.keys())


def window_mask(tid, fts):
    mask = [False] * len(fts)
    for d in raw.get(tid, []):
        kw = {k: d.get(k) for k in FIELDS}
        e = EvidenceEffect(**{**kw, "direction": _canon_direction(kw.get("direction"))})
        if e.grounded and e.start_timestamp and e.end_timestamp:
            mask = [a or b for a, b in zip(mask, horizon_window_mask(fts, e.start_timestamp, e.end_timestamp))]
    return mask


def levels(hv, hts):
    def wd(t):
        p = _parse(t); return p.weekday() if p else -1
    wk = [v for v, t in zip(hv, hts) if 0 <= wd(t) < 5]
    we = [v for v, t in zip(hv, hts) if wd(t) >= 5]
    byday: dict = {}
    for v, t in zip(hv, hts):
        p = _parse(t)
        if p:
            byday.setdefault(p.date(), []).append(v)
    dm = sorted(statistics.mean(x) for x in byday.values()) if byday else []
    L = {"overall": round(statistics.mean(hv), 3), "recent": round(statistics.mean(hv[-24:]), 3)}
    if wk:
        L["weekday"] = round(statistics.mean(wk), 3)
    if we:
        L["weekend"] = round(statistics.mean(we), 3)
    if dm:
        L["low_day"], L["high_day"] = round(dm[0], 3), round(dm[-1], 3)
    return L


SYS = (
    "You map a described real-world event to the historical regime the time series will "
    "most RESEMBLE during the event window, so its magnitude can be read from data. You "
    "get the event evidence, the base forecast's level in the window, and historical "
    "reference levels. Choose the single reference the window will most resemble, or "
    "'none' if the event should not change the level. Example: a public holiday on a "
    "weekday usually resembles 'weekend'; a maintenance/outage 'low_day'; a promotion "
    "'high_day'. Respond ONLY as JSON {\"reference\": \"<key>\"} using a provided key or 'none'."
)


def choose_reference(evidence, base_win, L):
    user = json.dumps({"event_evidence": list(evidence)[:8], "base_forecast_window_level": round(base_win, 3),
                       "historical_reference_levels": L, "allowed_references": list(L) + ["none"]},
                      ensure_ascii=False)
    try:
        resp = host.llm_client.complete(system=SYS, messages=[{"role": "user", "content": user}], temperature=0.0)
        return str(parse_json_object(resp.text).get("reference", "none"))
    except Exception as exc:
        print(f"    llm error: {exc}", flush=True)
        return "none"


out = dict(existing)
for i, tid in enumerate(todo, 1):
    t = tasks.get(tid)
    if t is None or not t.gt_evidence:
        out[tid] = "none"; continue
    n = t.numeric
    base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    fts = [str(x) for x in t.future_timestamps]
    mask = window_mask(tid, fts)
    if not any(mask):
        out[tid] = "none"
    else:
        hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
        bw = statistics.mean([base[i] for i in range(len(base)) if mask[i]])
        out[tid] = choose_reference(t.gt_evidence, bw, levels(hv, hts))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[{i}/{len(todo)}] {tid}: ref={out[tid]} (saved)", flush=True)

print(f"done. total refs: {len(out)}", flush=True)
host.close(); fs.close()
