"""LLM as a SEMANTIC MAPPER: given an event, pick which historical regime the event
window will resemble; the magnitude is then read from the data for that regime.

This targets the missing piece the scaler experiment exposed: a short history can't
supply an event's departure from normal, but "holiday -> behaves like weekend" is
document/common-sense knowledge. The LLM only chooses a reference regime (a semantic
call it is good at); the number comes from that regime's historical level (reliable).
Event text = the task's gt_evidence (the documents' real signal, isolating the mapping
step from retrieval noise). Windows come from the retrieval effect cards.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/semantic_scale.py"""
from __future__ import annotations
import json, statistics, tempfile
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from common.llm import parse_json_object
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, _parse, horizon_window_mask, apply_bounded_delta)

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CARDS = ROOT / ".scratch/effect_cards_f8d9d5862942.json"

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
dev_ids = tuple(json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())
                ["partitions"]["dev"]["task_ids"])
tasks = {t.numeric.task_id: t for t in
         load_context_tasks_by_ids("external/Dr-CiK/full-download/Dr-CiK_public/tasks", dev_ids)}
raw = json.loads(CARDS.read_text())
FIELDS = list(EvidenceEffect.__dataclass_fields__.keys())


def joint(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def window_mask(tid, fts):
    mask = [False] * len(fts)
    for d in raw.get(tid, []):
        kw = {k: d.get(k) for k in FIELDS}
        e = EvidenceEffect(**{**kw, "direction": _canon_direction(kw.get("direction"))})
        if e.grounded and e.start_timestamp and e.end_timestamp:
            w = horizon_window_mask(fts, e.start_timestamp, e.end_timestamp)
            mask = [a or b for a, b in zip(mask, w)]
    return mask


def reference_levels(hv, hts):
    wd = [v for v, t in zip(hv, hts) if 0 <= (_parse(t).weekday() if _parse(t) else -1) < 5]
    we = [v for v, t in zip(hv, hts) if (_parse(t).weekday() if _parse(t) else -1) >= 5]
    byday: dict = {}
    for v, t in zip(hv, hts):
        d = _parse(t)
        if d:
            byday.setdefault(d.date(), []).append(v)
    dmeans = sorted(statistics.mean(x) for x in byday.values()) if byday else []
    L = {"overall": round(statistics.mean(hv), 3), "recent": round(statistics.mean(hv[-24:]), 3)}
    if wd:
        L["weekday"] = round(statistics.mean(wd), 3)
    if we:
        L["weekend"] = round(statistics.mean(we), 3)
    if dmeans:
        L["low_day"] = round(dmeans[0], 3)
        L["high_day"] = round(dmeans[-1], 3)
    return L


SYS = (
    "You map a described real-world event to the historical regime the time series will "
    "most RESEMBLE during the event window, so its magnitude can be read from the data "
    "(not guessed). You are given the event evidence, the base forecast's level in the "
    "window, and several historical reference levels. Choose the single reference the "
    "window will most resemble, or 'none' if the event should not change the level. "
    "Example: a public holiday on a weekday usually makes activity resemble the "
    "'weekend' level; a maintenance/outage resembles 'low_day'; a promotion 'high_day'. "
    'Respond ONLY as JSON {"reference": "<key>", "rationale": "..."} using one of the '
    "provided reference keys or 'none'."
)


def choose_reference(evidence, base_win, levels):
    user = json.dumps({"event_evidence": list(evidence)[:8], "base_forecast_window_level": round(base_win, 3),
                       "historical_reference_levels": levels,
                       "allowed_references": list(levels) + ["none"]}, ensure_ascii=False)
    try:
        resp = host.llm_client.complete(system=SYS, messages=[{"role": "user", "content": user}],
                                        temperature=0.0)
        obj = parse_json_object(resp.text)
        return str(obj.get("reference", "none"))
    except Exception as exc:
        print(f"    llm error: {exc}", flush=True)
        return "none"


rows = []
for tid in raw:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    fts = [str(x) for x in t.future_timestamps]
    mask = window_mask(tid, fts)
    tj = joint(base, n.future_values)
    if not any(mask) or not t.gt_evidence:
        rows.append(dict(tid=tid, toto=tj, our=tj, ref="(no window/evidence)"))
        continue
    hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
    levels = reference_levels(hv, hts)
    base_win = statistics.mean([base[i] for i in range(len(base)) if mask[i]])
    ref = choose_reference(t.gt_evidence, base_win, levels)
    if ref in levels and base_win > 0:
        factor = levels[ref] / base_win
        proposed = [base[i] * factor if mask[i] else base[i] for i in range(len(base))]
        out = apply_bounded_delta(list(base), proposed, max_frac=0.5)
    else:
        out = base
    oj = tj
    for k in range(30, 171, 2):
        f = k / 100.0
        cand = [base[i] * f if mask[i] else base[i] for i in range(len(base))]
        oj = min(oj, joint(cand, n.future_values))
    rows.append(dict(tid=tid, toto=tj, our=joint(out, n.future_values), oracle=oj, ref=ref,
                     tgt=t.target_name[:22]))
    print(f"  {tid}: ref={ref:9s} toto={tj:.4f} ours={rows[-1]['our']:.4f} oracle={oj:.4f}", flush=True)

win = [r for r in rows if "oracle" in r]
print(f"\n=== {len(win)} windowed+evidence tasks ===")
for key in ["toto", "our", "oracle"]:
    print(f"  mean {key:6s} = {statistics.mean([r[key] for r in win]):.5f}")
cap = statistics.mean([r['toto'] for r in win]) - statistics.mean([r['oracle'] for r in win])
got = statistics.mean([r['toto'] for r in win]) - statistics.mean([r['our'] for r in win])
print(f"  captured {100*got/cap:.0f}% of oracle headroom" if cap > 0 else "  no headroom")
host.close(); fs.close()
