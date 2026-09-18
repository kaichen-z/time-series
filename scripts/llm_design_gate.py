"""LLM designs a CONFIDENCE GATE on the semantic reference mapping, so the reliable
mappings (holiday->weekend) are applied and the noisy ones (single-day extremes on an
already-accurate series) are damped -- the piece the hand gate couldn't separate.

Uses cached semantic refs (.scratch/semantic_refs.json). The LLM writes
    estimate(history_values, history_timestamps, future_timestamps, window_mask,
             reference_level, base_level) -> per-step factors
deciding, via a history backtest, how much to trust moving the window toward the
reference. Scored on the 15 windowed dev tasks; compared to the hand gate (+0.0064).

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/llm_design_gate.py"""
from __future__ import annotations
import json, re, statistics, tempfile
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
from evolving_loop.adjustment.llm_design import design_loop, make_gate_scaler, safe_compile

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CARDS = ROOT / ".scratch/effect_cards_f8d9d5862942.json"
REFS = json.loads((ROOT / ".scratch/semantic_refs.json").read_text())

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


def wmask(tid, fts):
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
    wkday = [v for v, t in zip(hv, hts) if 0 <= wd(t) < 5]
    wkend = [v for v, t in zip(hv, hts) if wd(t) >= 5]
    byday: dict = {}
    for v, t in zip(hv, hts):
        p = _parse(t)
        if p:
            byday.setdefault(p.date(), []).append(v)
    dm = sorted(statistics.mean(x) for x in byday.values()) if byday else []
    L = {"overall": statistics.mean(hv), "recent": statistics.mean(hv[-24:])}
    if wkday:
        L["weekday"] = statistics.mean(wkday)
    if wkend:
        L["weekend"] = statistics.mean(wkend)
    if dm:
        L["low_day"], L["high_day"] = dm[0], dm[-1]
    return L


DATA = []
for tid in raw:
    t = tasks.get(tid)
    if t is None or not t.gt_evidence:
        continue
    n = t.numeric
    base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    fts = [str(x) for x in t.future_timestamps]
    mask = wmask(tid, fts)
    if not any(mask):
        continue
    hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
    L = levels(hv, hts)
    bw = statistics.mean([base[i] for i in range(len(base)) if mask[i]])
    ref = REFS.get(tid, "none")
    ref_level = L[ref] if ref in L else bw
    DATA.append(dict(tid=tid, base=base, truth=n.future_values, mask=mask, hv=hv, hts=hts,
                     fts=fts, bw=bw, ref=ref, ref_level=ref_level, toto=joint(base, n.future_values)))

toto_mean = statistics.mean([d["toto"] for d in DATA])
print(f"windowed tasks: {len(DATA)}  toto mean {toto_mean:.5f}", flush=True)


def evaluate(scaler):
    total = 0.0
    for d in DATA:
        if d["ref"] == "none" or d["bw"] <= 0:
            continue
        factors = scaler(d["hv"], d["hts"], d["fts"], d["mask"], d["ref_level"], d["bw"])
        if factors is None:
            continue
        proposed = [d["base"][i] * factors[i] if d["mask"][i] else d["base"][i] for i in range(len(d["base"]))]
        out = apply_bounded_delta(list(d["base"]), proposed, max_frac=0.5)
        delta = d["toto"] - joint(out, d["truth"])
        total += delta if delta >= 0 else 5.0 * delta
    return total


SYS = (
    "You write a Python confidence GATE. A semantic step already chose a historical "
    "reference_level the event window should resemble; you decide HOW MUCH to trust it, "
    "from the data, and output the scaling. Signature:\n"
    "    def estimate(history_values, history_timestamps, future_timestamps, window_mask, "
    "reference_level, base_level):\n"
    "Return a list of one factor PER future step: 1.0 OUTSIDE the window; inside, move the "
    "base toward reference_level by a confidence in [0,1]: factor = 1 + (reference_level/"
    "base_level - 1) * confidence. Estimate confidence by BACKTESTING the reference on "
    "history: does the pattern implied by reference_level actually recur / predict "
    "held-out history? A reference matching a STABLE recurring regime (e.g. weekend level "
    "that repeats every week) deserves high confidence; one matching a single-day extreme "
    "or noise deserves ~0. When base_level is already stable/accurate-looking or the gap "
    "is tiny, return confidence ~0 (factors near 1.0). Rules: pure function, NO imports, "
    "NO attribute access. Helpers: mean, median, pstdev, min, max, sum, len, abs, round, "
    "sorted, range, enumerate, zip, list, tuple, float, int, weekday(ts), hour(ts), "
    'dayofyear(ts). Respond ONLY as JSON {"code_lines": ["def estimate(...):", "    ...", ...]}.'
)


def _extract(text):
    try:
        obj = parse_json_object(text)
        if isinstance(obj, dict) and isinstance(obj.get("code_lines"), list):
            return "\n".join(str(x) for x in obj["code_lines"])
        if isinstance(obj, dict) and isinstance(obj.get("code"), str):
            return obj["code"]
    except Exception:
        pass
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def proposer(archive):
    hint = ""
    if archive:
        hint = f"\n\nYour current best scored {archive[0][0]:.4f}. Improve it:\n```python\n{archive[0][1]}\n```"
    user = ("Design/improve the confidence gate. Higher score = keep the reliable "
            "mappings and damp the ones that would regress." + hint)
    resp = host.llm_client.complete(system=SYS, messages=[{"role": "user", "content": user}],
                                    temperature=0.4)
    return _extract(resp.text)


print("\n== LLM designing a confidence gate ==", flush=True)
archive, log = design_loop(proposer, evaluate, wrap=make_gate_scaler, rounds=8, keep=4)
for r in log:
    extra = f"score={r['score']:.4f}" if r["status"] == "ok" else r.get("error", "")
    print(f"  round {r['round']}: {r['status']}  {extra}", flush=True)

if archive:
    best_score, best_code = archive[0]
    scaler = make_gate_scaler(safe_compile(best_code))
    js = []
    for d in DATA:
        out = d["base"]
        if d["ref"] != "none" and d["bw"] > 0:
            f = scaler(d["hv"], d["hts"], d["fts"], d["mask"], d["ref_level"], d["bw"])
            if f is not None:
                proposed = [d["base"][i] * f[i] if d["mask"][i] else d["base"][i] for i in range(len(d["base"]))]
                out = apply_bounded_delta(list(d["base"]), proposed, max_frac=0.5)
        jj = joint(out, d["truth"])
        js.append(jj)
        tag = "improved" if jj < d["toto"] - 1e-6 else ("REGRESS" if jj > d["toto"] + 1e-6 else "")
        print(f"  {d['tid']}: ref={d['ref']:9s} toto={d['toto']:.4f} ours={jj:.4f} {tag}")
    print(f"\n== designed gate: score {best_score:.4f} | mean joint {statistics.mean(js):.5f} "
          f"vs toto {toto_mean:.5f} (hand gate was 0.48825) ==")
    Path(".scratch/designed_gate_best.py").write_text(best_code)
    print("\n--- code ---\n" + best_code)
else:
    print("no valid design produced")
host.close(); fs.close()
