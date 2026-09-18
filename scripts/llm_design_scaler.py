"""LLM designs a WINDOW SCALER: given the document-localized event window, estimate the
multiplicative scale for those steps from history. This targets the real headroom the
stratify diagnostic found (oracle can cut error 33% inside the doc window; regime-fill
missed it and regressed). The scale must come from data, not the truth.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/llm_design_scaler.py"""
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
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, horizon_window_mask, apply_bounded_delta)
from evolving_loop.adjustment.llm_design import design_loop, make_window_scaler

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


def effects_for(tid):
    out = []
    for d in raw.get(tid, []):
        kw = {k: d.get(k) for k in FIELDS}
        kw["direction"] = _canon_direction(kw.get("direction"))
        out.append(EvidenceEffect(**kw))
    return out


def joint(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


DATA = []
for tid in raw:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    fts = [str(x) for x in t.future_timestamps]
    mask = [False] * len(fts)
    for e in effects_for(tid):
        if e.grounded and e.start_timestamp and e.end_timestamp:
            w = horizon_window_mask(fts, e.start_timestamp, e.end_timestamp)
            mask = [a or b for a, b in zip(mask, w)]
    DATA.append(dict(tid=tid, base=base, truth=n.future_values, fts=fts, window=mask,
                     hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                     toto_j=joint(base, n.future_values), has_win=any(mask)))

WIN = [d for d in DATA if d["has_win"]]
print(f"tasks with a document window: {len(WIN)}/{len(DATA)}", flush=True)


def evaluate(scaler):
    total = 0.0
    for d in WIN:
        factors = scaler(d["hv"], d["hts"], d["fts"], d["window"])
        if factors is None:
            continue
        proposed = [d["base"][i] * factors[i] if d["window"][i] else d["base"][i]
                    for i in range(len(d["base"]))]
        out = apply_bounded_delta(list(d["base"]), proposed, max_frac=0.5)
        delta = d["toto_j"] - joint(out, d["truth"])
        total += delta if delta >= 0 else 5.0 * delta
    return total


SYS = (
    "You design a Python function that estimates how much to rescale a base time-series "
    "forecast INSIDE a known event window (the documents already told us WHERE something "
    "happens; you estimate HOW MUCH, from the data). Signature:\n"
    "    def estimate(history_values, history_timestamps, future_timestamps, window_mask):\n"
    "window_mask[i] is True for the future steps in the event window. Return a list of one "
    "multiplicative factor PER future step: use 1.0 OUTSIDE the window; inside, a factor "
    "<1 lowers and >1 raises the forecast. Return None to make no change.\n"
    "Estimate the in-window factor from HISTORY: e.g. compare the level of history steps "
    "that resemble the window (same weekday/weekend, same hour-of-day, similar recent "
    "level) to the overall/base level. Be robust on short noisy history (48-168 pts); when "
    "the window's analog is not clearly different from normal, return factors near 1.0. "
    "Rules: pure function, NO imports, NO attribute access. Helpers available: mean, "
    "median, pstdev, min, max, sum, len, abs, round, sorted, range, enumerate, zip, list, "
    "tuple, float, int, weekday(ts), hour(ts), dayofyear(ts).\n"
    'Respond with ONLY a JSON object: {"code_lines": ["def estimate(history_values, '
    'history_timestamps, future_timestamps, window_mask):", "    ...", ...]} where each '
    "element is exactly one source line (4-space indented, no trailing newline)."
)

from common.llm import parse_json_object


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
        top = archive[0]
        hint = f"\n\nYour current best scored {top[0]:.4f}. Improve it:\n```python\n{top[1]}\n```"
    user = ("Design/improve estimate. Higher score = more error reduction inside event "
            "windows with fewer regressions across the tasks." + hint)
    resp = host.llm_client.complete(system=SYS, messages=[{"role": "user", "content": user}],
                                    temperature=0.4)
    return _extract(resp.text)


# oracle headroom on these windowed tasks (best constant scale, uses truth = upper bound)
def oracle(d):
    best = d["toto_j"]
    for k in range(30, 171, 2):
        f = k / 100.0
        cand = [d["base"][i] * f if d["window"][i] else d["base"][i] for i in range(len(d["base"]))]
        best = min(best, joint(cand, d["truth"]))
    return best


toto_mean = statistics.mean([d["toto_j"] for d in WIN])
oracle_mean = statistics.mean([oracle(d) for d in WIN])
print(f"toto mean = {toto_mean:.5f} | oracle ceiling = {oracle_mean:.5f} "
      f"(headroom {toto_mean - oracle_mean:.5f})", flush=True)

print("\n== LLM designing a window scaler ==", flush=True)
archive, log = design_loop(proposer, evaluate, wrap=make_window_scaler, rounds=8, keep=4)
for r in log:
    extra = f"score={r['score']:.4f}" if r["status"] == "ok" else r.get("error", "")
    print(f"  round {r['round']}: {r['status']}  {extra}", flush=True)

if archive:
    best_score, best_code = archive[0]
    scaler = make_window_scaler(__import__("evolving_loop.adjustment.llm_design",
             fromlist=["safe_compile"]).safe_compile(best_code))
    our_mean = statistics.mean([
        joint(apply_bounded_delta(list(d["base"]),
              [d["base"][i] * (scaler(d["hv"], d["hts"], d["fts"], d["window"]) or [1.0]*len(d["base"]))[i]
               if d["window"][i] else d["base"][i] for i in range(len(d["base"]))], max_frac=0.5), d["truth"])
        for d in WIN])
    print(f"\n== best designed scaler: score {best_score:.4f} ==")
    print(f"mean joint on windowed tasks: toto {toto_mean:.5f} | ours {our_mean:.5f} | "
          f"oracle {oracle_mean:.5f}")
    got = toto_mean - our_mean; cap = toto_mean - oracle_mean
    print(f"captured {100*got/cap:.0f}% of the oracle headroom" if cap > 0 else "no headroom")
    Path(".scratch/designed_scaler_best.py").write_text(best_code)
    print("\n--- code ---\n" + best_code)
else:
    print("no valid design produced")
host.close(); fs.close()
