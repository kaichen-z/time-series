"""LLM-as-designer, on real data: the LLM WRITES regime-estimator functions, each is
sandboxed + scored offline on the 20 dev tasks, the best are fed back, and it iterates.

Run:  TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/llm_design_regime.py

Compares the best LLM-designed regime against the hand-written weekend_weekday rule."""
from __future__ import annotations
import json, re, tempfile
from pathlib import Path
from statistics import mean

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import EvidenceEffect, _canon_direction
from evolving_loop.adjustment.coevolve import Regime, QualifyPolicy, IntegratePolicy
from evolving_loop.adjustment.dsl import _REGIMES
from evolving_loop.adjustment.llm_design import design_loop, make_regime_estimator

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
    DATA.append(dict(tid=tid, base=base, truth=n.future_values,
                     fts=[str(x) for x in t.future_timestamps],
                     hv=list(n.history_values), hts=[str(x) for x in t.history_timestamps],
                     effs=effects_for(tid), toto_j=joint(base, n.future_values)))
print(f"tasks: {len(DATA)}", flush=True)


def evaluate(est):
    """Score a regime estimator: downside-protected improvement over toto across tasks."""
    total = 0.0
    for d in DATA:
        allmask = tuple(True for _ in d["fts"])
        factors = est(d["hv"], d["hts"], None, d["fts"], allmask)
        if factors is None:
            continue
        avg = mean(factors)
        if avg == 1.0:
            continue
        regime = Regime("designed", tuple(factors), "decrease" if avg < 1.0 else "increase", abs(1.0 - avg))
        qualified = QualifyPolicy().apply(d["effs"], [regime], d["fts"])
        out, _ = IntegratePolicy().apply(d["base"], qualified)
        delta = d["toto_j"] - joint(out, d["truth"])
        total += delta if delta >= 0 else 5.0 * delta
    return total


SYS = (
    "You design a Python function that finds an exploitable regularity in a SHORT time "
    "series so a document-mentioned event can be quantified from the data. Signature:\n"
    "    def estimate(history_values, history_timestamps, future_timestamps):\n"
    "It returns a list of one multiplicative factor PER future timestamp (relative to the "
    "base level): <1 lowers, >1 raises, 1.0 unchanged; or return None when NO clear "
    "regularity is present (be conservative -- returning None avoids harm).\n"
    "Rules: pure function, NO imports, NO attribute access. Available helpers: mean, "
    "median, pstdev, min, max, sum, len, abs, round, sorted, range, enumerate, zip, list, "
    "tuple, float, int, and weekday(ts)->0..6, hour(ts)->0..23, dayofyear(ts)->1..366 "
    "(each takes an ISO timestamp string). Timestamps are hourly or daily; history is "
    "short (48-168 points). Ideas: a holiday makes a weekday behave like the weekend; a "
    "recurring low regime; hour-of-day or day-of-week profile. Return ONLY one ```python``` "
    "code block."
)


def _extract(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def proposer(archive):
    hint = ""
    if archive:
        top = archive[0]
        hint = f"\n\nYour current best scored {top[0]:.4f}. Improve on it:\n```python\n{top[1]}\n```"
    user = ("Design (or improve) the estimate function. Higher score = larger error "
            "reduction over the base forecast with fewer regressions." + hint)
    resp = host.llm_client.complete(system=SYS, messages=[{"role": "user", "content": user}],
                                    temperature=0.4)
    return _extract(resp.text)


baseline = evaluate(_REGIMES["weekend_weekday"])
print(f"hand-written weekend_weekday score: {baseline:.4f}", flush=True)

print("\n== LLM designing regime estimators ==", flush=True)
archive, log = design_loop(proposer, evaluate, rounds=8, keep=4)
for r in log:
    extra = f"score={r['score']:.4f}" if r["status"] == "ok" else r.get("error", "")
    print(f"  round {r['round']}: {r['status']}  {extra}", flush=True)

if archive:
    best_score, best_code = archive[0]
    print(f"\n== best LLM-designed regime: score {best_score:.4f} "
          f"(hand-written baseline {baseline:.4f}) ==")
    print("beats hand-written?" , "YES" if best_score > baseline else "no")
    print("\n--- code ---\n" + best_code)
    Path(".scratch/designed_regime_best.py").write_text(best_code)
else:
    print("no valid design produced")
host.close(); fs.close()
