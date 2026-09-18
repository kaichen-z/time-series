"""Stratify dev tasks by whether the DOCUMENT WINDOW actually contains usable signal.

For each task: take the union of grounded, windowed effect masks (where the documents
say something happens), and compute an ORACLE upper bound = the best constant scaling of
the base forecast inside that window (uses the truth, so it is an upper bound, not a
deployable result). If the oracle can reduce error there, the window holds real signal;
if not, the documents did not localize anything usable. Also report our deployable
regime-fill pipeline, to see how much of the oracle headroom we actually capture.

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/stratify_signal.py"""
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
from evolving_loop.adjustment.post_adjust import EvidenceEffect, _canon_direction, horizon_window_mask
from evolving_loop.adjustment.coevolve import Interaction, run_pipeline

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CARDS = ROOT / ".scratch/effect_cards_f8d9d5862942.json"
EPS = 0.005  # min oracle improvement to call a window "signal-bearing"

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


def union_window(effs, fts):
    mask = [False] * len(fts)
    for e in effs:
        if e.grounded and e.start_timestamp and e.end_timestamp:
            w = horizon_window_mask(fts, e.start_timestamp, e.end_timestamp)
            mask = [a or b for a, b in zip(mask, w)]
    return mask


def oracle_joint(base, truth, mask):
    if not any(mask):
        return None
    best = joint(base, truth)
    for k in range(30, 171, 2):          # best constant scale in [0.30, 1.70]
        f = k / 100.0
        cand = [base[i] * f if m else base[i] for i, m in enumerate(mask)]
        best = min(best, joint(cand, truth))
    return best


rows = []
for tid in raw:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    fts = [str(x) for x in t.future_timestamps]
    effs = effects_for(tid)
    mask = union_window(effs, fts)
    tj = joint(base, n.future_values)
    oj = oracle_joint(base, n.future_values, mask)
    our, _ = run_pipeline(Interaction(), base, effs, fts, list(n.history_values),
                          [str(x) for x in t.history_timestamps])
    ourj = joint(our, n.future_values)
    rows.append(dict(tid=tid, target=t.target_name, toto=tj, oracle=oj, our=ourj,
                     win=sum(mask), signal=(oj is not None and tj - oj > EPS)))

sig = [r for r in rows if r["signal"]]
w = 11
print(f"\n{'task':10s}{'target':28s}{'win':>4s}{'toto':>{w}s}{'oracle':>{w}s}{'ours':>{w}s}  signal")
for r in sorted(rows, key=lambda x: -( (x['toto']-x['oracle']) if x['oracle'] else -1)):
    o = f"{r['oracle']:.5f}" if r['oracle'] is not None else "   --   "
    print(f"{r['tid']:10s}{r['target'][:27]:28s}{r['win']:>4d}{r['toto']:{w}.5f}{o:>{w}s}"
          f"{r['our']:{w}.5f}  {'YES' if r['signal'] else ''}")

print(f"\n=== stratum: {len(sig)} SIGNAL tasks (oracle can beat toto in the doc window) ===")
if sig:
    for key in ["toto", "oracle", "our"]:
        vals = [r[key] for r in sig if r[key] is not None]
        print(f"  mean {key:7s} = {statistics.mean(vals):.5f}")
    cap = statistics.mean([r['toto'] for r in sig]) - statistics.mean([r['oracle'] for r in sig])
    got = statistics.mean([r['toto'] for r in sig]) - statistics.mean([r['our'] for r in sig])
    print(f"  oracle headroom = {cap:.5f} | we captured = {got:.5f} ({100*got/cap:.0f}% of the ceiling)"
          if cap > 0 else "  (no headroom)")
host.close(); fs.close()
