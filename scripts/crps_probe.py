"""Minimal CRPS concept check: does moving to a PROBABILISTIC view let document evidence
help, when point-forecasting can't? For each dev task we form a base predictive
distribution N(point_forecast, sigma) with sigma = historical volatility, then test three
document-driven edits INSIDE the event window and score mean CRPS vs the base:
  - widen:  inflate sigma in the window (admit more uncertainty around a known event)
  - shift:  move the mean toward the LLM-chosen regime level (as in point mode)
  - both.
Widening is near-zero-regression by construction, so if it improves CRPS, the
probabilistic framing is the promising direction. Uses cached toto point forecasts +
history std (no Toto re-inference needed).

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/crps_probe.py"""
from __future__ import annotations
import json, math, statistics, tempfile
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, _parse, horizon_window_mask)

ROOT = Path(".").resolve()
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CARDS = ROOT / ".scratch/effect_cards_f8d9d5862942.json"
REFS = json.loads((ROOT / ".scratch/semantic_refs.json").read_text())
WIDEN = 2.0
raw = json.loads(CARDS.read_text())


def crps_normal(mu, sigma, y):
    if sigma <= 1e-9:
        return abs(y - mu)
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


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
FIELDS = list(EvidenceEffect.__dataclass_fields__.keys())


def window(tid, fts):
    m = [False] * len(fts)
    for d in raw.get(tid, []):
        kw = {k: d.get(k) for k in FIELDS}
        e = EvidenceEffect(**{**kw, "direction": _canon_direction(kw.get("direction"))})
        if e.grounded and e.start_timestamp and e.end_timestamp:
            m = [a or b for a, b in zip(m, horizon_window_mask(fts, e.start_timestamp, e.end_timestamp))]
    return m


def reflevel(hv, hts, ref):
    def wd(t):
        p = _parse(t); return p.weekday() if p else -1
    if ref == "weekend":
        v = [x for x, t in zip(hv, hts) if wd(t) >= 5]; return statistics.mean(v) if v else None
    if ref == "weekday":
        v = [x for x, t in zip(hv, hts) if 0 <= wd(t) < 5]; return statistics.mean(v) if v else None
    if ref in ("low_day", "high_day"):
        bd = {}
        for x, t in zip(hv, hts):
            p = _parse(t)
            if p: bd.setdefault(p.date(), []).append(x)
        dm = sorted(statistics.mean(z) for z in bd.values()) if bd else []
        return (dm[0] if ref == "low_day" else dm[-1]) if dm else None
    return None


def mean_crps(mu, sig, truth):
    return statistics.mean(crps_normal(mu[i], sig[i], truth[i]) for i in range(len(truth)))


rows = []
for tid in raw:
    t = tasks.get(tid)
    if t is None:
        continue
    n = t.numeric
    point = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
    truth = list(n.future_values)
    fts = [str(x) for x in t.future_timestamps]
    hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
    mask = window(tid, fts)
    diffs = [hv[i + 1] - hv[i] for i in range(len(hv) - 1)]   # one-step volatility ~ forecast uncertainty
    sig0 = max(statistics.pstdev(diffs) if len(diffs) > 1 else 0.0, 1e-6)
    N = len(truth)
    base = mean_crps(point, [sig0] * N, truth)
    widen = mean_crps(point, [sig0 * WIDEN if mask[i] else sig0 for i in range(N)], truth)
    ref = REFS.get(tid, "none")
    lvl = reflevel(hv, hts, ref) if ref not in ("none", "overall", "recent") else None
    bw = statistics.mean([point[i] for i in range(N) if mask[i]]) if any(mask) else 0.0
    factor = (lvl / bw) if (lvl is not None and bw > 0) else 1.0
    mu_shift = [point[i] * factor if mask[i] else point[i] for i in range(N)]
    shift = mean_crps(mu_shift, [sig0] * N, truth)
    both = mean_crps(mu_shift, [sig0 * WIDEN if mask[i] else sig0 for i in range(N)], truth)
    rows.append(dict(tid=tid, base=base, widen=widen, shift=shift, both=both,
                     win=any(mask), ref=ref))

win = [r for r in rows if r["win"]]
print(f"\n== CRPS concept check on {len(rows)} dev tasks ({len(win)} windowed) ==")
for label in ["base", "widen", "shift", "both"]:
    print(f"  mean {label:6s} = {statistics.mean([r[label] for r in rows]):.5f} "
          f"| windowed-only = {statistics.mean([r[label] for r in win]):.5f}")
b = statistics.mean([r['base'] for r in win])
for label in ["widen", "shift", "both"]:
    m = statistics.mean([r[label] for r in win]); reg = sum(1 for r in win if r[label] > r['base'] + 1e-9)
    print(f"  {label:6s} vs base (windowed): {b - m:+.5f}  ({'improves' if m < b else 'worse'}), "
          f"regressions {reg}/{len(win)}")
host.close(); fs.close()
