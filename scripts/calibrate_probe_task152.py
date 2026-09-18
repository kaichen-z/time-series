"""Probe: can the historical series itself calibrate a document event's magnitude?

Run from repo root:  PYTHONPATH=$PWD .venv/bin/python scripts/calibrate_probe_task152.py

The retrieved document effect for task_152 (July-4 holiday -> lower road occupancy)
gives a direction and window but a fuzzy magnitude. The KEY IDEA (data<->text
coupling): a real, document-explained regularity is already observable in the short
history -- weekends have ~37% lower occupancy (business activity drops). A public
holiday makes a weekday behave like a weekend, so the weekend/weekday level ratio
CALIBRATES the holiday adjustment magnitude, instead of a hand-picked cap.

Result (this probe): toto joint 0.46357; fixed -30% cap 0.32638; history-calibrated
(-37.4%) 0.30510 -- calibration beats the fixed cap and lands near the truth."""
from __future__ import annotations
import json, statistics, tempfile
from datetime import datetime
from pathlib import Path

from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.cli import _read_canonical
from evolving_loop.v2.real.contracts import RealEvolutionManifestV2
from evolving_loop.v2.real.host import build_real_host, _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
DOC_WINDOW = ("2024-07-04T00:00:00", "2024-07-04T23:00:00")  # the holiday day

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
t = {x.numeric.task_id: x for x in load_context_tasks_by_ids(
    "external/Dr-CiK/full-download/Dr-CiK_public/tasks", dev_ids)[:8]}["task_152"]
n = t.numeric
base = list(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
truth = n.future_values
fts = [str(x) for x in t.future_timestamps]
hts = [datetime.fromisoformat(str(x).replace("Z", "+00:00")) for x in t.history_timestamps]
hv = list(n.history_values)


def joint(fc):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


mask = horizon_window_mask(fts, *DOC_WINDOW)
wk = statistics.mean([v for ts, v in zip(hts, hv) if ts.weekday() < 5])
we = statistics.mean([v for ts, v in zip(hts, hv) if ts.weekday() >= 5])
factor = we / wk  # the document-explained regime observed in history

cal = apply_bounded_delta(base, [base[i] * factor if m else base[i]
                                 for i, m in enumerate(mask)], max_frac=0.5)
fixed = apply_bounded_delta(base, [base[i] * 0.70 if m else base[i]
                                   for i, m in enumerate(mask)], max_frac=0.5)

win = lambda s: statistics.mean([s[i] for i, m in enumerate(mask) if m])
print(f"weekend/weekday factor = {factor:.3f}  (calibrated holiday move = {100*(factor-1):.1f}%)")
print(f"window means -> toto {win(base):.3f} | calibrated {win(cal):.3f} | truth {win(truth):.3f}")
print(f"joint  toto={joint(base):.5f}  fixed(-30%)={joint(fixed):.5f}  calibrated={joint(cal):.5f}")
host.close(); fs.close()
