"""Offline evaluation of the Level-0 DSL policies on real dev data.

Run from the repo root:  PYTHONPATH=$PWD .venv/bin/python scripts/eval_dsl_offline.py

No LLM calls: uses the cache_only toto ForecastStore (base) + the cached retrieval
effects (.scratch/dev_cards.json, produced by scripts/evidence_headroom_probe.py) +
real task truth/timestamps. Scores each policy's joint error (smae+srmse)/2 vs toto,
so we can see whether GROUNDED_EVENT actually reduces error on the tasks where it
fires (the "is the discarded document signal real value?" question)."""
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
from evolving_loop.adjustment.post_adjust import EvidenceEffect, _canon_direction
from evolving_loop.adjustment.dsl import (
    IDENTITY, EVENT_SCALE, EVENT_SCALE_RELAXED_ENTITY, GROUNDED_EVENT, apply_policy,
)

ROOT = Path(".").resolve()
MANIFEST = ROOT / "configs/evolution_v2/real/real-30m-toto-claude-server.json"
SPLIT = ROOT / "splits/drcik_public_80_20_99_v3.json"
TASKS_DIR = ROOT / "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
IDENTITY_HASH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
CAP = 5.0

manifest = RealEvolutionManifestV2.from_payload(_read_canonical(MANIFEST))
host = build_real_host(manifest, repo_root=ROOT, output_dir=Path(tempfile.mkdtemp()),
                       projection_train_size=8, projection_dev_size=3, projection_fold_count=2)
mroot = host.source_repo
port = read_policy_file(str(mroot / "policies.py"))
scr = _load_screening_policy(str(mroot / "dictionary.py"))
toto_fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                        mroot / "methods.py", mroot / "skills.py", port, None,
                        screening_hash=scr.fingerprint(), runtime_identity={}, cache_only=True,
                        identity_hash_override=IDENTITY_HASH)

dev_ids = tuple(json.loads(SPLIT.read_text())["partitions"]["dev"]["task_ids"])
tasks = load_context_tasks_by_ids(str(TASKS_DIR), dev_ids)[:8]
by_id = {t.numeric.task_id: t for t in tasks}

raw = json.loads(Path(".scratch/dev_cards.json").read_text())
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


POLICIES = {
    "identity": IDENTITY, "event_scale": EVENT_SCALE,
    "relaxed_entity": EVENT_SCALE_RELAXED_ENTITY, "grounded_event": GROUNDED_EVENT,
}

rows, fired_tasks = [], []
agg = {k: [] for k in POLICIES}
for tid, t in by_id.items():
    n = t.numeric
    base = tuple(toto_fs.forecast("toto_2_0", tuple(n.history_values),
                                  n.prediction_length, n.frequency))
    truth, fts = n.future_values, t.future_timestamps
    effs = effects_for(tid)
    r = {"task": tid}
    for name, pol in POLICIES.items():
        out, fired = apply_policy(pol, base, effs, fts)
        r[name] = joint(out, truth)
        agg[name].append(r[name])
        if name == "grounded_event":
            r["ge_fired"] = len(fired)
            if fired:
                fired_tasks.append(tid)
    rows.append(r)

hdr = f"{'task':10s} {'toto':>9s} {'event':>9s} {'relaxE':>9s} {'grounded':>9s} {'geFired':>7s}"
print("\n== offline DSL eval on dev (joint = (sMAE+sRMSE)/2, lower better) ==")
print(hdr)
for r in rows:
    print(f"{r['task']:10s} {r['identity']:9.5f} {r['event_scale']:9.5f} "
          f"{r['relaxed_entity']:9.5f} {r['grounded_event']:9.5f} {r['ge_fired']:7d}")
print("-" * len(hdr))
print(f"{'MEAN(all)':10s} " + " ".join(f"{statistics.mean(agg[k]):9.5f}"
      for k in ["identity", "event_scale", "relaxed_entity", "grounded_event"]))

if fired_tasks:
    print(f"\n-- stratum: tasks where grounded_event fires ({fired_tasks}) --")
    sub = [r for r in rows if r["task"] in fired_tasks]
    for k in ["identity", "event_scale", "relaxed_entity", "grounded_event"]:
        print(f"  {k:16s} mean joint = {statistics.mean([r[k] for r in sub]):.5f}")
else:
    print("\n-- grounded_event fired on NO task (nothing to compare in the fired stratum) --")

host.close(); toto_fs.close()
