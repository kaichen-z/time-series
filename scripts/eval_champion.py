"""Evaluate the EVOLVED champion controller (merged-evidence) on train/dev/test -> the proper
80/20/99 generalization of the full evolved policy (not just the CorDP gate). Whole-horizon
drcik_point_metrics (cap=5.0), the trustworthy metric. Evidence is the MERGED source
(effects_from_cordp), so RegimeAdjust/SemanticAdjust no longer no-op on test.
Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD .venv/bin/python scripts/eval_champion.py"""
from __future__ import annotations
import json, statistics
from pathlib import Path
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from common.metrics import drcik_point_metrics
from evolving_loop.adjustment.post_adjust import effects_from_cordp
from evolving_loop.adjustment.math_menu import build_math_menu
from evolving_loop.adjustment.controller import (
    Controller, SelectBase, MenuAdjust, SemanticAdjust, RegimeAdjust, CorDPAdjust, run_controller)

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
REFS = json.loads((ROOT / ".scratch/semantic_refs.json").read_text()) if (ROOT / ".scratch/semantic_refs.json").exists() else {}

# the evolved (merged-evidence) champion
CHAMP = Controller(steps=(
    SelectBase(candidate="toto_2_0"),
    MenuAdjust(prefer=("decay", "regime", "trend"), cap=0.29),
    SemanticAdjust(cap=0.35, require_significant=True),
    RegimeAdjust(regime="weekend_weekday", cap=0.07, fill_unknown_direction=True),
    CorDPAdjust(conf_min=0.87, wfrac_max=0.25, mult_lo=0.6, mult_hi=1.6, shrink=1.0),
))

mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def evalsplit(part):
    cp = ROOT / f".scratch/cordp_cards_{part}.json"
    if not cp.exists():
        print(f"{part}: no cordp cards"); return
    cards = json.loads(cp.read_text())
    ids = tuple(t for t in SPLIT[part]["task_ids"] if t in cards)
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(
        "external/Dr-CiK/full-download/Dr-CiK_public/tasks", ids)}
    rows = []
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        n = t.numeric; truth = list(n.future_values)
        base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        if len(base) != len(truth):
            continue
        c = cards.get(tid) or {}
        corrs = tuple((r[0], r[1], r[2]) for r in (c.get("corrections") or []) if len(r) >= 3)
        fts = [str(x) for x in t.future_timestamps]
        hv, hts = list(n.history_values), [str(x) for x in t.history_timestamps]
        out, _ = run_controller(CHAMP, {"toto_2_0": base}, effects_from_cordp(corrs), hv, hts, fts,
                                semantic_ref=REFS.get(tid, "none"), cordp_conf=float(c.get("confidence") or 0.0),
                                cordp_corrections=corrs, menu=build_math_menu(hv, hts, fts))
        mb = drcik_point_metrics(truth, list(base), cap=CAP)
        mo = drcik_point_metrics(truth, list(out), cap=CAP)
        touched = any(abs(a - b) > 1e-9 for a, b in zip(base, out))
        rows.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"], touched))
    bs = statistics.mean(r[0] for r in rows); br = statistics.mean(r[1] for r in rows)
    os_ = statistics.mean(r[2] for r in rows); or_ = statistics.mean(r[3] for r in rows)
    tw = sum(1 for r in rows if r[4] and (r[2]+r[3]) < (r[0]+r[1]) - 1e-9)
    tr = sum(1 for r in rows if r[4] and (r[2]+r[3]) > (r[0]+r[1]) + 1e-9)
    tt = sum(1 for r in rows if r[4])
    print(f"  {part:11s} ({len(rows)} tasks, {tt} touched): "
          f"sMAE {bs:.4f}->{os_:.4f} ({(bs-os_)/bs:+.2%}) | sRMSE {br:.4f}->{or_:.4f} ({(br-or_)/br:+.2%}) "
          f"| wins {tw} reg {tr}")


print("== evolved champion (merged evidence), whole-horizon drcik_point_metrics cap=5.0 ==")
print("  controller: SelectBase -> MenuAdjust -> SemanticAdjust -> RegimeAdjust -> CorDPAdjust(conf>=0.87)")
for part in ("train", "dev", "public_test"):
    evalsplit(part)
fs.close()
