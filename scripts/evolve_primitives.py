"""LLM-authored Loop-B correction PRIMITIVES, kept only if they beat the flat CorDP baseline.

This is the "LLM writes the code, evolution keeps what generalizes, the kernel guarantees safety"
pattern applied to the DECISION primitives (not just retrieval features). The LLM proposes new ways
to turn document multipliers into a corrected forecast (ramp / decay / partial-trust / taper / ...);
each is sandboxed (llm_design.safe_compile_primitive) and then FORCED through the fixed safety
kernel -- every step outside a grounded window is reset to base, and the whole thing is clamped to
base +/- 50% -- so a bad primitive can do no harm. We keep a primitive only if it beats the flat
scaling baseline (== CorDPAdjust) on TRAIN, and we REPORT its TEST number so overfit survivors are
visible. The surviving primitives' SOURCE is the auditable "discovery".

Run: TMPDIR=$PWD/.scratch PYTHONPATH=$PWD [N_PROPOSE=6] .venv/bin/python scripts/evolve_primitives.py
"""
from __future__ import annotations
import json, os, shutil, statistics
from pathlib import Path

from common.metrics import drcik_point_metrics
from evolving_loop.data import load_context_tasks_by_ids
from evolving_loop.v2.real.host import _load_screening_policy
from numerical_agent.evolution.portfolio import read_policy_file
from numerical_agent.evolution.forecast_store import ForecastStore
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask
from scripts.llm_design import safe_compile_primitive, propose_primitives

ROOT = Path(".").resolve(); CAP = 5.0
IDH = "90a281166723e9ea43e58e9468c286675dbe1dab430a5a7d3e6b38526a4313c2"
TASKS = "external/Dr-CiK/full-download/Dr-CiK_public/tasks"
SPLIT = json.loads((ROOT / "splits/drcik_public_80_20_99_v3.json").read_text())["partitions"]
mr = ROOT / "runs/method_evolution/v001"
port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
fs = ForecastStore(ROOT / "runs/champion_forecasts/gpt56sol_high_toto_balanced_v3_20260906",
                   mr / "methods.py", mr / "skills.py", port, None, screening_hash=scr.fingerprint(),
                   runtime_identity={}, cache_only=True, identity_hash_override=IDH)


def load(part):
    """Only tasks that HAVE >=1 document correction window (where primitives differ from base)."""
    cards = json.loads((ROOT / f".scratch/cordp_cards_{part}.json").read_text())
    ids = tuple(t for t in SPLIT[part]["task_ids"] if t in cards)
    tasks = {t.numeric.task_id: t for t in load_context_tasks_by_ids(TASKS, ids)}
    data = []
    for tid in ids:
        t = tasks.get(tid)
        if t is None:
            continue
        n = t.numeric; truth = list(n.future_values); H = len(truth)
        base = tuple(fs.forecast("toto_2_0", tuple(n.history_values), n.prediction_length, n.frequency))
        if len(base) != H:
            continue
        c = cards.get(tid) or {}
        fts = [str(x) for x in t.future_timestamps]
        corr_idx = []
        for r in (c.get("corrections") or []):
            if len(r) < 3:
                continue
            mask = horizon_window_mask(fts, str(r[0]), str(r[1]))
            on = [i for i, m in enumerate(mask) if m]
            if on:
                corr_idx.append([on[0], on[-1] + 1, float(r[2])])
        if not corr_idx:
            continue
        data.append(dict(tid=tid, base=list(base), truth=truth, hist=list(n.history_values), corr=corr_idx))
    return data


def apply_primitive(fn, base, history, corr):
    """Run an LLM primitive, then FORCE safety: outside grounded windows -> base; clamp base+/-50%."""
    H = len(base)
    proposed = fn(base, history, corr)
    covered = [False] * H
    for s, e, _m in corr:
        for i in range(max(0, int(s)), min(H, int(e))):
            covered[i] = True
    grounded = [proposed[i] if covered[i] else base[i] for i in range(H)]
    return list(apply_bounded_delta(base, grounded))


def _joint(fc, truth):
    m = drcik_point_metrics(truth, fc, cap=CAP)
    return (m["smae"] + m["srmse"]) / 2.0


def gain(fn, data):
    """Whole-horizon do-no-harm gain vs Toto (positive = better). Downside-penalized."""
    ds = [_joint(d["base"], d["truth"]) - _joint(apply_primitive(fn, d["base"], d["hist"], d["corr"]), d["truth"])
          for d in data]
    return statistics.mean(ds) - 0.5 * statistics.mean(min(0.0, x) for x in ds) if ds else 0.0


def cv_gain(fn, data, K=3):
    """Mean +/- std of held-out per-fold gain (error bar). Primitives have no fitted params, so this
    is a variance estimate of the gain, and TEST is the real generalization check."""
    scores = [gain(fn, [d for i, d in enumerate(data) if i % K == f]) for f in range(K)]
    scores = [s for s in scores if s is not None]
    return (statistics.mean(scores), statistics.pstdev(scores) if len(scores) > 1 else 0.0)


def whole(fn, data):
    """Aggregate sMAE/sRMSE vs Toto over the (touched) set, with wins/reg."""
    rows, wins, reg = [], 0, 0
    for d in data:
        out = apply_primitive(fn, d["base"], d["hist"], d["corr"])
        mb = drcik_point_metrics(d["truth"], d["base"], cap=CAP)
        mo = drcik_point_metrics(d["truth"], out, cap=CAP)
        rows.append((mb["smae"], mb["srmse"], mo["smae"], mo["srmse"]))
        if mo["smae"] + mo["srmse"] < mb["smae"] + mb["srmse"] - 1e-9: wins += 1
        elif mo["smae"] + mo["srmse"] > mb["smae"] + mb["srmse"] + 1e-9: reg += 1
    bs = statistics.mean(r[0] for r in rows); br = statistics.mean(r[1] for r in rows)
    os_ = statistics.mean(r[2] for r in rows); or_ = statistics.mean(r[3] for r in rows)
    return f"sMAE {bs:.4f}->{os_:.4f} ({(bs-os_)/bs:+.2%}) | sRMSE {br:.4f}->{or_:.4f} ({(br-or_)/br:+.2%}) | wins {wins} reg {reg}"


# flat scaling in-window == the current CorDPAdjust behaviour (the baseline to beat)
FLAT = safe_compile_primitive("flat_scale", (
    "def flat_scale(base, history, corrections):\n"
    "    out = list(base)\n"
    "    for s, e, m in corrections:\n"
    "        for i in range(int(s), int(e)):\n"
    "            if 0 <= i < len(out):\n"
    "                out[i] = base[i] * m\n"
    "    return out\n"))

TR, TE = load("train"), load("public_test")
print(f"== LLM-authored Loop-B primitives (keep iff beats flat CorDP on TRAIN; TEST = honest check) ==")
print(f"tasks with document windows: train {len(TR)} | test {len(TE)}\n")

flat_tr = cv_gain(FLAT, TR); flat_te = gain(FLAT, TE)
print(f"BASELINE flat CorDP: train CV-gain {flat_tr[0]:+.4f}±{flat_tr[1]:.4f} | test gain {flat_te:+.4f}")
print(f"   flat test (touched): {whole(FLAT, TE)}\n")

from common.llm import ClaudeCLIClient, ClaudeCLIConfig
client = ClaudeCLIClient(ClaudeCLIConfig(binary=shutil.which("claude") or "claude", model="haiku",
                                         cache_dir=str(ROOT / ".scratch/prim-design-cache")))
try:
    proposed = propose_primitives(client, {"flat_scale"}, n=int(os.environ.get("N_PROPOSE", "6")))
except Exception as e:
    print(f"LLM proposer failed: {type(e).__name__}: {e}"); proposed = {}
print(f"LLM proposed {len(proposed)} that compiled: {list(proposed)}\n")

kept = []
for name, fn in proposed.items():
    tr = cv_gain(fn, TR); te = gain(fn, TE)
    beats_train = tr[0] >= flat_tr[0] + 0.0003
    generalizes = te >= flat_te - 1e-9
    tag = "✓ KEEP" if beats_train else "✗ drop"
    gen = "  ↑generalizes(test)" if (beats_train and generalizes) else ("  ↓overfit(test<flat)" if beats_train else "")
    print(f"  {name:28s} train CV {tr[0]:+.4f}±{tr[1]:.4f} | test {te:+.4f}  {tag}{gen}")
    if beats_train:
        kept.append((name, fn, te, generalizes))

print(f"\n== survivors (beat flat on train): {[k[0] for k in kept] or 'NONE'} ==")
winners = [k for k in kept if k[3]]
print(f"== that ALSO generalize (beat flat on test): {[k[0] for k in winners] or 'NONE'} ==\n")
for name, fn, te, gen in (winners or kept):
    print(f"--- {name}  (test gain {te:+.4f}, {'generalizes' if gen else 'overfit'}) ---")
    print(fn.source)
    print(f"   test (touched): {whole(fn, TE)}\n")
fs.close()
