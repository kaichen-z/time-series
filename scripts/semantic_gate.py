"""Offline: apply a DATA GATE on top of the LLM's semantic reference choices, to filter
its over-confident picks. Uses cached refs (.scratch/semantic_refs.json) -- no LLM.

Compares gate variants: raw (no gate), drop 'recent', stable-only (weekday/weekend),
and a significance+sample gate. The gate is exactly the auditable rule layer that the
neurosymbolic design calls for -- LLM supplies the semantic mapping, the rule vets it."""
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
from evolving_loop.adjustment.post_adjust import (
    EvidenceEffect, _canon_direction, _parse, horizon_window_mask, apply_bounded_delta)

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


def levels_and_counts(hv, hts):
    def wd(t):
        p = _parse(t); return p.weekday() if p else -1
    wkday = [v for v, t in zip(hv, hts) if 0 <= wd(t) < 5]
    wkend = [v for v, t in zip(hv, hts) if wd(t) >= 5]
    byday: dict = {}
    for v, t in zip(hv, hts):
        p = _parse(t)
        if p:
            byday.setdefault(p.date(), []).append(v)
    dmeans = sorted(statistics.mean(x) for x in byday.values()) if byday else []
    L = {"overall": statistics.mean(hv), "recent": statistics.mean(hv[-24:])}
    C = {"overall": len(hv), "recent": min(24, len(hv))}
    if wkday:
        L["weekday"], C["weekday"] = statistics.mean(wkday), len(wkday)
    if wkend:
        L["weekend"], C["weekend"] = statistics.mean(wkend), len(wkend)
    if dmeans:
        L["low_day"], C["low_day"] = dmeans[0], 1
        L["high_day"], C["high_day"] = dmeans[-1], 1
    return L, C


# gate variants: given ref, levels, counts, base_win -> bool (apply?)
STABLE = {"weekday", "weekend"}
def g_raw(ref, L, C, bw): return ref in L
def g_no_recent(ref, L, C, bw): return ref in L and ref != "recent"
def g_stable(ref, L, C, bw): return ref in STABLE
def g_sig(ref, L, C, bw):
    if ref not in L or ref == "recent" or bw <= 0:
        return False
    reldiff = abs(L[ref] - bw) / bw
    return C.get(ref, 0) >= 3 and reldiff >= 0.15   # enough samples + a real gap

GATES = {"raw": g_raw, "no_recent": g_no_recent, "stable_only": g_stable, "sig+stable_days": g_sig}

TASKDATA = []
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
    L, C = levels_and_counts(list(n.history_values), [str(x) for x in t.history_timestamps])
    bw = statistics.mean([base[i] for i in range(len(base)) if mask[i]])
    TASKDATA.append(dict(tid=tid, base=base, truth=n.future_values, mask=mask,
                         L=L, C=C, bw=bw, ref=REFS.get(tid, "none"), toto=joint(base, n.future_values)))

print(f"windowed+evidence tasks: {len(TASKDATA)}\n")
toto_mean = statistics.mean([d["toto"] for d in TASKDATA])
print(f"{'gate':16s}{'mean joint':>12s}{'vs toto':>10s}   acted / regressed")
print(f"{'toto':16s}{toto_mean:12.5f}{0.0:>10.4f}")
for gname, gfn in GATES.items():
    js, acted, regressed = [], 0, 0
    for d in TASKDATA:
        if gfn(d["ref"], d["L"], d["C"], d["bw"]) and d["bw"] > 0:
            factor = d["L"][d["ref"]] / d["bw"]
            proposed = [d["base"][i] * factor if d["mask"][i] else d["base"][i] for i in range(len(d["base"]))]
            out = apply_bounded_delta(list(d["base"]), proposed, max_frac=0.5)
            acted += 1
        else:
            out = d["base"]
        jj = joint(out, d["truth"])
        if jj > d["toto"] + 1e-6:
            regressed += 1
        js.append(jj)
    m = statistics.mean(js)
    print(f"{gname:16s}{m:12.5f}{toto_mean - m:>+10.4f}   {acted} / {regressed}")
host.close(); fs.close()
