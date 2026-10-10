#!/usr/bin/env python3
"""Build the frozen task pack for the Dr-CiK-budget-aligned evolution on ONE official dataset.

Dr-CiK runsheet v3.3.2 used 80 Train tasks and a 3-fold split of them (folds 0,1 VISIBLE, fold 2 HIDDEN) for every
agent stage (L4, L5-R1, L7); L5-R2 re-split the same tasks with a new seed (prep_mh2). This script reproduces that
structure on the official TimesX / Time-MMD Train partition:

  * 80 official Train tasks, deterministic, whole-group 3-fold (no group in two folds), target sizes 27/27/26;
  * primary folds (L4, L5-R1, L7) and secondary folds (L5-R2, different seed, same 80 tasks, whole-group);
  * agent-visible views (NO labels, NO group_id): history, timestamps, frequency, horizon, target description,
    frozen anchor forecasts (Toto/TimesFM/Moirai/Chronos/Granite where cached + the 5 frozen combined candidates),
    official documents with the audited Haiku event cards attached;
  * private truth + Toto reference errors, kept outside the agents' sandbox;
  * a receipt with every input SHA-256, coverage/fallback counts, fold task/group IDs and counts.

Official Dev / sealed Test IDs, labels and event cards never enter the pack (asserted).
usage: build_pack.py --dataset {timesx,time_mmd} --out PACK_DIR [paths...]"""
import argparse, hashlib, json, math, sys
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("--dataset", choices=("timesx", "time_mmd"), required=True)
P.add_argument("--out", type=Path, required=True)
P.add_argument("--repo", type=Path, required=True, help="time-series repo checkout (official loader + handoff/official_ts_llm)")
P.add_argument("--official-root", type=Path, required=True,
               help="directory holding repos/{TimesX-project,MM-TSFlib} and artifacts/official_ts_alignment")
P.add_argument("--tasks", type=int, default=80)
A = P.parse_args()
for need in (A.repo / "evolving_loop/official_benchmark_loader.py", A.repo / "handoff/official_ts_llm/alignment_manifest.json",
             A.official_root / "artifacts/official_ts_alignment", A.official_root / "repos"):
    if not need.exists(): raise SystemExit(f"preflight: missing {need}")
if A.out.exists(): raise SystemExit(f"refusing to overwrite existing pack {A.out}")
sys.path.insert(0, str(A.repo))
from evolving_loop.official_benchmark_loader import load_official_timesx, load_official_time_mmd, load_anchor_cache  # noqa: E402

DS = A.dataset; H_ = A.repo / "handoff/official_ts_llm"; ART = A.official_root / "artifacts/official_ts_alignment"
SEED_PRIMARY, SEED_SECONDARY = f"drcik-aligned-v1:{DS}:primary", f"drcik-aligned-v1:{DS}:secondary"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def hkey(seed, s): return hashlib.sha256(f"{seed}\0{s}".encode()).hexdigest()


inputs = {}
manifest = H_ / "alignment_manifest.json"; inputs["alignment_manifest"] = manifest
cards_f = H_ / f"haiku_output/split/{DS}_events_valid.jsonl"; inputs["event_cards"] = cards_f
index_f = H_ / ("timesx_train_dev_documents.jsonl" if DS == "timesx" else "time_mmd_train_dev_task_ids.jsonl"); inputs["task_index"] = index_f
src_root = A.official_root / "repos" / ("TimesX-project" if DS == "timesx" else "MM-TSFlib")
loader = load_official_timesx if DS == "timesx" else load_official_time_mmd
parts = loader(manifest_path=manifest, repository_root=src_root, anchor_cache=None, allow_unbound_anchors=True)

# ---- anchors: frozen caches (coverage + fallback are reported, never silently filled) ----
cache_files = {"toto_2_0": f"{DS}_toto_anchors.json", "timesfm_2_5": f"{DS}_timesfm_anchors.json", "moirai_2_0": f"{DS}_moirai_anchors.json",
               "chronos_bolt": f"{DS}_chronos_anchors.json", "granite_ttm_r2": f"{DS}_granite_anchors.json"}
anchors = {}
for name, fn in cache_files.items():
    for p in sorted(ART.glob(fn.replace(".json", "*.json"))):  # base file + shards (e.g. timesfm shard0/1)
        inputs[f"anchor:{p.name}"] = p
        for tid, fc in load_anchor_cache(p).items():
            if name in fc: anchors.setdefault(tid, {}).setdefault(name, list(fc[name]))
comb_f = ART / "combined_v1" / f"{DS}_combined.jsonl"; inputs["combined_candidates"] = comb_f
combined = {}
with open(comb_f) as f:
    for line in f:
        r = json.loads(line); combined[r["task_id"]] = r["forecasts"]

# ---- official Train only; choose 80 tasks with whole-group folds ----
train_ids = set()
with open(index_f) as f:
    for line in f:
        r = json.loads(line)
        if r["split"] == "train": train_ids.add(r["task_id"])
dev_ids = {w.task_id for w in parts.dev}; sealed = set(parts.sealed_test_task_ids)
groups = {}
for w in parts.train:
    assert w.task_id in train_ids, w.task_id
    if "toto_2_0" not in anchors.get(w.task_id, {}): continue  # reference forecast required (reported below)
    groups.setdefault(w.group_id, []).append(w)
for g in groups: groups[g].sort(key=lambda w: hkey("task", w.task_id))
quota = [A.tasks // 3 + (1 if i < A.tasks % 3 else 0) for i in range(3)]  # 27/27/26


def make_folds(seed, fixed_tasks=None):
    order = sorted(groups, key=lambda g: (hkey(seed, g), g))
    if fixed_tasks is not None:  # re-split an existing task set (secondary folds): whole groups, round-robin
        gs = [g for g in order if any(w.task_id in fixed_tasks for w in groups[g])]
        folds = [[], [], []]
        for i, g in enumerate(gs):
            folds[i % 3] += [w for w in groups[g] if w.task_id in fixed_tasks]
        return folds
    folds_groups = [order[i::3] for i in range(3)]  # whole groups assigned to folds
    folds = []
    for fg, q in zip(folds_groups, quota):
        queues = {g: list(groups[g]) for g in fg}; sel = []
        while len(sel) < q and any(queues.values()):
            for g in fg:
                if queues[g] and len(sel) < q: sel.append(queues[g].pop(0))
        if len(sel) < q: raise SystemExit(f"fold quota {q} not reachable with whole groups {fg}")
        folds.append(sel)
    return folds


primary = make_folds(SEED_PRIMARY)
chosen = {w.task_id: w for f in primary for w in f}
secondary = make_folds(SEED_SECONDARY, fixed_tasks=set(chosen))
if [{w.task_id for w in f} for f in secondary] == [{w.task_id for w in f} for f in primary]:
    secondary = secondary[1:] + secondary[:1]  # only possible with 3 groups (TimesX): force a different hidden fold
assert len(chosen) == A.tasks
for F in (primary, secondary):
    gsets = [{w.group_id for w in f} for f in F]
    assert not (gsets[0] & gsets[1] or gsets[0] & gsets[2] or gsets[1] & gsets[2]), "group crosses folds"
    assert sum(len(f) for f in F) == A.tasks
assert not (set(chosen) & (dev_ids | sealed)), "Dev/Test leaked into the pack"

# ---- event cards for the chosen tasks' documents ----
need = {d.document_id for w in chosen.values() for d in w.documents}
cards = {}
with open(cards_f) as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in need: cards[r["document_id"]] = r["events"]
missing_cards = sorted(need - set(cards))

# ---- views (no labels, no group_id) / private truth ----
def jt(f, y):  # Dr-CiK joint error: sMAE + sRMSE, each capped at 5 (scale = mean |truth|)
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


views, truth, base_jt, coverage = {}, {}, {}, {}
for tid, w in sorted(chosen.items()):
    H = len(w.truth); mf = {}
    for name, fc in anchors.get(tid, {}).items():
        if len(fc) == H and all(math.isfinite(x) for x in fc): mf[name] = fc
    for name, fc in (combined.get(tid) or {}).items():
        if len(fc) == H and all(math.isfinite(x) for x in fc): mf[name] = [float(x) for x in fc]
    for name in mf: coverage[name] = coverage.get(name, 0) + 1
    views[tid] = dict(tid=tid, dataset=DS, H=H, freq=w.frequency, history=list(w.history),
                      history_timestamps=list(w.history_timestamps), future_timestamps=list(w.future_timestamps),
                      target_description=w.target_description, method_forecasts=mf,
                      documents=[dict(document_id=d.document_id, content=d.content, events=cards.get(d.document_id, [])) for d in w.documents])
    truth[tid] = list(w.truth); base_jt[tid] = jt(mf["toto_2_0"], truth[tid])

# Protocol v2: series / group identity is removed from agent views (a generic target description), so programs cannot
# key rules on particular series or groups; the identifying names go to a private list used by the static check.
import re as _re
terms = set()
for tid, w in chosen.items():
    terms.add(w.group_id.split(":")[0])
    m = _re.search(r"records (.+?) data in the (.+?) (?:domain|exchange|based)", w.target_description)
    if m: terms.update({m.group(1).strip(), m.group(2).strip()})
    m = _re.search(r"Time-MMD (\S+) target series", w.target_description)
    if m: terms.add(m.group(1))
for v in views.values(): v["target_description"] = "TimesX target series" if DS == "timesx" else "Time-MMD target series"
out = A.out; (out / "shared").mkdir(parents=True, exist_ok=True); (out / "private").mkdir(parents=True, exist_ok=True)
json.dump(sorted(t for t in terms if len(t) >= 3), open(out / "private/forbidden_terms.json", "w"), indent=1)
fold_ids = lambda F: [[w.task_id for w in f] for f in F]
json.dump(views, open(out / "shared/views_train.json", "w"))
# Protocol v2: one private file per fold so that F1 (selection) and F2 (final test) truth can be opened only by
# final_select.py (F2 only after the final program is locked). F0 is the agents' feedback fold.
for i, name in enumerate(("F0_feedback", "F1_selection", "F2_final_test")):
    ids = [w.task_id for w in primary[i]]
    json.dump(dict(fold=name, task_ids=ids, truth={t: truth[t] for t in ids}, base_jt={t: base_jt[t] for t in ids}),
              open(out / f"private/eval_{name}.json", "w"))
receipt = dict(
    schema="drcik-aligned-pack-v1", dataset=DS, uses_official_train_only=True, uses_external_dev=False, uses_test_ids_or_labels=False,
    task_count=len(chosen), seeds=dict(primary=SEED_PRIMARY, secondary=SEED_SECONDARY),
    folds_primary=[dict(fold=i, role=("F0_feedback", "F1_selection", "F2_final_test")[i], n=len(f), task_ids=[w.task_id for w in f], group_ids=sorted({w.group_id for w in f})) for i, f in enumerate(primary)],
    folds_secondary=[dict(fold=i, role="hidden" if i == 2 else "visible", n=len(f), task_ids=[w.task_id for w in f], group_ids=sorted({w.group_id for w in f})) for i, f in enumerate(secondary)],
    official_train_groups=len(groups), anchor_coverage={k: f"{v}/{len(chosen)}" for k, v in sorted(coverage.items())},
    documents=len(need), documents_missing_event_cards=len(missing_cards), events=sum(len(v) for v in cards.values()),
    reference="toto_2_0", internal_fitness="Toto-relative sMAE+sRMSE (cap 5), PEN 0.5, STD_W 0.25 (Dr-CiK v3.3.2)",
    input_sha256={k: sha(v) for k, v in sorted(inputs.items())}, source_fingerprint=parts.source_fingerprint,
    protocol="v2: F0 feedback (agents, aggregate only) / F1 host-only selection after all stages / F2 final test once after lock",
    output_sha256={"shared/views_train.json": sha(out / "shared/views_train.json"),
                   **{f"private/eval_{n}.json": sha(out / f"private/eval_{n}.json") for n in ("F0_feedback", "F1_selection", "F2_final_test")}})
json.dump(receipt, open(out / "pack_receipt.json", "w"), indent=1)
print(json.dumps(dict(dataset=DS, tasks=len(chosen), primary=[len(f) for f in primary], secondary=[len(f) for f in secondary],
                      groups_primary=[len(r["group_ids"]) for r in receipt["folds_primary"]], coverage=receipt["anchor_coverage"],
                      missing_cards=len(missing_cards), events=receipt["events"])))
