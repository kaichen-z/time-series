#!/usr/bin/env python3
"""Build the frozen FULL-Train task pack (protocol v5) for ONE official dataset (TimesX or Time-MMD).

  * ALL official Train and Dev windows (closure asserted against the handoff index); official Test is NOT built (no Test
    labels are read; sealed Test ids are used only to assert that none entered the pack). Split mode (--split):
      train_2to1: Train split by WHOLE isolation unit (group_id prefix: TimesX entity group, Time-MMD domain - a domain never
                  appears in two parts, whatever the horizon) into F0 feedback (agents; opaque handles, aggregate feedback) and
                  F1 selection (host-only, once after all stages): F1 = round(units/3) whole units, the subset closest to 1/3 of
                  the Train tasks;
                  FINAL_dev = official Dev, opened once after the final program is locked.
      all_train:  F0 feedback = ALL Train; F1 selection = official Dev; the final check (official Test) is a separately
                  approved step outside this package.
  * views (NO labels): history, timestamps, frequency, horizon, a GENERIC target description (series/group identity removed;
    names go to private/forbidden_terms.json), frozen anchor forecasts (Toto/TimesFM/Moirai/Chronos/Granite where cached +
    the 5 frozen combined candidates) and document REFERENCES; documents (content + audited Haiku event cards) are stored
    once in shared/store/documents.json; forecasts as little-endian float32 in shared/store (viewstore.py) (Time-MMD tasks share retrieved documents). Run dirs replace task/document ids by per-run
    opaque handles (anon.py).
usage: build_pack.py --dataset {timesx,time_mmd} --split {train_2to1,all_train} --out PACK --repo TS_REPO --official-root ROOT"""
import argparse, hashlib, json, math, re, sys
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("--dataset", choices=("timesx", "time_mmd"), required=True)
P.add_argument("--out", type=Path, required=True)
P.add_argument("--repo", type=Path, required=True, help="time-series repo checkout (official loader + handoff/official_ts_llm)")
P.add_argument("--official-root", type=Path, required=True, help="dir with repos/{TimesX-project,MM-TSFlib} and artifacts/official_ts_alignment")
P.add_argument("--split", choices=("train_2to1", "all_train"), required=True,
               help="train_2to1: F0 = ~2/3 of Train tasks (whole groups), F1 = rest of Train, FINAL = official Dev; "
                    "all_train: F0 = ALL Train, F1 = official Dev, FINAL = official Test (not built here; separate approval)")
A = P.parse_args()
for need in (A.repo / "evolving_loop/official_benchmark_loader.py", A.repo / "handoff/official_ts_llm/alignment_manifest.json",
             A.official_root / "artifacts/official_ts_alignment", A.official_root / "repos"):
    if not need.exists(): raise SystemExit(f"preflight: missing {need}")
if A.out.exists(): raise SystemExit(f"refusing to overwrite existing pack {A.out}")
sys.path.insert(0, str(A.repo))
from evolving_loop.official_benchmark_loader import load_official_timesx, load_official_time_mmd, load_anchor_cache  # noqa: E402

DS = A.dataset; H_ = A.repo / "handoff/official_ts_llm"; ART = A.official_root / "artifacts/official_ts_alignment"
SEED = f"full-train-v5:{DS}"  # same hash order for both split modes


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def chunk_shas(p, size=64 << 20):
    out = []
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(size), b""): out.append(hashlib.sha256(c).hexdigest())
    return dict(chunk_bytes=size, sha256=out)


def hkey(s): return hashlib.sha256(f"{SEED}\0{s}".encode()).hexdigest()


def jt(f, y):  # Dr-CiK joint error: sMAE + sRMSE, each capped at 5 (scale = mean |truth|)
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


inputs = {"alignment_manifest": H_ / "alignment_manifest.json", "event_cards": H_ / f"haiku_output/split/{DS}_events_valid.jsonl",
          "task_index": H_ / ("timesx_train_dev_documents.jsonl" if DS == "timesx" else "time_mmd_train_dev_task_ids.jsonl")}
src_root = A.official_root / "repos" / ("TimesX-project" if DS == "timesx" else "MM-TSFlib")
parts = (load_official_timesx if DS == "timesx" else load_official_time_mmd)(
    manifest_path=inputs["alignment_manifest"], repository_root=src_root, anchor_cache=None, allow_unbound_anchors=True)
index = {}
with open(inputs["task_index"]) as f:
    for line in f:
        r = json.loads(line); index[r["task_id"]] = r["split"]
assert {w.task_id for w in parts.train} == {t for t, s in index.items() if s == "train"}, "Train closure vs handoff index"
assert {w.task_id for w in parts.dev} == {t for t, s in index.items() if s == "dev"}, "Dev closure vs handoff index"

# ---- time boundary (purge): no Train prediction target may reach into the Dev period ----
# TimesX: internal date split (Dev = target start >= 2024-09-01). Every Train window whose target END is on/after the
# boundary is dropped (a purge gap of one full horizon before the boundary). Time-MMD: the official chronological split
# already ends every Train target before the first Dev target (origin <= n_train - h), asserted below via the loader rule.
BOUNDARY = "2024-09-01" if DS == "timesx" else None
if DS == "timesx":
    assert all(w.future_timestamps[0][:10] >= BOUNDARY for w in parts.dev) and all(w.future_timestamps[0][:10] < BOUNDARY for w in parts.train)
    purged = [w for w in parts.train if w.future_timestamps[-1][:10] >= BOUNDARY]
else:
    purged = []
_pid = {w.task_id for w in purged}; train_ws = [w for w in parts.train if w.task_id not in _pid]
if DS == "timesx":
    assert max(w.future_timestamps[-1][:10] for w in train_ws) < BOUNDARY <= min(w.future_timestamps[0][:10] for w in parts.dev), "target overlap"
PURGE = dict(rule="drop Train windows whose prediction target ends on/after the Train/Dev boundary" if DS == "timesx" else
             "none needed: official chronological split, every Train target ends before the first Dev target",
             boundary=BOUNDARY, max_horizon=max(len(w.truth) for w in (*parts.train, *parts.dev)), purged=len(purged),
             purged_by_group={g: sum(w.group_id == g for w in purged) for g in sorted({w.group_id for w in purged})},
             purged_task_ids=sorted(_pid), train_after_purge=len(train_ws), dev=len(parts.dev),
             target_overlap_windows_after_purge=0)

# ---- frozen anchors (all shards) + combined candidates ----
names = {"toto_2_0": "toto", "timesfm_2_5": "timesfm", "moirai_2_0": "moirai", "chronos_bolt": "chronos", "granite_ttm_r2": "granite"}
anchors = {}
for name, short in names.items():
    for p in sorted(ART.glob(f"{DS}_{short}_anchors*.json")):
        inputs[f"anchor:{p.name}"] = p
        for tid, fc in load_anchor_cache(p).items():
            if name in fc: anchors.setdefault(tid, {}).setdefault(name, list(fc[name]))
inputs["combined_candidates"] = ART / "combined_v1" / f"{DS}_combined.jsonl"
combined = {}
with open(inputs["combined_candidates"]) as f:
    for line in f:
        r = json.loads(line); combined[r["task_id"]] = r["forecasts"]

# ---- split (parameterised; see --split) ----
# isolation unit = the underlying series/domain (group_id prefix), NOT domain x horizon: Time-MMD's 36 groups are 9 domain
# series x 4 horizons, so splitting by group_id would put the same series (overlapping windows) in F0 and F1
def unit(w): return w.group_id.split(":")[0]
groups = {}
for w in train_ws: groups.setdefault(unit(w), []).append(w)
order = sorted(groups, key=lambda g: (hkey(g), g))
if A.split == "train_2to1":
    # confirmed protocol: F0 ~ 2/3 of the GROUPS, F1 ~ 1/3 of the groups (whole isolation units). F1 = round(n_units / 3)
    # units (>= 1); among those subsets the one whose task count is closest to 1/3 of the Train tasks (ties -> hash order).
    # TimesX: 3 entity groups -> 1 in F1; Time-MMD: 9 domains -> 3 in F1.
    from itertools import combinations
    target, k = len(train_ws) / 3, max(1, round(len(order) / 3))
    best = min(((list(c), sum(len(groups[g]) for g in c)) for c in combinations(order, k)),
               key=lambda x: (abs(x[1] - target), [order.index(g) for g in x[0]]))
    f1_groups = set(best[0]); f0_groups = set(order) - f1_groups
    F = {"F0_feedback": [w for g in order if g in f0_groups for w in groups[g]],
         "F1_selection": [w for g in order if g in f1_groups for w in groups[g]],
         "FINAL_dev": list(parts.dev)}
else:
    F = {"F0_feedback": [w for g in order for w in groups[g]], "F1_selection": list(parts.dev)}  # FINAL = official Test, not built
assert not ({w.task_id for w in F["F0_feedback"]} & {w.task_id for w in F["F1_selection"]})
sealed = set(parts.sealed_test_task_ids)
assert not ({w.task_id for ws in F.values() for w in ws} & sealed), "Test leaked into the pack"
# ---- documents (deduplicated) with event cards; views with document references ----
need = {d.document_id for ws in F.values() for w in ws for d in w.documents}
cards = {}
with open(inputs["event_cards"]) as f:
    for line in f:
        r = json.loads(line)
        if r["document_id"] in need: cards[r["document_id"]] = r["events"]
# ---- information-time filter: every fact / event card an agent sees must be dated STRICTLY BEFORE the forecast origin ----
# origin date: TimesX = first target timestamp; Time-MMD = start_date of the first target row (the loader takes a task's
# documents from that row, whose retrieved facts can be dated inside the target period). Facts without a verifiable date
# are dropped. Deterministic calendar information (TimesX holiday_info) is kept but marked calendar=True.
ISO = re.compile(r"(\d{4}-\d{2}-\d{2})"); HEAD = re.compile(r"(\d{4}-\d{2}-\d{2}):\s")
doc_origin = {}
if DS == "timesx":
    for ws in F.values():
        for w in ws:
            for d in w.documents: doc_origin[d.document_id] = w.future_timestamps[0][:10]
else:
    import csv
    from evolving_loop.official_benchmark_loader import _opaque_id
    for rec_ in json.load(open(inputs["alignment_manifest"]))["time_mmd"]["data_files"]:
        rp = Path(rec_["path"]); rp = Path(*rp.parts[2:]) if rp.parts[:2] == ("repos", "MM-TSFlib") else rp
        with open(src_root / rp, newline="", encoding="utf-8-sig") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                for fld in ("Final_Search_2", "Final_Search_4", "Final_Search_6"):
                    doc_origin[_opaque_id("time_mmd_document", f"{rp}:{i}:{fld}")] = row["start_date"][:10]
assert need <= set(doc_origin), "documents without a known forecast origin"
TF = dict(facts_kept=0, facts_dropped_on_or_after_origin=0, facts_dropped_undated=0, events_kept=0, events_dropped_on_or_after_origin=0,
          events_dropped_undated=0, calendar_documents=0, calendar_events=0, documents_emptied=0, documents_dropped_future_covariates=0)


def ev_date(e):
    ts = [str(x)[:10] for x in (e.get("time_start"), e.get("time_end")) if x and ISO.fullmatch(str(x)[:10])]
    return max(ts) if ts else None


def filter_doc(did, content, events):
    o = doc_origin[did]; kind = did.rsplit("_", 1)[-1] if DS == "timesx" else "facts"
    if DS == "timesx" and did.endswith("_holiday_info"):  # deterministic calendar covariate (known in advance)
        TF["calendar_documents"] += 1; TF["calendar_events"] += len(events)
        return dict(content=content, events=[dict(e, calendar=True) for e in events], calendar=True)
    kept = []
    for e in events:
        d = ev_date(e)
        if d is None: TF["events_dropped_undated"] += 1
        elif d >= o: TF["events_dropped_on_or_after_origin"] += 1
        else: kept.append(e); TF["events_kept"] += 1
    if DS == "time_mmd":  # "Available facts are as follows: D1: facts D2: facts ..." -> keep blocks with date < origin
        parts_ = HEAD.split(content); blocks = []
        if parts_[0].strip().strip(":").strip() and parts_[0].strip() != "Available facts are as follows:": TF["facts_dropped_undated"] += 1
        for d, txt in zip(parts_[1::2], parts_[2::2]):
            if d < o and all(x < o for x in ISO.findall(txt)): blocks.append(f"{d}: {txt.strip()}"); TF["facts_kept"] += 1
            else: TF["facts_dropped_on_or_after_origin"] += 1
        content = ("Available facts are as follows: " + " ".join(blocks)) if blocks else ""
    elif kind == "scenario":  # official header (target period = the known forecast timestamps) + only dated, pre-origin events
        head = content.split("Recent related events")[0].strip()
        assert re.fullmatch(r"Prediction target period: from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\.", head), "unexpected scenario header"
        content = head + (" Recent related events (dated before the forecast origin): " + " ".join(
            f"<{i + 1}> {e.get('time_start') or ''}..{e.get('time_end') or ''}: {e.get('evidence', '')}" for i, e in enumerate(kept)) if kept else "")
        TF["facts_kept"] += len(kept)
    elif kind == "covariates_info":  # summary statistics over a past window; every date in it must be before the origin
        if any(d >= o for d in ISO.findall(content)): TF["documents_dropped_future_covariates"] += 1; return None
    if not content and not kept: TF["documents_emptied"] += 1; return None
    return dict(content=content, events=kept)


docs, views, truth, base_jt, coverage, dropped = {}, {}, {}, {}, {}, []
terms = set()
for part, ws in F.items():
    for w in ws:
        H = len(w.truth); mf = {}
        for name, fc in anchors.get(w.task_id, {}).items():
            if len(fc) == H and all(math.isfinite(x) for x in fc): mf[name] = fc
        for name, fc in (combined.get(w.task_id) or {}).items():
            if len(fc) == H and all(math.isfinite(x) for x in fc): mf[name] = [float(x) for x in fc]
        if "toto_2_0" not in mf: dropped.append(w.task_id); continue  # reference forecast required (reported)
        for name in mf: coverage[name] = coverage.get(name, 0) + 1
        refs = []
        for d in w.documents:
            if d.document_id not in docs: docs[d.document_id] = filter_doc(d.document_id, d.content, cards.get(d.document_id, []))
            if docs[d.document_id] is not None: refs.append(d.document_id)
        terms.add(w.group_id.split(":")[0])
        m = re.search(r"records (.+?) data in the (.+?) (?:domain|exchange|based)", w.target_description)
        if m: terms.update({m.group(1).strip(), m.group(2).strip()})
        m = re.search(r"Time-MMD (\S+) target series", w.target_description)
        if m: terms.add(m.group(1))
        views[w.task_id] = dict(dataset=DS, H=H, freq=w.frequency, history=list(w.history), history_timestamps=list(w.history_timestamps),
                                future_timestamps=list(w.future_timestamps), target_description=f"{'TimesX' if DS == 'timesx' else 'Time-MMD'} target series",
                                method_forecasts=mf, document_refs=refs)
        truth[w.task_id] = list(w.truth); base_jt[w.task_id] = jt(mf["toto_2_0"], truth[w.task_id])

out = A.out; (out / "shared").mkdir(parents=True); (out / "private").mkdir(parents=True)
sys.path.insert(0, str(Path(__file__).resolve().parent)); import viewstore  # noqa: E402
docs = {k: v for k, v in docs.items() if v is not None}
# independent post-filter verification (non-calendar content): no fact block / event card dated on or after the origin
_bad = 0
for k, v in docs.items():
    if v.get("calendar"): continue
    o = doc_origin[k]
    _bad += sum((ev_date(e) or "9999") >= o for e in v["events"])
    if DS == "time_mmd": _bad += sum(x >= o for x in ISO.findall(v["content"]))
    elif k.endswith("_covariates_info"): _bad += sum(x >= o for x in ISO.findall(v["content"]))
    elif k.endswith("_scenario"): _bad += sum(x >= o for x in ISO.findall(v["content"].split("Recent related events")[-1]) if "Recent related events" in v["content"])
assert _bad == 0, f"{_bad} non-calendar facts dated on/after the forecast origin remain"
TF["non_calendar_future_facts_after_filter"] = _bad
viewstore.save(out / "shared/store", views, docs)
# decode equivalence: every stored forecast vs the original float64 values (float32 rounding only)
import numpy as np  # noqa: E402
_st = viewstore.Store(out / "shared/store"); _maxabs = _maxrel = 0.0
for _k, _v in views.items():
    for _m, (_o, _n) in _st.meta[_k]["_fc"].items():
        _a = np.asarray(_v["method_forecasts"][_m], dtype=np.float64); _b = _st.fc[_o:_o + _n].astype(np.float64)
        _d = np.abs(_a - _b); _maxabs = max(_maxabs, float(_d.max(initial=0)))
        _maxrel = max(_maxrel, float((_d / np.maximum(np.abs(_a), 1e-12)).max(initial=0)))
DECODE = dict(max_abs_error=_maxabs, max_rel_error=_maxrel, note="float32 storage of float64 forecasts; history/truth stay float64 JSON")
json.dump(sorted(t for t in terms if len(t) >= 3), open(out / "private/forbidden_terms.json", "w"), indent=1)
counts = {}
for part, ws in F.items():
    ids = [w.task_id for w in ws if w.task_id in views]
    json.dump(dict(part=part, task_ids=ids, truth={t: truth[t] for t in ids}, base_jt={t: base_jt[t] for t in ids}),
              open(out / f"private/eval_{part}.json", "w"))
    counts[part] = dict(tasks=len(ids), isolation_units=sorted({unit(w) for w in ws}), group_ids=sorted({w.group_id for w in ws}))
receipt = dict(schema="full-train-pack-v5", dataset=DS, seed=SEED, split_mode=A.split, protocol=(
    "v5 train_2to1: ALL official Train split by whole group into F0 feedback (~2/3 of tasks; agents see opaque handles and one "
    "quantised aggregate fitness) and F1 selection (rest of Train; host-only, once, after all stages, fail closed); official Dev = "
    "one-time final validation after lock; official Test sealed" if A.split == "train_2to1" else
    "v5 all_train: F0 = ALL official Train (agents, aggregate feedback); F1 = official Dev (host-only, once, after all stages, fail "
    "closed); official Test = one-time final evaluation after lock, separate approval (Test labels not read here)"),
    official_counts=dict(train=len(parts.train), dev=len(parts.dev), sealed_test_ids=len(sealed)),
    time_boundary=PURGE, information_time_filter=dict(TF, rule="facts/event cards kept only if dated strictly before the forecast origin; "
                                                      "undated facts dropped; TimesX holiday_info kept as calendar=True"),
    scaler="TimesX: raw values, no fitted statistics; Time-MMD: z-score fitted on the official Train segment only (loader)",
    parts=counts, dropped_without_toto_anchor=dropped, documents=len(docs), documents_missing_event_cards=len(need - set(cards)),
    anchor_coverage={k: f"{v}/{len(views)}" for k, v in sorted(coverage.items())}, uses_test_labels=False,
    input_sha256={k: sha(v) for k, v in sorted(inputs.items())}, source_fingerprint=parts.source_fingerprint,
    decode_equivalence=DECODE,
    store_layout=dict(file="shared/store/forecasts.f32", dtype="float32", endianness="little", shape=[(out / "shared/store/forecasts.f32").stat().st_size // 4],
                      index="shared/store/views_meta.json: view._fc = {method: [offset, length]} (float32 elements)",
                      chunk_sha256=chunk_shas(out / "shared/store/forecasts.f32")),
    output_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
json.dump(receipt, open(out / "pack_receipt.json", "w"), indent=1)
print(json.dumps(dict(dataset=DS, official=receipt["official_counts"], parts={k: (v["tasks"], v["isolation_units"]) for k, v in counts.items()},
                      dropped=len(dropped), documents=len(docs), coverage=receipt["anchor_coverage"])))
