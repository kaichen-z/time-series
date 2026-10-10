#!/usr/bin/env python3
"""Split / closure / storage proof for a protocol-v5 pack (no model, no Test labels):
  * the Train parts together are EXACTLY the official Train ids of the handoff index, the Dev part EXACTLY the official Dev ids;
  * parts are pairwise disjoint by task; Train parts (train_2to1) are disjoint by isolation unit (whole domain/entity group);
  * the view store holds exactly the pack tasks (no Test id); the truth files match their ids;
  * every file matches the receipt's SHA-256; forecasts.f32 matches its chunk SHA-256s, dtype/endianness/shape;
  * every agent-visible view has no identity keys and the single generic target description;
  * (--run) a run dir holds only F0 tasks, under opaque row handles / document ids, no task ids in its store text.
usage: closure_proof.py --pack PACK --task-index HANDOFF_TASK_INDEX [--run RUN_DIR]"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import viewstore  # noqa: E402
P = argparse.ArgumentParser(); P.add_argument("--pack", type=Path, required=True); P.add_argument("--task-index", type=Path, required=True)
P.add_argument("--run", type=Path); A = P.parse_args()
rec = json.load(open(A.pack / "pack_receipt.json")); mode = rec["split_mode"]
names = ["F0_feedback", "F1_selection"] + (["FINAL_dev"] if mode == "train_2to1" else [])
F = {n: json.load(open(A.pack / f"private/eval_{n}.json")) for n in names}
ids = {n: set(f["task_ids"]) for n, f in F.items()}
split = {r["task_id"]: r["split"] for r in json.load(open(A.task_index))["rows"]}
train = {t for t, s in split.items() if s == "train"}; dev = {t for t, s in split.items() if s == "dev"}
train_parts = ["F0_feedback", "F1_selection"] if mode == "train_2to1" else ["F0_feedback"]
dev_part = "FINAL_dev" if mode == "train_2to1" else "F1_selection"
S = viewstore.Store(A.pack / "shared/store"); keys = set(S.keys())
checks = {}
checks["tasks_pairwise_disjoint"] = all(not (ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:])
tb = rec["time_boundary"]; purged = set(tb["purged_task_ids"])
checks["train_parts_union_plus_purged_equals_official_train"] = (set().union(*(ids[n] for n in train_parts)) | purged) == train and not (purged & set().union(*ids.values()))
checks["purge_counts_consistent"] = tb["purged"] == len(purged) and tb["train_after_purge"] == sum(len(ids[n]) for n in train_parts) and purged <= train
checks["dev_part_equals_official_dev"] = ids[dev_part] == dev
if mode == "train_2to1":
    u = {n: set(rec["parts"][n]["isolation_units"]) for n in train_parts}
    checks["train_parts_disjoint_by_isolation_unit"] = not (u["F0_feedback"] & u["F1_selection"])
checks["store_keys_equal_pack_tasks_no_test"] = keys == set().union(*ids.values()) and keys <= set(split)
if tb["boundary"]:  # TimesX internal date split: recheck from the stored timestamps
    M = S.meta; B = tb["boundary"]
    tr_end = max(M[t]["future_timestamps"][-1][:10] for n in train_parts for t in ids[n]); dv_start = min(M[t]["future_timestamps"][0][:10] for t in ids[dev_part])
    checks["no_train_target_reaches_dev_period"] = tr_end < B <= dv_start
    checks["history_before_origin"] = all(M[t]["history_timestamps"][-1] < M[t]["future_timestamps"][0] for t in M)
tf = rec["information_time_filter"]
checks["zero_non_calendar_future_facts"] = tf["non_calendar_future_facts_after_filter"] == 0
checks["event_cards_dated_or_calendar"] = all(d.get("calendar") or all(e.get("time_start") or e.get("time_end") for e in d["events"]) for d in S.docs.values())
checks["counts_match_official"] = (len(train), len(dev)) == (rec["official_counts"]["train"], rec["official_counts"]["dev"]) and not rec["dropped_without_toto_anchor"]
checks["truth_files_match_ids"] = all(set(f["truth"]) == ids[n] == set(f["base_jt"]) and f["task_ids"] == list(dict.fromkeys(f["task_ids"])) for n, f in F.items())
checks["uses_no_test_labels"] = rec["uses_test_labels"] is False and not (A.pack / "private/eval_FINAL_test.json").exists()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


files = {p.relative_to(A.pack).as_posix() for p in A.pack.rglob("*") if p.is_file()} - {"pack_receipt.json"}
checks["receipt_sha256_all_files"] = files == set(rec["output_sha256"]) and all(sha(A.pack / f) == h for f, h in rec["output_sha256"].items())
lay = rec["store_layout"]; fc = A.pack / lay["file"]; cs = []
with open(fc, "rb") as f:
    for c in iter(lambda: f.read(lay["chunk_sha256"]["chunk_bytes"]), b""): cs.append(hashlib.sha256(c).hexdigest())
checks["store_chunk_sha256"] = cs == lay["chunk_sha256"]["sha256"]
checks["store_dtype_shape"] = (lay["dtype"], lay["endianness"]) == ("float32", "little") and fc.stat().st_size == 4 * lay["shape"][0]
spans = sorted(tuple(x) for m in S.meta.values() for x in m["_fc"].values())
checks["store_offsets_tile_file_exactly"] = spans[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in zip(spans, spans[1:])) and spans[-1][0] + spans[-1][1] == lay["shape"][0]
checks["store_decode_rel_error_le_1e-7"] = rec["decode_equivalence"]["max_rel_error"] <= 1e-7
ID_KEYS = ("tid", "task_id", "group_id", "entity_name", "series", "variable")
checks["pack_views_no_identity_keys"] = not any(k in m for m in S.meta.values() for k in ID_KEYS)
checks["pack_views_single_target_description"] = len({m["target_description"] for m in S.meta.values()}) == 1
if A.run:
    R = viewstore.Store(A.run / "shared/store"); re_ = json.load(open(A.run / "private/eval_data.json"))
    hm = json.load(open(A.run / "private/handle_map.json")); txt = (A.run / "shared/store/views_meta.json").read_text()
    checks["run_dir_views_only_F0"] = set(hm.values()) == ids["F0_feedback"] and set(R.keys()) == set(hm)
    checks["run_dir_truth_only_F0"] = set(re_["truth"]) == set(hm) and len(re_["folds"]) == 1
    checks["run_dir_opaque_handles"] = all(h.startswith("r_") for h in R.keys()) and not any(t in txt for t in list(ids["F0_feedback"])[:2000])
    checks["run_dir_no_identity_keys"] = not any(k in m for m in R.meta.values() for k in ID_KEYS)
    checks["run_dir_opaque_document_ids"] = all(r.startswith("d_") for m in R.meta.values() for r in m["document_refs"]) and all(d.startswith("d_") for d in R.docs)
    checks["run_dir_no_private_files_in_shared"] = not any(p.name.startswith("eval_") or p.name == "handle_map.json" for p in (A.run / "shared").rglob("*"))
out = dict(dataset=rec["dataset"], split_mode=mode, counts={n: len(s) for n, s in ids.items()}, official=rec["official_counts"],
           isolation_units={n: rec["parts"][n]["isolation_units"] for n in names}, checks=checks, ok=all(checks.values()))
print(json.dumps(out, indent=1)); raise SystemExit(0 if out["ok"] else 1)
