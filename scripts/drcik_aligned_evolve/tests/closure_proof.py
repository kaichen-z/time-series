#!/usr/bin/env python3
"""Split / closure proof for a protocol-v2 pack (no model): F0 feedback / F1 selection / F2 final test are disjoint by
task AND by group, cover exactly the 80 pack tasks, all come from the official Train partition, contain no official Dev /
sealed Test id; agent views carry no group/series identity; a run dir (from a dry-run) holds only F0 tasks.
usage: closure_proof.py --pack PACK --task-index HANDOFF_TASK_INDEX [--run RUN_DIR]"""
import argparse, json
from pathlib import Path
P = argparse.ArgumentParser(); P.add_argument("--pack", type=Path, required=True); P.add_argument("--task-index", type=Path, required=True)
P.add_argument("--run", type=Path); A = P.parse_args()
rec = json.load(open(A.pack / "pack_receipt.json"))
F = {n: json.load(open(A.pack / f"private/eval_{n}.json")) for n in ("F0_feedback", "F1_selection", "F2_final_test")}
ids = {n: set(f["task_ids"]) for n, f in F.items()}
groups = {r["role"]: set(r["group_ids"]) for r in rec["folds_primary"]}
split = {}
for l in open(A.task_index):
    r = json.loads(l); split[r["task_id"]] = r["split"]
V = json.load(open(A.pack / "shared/views_train.json"))
names = [n for n in ids]; checks = {}
checks["tasks_pairwise_disjoint"] = all(not (ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:])
checks["groups_pairwise_disjoint"] = all(not (groups[a] & groups[b]) for i, a in enumerate(names) for b in names[i + 1:])
checks["union_equals_pack_80"] = set().union(*ids.values()) == set(V) and len(V) == rec["task_count"] == 80
checks["all_official_train"] = all(split.get(t) == "train" for t in V)
checks["no_dev_or_test_ids"] = not any(split.get(t) == "dev" for t in V) and rec["uses_test_ids_or_labels"] is False
checks["truth_files_match_ids"] = all(set(f["truth"]) == ids[n] == set(f["base_jt"]) for n, f in F.items())
checks["views_have_no_identity"] = len({v["target_description"] for v in V.values()}) == 1 and all("group_id" not in v for v in V.values())
if A.run:
    rv = json.load(open(A.run / "shared/views_train.json")); re_ = json.load(open(A.run / "private/eval_data.json"))
    checks["run_dir_views_only_F0"] = set(rv) == ids["F0_feedback"]
    checks["run_dir_truth_only_F0"] = set(re_["truth"]) == ids["F0_feedback"] and len(re_["folds"]) == 1
out = dict(dataset=rec["dataset"], counts={n: len(s) for n, s in ids.items()}, groups={n: len(g) for n, g in groups.items()},
           checks=checks, ok=all(checks.values()))
print(json.dumps(out, indent=1)); raise SystemExit(0 if out["ok"] else 1)
