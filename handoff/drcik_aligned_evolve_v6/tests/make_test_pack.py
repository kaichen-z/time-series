#!/usr/bin/env python3
"""TEST ONLY: deterministic small sub-pack of a full v5 pack, so the model-free end-to-end tests (fake agents through
L4 -> L5 -> L7 -> final selection) run in minutes. Every part keeps its role; N tasks per part are taken in sha256 order of
the task id (seeded), so the sub-pack is reproducible. The receipt is marked test_subsample and must never be used for a
real run (orchestrate.py refuses it unless --test-fake-codex / --dry-run / --leak-test).
usage: make_test_pack.py --pack FULL_PACK --out SUBPACK [--n 48]"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import viewstore  # noqa: E402
P = argparse.ArgumentParser(); P.add_argument("--pack", type=Path, required=True); P.add_argument("--out", type=Path, required=True)
P.add_argument("--n", type=int, default=48); A = P.parse_args()
if A.out.exists(): raise SystemExit(f"{A.out} exists")
rec = json.load(open(A.pack / "pack_receipt.json")); (A.out / "private").mkdir(parents=True)
keep = []
for f in sorted((A.pack / "private").glob("eval_*.json")):
    e = json.load(open(f)); ids = sorted(e["task_ids"], key=lambda t: hashlib.sha256(f"test-subpack\0{t}".encode()).hexdigest())[:A.n]
    json.dump(dict(part=e["part"], task_ids=ids, truth={t: e["truth"][t] for t in ids}, base_jt={t: e["base_jt"][t] for t in ids}), open(A.out / f"private/{f.name}", "w"))
    rec["parts"][e["part"]]["tasks"] = len(ids); keep += ids
shutil.copy(A.pack / "private/forbidden_terms.json", A.out / "private/forbidden_terms.json")
S = viewstore.Store(A.pack / "shared/store"); docs = {r for t in keep for r in S.meta[t]["document_refs"]}
viewstore.copy_subset(S, A.out / "shared/store", {t: t for t in keep}, {d: d for d in docs})
rec.update(test_subsample=dict(source_receipt_sha256=hashlib.sha256((A.pack / "pack_receipt.json").read_bytes()).hexdigest(), n_per_part=A.n),
           output_sha256={p.relative_to(A.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(A.out.rglob("*")) if p.is_file()})
json.dump(rec, open(A.out / "pack_receipt.json", "w"), indent=1)
print(json.dumps(dict(out=str(A.out), tasks=len(keep), parts={k: v["tasks"] for k, v in rec["parts"].items()})))
