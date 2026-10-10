#!/usr/bin/env python3
"""One-command, model-free verification of the protocol-v6 package (no codex/LLM call, no Test labels).
Run from a fresh extraction:   python -I -S -B run_tests.py --split {train_2to1,all_train} [--work DIR] [--datasets timesx,time_mmd] [--n 48]
All outputs go to a fresh 0700 work dir OUTSIDE the package; the package bytes are hashed before and after.
Per dataset, on the FULL pack packs/<dataset>_<split>:
  closure / split / storage proof (official Train/Dev closure, disjointness, receipt SHA-256, float32 chunk SHA-256, layout);
  sandbox leak test (bubblewrap) on a full-size run dir, plus the run-dir closure proof (only F0, opaque handles);
on a deterministic TEST sub-pack (tests/make_test_pack.py, --n tasks per part; refused by real runs):
  dry-run, anti-memorisation (synthetic attacks + the 8 old champions + identity probe), full fake-agent end-to-end
  (L4 -> L5 -> L7 -> final selection on F1 -> final check per split mode), access log, group-specific-rule selection test,
  fail-closed final-selection tests. The fake agent replaces codex and a dry-run guard is installed."""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = argparse.ArgumentParser(); P.add_argument("--work", type=Path); P.add_argument("--datasets", default="timesx,time_mmd")
P.add_argument("--split", choices=("train_2to1", "all_train"), required=True); P.add_argument("--n", type=int, default=48); A = P.parse_args()
WORK = A.work.resolve() if A.work else Path(tempfile.mkdtemp(prefix="drcik_v5_tests_"))
if A.work:
    if WORK.exists(): raise SystemExit(f"{WORK} exists")
    WORK.mkdir(parents=True, mode=0o700)
os.chmod(WORK, 0o700)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", ROUNDS_POLL="1")
PY = [sys.executable, "-B"]


def tree_sha():
    h = {}
    for p in sorted(HERE.rglob("*")):
        if p.is_file(): h[str(p.relative_to(HERE))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


def run(name, args, ok_codes=(0,), timeout=3600):
    t0 = time.time(); r = subprocess.run(PY + args, env=ENV, capture_output=True, text=True, timeout=timeout)
    rec = dict(step=name, rc=r.returncode, ok=r.returncode in ok_codes, seconds=round(time.time() - t0), tail=(r.stdout + r.stderr)[-600:])
    print(f"[{'PASS' if rec['ok'] else 'FAIL'}] {name} ({rec['seconds']}s)", flush=True); return rec, r.stdout


before = tree_sha(); results = []
rec, _ = run("v6 host-pinned fixed-candidate contract", [str(HERE / "tests/test_fixed_candidates.py")]); results.append(rec)
EXPECT_ACCESS = (dict(F1_opens=1, FINAL_opens=1, split_mode=A.split, FINAL_opened_after_lock=True, locked=True) if A.split == "train_2to1"
                 else dict(F1_opens=1, FINAL_opens=0, split_mode=A.split, FINAL_opened_after_lock=False, locked=True))
for ds in A.datasets.split(","):
    full = HERE / "packs" / f"{ds}_{A.split}"; w = WORK / ds; w.mkdir(mode=0o700); idx = HERE / f"fixtures/{ds}_split_index.json"
    rec, _ = run(f"{ds}: closure/split/storage proof (full pack)", [str(HERE / "tests/closure_proof.py"), "--pack", str(full), "--task-index", str(idx)]); results.append(rec)
    forbid = [str(full / "private"), str(HERE / "fixtures"), str(HERE / "tests/cheat_samples"), str(Path.home() / ".codex")]
    rec, _ = run(f"{ds}: sandbox leak test (full-size run dir)", [str(HERE / "orchestrate.py"), "--dataset", ds, "--pack", str(full), "--out", str(w / "leak"), "--leak-test",
                                                                  *[x for f in forbid for x in ("--forbid", f)]]); results.append(rec)
    rec, _ = run(f"{ds}: run-dir closure proof (full-size F0 run dir)", [str(HERE / "tests/closure_proof.py"), "--pack", str(full), "--task-index", str(idx),
                                                                         "--run", str(w / "leak/leak/L4_A")]); results.append(rec)
    rec, _ = run(f"{ds}: v6 baseline = frozen reference (not Toto) and seed == reference (full-size F0 run dir)",
                 [str(HERE / "tests/test_reference_baseline.py"), "--run", str(w / "leak/leak/L4_A")]); results.append(rec)
    pack = w / "subpack"
    rec, _ = run(f"{ds}: build test sub-pack (n={A.n}/part)", [str(HERE / "tests/make_test_pack.py"), "--pack", str(full), "--out", str(pack), "--n", str(A.n)]); results.append(rec)
    rec, _ = run(f"{ds}: dry-run (sub-pack)", [str(HERE / "orchestrate.py"), "--dataset", ds, "--pack", str(pack), "--out", str(w / "dry"), "--dry-run"]); results.append(rec)
    cheats = [x for f in sorted((HERE / "tests/cheat_samples").glob(f"{ds}_*.py"))
              for x in ("--cheat", {"forecast": "numerical", "retrieve": "retrieval", "adjust": "decision"}[f.stem.split("best_")[1]] + ":" + str(f))]
    rec, _ = run(f"{ds}: anti-memorisation + identity probe + {len(cheats) // 2} old champions",
                 [str(HERE / "tests/test_anti_memo.py"), "--dataset", ds, "--pack", str(pack), "--out", str(w / "antimemo"), *cheats]); results.append(rec)
    rec, _ = run(f"{ds}: fake-agent end-to-end (L4-L5-L7-final)", [str(HERE / "orchestrate.py"), "--dataset", ds, "--pack", str(pack), "--out", str(w / "fake_e2e"),
                                                                  "--test-fake-codex", str(HERE / "tests/fake_codex.py")]); results.append(rec)
    acc = json.load(open(w / "fake_e2e/final/access_log.json")) if (w / "fake_e2e/final/access_log.json").exists() else None
    results.append(dict(step=f"{ds}: access log = F1 once; final check {'Dev once after lock' if A.split == 'train_2to1' else 'not opened (Test pending approval)'}",
                        ok=acc == EXPECT_ACCESS, access=acc))
    print(f"[{'PASS' if results[-1]['ok'] else 'FAIL'}] {results[-1]['step']}", flush=True)
    rec, _ = run(f"{ds}: group-specific rule gains nothing on F1", [str(HERE / "tests/test_group_rule_selection.py"), "--run-out", str(w / "fake_e2e"),
                                                                   "--pack", str(pack), "--out", str(w / "group_rule")]); results.append(rec)
    rec, _ = run(f"{ds}: final selection fails closed", [str(HERE / "tests/test_fail_closed.py"), "--run-out", str(w / "fake_e2e"), "--pack", str(pack),
                                                        "--out", str(w / "fail_closed")]); results.append(rec)
after = tree_sha()
results.append(dict(step="package bytes unchanged by the test run", ok=before == after, files=len(before)))
print(f"[{'PASS' if results[-1]['ok'] else 'FAIL'}] {results[-1]['step']} ({len(before)} files)", flush=True)
summary = dict(split_mode=A.split, ok=all(r["ok"] for r in results), passed=sum(r["ok"] for r in results), total=len(results), work=str(WORK), results=results,
               package_sha256_manifest=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest())
json.dump(summary, open(WORK / "run_tests_summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("ok", "passed", "total", "work", "package_sha256_manifest")}))
sys.exit(0 if summary["ok"] else 1)
