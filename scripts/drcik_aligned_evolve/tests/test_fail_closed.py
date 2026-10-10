#!/usr/bin/env python3
"""final_select.py fail-closed tests (no model), on a copy of a completed fake run:
  all_error   : every candidate (incl. seed) raises on F1 -> exit != 0, no LOCK, SELECTION_FAILED.json, F2 never opened;
  seed_error  : only the seed raises -> a non-seed error-free candidate is locked; F1 = 1, F2 = 1 after lock;
  partial     : half of the stage candidates raise -> the locked candidate has 0 F1 runtime errors and is not an erroring one.
usage: test_fail_closed.py --run-out COMPLETED_FAKE_RUN --pack PACK --out WORKDIR"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser(); P.add_argument("--run-out", type=Path, required=True); P.add_argument("--pack", type=Path, required=True)
P.add_argument("--out", type=Path, required=True); A = P.parse_args()
if A.out.exists(): raise SystemExit(f"{A.out} exists")
A.out.mkdir(parents=True)
BOOM = "\n_ORIG_FORECAST = forecast\ndef forecast(view):\n    raise RuntimeError('injected failure')\n"
STAGE_RUNS = ["L4_A", "L4_B1", "L4_B2", "L4_B3", "L5_S", "L5_C", "L5_I1", "L5_I2", "L5_I3", "L5_R2", "L7"]


def case(name, break_seed, break_runs):
    out = A.out / name; shutil.copytree(A.run_out, out, ignore=shutil.ignore_patterns("final", "ws_*"))
    seeds = A.out / f"{name}_seeds"; shutil.copytree(HERE / "seeds", seeds)
    if break_seed: (seeds / "seed_forecast.py").write_text((seeds / "seed_forecast.py").read_text() + BOOM)
    for r in break_runs:
        p = out / r / "shared/best_forecast.py"; p.write_text(p.read_text() + BOOM)
    rc = subprocess.run([sys.executable, str(HERE / "final_select.py"), "--out", str(out), "--pack", str(A.pack), "--seeds", str(seeds)],
                        capture_output=True, text=True).returncode
    fin = out / "final"; acc = json.load(open(fin / "access_log.json")) if (fin / "access_log.json").exists() else None
    lock = json.load(open(fin / "LOCK.json")) if (fin / "LOCK.json").exists() else None
    return rc, lock, acc, (fin / "SELECTION_FAILED.json").exists(), (fin / "F2_final_test.json").exists()


res = {}
# all candidates error (the L4 stage seed and routed/R1 candidates inherit forecast from L4 final; break every run's forecast)
rc, lock, acc, failed, f2 = case("all_error", True, STAGE_RUNS)
res["all_error"] = dict(ok=rc != 0 and lock is None and failed and not f2 and acc is not None and acc["F2_opens"] == 0, rc=rc, access=acc)
rc, lock, acc, failed, f2 = case("seed_error", True, [])
res["seed_error"] = dict(ok=rc == 0 and lock is not None and lock["chosen"] != "seed" and acc == dict(F1_opens=1, F2_opens=1, F2_opened_after_lock=True, locked=True),
                         chosen=lock and lock["chosen"], access=acc)
broken = STAGE_RUNS[::2]
rc, lock, acc, failed, f2 = case("partial", False, broken)
errs = {row["candidate"]: row["runtime_errors"] for row in lock["F1_table"]} if lock else {}
res["partial"] = dict(ok=rc == 0 and lock is not None and errs.get(lock["chosen"], 1) == 0, chosen=lock and lock["chosen"],
                      erroring_candidates=sorted(k for k, v in errs.items() if v), access=acc)
ok = all(v["ok"] for v in res.values()); print(json.dumps(dict(ok=ok, **res), indent=1)); sys.exit(0 if ok else 1)
