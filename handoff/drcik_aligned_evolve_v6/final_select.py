#!/usr/bin/env python3
"""Protocol v6 final selection and final check (host-only; agents never see F1 / the final-check tasks or anything derived from them).

After ALL stages are frozen:
  1. collect host-pinned raw/ensemble baselines plus every frozen candidate program (seed, L4/L5/L7 champions);
  2. score each ONCE on F1 (candidates with any runtime error there are not selectable; fail closed if none), then select
     the lowest mean joint error; exact ties keep the earlier candidate. Robust gain relative to the frozen reference is
     recorded as a diagnostic, not used to choose the final candidate;
  3. write OUT/final/LOCK.json (chosen modules + SHA-256) - the program cannot change after this;
  4. split_mode train_2to1: only then open the official Dev labels and score the locked program ONCE (OUT/final/FINAL_dev.json).
     split_mode all_train: F1 IS official Dev; the final check would be official Test, which needs a separate approval and is NOT
     run here (OUT/final/FINAL_PENDING.json). Test labels are never read by this package.
Every opening of a label file is counted in OUT/final/access_log.json (expected: F1 = 1 before lock; FINAL = 1 after lock, or 0 for all_train).
Programs are scored on an anonymised copy of the fold (fresh opaque handles, no task/document ids), even host-side.
usage: final_select.py --out OUT --pack PACK [--seeds DIR]"""
import argparse, hashlib, json, math, shutil, statistics, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pipeline_runner as PR  # noqa: E402
from anon import anonymize  # noqa: E402
import reference as REFERENCE  # noqa: E402
P = argparse.ArgumentParser(); P.add_argument("--out", type=Path, required=True); P.add_argument("--pack", type=Path, required=True)
P.add_argument("--seeds", type=Path, default=HERE / "seeds", help="seed modules (tests may point elsewhere)")
P.add_argument("--fixed-candidates", type=Path, default=HERE / "fixed_candidates"); A = P.parse_args()
OUT, PACK = A.out.resolve(), A.pack.resolve(); FIN = OUT / "final"; FIN.mkdir(exist_ok=True)
SPLIT = json.load(open(PACK / "pack_receipt.json"))["split_mode"]; assert SPLIT in ("train_2to1", "all_train")
ROLES = ("numerical", "retrieval", "decision"); MOD = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}
LOG = dict(F1_opens=[], FINAL_opens=[])
if (FIN / "LOCK.json").exists(): raise SystemExit("final program already locked; refusing to reselect")


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jt(f, y):
    y = np.asarray(y, dtype=np.float64); d = np.asarray(f, dtype=np.float64) - y; sc = float(np.abs(y).mean()) + 1e-12
    return min(5.0, float(np.abs(d).mean()) / sc) + min(5.0, math.sqrt(float((d * d).mean())) / sc)


def robust(g): return statistics.mean(g) + 0.5 * statistics.mean(min(0.0, x) for x in g)


def open_fold(name):
    if name == "FINAL_dev" and not (FIN / "LOCK.json").exists(): raise RuntimeError("the final check may only be opened after the final program is locked")
    if name.startswith("FINAL") and SPLIT != "train_2to1": raise RuntimeError("all_train: the final check (official Test) needs separate approval")
    LOG["F1_opens" if name.startswith("F1") else "FINAL_opens"].append(time.time())
    return json.load(open(PACK / f"private/eval_{name}.json"))


class Fold:
    """one anonymised store for a fold; every candidate is scored on it, then it is deleted."""
    def __init__(self, fold, tag):
        self.fold, self.dir = fold, FIN / f"_store_{tag}"; shutil.rmtree(self.dir, ignore_errors=True)
        self.hmap = anonymize(PACK / "shared/store", fold["task_ids"], self.dir)

    def score(self, mods, tag):
        out = FIN / f"_out_{tag}"
        fin, _, _, errs = PR.read_output(_run(mods, self.dir, out)); f = self.fold
        if not hasattr(self, "ref"):  # frozen reference baseline on this fold (computed once, host-side)
            self.ref = REFERENCE.reference_jt(self.dir, {h: f["truth"][t] for h, t in self.hmap.items()})
        e = [jt(fin(h), f["truth"][t]) for h, t in self.hmap.items()]
        g = [self.ref[h] - x for h, x in zip(self.hmap, e)]; shutil.rmtree(out)
        return dict(mean_joint_error=statistics.mean(e), robust_gain_vs_reference=robust(g), mean_gain_vs_reference=statistics.mean(g),
                    better_than_reference=sum(x > 1e-9 for x in g), worse_than_reference=sum(x < -1e-9 for x in g),
                    runtime_errors=len(errs), n=len(g))

    def close(self): shutil.rmtree(self.dir, ignore_errors=True)


def _run(mods, store, out):
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, str(HERE / "pipeline_runner.py"), *(str(mods[r]) for r in ROLES), str(store), str(out)], check=True, timeout=7200)
    return out


def access(**kw):
    json.dump(dict(F1_opens=len(LOG["F1_opens"]), FINAL_opens=len(LOG["FINAL_opens"]), split_mode=SPLIT, **kw), open(FIN / "access_log.json", "w"), indent=1)


def mods(run): return {r: OUT / run / f"shared/best_{MOD[r]}.py" for r in ROLES}


def load_fixed(root):
    root = root.resolve(); manifest = root / "manifest.json"; spec = json.load(open(manifest))
    assert spec.get("schema_version") == 1 and spec.get("candidates"), "invalid fixed-candidate manifest"
    retrieve, decision = root / spec["retrieval"], root / spec["decision"]
    assert retrieve.is_file() and decision.is_file(), "fixed passthrough modules missing"
    names, fixed = set(), []
    for row in spec["candidates"]:
        name, forecast = row["name"], root / row["forecast"]
        assert name.startswith("fixed:") and name not in names and forecast.is_file(), "invalid fixed candidate"
        names.add(name); fixed.append((name, dict(numerical=forecast, retrieval=retrieve, decision=decision)))
    files = sorted({p for _, m in fixed for p in m.values()} | {manifest})
    provenance = dict(manifest_sha256=sha(manifest), files_sha256={str(p.relative_to(root)): sha(p) for p in files},
                      candidate_order=[n for n, _ in fixed], immutable_host_injected=True)
    return fixed, provenance


l4 = json.load(open(OUT / "receipts/L4.json")); s4 = mods(l4["final"])
fixed, fixed_provenance = load_fixed(A.fixed_candidates)
REF_SHA = REFERENCE.reference_sha256()
cands = list(fixed)
cands += [("seed", {r: A.seeds / f"seed_{MOD[r]}.py" for r in ROLES})]
cands += [(f"L4:{n}", mods(n)) for n in ("L4_A", "L4_B1", "L4_B2", "L4_B3")]
cands += [(f"L5R1:{n}", dict(s4, decision=OUT / n / "shared/best_adjust.py")) for n in ("L5_S", "L5_C", "L5_I1", "L5_I2", "L5_I3")]
cands += [("L5:routed", dict(s4, decision=OUT / "routed_adjust.py")), ("L5:R2", mods("L5_R2")), ("L7", mods("L7"))]
f1 = Fold(open_fold("F1_selection"), "F1")
table = [(name, m, f1.score(m, f"F1_{i}")) for i, (name, m) in enumerate(cands)]
f1.close(); del f1
# Fail closed if a host-pinned candidate changed during the single selection pass.
_, fixed_after = load_fixed(A.fixed_candidates)
if fixed_after != fixed_provenance:
    json.dump(dict(status="FAILED_CLOSED", reason="fixed candidate files changed during selection"),
              open(FIN / "SELECTION_FAILED.json", "w"), indent=1)
    access(FINAL_opened_after_lock=False, locked=False)
    raise SystemExit("final selection FAILED CLOSED: fixed candidate integrity changed")
# pre-registered: a candidate with ANY runtime error on F1 (a module falling back silently) is not selectable.
# If no candidate is error-free, FAIL CLOSED: nothing is selected, nothing is locked, the final check is never opened.
ok_i = [i for i in range(len(table)) if table[i][2]["runtime_errors"] == 0]
if not ok_i:
    json.dump(dict(status="FAILED_CLOSED", reason="every F1 candidate has runtime errors", F1_table=[dict(candidate=n, **s) for n, _, s in table]),
              open(FIN / "SELECTION_FAILED.json", "w"), indent=1)
    access(FINAL_opened_after_lock=False, locked=False)
    raise SystemExit("final selection FAILED CLOSED: no error-free candidate on F1 (no lock, final check not opened)")
# protocol v6: lowest mean joint error over ALL candidates (fixed baselines first, so exact ties keep the earlier one)
best_i = min(ok_i, key=lambda i: (table[i][2]["mean_joint_error"], i))
name, chosen, _ = table[best_i]
for r, p in chosen.items(): shutil.copy(p, FIN / f"final_{MOD[r]}.py")
if REFERENCE.reference_sha256() != REF_SHA:
    json.dump(dict(status="FAILED_CLOSED", reason="reference forecaster changed during selection"), open(FIN / "SELECTION_FAILED.json", "w"), indent=1)
    access(FINAL_opened_after_lock=False, locked=False)
    raise SystemExit("final selection FAILED CLOSED: reference integrity changed")
json.dump(dict(protocol_version=6, chosen=name, split_mode=SPLIT, fixed_candidates=fixed_provenance,
               selection_rule="lowest mean joint error; runtime-error candidates excluded; ties -> earlier candidate",
               reference_sha256=REF_SHA,
               modules_sha256={f"final_{MOD[r]}.py": sha(FIN / f"final_{MOD[r]}.py") for r in ROLES},
               F1_table=[dict(candidate=n, **s) for n, _, s in table], t=time.time()), open(FIN / "LOCK.json", "w"), indent=1)
if SPLIT == "train_2to1":
    fd = Fold(open_fold("FINAL_dev"), "FINAL")
    res = fd.score({r: FIN / f"final_{MOD[r]}.py" for r in ROLES}, "FINAL"); fd.close()
    json.dump(dict(locked=name, final_check="official Dev", FINAL=res), open(FIN / "FINAL_dev.json", "w"), indent=1)
    access(FINAL_opened_after_lock=True, locked=True)
else:
    res = "pending: official Test final check needs separate approval (not run by this package)"
    json.dump(dict(locked=name, final_check="official Test", status=res, modules_sha256=json.load(open(FIN / "LOCK.json"))["modules_sha256"]),
              open(FIN / "FINAL_PENDING.json", "w"), indent=1)
    access(FINAL_opened_after_lock=False, locked=True)
print(json.dumps(dict(chosen=name, split_mode=SPLIT, F1={n: round(s["mean_joint_error"], 5) for n, _, s in table}, FINAL=res)))
