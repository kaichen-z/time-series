#!/usr/bin/env python3
"""Protocol v2 final selection and test (host-only; agents never see F1/F2 or anything derived from them).

After ALL stages are frozen:
  1. collect every frozen candidate program (seed, each L4 run champion, the L4 stage seed, each L5-R1 run champion with the
     L4 numerical/retrieval modules, the routed L5 function, the L5-R2 champion, the L7 champion);
  2. score each ONCE on F1 (candidates with any runtime error there are not selectable) (selection groups, never seen during evolution): robust gain = mean(BJT - jt) + 0.5 *
     mean(negative part); pick the max, ties -> the earlier/simpler candidate in the order above (seed first);
  3. write OUT/final/LOCK.json (chosen modules + SHA-256) - the program cannot change after this;
  4. only then open F2 and score the locked program ONCE; write OUT/final/F2_final_test.json.
Every opening of an F1/F2 file is counted in OUT/final/access_log.json (expected: F1 = 1 before lock, F2 = 1 after lock).
usage: final_select.py --out OUT --pack PACK"""
import argparse, hashlib, json, math, shutil, statistics, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
P = argparse.ArgumentParser(); P.add_argument("--out", type=Path, required=True); P.add_argument("--pack", type=Path, required=True); A = P.parse_args()
OUT, PACK = A.out.resolve(), A.pack.resolve(); FIN = OUT / "final"; FIN.mkdir(exist_ok=True)
ROLES = ("numerical", "retrieval", "decision"); MOD = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}
LOG = dict(F1_opens=[], F2_opens=[])
if (FIN / "LOCK.json").exists(): raise SystemExit("final program already locked; refusing to reselect")


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jt(f, y):
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


def robust(g): return statistics.mean(g) + 0.5 * statistics.mean(min(0.0, x) for x in g)


def open_fold(name):
    if name == "F2_final_test" and not (FIN / "LOCK.json").exists(): raise RuntimeError("F2 may only be opened after the final program is locked")
    LOG["F1_opens" if name.startswith("F1") else "F2_opens"].append(time.time())
    return json.load(open(PACK / f"private/eval_{name}.json"))


def score(mods, fold, tag):
    sys.path.insert(0, str(HERE)); from anon import anonymize   # programs never see real task/document ids, even host-side
    V = json.load(open(PACK / "shared/views_train.json")); views = FIN / f"_views_{tag}.json"
    av, hmap = anonymize({t: V[t] for t in fold["task_ids"]}); json.dump(av, open(views, "w")); out = FIN / f"_out_{tag}.json"
    subprocess.run([sys.executable, str(HERE / "pipeline_runner.py"), *(str(mods[r]) for r in ROLES), str(views), str(out)], check=True, timeout=3600)
    o = json.load(open(out)); views.unlink(); out.unlink()
    g = [fold["base_jt"][t] - jt(o["forecasts"][h], fold["truth"][t]) for h, t in hmap.items()]
    return dict(robust_gain=robust(g), mean_gain=statistics.mean(g), better=sum(x > 1e-9 for x in g), worse=sum(x < -1e-9 for x in g), runtime_errors=len(o["errors"]))


def mods(run): return {r: OUT / run / f"shared/best_{MOD[r]}.py" for r in ROLES}


l4 = json.load(open(OUT / "receipts/L4.json")); s4 = mods(l4["final"])
cands = [("seed", {r: HERE / f"seeds/seed_{MOD[r]}.py" for r in ROLES})]
cands += [(f"L4:{n}", mods(n)) for n in ("L4_A", "L4_B1", "L4_B2", "L4_B3")]
cands += [(f"L5R1:{n}", dict(s4, decision=OUT / n / "shared/best_adjust.py")) for n in ("L5_S", "L5_C", "L5_I1", "L5_I2", "L5_I3")]
cands += [("L5:routed", dict(s4, decision=OUT / "routed_adjust.py")), ("L5:R2", mods("L5_R2")), ("L7", mods("L7"))]
f1 = open_fold("F1_selection")
table = [(name, m, score(m, f1, f"F1_{i}")) for i, (name, m) in enumerate(cands)]
del f1
# pre-registered: a candidate with ANY runtime error on F1 (a module falling back silently) is not selectable
ok_i = [i for i in range(len(table)) if table[i][2]["runtime_errors"] == 0] or [0]
best_i = max(ok_i, key=lambda i: (table[i][2]["robust_gain"], -i))
name, chosen, _ = table[best_i]
for r, p in chosen.items(): shutil.copy(p, FIN / f"final_{MOD[r]}.py")
json.dump(dict(chosen=name, modules_sha256={f"final_{MOD[r]}.py": sha(FIN / f"final_{MOD[r]}.py") for r in ROLES},
               F1_table=[dict(candidate=n, **s) for n, _, s in table], t=time.time()), open(FIN / "LOCK.json", "w"), indent=1)
f2 = open_fold("F2_final_test")
res = score({r: FIN / f"final_{MOD[r]}.py" for r in ROLES}, f2, "F2")
json.dump(dict(locked=name, F2=res), open(FIN / "F2_final_test.json", "w"), indent=1)
json.dump(dict(F1_opens=len(LOG["F1_opens"]), F2_opens=len(LOG["F2_opens"]), F2_opened_after_lock=True), open(FIN / "access_log.json", "w"), indent=1)
print(json.dumps(dict(chosen=name, F1={n: round(s["robust_gain"], 5) for n, _, s in table}, F2=res)))
