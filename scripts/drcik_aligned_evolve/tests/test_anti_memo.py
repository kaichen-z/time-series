#!/usr/bin/env python3
"""Anti-memorisation regression (no model calls). Builds a fresh run dir through the orchestrator's dry-run, starts the
evaluator daemon (non-sync) and submits attack programs + one legitimate change through submit.py. Every attack must be
rejected; the legitimate change must be scored; no agent-readable file may contain per-task truth or task-level values.
usage: test_anti_memo.py --dataset D --pack PACK --out OUT [--cheat ROLE:PATH ...]   (real cheating champions to replay)"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser(); P.add_argument("--dataset", required=True); P.add_argument("--pack", type=Path, required=True)
P.add_argument("--out", type=Path, required=True); P.add_argument("--cheat", action="append", default=[]); A = P.parse_args()
OUT = A.out.resolve()
if OUT.exists(): raise SystemExit(f"{OUT} exists")
subprocess.run([sys.executable, str(HERE / "orchestrate.py"), "--dataset", A.dataset, "--pack", str(A.pack.resolve()), "--out", str(OUT / "prep"), "--dry-run"],
               check=True, stdout=subprocess.DEVNULL)
run = OUT / "run"; shutil.copytree(OUT / "prep/dry/L4_A", run); (run / "private/state.json").unlink()
ed = json.load(open(run / "private/eval_data.json")); V = json.load(open(run / "shared/views_train.json"))
vis = ed["folds"][0]
d = subprocess.Popen([sys.executable, str(HERE / "evald.py"), str(run), "1000"], stdout=open(OUT / "evald.log", "w"), stderr=subprocess.STDOUT)


def submit(agent, role, code):
    p = OUT / f"cand_{agent}.py"; p.write_text(code)
    r = subprocess.run([sys.executable, str(HERE / "submit.py"), str(p), "--role", role], env=dict(os.environ, RUN_DIR=str(run), AGENT_ID=agent),
                       capture_output=True, text=True, timeout=1800)
    return r.stdout.strip()


roles = json.load(open(run / "stage.json"))["roles"]; agent_of = {v: k for k, v in roles.items()}
seed_fc = (run / "shared/best_forecast.py").read_text()
# exact truths of the smallest-H visible tasks (what a memoriser would need)
small = sorted(vis, key=lambda t: V[t]["H"])
k = max(3, int(0.1 * len(vis)) + 1); pick = small[:k]
table = {round(V[t]["history"][-1], 9): ed["truth"][t] for t in pick}
fp_lookup = "_T = " + repr(table) + "\n"
attacks = {
    "lookup_table_per_task": ("numerical", seed_fc + "\n_F0 = forecast\n" + fp_lookup +
                              "def forecast(view):\n    k = round(view['history'][-1], 9)\n    return list(_T[k]) if k in _T else _F0(view)\n"),
    "compressed_table_base64_zlib": ("numerical", "import base64, zlib\n_D = 'eJzLSM3JyVcozy/KSQEAGgQEXQ=='\n" + seed_fc),
    "big_integer_encoding": ("numerical", seed_fc + "\n_BIG = " + "7" * 4000 + "\n"),
    "bytes_string_encoding": ("numerical", seed_fc + "\n_B = " + repr(bytes(range(256)) * 12) + "\n"),
    "hash_branching": ("numerical", seed_fc + "\n_F0 = forecast\ndef forecast(view):\n    h = hash(tuple(view['history']))\n    return _F0(view)\n"),
    "exec_eval": ("numerical", seed_fc + "\nexec('x = 1')\n"),
    "file_read": ("numerical", seed_fc + "\n_X = open('/etc/hostname').read()\n"),
    "oversize_code": ("numerical", seed_fc + "\n# " + "x" * 25000 + "\n"),
    # compact exact memorisation that passes the static limits -> must be caught at run time
    "compact_memorisation_runtime": ("numerical", None),
}
compact = {round(V[t]["history"][-1], 6): [round(x, 6) for x in ed["truth"][t]] for t in pick}
n_numbers = sum(len(v) + 1 for v in compact.values())
attacks["compact_memorisation_runtime"] = ("numerical", seed_fc + "\n_F0 = forecast\n_T = " + repr(compact) +
                                           "\ndef forecast(view):\n    k = round(view['history'][-1], 6)\n    return list(_T[k]) if k in _T else _F0(view)\n") \
    if n_numbers <= 290 else ("numerical", None)
for spec in A.cheat:
    role, path = spec.split(":", 1); attacks[f"replayed_champion:{Path(path).parent.parent.name}/{Path(path).name}"] = (role, Path(path).read_text())
legit = ("numerical", seed_fc.replace("(0.4, 0.6, 0.0)", "(0.45, 0.55, 0.0)").replace("(0.7, 0.2, 0.1), 0.3", "(0.65, 0.25, 0.1), 0.3"))

report = {}
for name, (role, code) in attacks.items():
    if code is None: report[name] = dict(skipped=f"needs {n_numbers} numbers (>290) for this dataset"); continue
    out = submit(agent_of[role], role, code); rej = out.startswith("ERROR")
    report[name] = dict(role=role, rejected=rej, response=out[:220])
out = submit(agent_of["numerical"], "numerical", legit[1]); report["legitimate_change"] = dict(rejected=out.startswith("ERROR"), response=out[:300])
(run / "STOP").write_text("x"); d.wait(timeout=600)
# agent-visible files must not carry per-task truth / task ids / per-task values
leaks = []
truth_vals = {round(x, 4) for t in vis for x in ed["truth"][t]}
for f in list((run / "shared/traces").glob("*")) + list((run / "results").glob("*.json")) + list((run / "shared/attempts").glob("*.json")):
    txt = f.read_text()
    if any(t in txt for t in vis): leaks.append(f"{f.relative_to(run)}: contains task ids")
    if re.search(r'"(truth|visible_per_task|worst5|best5|runtime_errors)"', txt): leaks.append(f"{f.relative_to(run)}: per-task field")
ok = (all(v.get("rejected") or "skipped" in v for k, v in report.items() if k != "legitimate_change") and not report["legitimate_change"]["rejected"] and not leaks)
res = dict(dataset=A.dataset, ok=ok, attacks=report, agent_visible_leaks=leaks)
json.dump(res, open(OUT / "anti_memo_report.json", "w"), indent=1); print(json.dumps(res, indent=1)); sys.exit(0 if ok else 1)
