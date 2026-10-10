#!/usr/bin/env python3
"""ONE real-model smoke episode (owner-approved): real codex (gpt-5.6-sol, high) -> sol56 shim -> bubblewrap sandbox.
Checks that codex starts and reaches the API inside the sandbox, can read shared/, write its workspace and submit one
module through submit.py; the evaluator scores it. Exactly one codex call is made. Hidden data stays outside the sandbox.
usage: smoke_real.py --pack PACK --out OUT --real-codex CODEX_BIN"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser(); P.add_argument("--pack", type=Path, required=True); P.add_argument("--out", type=Path, required=True)
P.add_argument("--real-codex", type=Path, required=True); A = P.parse_args()
OUT = A.out.resolve()
if OUT.exists(): raise SystemExit(f"{OUT} exists")
# build one L4_B1-style run dir with the orchestrator's own dry-run machinery (no model), then use its first run dir
subprocess.run([sys.executable, str(HERE / "orchestrate.py"), "--dataset", json.load(open(A.pack / "pack_receipt.json"))["dataset"], "--pack", str(A.pack),
                "--out", str(OUT / "prep"), "--dry-run"], check=True, stdout=subprocess.DEVNULL)
run = OUT / "run"; shutil.copytree(OUT / "prep/dry/L4_B1", run)
for p in (run / "private/state.json",): p.unlink(missing_ok=True)
(OUT / "ledger").mkdir(); shutil.copy(OUT / "prep/ledger/caps.json", OUT / "ledger/caps.json")
bindir = OUT / "bin"; bindir.mkdir()
(bindir / "codex").write_text(f"#!/bin/bash\nexec {sys.executable} {HERE / 'sol56_codex_shim.py'} \"$@\"\n"); (bindir / "codex").chmod(0o755)
(bindir / "sandboxed_codex").write_text(f"#!/bin/bash\nexec {sys.executable} {HERE / 'sandbox_codex.py'} \"$@\"\n"); (bindir / "sandboxed_codex").chmod(0o755)
chome = OUT / "codex_home"; chome.mkdir()
for f in ("auth.json", "config.toml"):
    if (Path.home() / ".codex" / f).exists(): shutil.copy(Path.home() / ".codex" / f, chome / f)
real = A.real_codex.resolve()
env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", SOL56_ROOT=str(OUT), SOL56_STAGE="smoke", SOL56_REAL_CODEX=str(bindir / "sandboxed_codex"),
           SANDBOX_INNER=str(real), SANDBOX_CODEX_DIR=str(real.parents[2]), SANDBOX_CODEX_HOME=str(chome), SANDBOX_SUBMIT=str(HERE / "submit.py"))
d = subprocess.Popen([sys.executable, str(HERE / "evald.py"), str(run), "5"], env=env, stdout=open(OUT / "evald.log", "w"), stderr=subprocess.STDOUT)
smoke_task = (f"SMOKE TEST (one episode). Do exactly this and nothing else:\n"
              f"1. Read the file $RUN_DIR/shared/best_forecast.py.\n"
              f"2. Copy it unchanged to my_module.py in your current working directory.\n"
              f"3. Run: python3 {HERE / 'submit.py'} my_module.py --role numerical --note smoke\n"
              f"4. Report the JSON line that command printed, then stop. Make no other submission.\n")
tf = OUT / "SMOKE_TASK.md"; tf.write_text(smoke_task)
t0 = time.time()
r = subprocess.run([sys.executable, str(HERE / "run_agent.py"), str(run), "B1", "1", "1"], env=dict(env, TASK_FILE=str(tf), EP_INDEX="1"),
                   capture_output=True, text=True, timeout=7500)
(run / "STOP").write_text("x"); d.wait(timeout=600)
L = [json.loads(l) for l in open(OUT / "ledger/ledger.jsonl")] if (OUT / "ledger/ledger.jsonl").exists() else []
st = json.load(open(run / "private/state.json"))
res = [json.load(open(p)) for p in (run / "results").glob("*.json")]
log = (run / "ws_B1/episode1.log").read_text() if (run / "ws_B1/episode1.log").exists() else ""
out = dict(seconds=round(time.time() - t0), run_agent_rc=r.returncode, codex_calls=sum(x["kind"] == "commit" for x in L),
           ledger=[{k: x.get(k) for k in ("kind", "stage", "tokens", "reported", "rc", "turns", "gross_input", "cached_input", "output")} for x in L if x["kind"] == "commit"],
           submissions=st["n"], results=[{k: x.get(k) for k in ("error", "visible_fitness", "hidden_check", "accepted")} for x in res],
           episode_log_tail=log[-1500:])
json.dump(out, open(OUT / "smoke_report.json", "w"), indent=1); print(json.dumps(out, indent=1))
