"""Run ONE episode of one Codex research agent (CORAL-style heartbeats), as v3.3.2 agentic_run_agent.py / run_agent.py,
called by rounds.py with EP_INDEX = round number. Model gpt-5.6-sol with reasoning effort high; the `codex` on PATH is the
sol56 accounting shim (ledger per dataset, stage label in SOL56_STAGE).
usage: run_agent.py <run_dir> <agent_id> <episodes> <per_episode> [focus]   (env TASK_FILE, EP_INDEX)"""
import json, os, subprocess, sys, time
from pathlib import Path
RUN = Path(sys.argv[1]).resolve(); AID = sys.argv[2]; EPS = int(sys.argv[3]); PER = int(sys.argv[4])
FOCUS = sys.argv[5] if len(sys.argv) > 5 else ""
WS = RUN / f"ws_{AID}"; WS.mkdir(parents=True, exist_ok=True)
TASK = Path(os.environ["TASK_FILE"]).read_text()
MODEL, EFFORT = "gpt-5.6-sol", "high"
EPI = int(os.environ["EP_INDEX"])


def my_accepts():
    n = 0
    for p in (RUN / "shared/attempts").glob(f"*_{AID}.json"):
        try: n += bool(json.load(open(p)).get("accepted"))
        except Exception: pass
    return n


PF = WS / "prev_accepts.json"; prev = json.load(open(PF)) if PF.exists() else 0
hb = ["Reflection: during work, record observations in shared/notes/ as you go."]
if EPI > 1 and EPI % 2 == 1:
    hb.append("Consolidation: review all notes you can see, merge/clean them, and turn any reusable analysis into a script in shared/skills/ with a README.")
acc = my_accepts()
if EPI > 1 and acc == prev:
    hb.append("Redirection: your last episode produced no accepted improvement. Reassess: is the current direction likely to generalise? Consider a different mechanism, and write down in shared/notes/ what NEVER worked.")
json.dump(acc, open(PF, "w"))
prompt = (TASK + f"\n\n## This episode\nYou are agent `{AID}`. Episode {EPI}/{EPS}. Make at most {PER} submissions in this episode, then stop. "
          + (f"Suggested starting focus: {FOCUS}. " if FOCUS else "") + " ".join(hb)
          + " Your working directory is private scratch space; shared material is under $RUN_DIR/shared/. End by writing a short note of what you tried and learned.")
env = dict(os.environ, RUN_DIR=str(RUN), AGENT_ID=AID); t0 = time.time()
with open(WS / f"episode{EPI}.log", "w") as lf:
    try:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "--add-dir", str(RUN / "shared"), "--add-dir", str(RUN / "queue"),
                        "-m", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"', "-C", str(WS), "-o", str(WS / f"episode{EPI}_final.md"), "-"],
                       input=prompt, text=True, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
    except subprocess.TimeoutExpired:
        lf.write("\n[episode timeout]\n")
print(f"{AID} episode {EPI} done in {round(time.time() - t0)}s, accepted so far {my_accepts()}", flush=True)
