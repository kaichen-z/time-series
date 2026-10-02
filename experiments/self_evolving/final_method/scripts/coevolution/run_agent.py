"""Run one Codex research agent for several episodes against a run directory (CORAL-style heartbeats).
usage: run_agent.py <run_dir> <agent_id> [episodes=4] [per_episode=5] [focus]"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_backend import run_agent
RUN = Path(sys.argv[1]).resolve(); AID = sys.argv[2]
EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4; PER = int(sys.argv[4]) if len(sys.argv) > 4 else 5
FOCUS = sys.argv[5] if len(sys.argv) > 5 else ""
WS = RUN / f"ws_{AID}"; WS.mkdir(parents=True, exist_ok=True)
TASK = Path(os.environ.get("TASK_FILE") or Path(__file__).parent / "TASK.md").read_text()
MODEL = os.environ.get("AGENT_MODEL", "gpt-6-sol")  # or opus / haiku / claude-* (Claude Code CLI)


def my_accepts():
    n = 0
    for p in (RUN / "shared/attempts").glob(f"*_{AID}.json"):
        try: n += bool(json.load(open(p)).get("accepted"))
        except Exception: pass
    return n


prev = 0
for ep in range(1, EPS + 1):
    hb = ["Reflection: during work, record observations in shared/notes/ as you go."]
    if ep > 1 and ep % 2 == 1:
        hb.append("Consolidation: review all notes (yours and other agents'), merge/clean them, and turn any reusable analysis into a script in shared/skills/ with a README.")
    acc = my_accepts()
    if ep > 1 and acc == prev:
        hb.append("Redirection: your last episode produced no accepted improvement. Reassess: is the current direction likely to generalise? Consider a different role or mechanism of the pipeline, and write down in shared/notes/ what NEVER worked.")
    prev = acc
    prompt = (TASK + f"\n\n## This episode\nYou are agent `{AID}`. Episode {ep}/{EPS}. "
              f"Make at most {PER} submissions in this episode, then stop. " + (f"Suggested starting focus: {FOCUS}. " if FOCUS else "") +
              " ".join(hb) + " Your working directory is private scratch space; shared material is under $RUN_DIR/shared/. "
              "End by writing a short note of what you tried and learned.")
    env = dict(os.environ, RUN_DIR=str(RUN), AGENT_ID=AID)
    t0 = time.time()
    with open(WS / f"episode{ep}.log", "w") as lf:
        try:
            run_agent(MODEL, prompt, WS, RUN, WS / f"episode{ep}_final.md", lf, env=env, timeout=7200)
        except subprocess.TimeoutExpired:
            lf.write("\n[episode timeout]\n")
    print(f"{AID} episode {ep} done in {round(time.time() - t0)}s, accepted so far {my_accepts()}", flush=True)
