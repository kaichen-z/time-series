"""Synchronous-round orchestrator (sol56). For round r = 1..E: start one episode (EP_INDEX=r) for EVERY agent of the group
at the same time (all see the same frozen champion), wait for all of them, wait for the barrier in drained() (queue empty, nothing in flight, every
dequeued request has a terminal result), then write RUN/round_close_r and wait for RUN/round_done_r.json (daemon picks the winner
by visible DESC, hidden DESC, agent id ASC, per-agent index ASC). Agent order in the list only fixes the tie-break id.
env ROUND_OFFSET (default 0). usage: rounds.py RUN_DIR RUNNER E PER_EPISODE AGENT[:FOCUS] [AGENT[:FOCUS] ...]
env passed through (AGENT_MODEL, TASK_FILE, SOL56_*); each agent episode log -> RUN/agent_<id>.log (append)."""
import os, subprocess, sys, time
from pathlib import Path

RUN = Path(sys.argv[1]).resolve(); RUNNER = sys.argv[2]; E = int(sys.argv[3]); PER = int(sys.argv[4])
AGENTS = [a.split(":", 1) + [""] if ":" not in a else a.split(":", 1) for a in sys.argv[5:]]
AGENTS = [(x[0], x[1]) for x in AGENTS]
OFF = int(os.environ.get("ROUND_OFFSET", "0"))  # daemon round number = OFF + r (phases of coevo_x share one daemon)


def drained():
    """Barrier: no queued request, no request in flight, and every dequeued request has a terminal result.
    The daemon writes private/inflight.json BEFORE removing a request from queue/ and deletes it only after the
    result file is written, so 'queue empty and no inflight' cannot be observed while a request is being scored."""
    if list((RUN / "queue").glob("*.json")) or (RUN / "private/inflight.json").exists(): return False
    deq = set((RUN / "private/dequeued.log").read_text().split()) if (RUN / "private/dequeued.log").exists() else set()
    done = set((RUN / "private/done.log").read_text().split()) if (RUN / "private/done.log").exists() else set()
    return deq == done and all((RUN / f"results/{r}.json").exists() for r in deq)


for r in range(1, E + 1):
    t0 = time.time(); procs = []
    for aid, focus in AGENTS:
        env = dict(os.environ, EP_INDEX=str(r))
        lf = open(RUN / f"agent_{aid}.log", "a")
        procs.append(subprocess.Popen([sys.executable, RUNNER, str(RUN), aid, str(E), str(PER), focus], env=env, stdout=lf, stderr=subprocess.STDOUT))
    for p in procs: p.wait()
    while not drained(): time.sleep(float(os.environ.get("ROUNDS_POLL", "5")))
    (RUN / f"round_close_{OFF + r}").write_text(str(time.time()))
    while not (RUN / f"round_done_{OFF + r}.json").exists(): time.sleep(5)
    print(f"round {r} closed in {round(time.time() - t0)}s", flush=True)
