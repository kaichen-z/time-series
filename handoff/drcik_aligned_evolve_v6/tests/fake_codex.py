#!/usr/bin/env python3
"""TEST ONLY: stands in for `codex exec` (no model call). Reads the episode prompt from stdin, determines the agent's role
(L4/L5 from stage.json, L7 from the 'submit with --role X' phase line) and submits 2 deterministic variants of the current
best module through submit.py, exercising daemon / rounds / acceptance / routing / phases end to end."""
import json, os, re, subprocess, sys
from pathlib import Path
prompt = sys.stdin.read(); run = Path(os.environ["RUN_DIR"]); aid = os.environ["AGENT_ID"]
rf = run / "shared/roles.json"
role = (json.load(open(rf)).get(aid) if rf.exists() else None) or re.findall(r"submit with `--role (\w+)`", prompt)[-1]
mod = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}[role]
src = (run / f"shared/best_{mod}.py").read_text(); ws = Path(os.getcwd()) if "-C" not in sys.argv else Path(sys.argv[sys.argv.index("-C") + 1])
ep = int(re.search(r"Episode (\d+)/", prompt).group(1))
for k in range(2):
    fac = 1 + 0.01 * ((hash((aid, ep, k)) % 7) - 3)
    if role == "numerical":
        n = f"_f{ep}_{k}_{aid}"; code = src + f"\n{n} = forecast\ndef forecast(view):\n    return [x * {fac} for x in {n}(view)]\n"
    elif role == "retrieval":
        n = f"_r{ep}_{k}_{aid}"; code = src + f"\n{n} = retrieve\ndef retrieve(view):\n    return [dict(c, multiplier=1 + (c['multiplier'] - 1) * {fac}) for c in {n}(view)]\n"
    else:
        n = f"_a{ep}_{k}_{aid}"; code = src + f"\n{n} = adjust\ndef adjust(view):\n    b = view['base_forecast']\n    return [bb + (a - bb) * {fac} for a, bb in zip({n}(view), b)]\n"
    p = ws / f"cand_{ep}_{k}.py"; p.write_text(code)
    r = subprocess.run([sys.executable, re.search(r"python3 (\S+submit\.py)", prompt).group(1), str(p), "--role", role, "--note", "fake"], capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-300:])
if "-o" in sys.argv: Path(sys.argv[sys.argv.index("-o") + 1]).write_text("fake episode done")
if "--json" in sys.argv:  # emulate codex --json usage accounting for the sol56 shim test
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 50}}), flush=True)
