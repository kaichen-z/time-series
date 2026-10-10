#!/usr/bin/env python3
"""TEST ONLY (active leak test): runs INSIDE the agent sandbox in place of codex and tries to read everything an agent must
not see. Prints a JSON verdict; exit 0 only if every forbidden read fails and every required access works.
Forbidden paths are passed in LEAK_FORBIDDEN (os.pathsep-separated)."""
import glob, json, os, sys
from pathlib import Path

argv = sys.argv[1:]; ws = Path(argv[argv.index("-C") + 1]); run = ws.parent
res = {"forbidden_readable": [], "required_failed": []}
for p in os.environ["LEAK_FORBIDDEN"].split(os.pathsep):
    try:
        if os.path.isdir(p): os.listdir(p); res["forbidden_readable"].append(p)
        else: open(p, "rb").read(16); res["forbidden_readable"].append(p)
    except OSError: pass
# other processes (daemon/orchestrator environment) must be invisible
for e in glob.glob("/proc/[0-9]*/environ"):
    try:
        if b"SOL56_ROOT" in open(e, "rb").read(): res["forbidden_readable"].append(e)
    except OSError: pass
for p, mode in ((run / "shared/store/views_meta.json", "r"), (run / "shared/store/forecasts.f32", "rb"), (run / "shared/store/viewstore.py", "r"), (run / "TASK.md", "r")):
    try: open(p, mode).read(16)
    except OSError: (res["required_failed"].append(str(p)) if p.name != "TASK.md" else None)
try: (run / "queue" / ".probe").write_text("x"); (run / "queue" / ".probe").unlink()
except OSError: res["required_failed"].append("queue write")
try: (ws / "probe.txt").write_text("x")
except OSError: res["required_failed"].append("workspace write")
ok = not res["forbidden_readable"] and not res["required_failed"]
print(json.dumps(dict(res, ok=ok))); sys.exit(0 if ok else 1)
