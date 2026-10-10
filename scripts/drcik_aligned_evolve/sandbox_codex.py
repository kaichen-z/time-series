#!/usr/bin/env python3
"""OS-level read isolation for agent episodes: run the real codex (or, in tests, a probe) inside bubblewrap with ONLY
  * system directories read-only (/usr, /etc and the usual /bin, /lib* links), fresh /proc (own PID namespace), /dev, /tmp;
  * the codex installation read-only and a per-run CODEX_HOME (auth/config copy) read-write;
  * RUN/shared and RUN/queue read-write, RUN/results read-only, the agent's own workspace RUN/ws_<id> read-write;
  * submit.py read-only.
RUN/private (truth, hidden folds, daemon state), the task packs, the official data repositories, the ledger and every
other path on the host are NOT mounted, so they cannot be read. The sol56 shim (outside the sandbox) calls this file as
SOL56_REAL_CODEX; argv is the codex argv. Env: SANDBOX_INNER (binary to run inside), SANDBOX_CODEX_DIR (install dir to
mount), SANDBOX_CODEX_HOME, SANDBOX_SUBMIT."""
import os, sys
from pathlib import Path

argv = sys.argv[1:]
def opt(flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag and i + 1 < len(argv)]

ws = Path(opt("-C")[0]).resolve(); run = ws.parent
assert ws.name.startswith("ws_") and (run / "stage.json").exists(), "sandbox: unexpected workspace layout"
inner = os.environ["SANDBOX_INNER"]; cdir = os.environ.get("SANDBOX_CODEX_DIR", str(Path(inner).parent))
chome = os.environ["SANDBOX_CODEX_HOME"]; submit = os.environ["SANDBOX_SUBMIT"]
cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--new-session",
       "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
# DNS: /etc/resolv.conf is usually a symlink into /run (systemd-resolved); mount only the resolver dir, read-only
for p in ("/run/systemd/resolve", "/run/resolvconf"):
    if os.path.isdir(p): cmd += ["--ro-bind", p, p]
for p in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
    if os.path.islink(p): cmd += ["--symlink", os.readlink(p), p]
    elif os.path.isdir(p): cmd += ["--ro-bind", p, p]
# clean environment: only what the agent needs (no SOL56_*/SANDBOX_* paths of the host)
keep = {k: os.environ[k] for k in ("RUN_DIR", "AGENT_ID", "LANG", "LC_ALL", "TERM", "LEAK_FORBIDDEN") if k in os.environ}
cmd += ["--clearenv", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", *[x for k, v in keep.items() for x in ("--setenv", k, v)]]
cmd += ["--ro-bind", cdir, cdir, "--bind", chome, chome, "--setenv", "CODEX_HOME", chome, "--setenv", "HOME", "/tmp",
        "--bind", str(run / "shared"), str(run / "shared"), "--bind", str(run / "queue"), str(run / "queue"),
        "--ro-bind", str(run / "results"), str(run / "results"), "--bind", str(ws), str(ws),
        "--ro-bind", submit, submit, "--chdir", str(ws), inner, *argv]
os.execvpe("bwrap", cmd, dict(keep, PATH="/usr/local/bin:/usr/bin:/bin"))  # bwrap itself (PID 1 inside) also gets the clean env
