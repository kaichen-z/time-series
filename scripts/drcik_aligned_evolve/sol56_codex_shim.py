#!/usr/bin/env python3
"""`codex` shim for the Dr-CiK full-chain gpt-5.6-sol reconstruction (sol56_code/new/bin first on PATH).

Every codex invocation of the run goes through here:
  1. only `codex exec` with `-m gpt-5.6-sol` is allowed (else exit 96 / 97, no call is made);
  2. ATOMIC admission: in ONE critical section (exclusive flock on the ledger lock) the shim
       - reclaims reservations whose owning shim process is dead (crash / kill -9) -> `reclaim` record, unit cap committed
         as its usage (conservative, the true usage is unknown),
       - checks open units < max_concurrent,
       - checks global: committed (including the pre-charged VOID entry) + open reservations + unit cap <= global,
       - and only then appends the `reserve` record.
     Global-budget failure -> exit 98 (the whole run stops). Stage usage is reported but never used as an admission cap;
     concurrency full -> release the lock, sleep, retry (never fails);
  3. the real codex runs with the same argv/stdin; stdout/stderr are STREAMED through unchanged (line-wise tee);
  4. (v3.3) codex always runs with `--json`; the usage of every `turn.completed` event is summed and committed as
     NET = input - cached_input + output (missing -> unit cap committed, reported=false), together with gross input,
     cached input, uncached input and output; overshoot above the unit cap is recorded. codex cannot be stopped mid-unit, so the global value is a SOFT stop threshold, not a hard cap.
Env: SOL56_ROOT (ledger dir), SOL56_STAGE, SOL56_REAL_CODEX. Ledger: $SOL56_ROOT/ledger/ledger.jsonl, caps.json."""
import ctypes, fcntl, json, os, re, signal, subprocess, sys, threading, time, uuid
from pathlib import Path

MODEL = "gpt-5.6-sol"
ROOT = Path(os.environ["SOL56_ROOT"]); STAGE = os.environ["SOL56_STAGE"]
REAL = os.environ.get("SOL56_REAL_CODEX", "/home/yiqi/.local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex")  # native binary (the node wrapper would spawn it as a grandchild)
NATIVE_ENV = {"CODEX_MANAGED_BY_NPM": "1", "CODEX_MANAGED_PACKAGE_ROOT": "/home/yiqi/.local/lib/node_modules/@openai/codex"}
LD = ROOT / "ledger"; LD.mkdir(parents=True, exist_ok=True)
CAPS = json.load(open(LD / "caps.json")); LEDGER = LD / "ledger.jsonl"; LOCK = LD / "ledger.lock"


def model_of(argv):
    for i, a in enumerate(argv):
        if a in ("-m", "--model") and i + 1 < len(argv): return argv[i + 1]
        if a.startswith("--model="): return a.split("=", 1)[1]
    return None


def alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def state():
    """committed tokens per stage (+_all), open reservations {id: rec}."""
    com, opn = {"_all": 0}, {}
    if LEDGER.exists():
        for ln in open(LEDGER):
            r = json.loads(ln)
            if r["kind"] == "reserve": opn[r["id"]] = r
            elif r["kind"] in ("commit", "reclaim", "void"):
                opn.pop(r.get("id"), None)
                com[r["stage"]] = com.get(r["stage"], 0) + r["tokens"]; com["_all"] += r["tokens"]
    return com, opn


def log(rec):
    with open(LEDGER, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())


def admit(cap, single):
    uid = uuid.uuid4().hex[:12]
    while True:
        with open(LOCK, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)  # one critical section: reclaim + concurrency + budgets + reserve
            com, opn = state()
            for r in list(opn.values()):
                if not alive(r["pid"]):
                    log(dict(kind="reclaim", id=r["id"], stage=r["stage"], cap=r["cap"], tokens=r["cap"], t=time.time()))
                    opn.pop(r["id"]); com[r["stage"]] = com.get(r["stage"], 0) + r["cap"]; com["_all"] += r["cap"]
            res_all = sum(r["cap"] for r in opn.values())
            if com["_all"] + res_all + cap > CAPS["global"]:
                log(dict(kind="refused", id=uid, stage=STAGE, cap=cap, pid=os.getpid(), t=time.time())); return None
            if len(opn) < CAPS["max_concurrent"]:
                log(dict(kind="reserve", id=uid, stage=STAGE, cap=cap, single=single, pid=os.getpid(), t=time.time()))
                return uid
        time.sleep(CAPS.get("poll_seconds", 2))


def main():
    argv = sys.argv[1:]
    if argv[:1] != ["exec"]:
        print("sol56 shim: only `codex exec` is allowed", file=sys.stderr); sys.exit(96)
    m = model_of(argv)
    if m != MODEL:
        print(f"sol56 shim: model {m!r} rejected (only {MODEL})", file=sys.stderr); sys.exit(97)
    single = "--ephemeral" in argv or "--output-schema" in argv
    if "--json" not in argv: argv = ["exec", "--json"] + argv[1:]  # (v3.3) usage accounting from turn.completed events
    cap = CAPS["unit_single"] if single else CAPS["unit_episode"]
    uid = admit(cap, single)
    if uid is None:
        print(f"sol56 shim: budget refused (stage {STAGE})", file=sys.stderr); sys.exit(98)
    def _pdeath():  # the real codex dies with the shim (kill -9 of the shim cannot leave an unaccounted codex running)
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG = 1
    # (v3.3.1) STREAMING: stdout is forwarded line by line as it arrives (tee) and only the small usage state is kept;
    # stderr is forwarded by a thread; no transcript is buffered in memory.
    p = subprocess.Popen([REAL] + argv, stdin=sys.stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         preexec_fn=_pdeath, env=dict(os.environ, **NATIVE_ENV))
    def _err():
        for chunk in iter(lambda: p.stderr.read1(65536), b""): sys.stderr.buffer.write(chunk); sys.stderr.buffer.flush()
    th = threading.Thread(target=_err, daemon=True); th.start()
    gin = cin = out = 0; nturn = 0; out_buf = sys.stdout.buffer
    for ln in iter(lambda: p.stdout.readline(1 << 20), b""):   # <= 1 MiB per read: a huge single line is streamed in chunks
        out_buf.write(ln); out_buf.flush()
        if b'"turn.completed"' in ln:
            try: ev = json.loads(ln)
            except ValueError: ev = None
            if isinstance(ev, dict) and ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
                u = ev["usage"]; nturn += 1
                gin += int(u.get("input_tokens", 0)); cin += int(u.get("cached_input_tokens", 0)); out += int(u.get("output_tokens", 0))
    rc = p.wait(); th.join(timeout=30)
    mt = nturn > 0
    tok = (gin - cin + out) if mt else cap   # NET = uncached input + output
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        log(dict(kind="commit", id=uid, stage=STAGE, cap=cap, tokens=tok, reported=bool(mt), overshoot=max(0, tok - cap),
                 gross_input=gin, cached_input=cin, uncached_input=gin - cin, output=out, turns=nturn,
                 rc=rc, t=time.time()))
    sys.exit(rc)


if __name__ == "__main__":
    main()
