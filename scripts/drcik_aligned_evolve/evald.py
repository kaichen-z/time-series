"""Evaluator daemon for one run directory of the Dr-CiK-budget-aligned evolution (generalised from the v3.3.2 daemons
coevolution/evald.py, meta_harness/hevald.py and coevo_x/hevald_x.py; same scoring and acceptance).

Agents submit ONE module (role numerical -> forecast.py, retrieval -> retrieve.py, decision -> adjust.py); the other
two modules are the run's current shared best. Which role an agent may submit is fixed in RUN/stage.json
("roles": {agent_id: role}) or, for phased runs (L7), by RUN/shared/phase.json.

Score (Dr-CiK v3.3.2 formulas per stage): per task gain = Toto joint error - final joint error (L4/L5) or the same
divided by the mean Toto joint error (L7), joint error = sMAE + sRMSE each capped at 5; fitness = robust gain (mean +
0.5 * mean negative part) over the F0 FEEDBACK tasks, the only tasks in a run dir (protocol v2). Synchronous rounds
(SYNC_ROUNDS=1, sync_round.py): eligible iff F0 fitness > frozen round base + eps; winner at round close. F1 (final
selection) and F2 (final test) are never available here; see final_select.py. Per-task numbers never leave private/; agents get fixed-precision aggregates only. usage: evald.py <run_dir> <budget_per_agent>"""
import io, json, math, os, shutil, statistics, subprocess, sys, time, tokenize, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import sync_round as SR  # noqa: E402
RUN = Path(sys.argv[1]).resolve(); BUDGET = int(sys.argv[2])
STAGE = json.load(open(RUN / "stage.json")); ED = json.load(open(RUN / "private/eval_data.json"))
# Protocol v2 (anti-overfit): the run dir holds truth for the F0 FEEDBACK tasks ONLY. F1 (selection) and F2 (final
# test) never enter a run dir; they are scored host-side by final_select.py after every stage is frozen.
FOLDS = ED["folds"]; assert len(FOLDS) == 1, "run eval_data must contain only the F0 feedback fold"; VIS = FOLDS[0]
TRUTH, BJT = ED["truth"], ED["base_jt"]
# v3.3.2 per-stage formulas: L4 coevolution/evald.py and L5 meta_harness/hevald.py use gain = BJT - jt (no scale), visible
# eps 1e-4, hidden tolerance 1e-9; L7 coevo_x/hevald_x.py divides by the dataset's mean Toto joint error, eps 1e-5, tol 1e-6.
SCALE = (sum(BJT[t] for t in VIS) / len(VIS)) if STAGE["scaled_gain"] else 1.0
PEN, EPS = 0.5, STAGE["eps"]
MOD = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}
SYNC = os.environ.get("SYNC_ROUNDS") == "1"


# ---- anti-memorisation guards (v2; applied to agent submissions only, never to seeds) ----
MAX_BYTES, MAX_NUMBERS, MAX_STRING_CHARS = 20_000, 300, 2_000
BANNED_NAMES = {"exec", "eval", "compile", "__import__", "open", "globals", "locals", "vars", "getattr", "setattr", "hash", "breakpoint", "input"}
MAX_NUMBER_CHARS = 3_000
BANNED_MODULES = {"base64", "zlib", "gzip", "bz2", "lzma", "pickle", "marshal", "codecs", "binascii", "importlib", "os", "sys",
                  "subprocess", "pathlib", "io", "shutil", "socket", "urllib", "json", "ctypes", "struct", "array", "zipfile", "tarfile", "hashlib", "hmac", "secrets", "random", "builtins", "inspect", "types", "functools", "operator"}
NEAR_PERFECT_REL, NEAR_PERFECT_FRAC = 1e-3, 0.10


def static_check(code):
    """Reject lookup-table / encoded-answer programs before running them."""
    if len(code.encode()) > MAX_BYTES: raise RuntimeError(f"rejected: code larger than {MAX_BYTES} bytes")
    nums = strs = nchars = 0; prev = None
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.NUMBER: nums += 1; nchars += len(tok.string)
        elif tok.type == tokenize.STRING and not (prev in (None, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.NL)):
            strs += len(tok.string)  # docstrings (statement-level strings) are not counted
        elif tok.type == tokenize.NAME and tok.string in BANNED_NAMES: raise RuntimeError(f"rejected: use of '{tok.string}'")
        if tok.type not in (tokenize.COMMENT, tokenize.NL): prev = tok.type
    if nums > MAX_NUMBERS: raise RuntimeError(f"rejected: {nums} numeric constants (max {MAX_NUMBERS}) - lookup tables are not allowed")
    if nchars > MAX_NUMBER_CHARS: raise RuntimeError(f"rejected: {nchars} characters of numeric literals (max {MAX_NUMBER_CHARS})")
    lits = [t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.STRING]
    low = " ".join(lits).lower()
    hit = [w for w in STAGE.get("forbidden_terms", []) if w.lower() in low]
    if hit: raise RuntimeError(f"rejected: series/group-specific names in string literals ({len(hit)} found) - rules must be general")
    if strs > MAX_STRING_CHARS: raise RuntimeError(f"rejected: {strs} characters of string literals (max {MAX_STRING_CHARS})")
    import ast
    for node in ast.walk(ast.parse(code)):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for m in mods:
            if m.split(".")[0] in BANNED_MODULES: raise RuntimeError(f"rejected: import of '{m}'")


def jt(f, y):
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


def robust(g): return statistics.mean(g) + PEN * statistics.mean(min(0.0, x) for x in g)


def public(d):
    """what agents may read: fixed precision (4 decimals) for scores; everything else is already aggregate."""
    d = dict(d)
    if "visible_fitness" in d: d["visible_fitness"] = round(d["visible_fitness"], 3)  # pre-registered quantisation 1e-3
    return d


def best(role): return RUN / f"shared/best_{MOD[role]}.py"


def score(mods, check_memo=False):
    o = RUN / "private/_out.json"
    subprocess.run([sys.executable, str(HERE / "pipeline_runner.py"), str(mods["numerical"]), str(mods["retrieval"]), str(mods["decision"]),
                    str(RUN / "shared/views_train.json"), str(o)], check=True, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    r = json.load(open(o)); errs = {t: jt(f, TRUTH[t]) for t, f in r["forecasts"].items()}
    per = {t: (BJT[t] - e) / SCALE for t, e in errs.items()}
    if check_memo:
        # base (numerical output) and final forecast are both checked: a memoriser may pre-compensate later stages
        near = [t for t in VIS if min(errs[t], jt(r["base"][t], TRUTH[t])) <= NEAR_PERFECT_REL * BJT[t]]
        if len(near) >= max(3, NEAR_PERFECT_FRAC * len(VIS)):
            raise RuntimeError(f"rejected as memorisation: {len(near)} visible tasks reproduced almost exactly (error <= {NEAR_PERFECT_REL} x Toto)")
    vis = {t: per[t] for t in VIS}
    # agents see only fixed-precision aggregates: no task ids, rankings or per-task values
    # v3: a single aggregate fitness (quantised to 1e-3 when shown) + total runtime-error count; no better/worse counts
    pub = dict(visible_fitness=robust(list(vis.values())), n_runtime_errors=sum(t in vis for t in r["errors"]))
    return pub, 0.0  # no hidden scalar exists during evolution (sync_round ties on it are neutral)


def main():
    for s in ("queue", "results", "private", "shared/attempts"): (RUN / s).mkdir(parents=True, exist_ok=True)
    st_f = RUN / "private/state.json"
    if st_f.exists(): st = json.load(open(st_f))
    else:
        pub, hid = score({r: best(r) for r in MOD})
        st = dict(best_visible=pub["visible_fitness"], best_hidden=0.0, used={}, n=0, history=[])
        json.dump(public(dict(pub, attempt=0, agent="seed", accepted=True)), open(RUN / "shared/attempts/0000_seed.json", "w"), indent=1)
        json.dump(st, open(st_f, "w"))
    if "--seed-only" in sys.argv: print(json.dumps(dict(seed_visible=st["best_visible"]))); return
    print("daemon up; seed F0 fitness", round(st["best_visible"], 5), flush=True)
    if SYNC: SR.init(st, st["best_visible"], st["best_hidden"])

    def apply_winner(w):
        cp = RUN / w["ref"]; role = w["ref"].split("/")[1]
        shutil.copy(cp, best(role)); st.update(best_visible=w["visible"], best_hidden=w["hidden"])
        af = RUN / f"shared/attempts/{cp.stem}.json"
        if af.exists(): a = json.load(open(af)); a["accepted"] = True; a["accepted_at_round_close"] = st["round"]; json.dump(a, open(af, "w"), indent=1)
        return w["visible"], w["hidden"]

    while True:
        items = sorted((RUN / "queue").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not items:
            if SYNC and SR.maybe_close(RUN, st, apply_winner, True): json.dump(st, open(st_f, "w")); continue
            if (RUN / "STOP").exists(): break
            time.sleep(3); continue
        q = items[0]; sub = json.load(open(q))
        (RUN / "private/inflight.json").write_text(json.dumps(dict(file=q.name)))  # in-flight marker BEFORE dequeue
        open(RUN / "private/dequeued.log", "a").write(q.stem + "\n"); q.unlink()
        agent, rid, role = sub.get("agent", "?"), sub["request_id"], sub.get("role"); out = dict(request_id=rid, agent=agent, role=role)
        try:
            allowed = STAGE["roles"].get(agent) if "roles" in STAGE else json.load(open(RUN / "shared/phase.json"))["phase"]
            if role != allowed: raise RuntimeError(f"agent {agent} may only submit role '{allowed}', got '{role}'")
            if st["used"].get(agent, 0) >= BUDGET: raise RuntimeError(f"budget exhausted ({BUDGET})")
            st["used"][agent] = st["used"].get(agent, 0) + 1; st["n"] += 1
            cp = RUN / f"shared/{role}/{st['n']:04d}_{agent}.py"; cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(sub["code"])
            static_check(sub["code"])
            mods = {r: best(r) for r in MOD}; mods[role] = cp
            pub, hid = score(mods, check_memo=True)
            ref_v = st["base_visible"] if SYNC else st["best_visible"]
            elig = pub["visible_fitness"] > ref_v + EPS
            if SYNC:
                idx = SR.next_index(st, agent); SR.record(st, agent, idx, pub["visible_fitness"], hid, elig, f"shared/{role}/{cp.name}")
                acc = False; out.update(eligible=elig, round=st["round"])
            else:
                acc = elig
                if acc: shutil.copy(cp, best(role)); st.update(best_visible=pub["visible_fitness"], best_hidden=hid)
            out.update(pub, accepted=acc, budget_left=BUDGET - st["used"][agent], attempt=st["n"], file=f"{role}/{cp.name}")
            st["history"].append(dict(n=st["n"], agent=agent, role=role, visible=pub["visible_fitness"], eligible=elig, t=time.time()))
            json.dump(dict(public(out), note=sub.get("note", "")), open(RUN / f"shared/attempts/{st['n']:04d}_{agent}.json", "w"), indent=1)
        except subprocess.CalledProcessError as e:
            out.update(error="crashed: " + (e.stderr or b"").decode()[-600:])
        except Exception as e:
            out.update(error=repr(e)[:400]); traceback.print_exc()
        json.dump(public(out), open(RUN / f"results/{rid}.json", "w"), indent=1); json.dump(st, open(st_f, "w"))
        open(RUN / "private/done.log", "a").write(q.stem + "\n"); (RUN / "private/inflight.json").unlink()
        print(f"[{st['n']}] {agent}/{role} -> {'ERR ' + out['error'][:80] if 'error' in out else ('eligible' if out.get('eligible', out.get('accepted')) else 'reject')} "
              f"F0 {out.get('visible_fitness', float('nan')):+.5f}", flush=True)
    print("daemon stopped", flush=True)


if __name__ == "__main__":
    main()
