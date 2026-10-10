#!/usr/bin/env python3
"""Dr-CiK-budget-aligned evolution (runsheet v3.3.2 L4 + L5 + L7) of the three-module document-aware forecaster on ONE
official dataset (TimesX or Time-MMD), built on a pack from build_pack.py.

This transfers Dr-CiK's stage / episode / submission / acceptance BUDGET and SELECTION PROTOCOL to a new three-module
implementation (Numerical forecast.py, Retrieval retrieve.py, Decision adjust.py). The module semantics are adapted; it
does NOT claim to reproduce Dr-CiK-specific configs (tl2 instructions, nrd4 calibration, validator) themselves.

Protocol v3 (anti-overfit): each pack's 80 official Train tasks = F0 feedback / F1 selection / F2 final test
(whole-group, disjoint). Run dirs hold ONLY F0, keyed by per-run opaque row handles without task/document/group/series
identity; agents get one quantised aggregate fitness + runtime-error count. Per-round acceptance on F0 only. After ALL
stages are frozen, final_select.py scores every candidate (+ seed) once on F1 (erroring candidates not selectable; fail
closed if none), locks the best, then scores F2 once.

Stages (episode = one synchronous round of one agent, <= 5 submissions; v3.3.2 rounds.py / sync_round.py):
  L4  group A: run L4_A with A1 numerical, A2 retrieval, A3 decision (shared notes, one champion);
      group B: independent runs L4_B1 / L4_B2 / L4_B3 (B1 numerical, B2 retrieval, B3 decision);
      6 agents x 4 rounds = 24 episodes, <= 120 submissions. gain = BJT - jt, eps 1e-4.
      group B champion = B run with the highest final F0 fitness (id asc on ties); the L5 seed = A or B champion by
      F0 fitness, tie -> B (v3.3.2 used the hidden fold here; v3 has none during evolution).
  L5  R1: S1 (single) + C1-C3 (shared) + I1-I3 (independent) = 7 agents x 4 rounds, Decision module only;
      routing: per (freq, H) cell the R1 champion with the best F0 mean gain (default: best overall);
      R2: R1-R3 (shared) x 4 rounds on F0, seed = routed function (v3.3.2 re-split folds here; v3 does not, as a
      re-split would expose F1/F2 groups). 40 episodes, <= 200 submissions; gain = BJT - jt, eps 1e-4.
  L7  numerical-a -> decision-a -> numerical-b -> decision-b, 2 agents x 2 rounds per phase = 16 episodes, <= 80
      submissions, one run, Retrieval frozen; gain = (BJT - jt) / mean BJT, eps 1e-5 (hevald_x.py).
Isolation: every agent episode runs inside bubblewrap (sandbox_codex.py): RUN/private, packs, official data and the
ledger are not mounted. Accounting: sol56 shim, gpt-5.6-sol + reasoning effort high, ledger per dataset, stage label
<dataset>:<stage>. Failure rules (v3.3.2): an episode stops only by its 2 h timeout; a stage with > 5 % failed
episodes stops the run after that stage; any shim refusal (global soft threshold) stops everything; no retries.
Restart safety: run directories are never overwritten. A completed stage writes OUT/receipts/<stage>.json and is skipped
on --resume; an interrupted stage is refused unless --restart-stage, which MOVES its run dirs to OUT/aborted/<time>/.
usage: orchestrate.py --dataset D --pack PACK --out OUT (--dry-run | --leak-test | --real-codex PATH [--resume] [--restart-stage])"""
import argparse, concurrent.futures as CF, hashlib, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
P = argparse.ArgumentParser()
P.add_argument("--dataset", choices=("timesx", "time_mmd"), required=True)
P.add_argument("--pack", type=Path, required=True)
P.add_argument("--out", type=Path, required=True)
P.add_argument("--dry-run", action="store_true")
P.add_argument("--leak-test", action="store_true", help="run the active leak probe inside the agent sandbox (no model)")
P.add_argument("--real-codex", type=Path, help="native codex binary (required for a real run)")
P.add_argument("--codex-home", type=Path, default=Path.home() / ".codex", help="source of auth.json/config.toml copied into the sandbox")
P.add_argument("--forbid", action="append", default=[], help="extra host paths the leak test must find unreadable (e.g. official data repos)")
P.add_argument("--resume", action="store_true"); P.add_argument("--restart-stage", action="store_true")
P.add_argument("--test-fake-codex", type=Path, help="TEST ONLY: scripted fake agent instead of codex (no model calls)")
A = P.parse_args()
DS = A.dataset; PACK = A.pack.resolve(); OUT = A.out.resolve()
ROLES = ("numerical", "retrieval", "decision"); MOD = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}
PER, R4, R5, R7, FAIL_FRAC = 5, 4, 4, 2, 0.05
MIN_CELL = 3  # minimum visible tasks per (freq,H) cell before per-method error means are shown
SCORING = {"L4": dict(scaled_gain=False, eps=1e-4), "L5": dict(scaled_gain=False, eps=1e-4), "L7": dict(scaled_gain=True, eps=1e-5)}
DESC = {
    "timesx": "Official TimesX (PostTime scope): commodity prices, FX rates and Google-search trends, history 96, horizon 12, daily/weekly. "
              "Documents per task: background, scenario (dated news events), holiday_info, covariates_info.",
    "time_mmd": "Official Time-MMD (MM-TSFlib): nine domains (agriculture, climate, economy, energy, environment, public health, security, "
                "social good, traffic), standardised series (train-fitted scaler), several horizons. Documents per task: retrieved facts "
                "(Final_Search_2/4/6) before the forecast origin; the views carry no timestamps."}
NAME = {"timesx": "TimesX", "time_mmd": "Time-MMD"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def jt(f, y):
    import math
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


def run_pipeline(mods, views, out):
    subprocess.run([sys.executable, str(HERE / "pipeline_runner.py"), str(mods["numerical"]), str(mods["retrieval"]), str(mods["decision"]),
                    str(views), str(out)], check=True, timeout=1800)
    return json.load(open(out))


def retrieval_coverage(views_f, retrieve_f):
    """event-bearing tasks, eligible events (up/down, conf >= 0.6), tasks with nonempty corrections, fallback tasks."""
    import importlib.util
    s = importlib.util.spec_from_file_location("r", retrieve_f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    V = json.load(open(views_f)); c = dict(tasks=len(V), event_bearing_tasks=0, events=0, eligible_events=0, nonempty_correction_tasks=0,
                                           corrections=0, fallback_tasks_with_eligible_events_but_no_correction=0, kinds={})
    for v in V.values():
        ev = [e for d in v["documents"] for e in d["events"]]; el = [e for e in ev if e.get("direction") in ("up", "down") and (e.get("confidence") or 0) >= 0.6]
        cs = m.retrieve(v); c["events"] += len(ev); c["eligible_events"] += len(el); c["corrections"] += len(cs)
        c["event_bearing_tasks"] += bool(ev); c["nonempty_correction_tasks"] += bool(cs); c["fallback_tasks_with_eligible_events_but_no_correction"] += bool(el) and not cs
        for x in cs: c["kinds"][x.get("kind", "?")] = c["kinds"].get(x.get("kind", "?"), 0) + 1
    return c


# ---------------------------------------------------------------- run directories (never overwritten)
def prep_run(name, seeds, stage, fold_key="F0", roles=None, phased=False, role_text="", notes_text=""):
    run = OUT / name
    if run.exists(): raise SystemExit(f"refusing to overwrite existing run dir {run} (use --resume / --restart-stage)")
    for s in ("shared/notes", "shared/skills", "shared/attempts", "shared/traces", "private", "queue", "results"): (run / s).mkdir(parents=True)
    # Protocol v3: a run dir gets ONLY the F0 feedback tasks (views + truth). F1/F2 never enter a run dir.
    f0 = json.load(open(PACK / "private/eval_F0_feedback.json")); V_all = json.load(open(PACK / "shared/views_train.json"))
    # v3: fresh opaque row handles per run (no task/document ids, shuffled); handle map host-private
    from anon import anonymize
    av, hmap = anonymize({t: V_all[t] for t in f0["task_ids"]})
    json.dump(av, open(run / "shared/views_train.json", "w")); json.dump(hmap, open(run / "private/handle_map.json", "w"))
    json.dump(dict(truth={h: f0["truth"][t] for h, t in hmap.items()}, base_jt={h: f0["base_jt"][t] for h, t in hmap.items()}, folds=[list(hmap)]),
              open(run / "private/eval_data.json", "w"))
    for r in ROLES: shutil.copy(seeds[r], run / f"shared/best_{MOD[r]}.py"); (run / f"shared/{r}").mkdir()
    json.dump(dict(stage=name, fold_key=fold_key, **SCORING[stage], **({"roles": roles} if roles else {}),
                   forbidden_terms=json.load(open(PACK / "private/forbidden_terms.json"))), open(run / "stage.json", "w"), indent=1)
    if phased: json.dump(dict(phase="numerical"), open(run / "shared/phase.json", "w"))
    if roles: json.dump(roles, open(run / "shared/roles.json", "w"), indent=1)  # agent -> module (not secret; agents see it)
    ed = json.load(open(run / "private/eval_data.json")); vis = set(ed["folds"][0]); V = json.load(open(run / "shared/views_train.json"))
    o = run_pipeline({r: run / f"shared/best_{MOD[r]}.py" for r in ROLES}, run / "shared/views_train.json", run / "private/_seed_out.json")
    # Anti-memorisation (v3): agents get NO per-task information (no truth, forecasts, per-task or per-method errors,
    # task identities or rankings). Only aggregates over >= MIN_CELL visible tasks of a (freq,H) cell and over all
    # visible tasks, rounded to 3 decimals.
    cells = {}
    for t in vis: cells.setdefault(f"{V[t]['freq']}|H{V[t]['H']}", []).append(t)
    def agg(ts):
        ms = sorted(set.intersection(*[set(V[t]["method_forecasts"]) for t in ts]))
        return dict(n=len(ts), mean_toto_joint_error=round(statistics.mean(ed["base_jt"][t] for t in ts), 3),
                    mean_seed_gain=round(statistics.mean(ed["base_jt"][t] - jt(o["forecasts"][t], ed["truth"][t]) for t in ts), 3),
                    mean_method_joint_error={m: round(statistics.mean(jt(V[t]["method_forecasts"][m], ed["truth"][t]) for t in ts), 3) for m in ms})
    summ = {c: (agg(ts) if len(ts) >= MIN_CELL else dict(n=len(ts), note=f"fewer than {MIN_CELL} visible tasks: withheld")) for c, ts in sorted(cells.items())}
    summ["ALL_VISIBLE"] = agg(sorted(vis))
    json.dump(summ, open(run / "shared/traces/visible_summary.json", "w"), indent=1)
    task = (HERE / "prompts/TASK_TEMPLATE.md").read_text()
    gain_txt = ("relative gain = (Toto error - final error) / (mean Toto error)" if SCORING[stage]["scaled_gain"] else "gain = Toto error - final error")
    for k, v in dict(DATASET_NAME=NAME[DS], STAGE=name, DATASET_DESC=DESC[DS], ROLE_TEXT=role_text, NOTES_TEXT=notes_text, GAIN=gain_txt,
                     SUBMIT=str(HERE / "submit.py"), ROLE_ARG="<your role>").items():
        task = task.replace("{" + k + "}", v)
    (run / "TASK.md").write_text(task); shutil.copy(run / "TASK.md", run / "shared/TASK.md")
    seed = subprocess.run([sys.executable, str(HERE / "evald.py"), str(run), "0", "--seed-only"], check=True, capture_output=True, text=True).stdout
    return run, json.loads(seed.strip().splitlines()[-1])


ROLE_TEXT = {
    "numerical": "**Your role: Numerical** (submit with `--role numerical`; your file must define `forecast(view)`).",
    "retrieval": "**Your role: Retrieval** (submit with `--role retrieval`; your file must define `retrieve(view)`).",
    "decision": "**Your role: Decision** (submit with `--role decision`; your file must define `adjust(view)`).",
    "phased": "**Your role is given in the episode text below** (numerical or decision, by the current phase; the Retrieval module is frozen in this stage).",
    "L4_A": "**Your role is fixed by your agent id**: A1 numerical (`forecast`), A2 retrieval (`retrieve`), A3 decision (`adjust`); submit only your own module.",
}
SHARED_NOTES = "- `$RUN_DIR/shared/notes/` and `shared/skills/` are shared with the other agents of this run."
INDEP_NOTES = "- You are the only agent of this run; `$RUN_DIR/shared/notes/` and `shared/skills/` are yours."
L4_ROLES = {"A1": "numerical", "A2": "retrieval", "A3": "decision", "B1": "numerical", "B2": "retrieval", "B3": "decision"}
PLAN = {
    "L4": dict(runs={"L4_A": ["A1", "A2", "A3"], "L4_B1": ["B1"], "L4_B2": ["B2"], "L4_B3": ["B3"]}, rounds=R4),
    "L5_R1": dict(runs={"L5_S": ["S1"], "L5_C": ["C1", "C2", "C3"], "L5_I1": ["I1"], "L5_I2": ["I2"], "L5_I3": ["I3"]}, rounds=R5),
    "L5_R2": dict(runs={"L5_R2": ["R1", "R2", "R3"]}, rounds=R5),
    "L7": dict(runs={"L7": ["n1a", "n2a", "d1a", "d2a", "n1b", "n2b", "d1b", "d2b"]}, rounds=R7),
}
L7_PHASES = [("numerical", "a", ["n1a:robust blending of the frozen forecasts by frequency and history", "n2a:level/trend handling and shrinkage"]),
             ("decision", "a", ["d1a:when to trust event-based corrections", "d2a:magnitude calibration relative to the base forecast"]),
             ("numerical", "b", ["n1b:robust blending of the frozen forecasts by frequency and history", "n2b:level/trend handling and shrinkage"]),
             ("decision", "b", ["d1b:when to trust event-based corrections", "d2b:magnitude calibration relative to the base forecast"])]
episodes = {k: sum(len(a) for a in v["runs"].values()) * v["rounds"] for k, v in PLAN.items()}
BUDGET = dict(L4=(episodes["L4"], episodes["L4"] * PER), L5=(episodes["L5_R1"] + episodes["L5_R2"], (episodes["L5_R1"] + episodes["L5_R2"]) * PER),
              L7=(episodes["L7"], episodes["L7"] * PER))
assert BUDGET == dict(L4=(24, 120), L5=(40, 200), L7=(16, 80)), BUDGET
SEEDS0 = {r: HERE / f"seeds/seed_{MOD[r]}.py" for r in ROLES}


def run_kwargs(stage, name, agents):
    if stage == "L4":
        return dict(stage="L4", roles={a: L4_ROLES[a] for a in agents}, role_text=ROLE_TEXT["L4_A"] if len(agents) > 1 else ROLE_TEXT[L4_ROLES[agents[0]]],
                    notes_text=SHARED_NOTES if len(agents) > 1 else INDEP_NOTES)
    if stage in ("L5_R1", "L5_R2"):
        return dict(stage="L5", roles={a: "decision" for a in agents}, role_text=ROLE_TEXT["decision"], notes_text=SHARED_NOTES if len(agents) > 1 else INDEP_NOTES,
                    fold_key="F0")  # v3: R2 also evolves on F0 (a secondary re-split would expose F1/F2 groups)
    return dict(stage="L7", phased=True, role_text=ROLE_TEXT["phased"], notes_text=SHARED_NOTES)


# ---------------------------------------------------------------- sandbox / accounting environment
def agent_env(stage, inner):
    bindir = OUT / "bin"; bindir.mkdir(parents=True, exist_ok=True); w = bindir / "codex"
    w.write_text(f"#!/bin/bash\nexec {sys.executable} {HERE / 'sol56_codex_shim.py'} \"$@\"\n"); w.chmod(0o755)
    sb = bindir / "sandboxed_codex"; sb.write_text(f"#!/bin/bash\nexec {sys.executable} {HERE / 'sandbox_codex.py'} \"$@\"\n"); sb.chmod(0o755)
    chome = OUT / "codex_home"; chome.mkdir(exist_ok=True)
    for f in ("auth.json", "config.toml"):
        if (A.codex_home / f).exists() and not (chome / f).exists(): shutil.copy(A.codex_home / f, chome / f)
    inner = Path(inner).resolve()
    if A.test_fake_codex or stage == "leak":
        # test stand-ins are copied into an otherwise empty dir, so the sandbox mounts nothing else of the package
        ib = OUT / "inner_bin"; ib.mkdir(exist_ok=True); shutil.copy(inner, ib / inner.name); (ib / inner.name).chmod(0o755)
        inner = ib / inner.name; cdir = ib
    else:
        cdir = inner.parents[2]  # native codex package dir (binary + vendored helpers), read-only
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", SOL56_ROOT=str(OUT), SOL56_STAGE=f"{DS}:{stage}", SOL56_REAL_CODEX=str(sb),
                SANDBOX_INNER=str(inner), SANDBOX_CODEX_DIR=str(cdir), SANDBOX_CODEX_HOME=str(chome),
                SANDBOX_SUBMIT=str(HERE / "submit.py"), SUBMIT_PY=str(HERE / "submit.py"))


def write_caps():
    (OUT / "ledger").mkdir(parents=True, exist_ok=True)
    if not (OUT / "ledger/caps.json").exists():
        json.dump({"global": 43_200_000, "unit_single": 40_000, "unit_episode": 300_000, "max_concurrent": 8, "poll_seconds": 2,
                   "stage_reporting_envelopes": {"L4": 26_400_000, "L5": 12_000_000, "L7": 4_800_000},
                   "note": "Dr-CiK v3.3.2 L4+L5+L7 envelopes per dataset; ONE global soft threshold"}, open(OUT / "ledger/caps.json", "w"), indent=1)


def ledger(stage=None):
    f = OUT / "ledger/ledger.jsonl"
    L = [json.loads(l) for l in open(f)] if f.exists() else []
    return [r for r in L if stage is None or r.get("stage") == f"{DS}:{stage}"]


def run_rounds(run, agents, rounds, budget, stage, phases=None):
    env = agent_env(stage, A.test_fake_codex or A.real_codex)
    d = subprocess.Popen([sys.executable, str(HERE / "evald.py"), str(run), str(budget)], env=dict(env, SYNC_ROUNDS="1"),
                         stdout=open(run / "evald.log", "a"), stderr=subprocess.STDOUT)
    try:
        sets = [(None, None, agents)] if phases is None else phases; off = 0
        for phase, tag, ags in sets:
            tf = run / "TASK.md"
            if phase:
                json.dump(dict(phase=phase), open(run / "shared/phase.json", "w"))
                tf = run / f"TASK_{phase}_{tag}.md"; tf.write_text((run / "TASK.md").read_text() + f"\n\n## Current phase\nPhase `{phase}` round-set `{tag}`: submit with `--role {phase}`.\n")
            r = subprocess.run([sys.executable, str(HERE / "rounds.py"), str(run), str(HERE / "run_agent.py"), str(rounds), str(PER), *ags],
                               env=dict(env, TASK_FILE=str(tf), ROUND_OFFSET=str(off)), stdout=open(run / "rounds.log", "a"), stderr=subprocess.STDOUT)
            if r.returncode != 0: raise RuntimeError(f"rounds.py failed in {run.name} (rc {r.returncode})")
            if d.poll() is not None: raise RuntimeError(f"evaluator daemon of {run.name} died (rc {d.returncode})")
            off += rounds
    finally:
        (run / "STOP").write_text(str(time.time()))
        try: d.wait(timeout=3600)
        except subprocess.TimeoutExpired: d.kill()
    if d.returncode != 0: raise RuntimeError(f"evaluator daemon of {run.name} exited rc {d.returncode}")


def state(run): return json.load(open(run / "private/state.json"))


def modules(run): return {r: run / f"shared/best_{MOD[r]}.py" for r in ROLES}


def parallel(jobs):
    with CF.ThreadPoolExecutor(len(jobs)) as ex:
        for f in [ex.submit(j) for j in jobs]: f.result()  # propagate the first failure


def route(r1_runs, seeds, run_dir):
    """v3.3.2 L5 routing (ensemble.py) generalised: per (freq,H) cell, the R1 champion with the best visible-fold mean gain."""
    from anon import anonymize
    f0 = json.load(open(PACK / "private/eval_F0_feedback.json")); V0 = json.load(open(PACK / "shared/views_train.json"))
    V, hmap = anonymize({t: V0[t] for t in f0["task_ids"]}); vis = list(V)
    ed = dict(truth={h: f0["truth"][t] for h, t in hmap.items()}, base_jt={h: f0["base_jt"][t] for h, t in hmap.items()})
    vf = run_dir / "_route_views.json"; json.dump(V, open(vf, "w"))
    cell = {t: f"{V[t]['freq']}|H{V[t]['H']}" for t in V}; gains = {}; names = list(r1_runs)
    for name, run in r1_runs.items():
        o = run_pipeline(dict(seeds, decision=run / "shared/best_adjust.py"), vf, run_dir / f"_route_{name}.json")
        gains[name] = {t: ed["base_jt"][t] - jt(o["forecasts"][t], ed["truth"][t]) for t in vis}
    overall = max(names, key=lambda n: (statistics.mean(gains[n].values()), -names.index(n)))
    table = {c: max(names, key=lambda n: (statistics.mean(gains[n][t] for t in vis if cell[t] == c), -names.index(n))) for c in sorted({cell[t] for t in vis})}
    src = {n: (r1_runs[n] / "shared/best_adjust.py").read_text() for n in r1_runs}
    code = ('"""Routed correction function (L5 routing): per (freq,H) cell, the R1 champion with the best visible-fold mean gain."""\n'
            f"_SOURCES = {json.dumps(src)}\nROUTE = {json.dumps(table)}\nDEFAULT = {json.dumps(overall)}\n_M = {{}}\n"
            "def _load(n):\n    if n not in _M:\n        ns = {}; exec(_SOURCES[n], ns); _M[n] = ns['adjust']\n    return _M[n]\n"
            "def adjust(view):\n    return _load(ROUTE.get(f\"{view['freq']}|H{view['H']}\", DEFAULT))(view)\n")
    p = run_dir / "routed_adjust.py"; p.write_text(code); json.dump(dict(route=table, default=overall), open(run_dir / "routing.json", "w"), indent=1)
    return p


# ---------------------------------------------------------------- modes
def pack_summary():
    pr = json.load(open(PACK / "pack_receipt.json"))
    return dict(receipt_sha256=sha(PACK / "pack_receipt.json"), task_count=pr["task_count"],
                folds_primary=[dict(n=f["n"], role=f["role"], groups=f["group_ids"], task_ids=f["task_ids"]) for f in pr["folds_primary"]],
                anchor_coverage=pr["anchor_coverage"], input_sha256=pr["input_sha256"], output_sha256=pr["output_sha256"],
                uses_test_ids_or_labels=pr["uses_test_ids_or_labels"], uses_external_dev=pr["uses_external_dev"])


def code_sha():
    return ({p.name: sha(p) for p in sorted(HERE.glob("*.py"))} | {f"seeds/{p.name}": sha(p) for p in sorted((HERE / "seeds").glob("*.py"))}
            | {"prompts/TASK_TEMPLATE.md": sha(HERE / "prompts/TASK_TEMPLATE.md")})


def dry_run():
    if OUT.exists(): raise SystemExit(f"dry-run needs a fresh --out ({OUT} exists)")
    OUT.mkdir(parents=True); write_caps()
    guard = OUT / "bin_guard"; guard.mkdir(); g = guard / "codex"; g.write_text("#!/bin/bash\necho 'dry-run: codex must not be called' >&2\nexit 99\n"); g.chmod(0o755)
    os.environ["PATH"] = f"{guard}:{os.environ['PATH']}"
    rep = dict(dataset=DS, mode="dry-run", model="gpt-5.6-sol", reasoning_effort="high", per_episode_max_submissions=PER,
               budget={k: dict(episodes=v[0], max_submissions=v[1]) for k, v in BUDGET.items()}, scoring=SCORING, failure_rules=dict(
                   episode_timeout_s=7200, stage_failed_episode_fraction_stop=FAIL_FRAC, global_threshold_refusal="stop all", retries=0), stages={})
    for st, spec in PLAN.items():
        rep["stages"][st] = {}
        for name, agents in spec["runs"].items():
            kw = run_kwargs(st, name, agents); run, seed = prep_run(f"dry/{name}", SEEDS0, **kw)
            rep["stages"][st][name] = dict(agents=agents, rounds=spec["rounds"], episodes=len(agents) * spec["rounds"], max_submissions=len(agents) * spec["rounds"] * PER,
                                           budget_per_agent=spec["rounds"] * PER, fold_key=kw.get("fold_key", "F0"), roles=kw.get("roles", "phase-driven"),
                                           seed_visible=round(seed["seed_visible"], 6),
                                           seed_note="L4 seed" if st == "L4" else "placeholder seed (real seed = previous stage's champion)")
    rep["l7_phases"] = [[p, t, [a.split(":")[0] for a in ags]] for p, t, ags in L7_PHASES]
    rep["selection"] = dict(L4="group A = L4_A; group B = best of L4_B1..3 by (F0 desc, id asc); stage seed = higher F0, tie -> B",
                            L5="R1 -> per (freq,H) cell routing on F0 -> R2 on F0; stage seed = R2 champion", L7="L7 champion",
                            final="final_select.py: every frozen stage candidate (+ seed) scored host-side on F1 once (erroring candidates excluded; fail closed if none); best locked; then F2 scored once")
    rep["retrieval_seed_coverage"] = retrieval_coverage(PACK / "shared/views_train.json", SEEDS0["retrieval"])
    rep["pack"] = pack_summary(); rep["code_sha256"] = code_sha()
    rep["llm_calls"] = len(ledger()); assert rep["llm_calls"] == 0, "dry-run must not call any model"
    json.dump(rep, open(OUT / "dry_run_report.json", "w"), indent=1)
    s = {k: {n: (v["episodes"], v["max_submissions"], v["seed_visible"]) for n, v in d.items()} for k, d in rep["stages"].items()}
    print(json.dumps(dict(dataset=DS, budget=rep["budget"], runs=s, retrieval_seed_coverage=rep["retrieval_seed_coverage"], llm_calls=0,
                          report=str(OUT / "dry_run_report.json")), indent=1))


def leak_test():
    if OUT.exists(): raise SystemExit(f"leak-test needs a fresh --out ({OUT} exists)")
    OUT.mkdir(parents=True); write_caps()
    run, _ = prep_run("leak/L4_A", SEEDS0, **run_kwargs("L4", "L4_A", ["A1", "A2", "A3"])); (run / "ws_A1").mkdir()
    probe = HERE / "tests/leak_probe.py"; env = agent_env("leak", probe)
    forb = [run / "private/eval_data.json", run / "private", run / "stage.json", PACK / "private/eval_F1_selection.json", PACK / "private/eval_F2_final_test.json", PACK, OUT / "ledger", HERE / "packs",
            HERE / "build_pack.py", A.codex_home, *map(Path, A.forbid)]
    env["LEAK_FORBIDDEN"] = os.pathsep.join(str(p) for p in forb)
    r = subprocess.run([sys.executable, str(HERE / "sandbox_codex.py"), "exec", "-C", str(run / "ws_A1"), "-m", "x"], env=env, capture_output=True, text=True)
    out = dict(rc=r.returncode, probe=r.stdout.strip()[-2000:], stderr=r.stderr.strip()[-500:], forbidden_checked=[str(p) for p in forb])
    json.dump(out, open(OUT / "leak_test_report.json", "w"), indent=1); print(json.dumps(out, indent=1))
    if r.returncode != 0: raise SystemExit("LEAK TEST FAILED")


def real_run():
    if not (A.real_codex or A.test_fake_codex): raise SystemExit("--real-codex PATH is required for a real run")
    if shutil.which("bwrap") is None: raise SystemExit("bubblewrap (bwrap) is required for agent isolation")
    if OUT.exists() and not (A.resume or A.restart_stage) and any(OUT.iterdir()): raise SystemExit(f"{OUT} exists; use --resume")
    OUT.mkdir(parents=True, exist_ok=True); write_caps(); (OUT / "receipts").mkdir(exist_ok=True)
    say = lambda m: open(OUT / "stages.log", "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + m + "\n")
    done = lambda st: (OUT / "receipts" / f"{st}.json").exists()

    def begin(st, names):
        existing = [n for n in names if (OUT / n).exists()]
        if existing:
            if not A.restart_stage: raise SystemExit(f"stage {st} was interrupted ({existing}); rerun with --restart-stage to archive and restart it")
            dst = OUT / "aborted" / time.strftime("%Y%m%dT%H%M%S"); dst.mkdir(parents=True)
            for n in existing: shutil.move(str(OUT / n), str(dst / n))
            say(f"{st}: archived interrupted runs {existing} -> {dst}")

    def finish(st, ledger_stage, runs, extra):
        L = ledger(ledger_stage); com = [r for r in L if r["kind"] in ("commit", "reclaim")]
        fails = [r for r in com if r["kind"] == "reclaim" or r.get("rc", 0) != 0 or not r.get("reported", False)]
        rec = dict(stage=st, runs={n: dict(submissions=state(OUT / n)["n"], best_F0=state(OUT / n)["best_visible"],
                                           used=state(OUT / n)["used"]) for n in runs},
                   episodes_accounted=len(com), failed_episodes=len(fails), refused=sum(r["kind"] == "refused" for r in L),
                   overshoot_units=sum(r.get("overshoot", 0) > 0 for r in com), net_tokens=sum(r["tokens"] for r in com), **extra)
        rec["stop"] = (rec["refused"] > 0 and "global soft threshold refusal") or (len(com) and len(fails) / len(com) > FAIL_FRAC and f"> {FAIL_FRAC:.0%} failed episodes") or None
        json.dump(rec, open(OUT / "receipts" / f"{st}.json", "w"), indent=1); say(f"{st} done: {json.dumps({k: rec[k] for k in ('episodes_accounted', 'failed_episodes', 'refused', 'stop')})}")
        if rec["stop"]: raise SystemExit(f"stopping after {st}: {rec['stop']}")
        return rec

    # L4
    names4 = list(PLAN["L4"]["runs"])
    if not done("L4"):
        begin("L4", names4)
        for n, a in PLAN["L4"]["runs"].items(): prep_run(n, SEEDS0, **run_kwargs("L4", n, a))
        say("L4 start"); parallel([lambda n=n, a=a: run_rounds(OUT / n, a, R4, R4 * PER, "L4") for n, a in PLAN["L4"]["runs"].items()])
        bchamp = sorted(["L4_B1", "L4_B2", "L4_B3"], key=lambda n: (-state(OUT / n)["best_visible"], n))[0]
        # v3: no hidden fold during evolution -> A vs B by F0 fitness, tie -> B (F1 decides the FINAL program later)
        final4 = "L4_A" if state(OUT / "L4_A")["best_visible"] > state(OUT / bchamp)["best_visible"] else bchamp
        finish("L4", "L4", names4, dict(group_A="L4_A", group_B_champion=bchamp, final=final4,
                                        modules_sha256={r: sha(p) for r, p in modules(OUT / final4).items()}))
    s4 = modules(OUT / json.load(open(OUT / "receipts/L4.json"))["final"])
    # L5 R1 + routing
    names5 = list(PLAN["L5_R1"]["runs"])
    if not done("L5_R1"):
        begin("L5_R1", names5)
        for n, a in PLAN["L5_R1"]["runs"].items(): prep_run(n, s4, **run_kwargs("L5_R1", n, a))
        say("L5 R1 start"); parallel([lambda n=n, a=a: run_rounds(OUT / n, a, R5, R5 * PER, "L5") for n, a in PLAN["L5_R1"]["runs"].items()])
        routed = route({n: OUT / n for n in names5}, s4, OUT)
        finish("L5_R1", "L5", names5, dict(routing=json.load(open(OUT / "routing.json")), routed_sha256=sha(routed)))
    # L5 R2
    if not done("L5_R2"):
        begin("L5_R2", ["L5_R2"]); prep_run("L5_R2", dict(s4, decision=OUT / "routed_adjust.py"), **run_kwargs("L5_R2", "L5_R2", PLAN["L5_R2"]["runs"]["L5_R2"]))
        say("L5 R2 start"); run_rounds(OUT / "L5_R2", PLAN["L5_R2"]["runs"]["L5_R2"], R5, R5 * PER, "L5")
        finish("L5_R2", "L5", ["L5_R2"], dict(final="L5_R2", modules_sha256={r: sha(p) for r, p in modules(OUT / "L5_R2").items()}))
    s5 = modules(OUT / "L5_R2")
    # L7
    if not done("L7"):
        begin("L7", ["L7"]); prep_run("L7", s5, **run_kwargs("L7", "L7", PLAN["L7"]["runs"]["L7"]))
        say("L7 start"); run_rounds(OUT / "L7", None, R7, R7 * PER, "L7", phases=L7_PHASES)
        finish("L7", "L7", ["L7"], dict(final="L7", modules_sha256={r: sha(p) for r, p in modules(OUT / "L7").items()}, code_sha256=code_sha()))
    # v3: host-only final selection on F1 (once), lock, then F2 final test (once)
    if not (OUT / "final/LOCK.json").exists():
        r = subprocess.run([sys.executable, str(HERE / "final_select.py"), "--out", str(OUT), "--pack", str(PACK)], capture_output=True, text=True)
        if r.returncode != 0: raise SystemExit("final selection failed: " + r.stderr[-800:])
        say("final selection locked + F2 final test written: " + r.stdout.strip()[-400:])
    say("all stages done")


if A.dry_run: dry_run()
elif A.leak_test: leak_test()
else: real_run()
