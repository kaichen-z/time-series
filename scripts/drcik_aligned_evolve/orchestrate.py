#!/usr/bin/env python3
"""Dr-CiK-budget-aligned evolution (runsheet v3.3.2 L4 + L5 + L7) of the three-module document-aware forecaster on ONE
official dataset (TimesX or Time-MMD), built on a pack from build_pack.py.

This transfers Dr-CiK's stage / episode / submission / acceptance BUDGET to a new three-module implementation
(Numerical forecast.py, Retrieval retrieve.py, Decision adjust.py). The module semantics are adapted; it does NOT claim to
reproduce Dr-CiK-specific configs (tl2 instructions, nrd4 calibration, validator) themselves.

Stages (episode = one synchronous round of one agent, <= 5 submissions; v3.3.2 rounds.py / sync_round.py acceptance):
  L4  6 agents x 4 rounds = 24 episodes, <= 120 submissions.
      run L4_A : A1 numerical, A2 retrieval, A3 decision (shared notes, one champion)
      runs L4_B1/B2/B3 : B1 numerical / B2 retrieval / B3 decision (independent, one agent each)
      -> L4 champion = run with the highest final visible fitness (hidden non-degradation holds by construction;
         tie -> L4_A). Its three modules seed L5.
  L5  R1: S1 (single) + C1-C3 (shared) + I1-I3 (independent) = 7 agents x 4 rounds = 28 episodes, decision module only;
      routing: per (freq, H) cell choose the R1 champion adjust with the best visible-fold mean gain (default: best overall);
      R2: R1-R3 (shared) x 4 rounds = 12 episodes on the SECONDARY whole-group folds (v3.3.2 prep_mh2 semantics),
      seed = routed function. 40 episodes, <= 200 submissions. L5 final = R2 champion.
  L7  numerical-a -> decision-a -> numerical-b -> decision-b, 2 agents x 2 rounds per phase = 16 episodes, <= 80
      submissions, one shared run (retrieval frozen), seed = L5 final.
Model: gpt-5.6-sol, reasoning effort high, every call through the sol56 accounting shim (ledger per dataset; stage label
<dataset>:<stage>). --dry-run builds every run directory, scores the seeds locally and prints the plan; it never starts
an agent (the codex on PATH is replaced by a guard that fails, and the ledger is checked to be empty).
usage: orchestrate.py --dataset {timesx,time_mmd} --pack PACK --out OUT [--dry-run]"""
import argparse, hashlib, json, os, shutil, statistics, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = argparse.ArgumentParser()
P.add_argument("--dataset", choices=("timesx", "time_mmd"), required=True)
P.add_argument("--pack", type=Path, required=True)
P.add_argument("--out", type=Path, required=True)
P.add_argument("--dry-run", action="store_true")
P.add_argument("--test-fake-codex", type=Path, help="TEST ONLY: replace codex by a scripted fake agent (no model calls)")
P.add_argument("--real-codex", default="/home/yiqi/.local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex")
A = P.parse_args()
DS = A.dataset; PACK = A.pack.resolve(); OUT = A.out.resolve(); OUT.mkdir(parents=True, exist_ok=True)
ROLES = ("numerical", "retrieval", "decision"); MOD = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}
PER, R4, R5, R7 = 5, 4, 4, 2
DESC = {
    "timesx": "Official TimesX (PostTime scope): commodity prices, FX rates and Google-search trends, history 96, horizon 12, daily/weekly. "
              "Documents per task: background, scenario (dated news events), holiday_info, covariates_info.",
    "time_mmd": "Official Time-MMD (MM-TSFlib): nine domains (agriculture, climate, economy, energy, environment, public health, security, "
                "social good, traffic), standardised series (train-fitted scaler), several horizons. Documents per task: retrieved facts "
                "(Final_Search_2/4/6) before the forecast origin."}
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


# ---------------------------------------------------------------- run directories
def prep_run(name, seeds, fold_key="folds", roles=None, phased=False, role_text="", notes_text=""):
    run = OUT / name
    if run.exists(): shutil.rmtree(run)
    for s in ("shared/notes", "shared/skills", "shared/attempts", "shared/traces", "private", "queue", "results"): (run / s).mkdir(parents=True)
    shutil.copy(PACK / "shared/views_train.json", run / "shared/views_train.json"); shutil.copy(PACK / "private/eval_data.json", run / "private/eval_data.json")
    for r in ROLES: shutil.copy(seeds[r], run / f"shared/best_{MOD[r]}.py"); (run / f"shared/{r}").mkdir()
    json.dump(dict(stage=name, fold_key=fold_key, **({"roles": roles} if roles else {})), open(run / "stage.json", "w"), indent=1)
    if phased: json.dump(dict(phase="numerical"), open(run / "shared/phase.json", "w"))
    ed = json.load(open(run / "private/eval_data.json")); F = ed[fold_key]; vis = set(F[0] + F[1]); V = json.load(open(run / "shared/views_train.json"))
    o = run_pipeline({r: run / f"shared/best_{MOD[r]}.py" for r in ROLES}, run / "shared/views_train.json", run / "private/_seed_out.json")
    with open(run / "shared/traces/visible.jsonl", "w") as fo:
        for t in sorted(vis):
            y = ed["truth"][t]
            fo.write(json.dumps(dict(tid=t, fold=0 if t in F[0] else 1, truth=y, toto_joint_error=round(ed["base_jt"][t], 4),
                                     method_joint_errors={m: round(jt(f, y), 4) for m, f in V[t]["method_forecasts"].items()},
                                     seed_base_forecast=o["base"][t], seed_corrections=o["corrections"][t], seed_final_forecast=o["forecasts"][t],
                                     seed_final_gain=round(ed["base_jt"][t] - jt(o["forecasts"][t], y), 4))) + "\n")
    task = (HERE / "prompts/TASK_TEMPLATE.md").read_text()
    for k, v in dict(DATASET_NAME=NAME[DS], STAGE=name, DATASET_DESC=DESC[DS], ROLE_TEXT=role_text, NOTES_TEXT=notes_text,
                     SUBMIT=str(HERE / "submit.py"), ROLE_ARG="<your role>" if phased else "<your role>").items():
        task = task.replace("{" + k + "}", v)
    (run / "TASK.md").write_text(task)
    seed = subprocess.run([sys.executable, str(HERE / "evald.py"), str(run), "0", "--seed-only"], check=True, capture_output=True, text=True).stdout
    return run, json.loads(seed.strip().splitlines()[-1])


ROLE_TEXT = {
    "numerical": "**Your role: Numerical** (submit with `--role numerical`; your file must define `forecast(view)`).",
    "retrieval": "**Your role: Retrieval** (submit with `--role retrieval`; your file must define `retrieve(view)`).",
    "decision": "**Your role: Decision** (submit with `--role decision`; your file must define `adjust(view)`).",
    "phased": "**Your role is given in the episode text below** (numerical or decision, by the current phase; the Retrieval module is frozen in this stage).",
}
SHARED_NOTES = "- `$RUN_DIR/shared/notes/` and `shared/skills/` are shared with the other agents of this run."
INDEP_NOTES = "- You are the only agent of this run; `$RUN_DIR/shared/notes/` and `shared/skills/` are yours."


# ---------------------------------------------------------------- execution (real runs only)
def shim_env(stage):
    bindir = OUT / "bin"; bindir.mkdir(exist_ok=True); w = bindir / "codex"
    target = A.test_fake_codex.resolve() if A.test_fake_codex else HERE / "sol56_codex_shim.py"
    w.write_text(f"#!/bin/bash\nexec {sys.executable} {target} \"$@\"\n"); w.chmod(0o755)
    return dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", SOL56_ROOT=str(OUT), SOL56_STAGE=f"{DS}:{stage}", SOL56_REAL_CODEX=A.real_codex,
                SUBMIT_PY=str(HERE / "submit.py"))


def run_rounds(run, agents, rounds, budget, stage, phases=None):
    env = shim_env(stage); denv = dict(env, SYNC_ROUNDS="1")
    d = subprocess.Popen([sys.executable, str(HERE / "evald.py"), str(run), str(budget)], env=denv, stdout=open(run / "evald.log", "a"), stderr=subprocess.STDOUT)
    try:
        if phases is None:
            subprocess.run([sys.executable, str(HERE / "rounds.py"), str(run), str(HERE / "run_agent.py"), str(rounds), str(PER), *agents],
                           env=dict(env, TASK_FILE=str(run / "TASK.md")), check=True, stdout=open(run / "rounds.log", "a"), stderr=subprocess.STDOUT)
        else:
            off = 0
            for phase, tag, ags in phases:
                json.dump(dict(phase=phase), open(run / "shared/phase.json", "w"))
                tf = run / f"TASK_{phase}_{tag}.md"; tf.write_text((run / "TASK.md").read_text() + f"\n\n## Current phase\nPhase `{phase}` round-set `{tag}`: submit with `--role {phase}`.\n")
                subprocess.run([sys.executable, str(HERE / "rounds.py"), str(run), str(HERE / "run_agent.py"), str(rounds), str(PER), *ags],
                               env=dict(env, TASK_FILE=str(tf), ROUND_OFFSET=str(off)), check=True, stdout=open(run / "rounds.log", "a"), stderr=subprocess.STDOUT)
                off += rounds
    finally:
        (run / "STOP").write_text(str(time.time())); d.wait(timeout=3600)


def final_state(run): return json.load(open(run / "private/state.json"))


def modules(run): return {r: run / f"shared/best_{MOD[r]}.py" for r in ROLES}


def parallel(jobs):
    ts = [threading.Thread(target=j) for j in jobs]; [t.start() for t in ts]; [t.join() for t in ts]


def route(r1_runs, seeds, run_dir):
    """v3.3.2 L5 routing generalised: per (freq,H) cell, the R1 champion adjust with the best visible-fold mean gain."""
    ed = json.load(open(PACK / "private/eval_data.json")); F = ed["folds"]; vis = F[0] + F[1]; V = json.load(open(PACK / "shared/views_train.json"))
    cell = {t: f"{V[t]['freq']}|H{V[t]['H']}" for t in V}; gains = {}
    for name, run in r1_runs.items():
        o = run_pipeline(dict(seeds, decision=run / "shared/best_adjust.py"), PACK / "shared/views_train.json", run_dir / f"_route_{name}.json")
        gains[name] = {t: ed["base_jt"][t] - jt(o["forecasts"][t], ed["truth"][t]) for t in vis}
    overall = max(gains, key=lambda n: (statistics.mean(gains[n].values()), -list(gains).index(n)))
    table = {}
    for c in sorted(set(cell[t] for t in vis)):
        ts = [t for t in vis if cell[t] == c]
        table[c] = max(gains, key=lambda n: (statistics.mean(gains[n][t] for t in ts), -list(gains).index(n)))
    src = {n: (r1_runs[n] / "shared/best_adjust.py").read_text() for n in r1_runs}
    code = ('"""Routed correction function (L5 routing): per (freq,H) cell, the R1 champion with the best visible-fold mean gain."""\n'
            f"_SOURCES = {json.dumps(src)}\nROUTE = {json.dumps(table)}\nDEFAULT = {json.dumps(overall)}\n_M = {{}}\n"
            "def _load(n):\n    if n not in _M:\n        ns = {}; exec(_SOURCES[n], ns); _M[n] = ns['adjust']\n    return _M[n]\n"
            "def adjust(view):\n    return _load(ROUTE.get(f\"{view['freq']}|H{view['H']}\", DEFAULT))(view)\n")
    p = run_dir / "routed_adjust.py"; p.write_text(code)
    json.dump(dict(route=table, default=overall), open(run_dir / "routing.json", "w"), indent=1)
    return p


# ---------------------------------------------------------------- plan
PLAN = {
    "L4": dict(runs={"L4_A": ["A1", "A2", "A3"], "L4_B1": ["B1"], "L4_B2": ["B2"], "L4_B3": ["B3"]}, rounds=R4),
    "L5_R1": dict(runs={"L5_S": ["S1"], "L5_C": ["C1", "C2", "C3"], "L5_I1": ["I1"], "L5_I2": ["I2"], "L5_I3": ["I3"]}, rounds=R5),
    "L5_R2": dict(runs={"L5_R2": ["R1", "R2", "R3"]}, rounds=R5),
    "L7": dict(runs={"L7": ["n1a", "n2a", "d1a", "d2a", "n1b", "n2b", "d1b", "d2b"]}, rounds=R7),
}
L4_ROLES = {"A1": "numerical", "A2": "retrieval", "A3": "decision", "B1": "numerical", "B2": "retrieval", "B3": "decision"}
episodes = {k: sum(len(a) for a in v["runs"].values()) * v["rounds"] for k, v in PLAN.items()}
budget = dict(L4=(episodes["L4"], episodes["L4"] * PER), L5=(episodes["L5_R1"] + episodes["L5_R2"], (episodes["L5_R1"] + episodes["L5_R2"]) * PER),
              L7=(episodes["L7"], episodes["L7"] * PER))
assert budget == dict(L4=(24, 120), L5=(40, 200), L7=(16, 80)), budget
seeds0 = {r: HERE / f"seeds/seed_{MOD[r]}.py" for r in ROLES}
caps = dict(global_=43_200_000, unit_single=40_000, unit_episode=300_000, max_concurrent=8, poll_seconds=2)
(OUT / "ledger").mkdir(exist_ok=True)
if not (OUT / "ledger/caps.json").exists():
    json.dump({"global": caps["global_"], **{k: v for k, v in caps.items() if k != "global_"},
               "stage_reporting_envelopes": {"L4": 26_400_000, "L5": 12_000_000, "L7": 4_800_000},
               "note": "Dr-CiK v3.3.2 L4+L5+L7 envelopes per dataset; one global soft threshold"}, open(OUT / "ledger/caps.json", "w"), indent=1)


def dry_run():
    guard = OUT / "bin_guard"; guard.mkdir(exist_ok=True); g = guard / "codex"
    g.write_text("#!/bin/bash\necho 'dry-run: codex must not be called' >&2\nexit 99\n"); g.chmod(0o755)
    os.environ["PATH"] = f"{guard}:{os.environ['PATH']}"
    report = dict(dataset=DS, mode="dry-run", model="gpt-5.6-sol", reasoning_effort="high", per_episode_max_submissions=PER,
                  budget={k: dict(episodes=v[0], max_submissions=v[1]) for k, v in budget.items()}, stages={})
    for st, spec in PLAN.items():
        report["stages"][st] = {}
        for name, agents in spec["runs"].items():
            if st == "L4":
                kw = dict(roles={a: L4_ROLES[a] for a in agents}, role_text=ROLE_TEXT[L4_ROLES[agents[0]]] if len(agents) == 1 else
                          "**Your role** is fixed per agent: A1 numerical, A2 retrieval, A3 decision (your agent id is given below).",
                          notes_text=SHARED_NOTES if len(agents) > 1 else INDEP_NOTES)
            elif st.startswith("L5"):
                kw = dict(roles={a: "decision" for a in agents}, role_text=ROLE_TEXT["decision"], notes_text=SHARED_NOTES if len(agents) > 1 else INDEP_NOTES,
                          fold_key="folds_secondary" if st == "L5_R2" else "folds")
            else:
                kw = dict(phased=True, role_text=ROLE_TEXT["phased"], notes_text=SHARED_NOTES)
            run, seed = prep_run(f"dry/{name}", seeds0, **kw)
            report["stages"][st][name] = dict(agents=agents, rounds=spec["rounds"], episodes=len(agents) * spec["rounds"],
                                              max_submissions=len(agents) * spec["rounds"] * PER, budget_per_agent=spec["rounds"] * PER,
                                              fold_key=kw.get("fold_key", "folds"), roles=kw.get("roles", "phase-driven"),
                                              seed_visible=round(seed["seed_visible"], 6), seed_hidden=round(seed["seed_hidden"], 6),
                                              seed_note="placeholder seed (real seed = previous stage's champion)" if st != "L4" else "L4 seed")
    report["l7_phases"] = [["numerical", "a", ["n1a", "n2a"]], ["decision", "a", ["d1a", "d2a"]], ["numerical", "b", ["n1b", "n2b"]], ["decision", "b", ["d1b", "d2b"]]]
    pr = json.load(open(PACK / "pack_receipt.json"))
    report["pack"] = dict(receipt_sha256=sha(PACK / "pack_receipt.json"), task_count=pr["task_count"],
                          folds_primary=[dict(n=f["n"], role=f["role"], groups=f["group_ids"], task_ids=f["task_ids"]) for f in pr["folds_primary"]],
                          folds_secondary=[dict(n=f["n"], role=f["role"], groups=f["group_ids"], task_ids=f["task_ids"]) for f in pr["folds_secondary"]],
                          anchor_coverage=pr["anchor_coverage"], input_sha256=pr["input_sha256"], output_sha256=pr["output_sha256"],
                          uses_test_ids_or_labels=pr["uses_test_ids_or_labels"], uses_external_dev=pr["uses_external_dev"])
    report["code_sha256"] = {p.name: sha(p) for p in sorted(HERE.glob("*.py"))} | {f"seeds/{p.name}": sha(p) for p in sorted((HERE / "seeds").glob("*.py"))} | {"prompts/TASK_TEMPLATE.md": sha(HERE / "prompts/TASK_TEMPLATE.md")}
    led = OUT / "ledger/ledger.jsonl"
    report["llm_calls"] = sum(1 for _ in open(led)) if led.exists() else 0
    assert report["llm_calls"] == 0, "dry-run must not call any model"
    json.dump(report, open(OUT / "dry_run_report.json", "w"), indent=1)
    s = {k: {n: (v["episodes"], v["max_submissions"], v["seed_visible"], v["seed_hidden"]) for n, v in d.items()} for k, d in report["stages"].items()}
    print(json.dumps(dict(dataset=DS, budget=report["budget"], runs=s, folds=[f["n"] for f in report["pack"]["folds_primary"]],
                          folds_secondary=[f["n"] for f in report["pack"]["folds_secondary"]], llm_calls=report["llm_calls"],
                          report=str(OUT / "dry_run_report.json")), indent=1))


def real_run():
    log = OUT / "stages.log"
    def say(m): open(log, "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + m + "\n")
    # L4
    runs4 = {}
    for name, agents in PLAN["L4"]["runs"].items():
        runs4[name], _ = prep_run(name, seeds0, roles={a: L4_ROLES[a] for a in agents},
                                  role_text=ROLE_TEXT[L4_ROLES[agents[0]]] if len(agents) == 1 else "**Your role** is fixed per agent: A1 numerical, A2 retrieval, A3 decision.",
                                  notes_text=SHARED_NOTES if len(agents) > 1 else INDEP_NOTES)
    say("L4 start"); parallel([lambda n=n, a=a: run_rounds(runs4[n], a, R4, R4 * PER, "L4") for n, a in PLAN["L4"]["runs"].items()])
    champ4 = max(runs4, key=lambda n: (final_state(runs4[n])["best_visible"], n == "L4_A")); s4 = modules(runs4[champ4]); say(f"L4 champion {champ4}")
    # L5 R1
    runs5 = {}
    for name, agents in PLAN["L5_R1"]["runs"].items():
        runs5[name], _ = prep_run(name, s4, roles={a: "decision" for a in agents}, role_text=ROLE_TEXT["decision"], notes_text=SHARED_NOTES if len(agents) > 1 else INDEP_NOTES)
    say("L5 R1 start"); parallel([lambda n=n, a=a: run_rounds(runs5[n], a, R5, R5 * PER, "L5") for n, a in PLAN["L5_R1"]["runs"].items()])
    routed = route(runs5, s4, OUT); say("L5 routing done")
    r2, _ = prep_run("L5_R2", dict(s4, decision=routed), fold_key="folds_secondary", roles={a: "decision" for a in PLAN["L5_R2"]["runs"]["L5_R2"]},
                     role_text=ROLE_TEXT["decision"], notes_text=SHARED_NOTES)
    say("L5 R2 start"); run_rounds(r2, PLAN["L5_R2"]["runs"]["L5_R2"], R5, R5 * PER, "L5"); s5 = modules(r2)
    # L7
    l7, _ = prep_run("L7", s5, phased=True, role_text=ROLE_TEXT["phased"], notes_text=SHARED_NOTES)
    phases = [("numerical", "a", ["n1a:robust blending of the frozen forecasts by frequency and history", "n2a:level/trend handling and shrinkage"]),
              ("decision", "a", ["d1a:when to trust event-based corrections", "d2a:magnitude calibration relative to the base forecast"]),
              ("numerical", "b", ["n1b:robust blending of the frozen forecasts by frequency and history", "n2b:level/trend handling and shrinkage"]),
              ("decision", "b", ["d1b:when to trust event-based corrections", "d2b:magnitude calibration relative to the base forecast"])]
    say("L7 start"); run_rounds(l7, None, R7, R7 * PER, "L7", phases=phases)
    fin = OUT / "final"; fin.mkdir(exist_ok=True)
    for r, p in modules(l7).items(): shutil.copy(p, fin / p.name)
    json.dump({p.name: sha(p) for p in fin.glob("*.py")}, open(fin / "final_sha256.json", "w"), indent=1); say("done; final modules frozen")


if A.dry_run: dry_run()
else: real_run()
