"""Evaluator daemon for one run directory of the Dr-CiK-budget-aligned evolution (generalised from the v3.3.2 daemons
coevolution/evald.py, meta_harness/hevald.py and coevo_x/hevald_x.py; same scoring and acceptance).

Agents submit ONE module (role numerical -> forecast.py, retrieval -> retrieve.py, decision -> adjust.py); the other
two modules are the run's current shared best. Which role an agent may submit is fixed in RUN/stage.json
("roles": {agent_id: role}) or, for phased runs (L7), by RUN/shared/phase.json.

Score (Dr-CiK v3.3.2, per stage, see SCALE below): per task gain = Toto joint error - final joint error (L4/L5) or the
same divided by the mean Toto joint error (L7), joint error = sMAE + sRMSE each capped at 5; fold robust gain = mean + 0.5 * mean(negative part); visible fitness = mean over the two
visible folds - 0.25 * std; hidden = robust gain on the hidden fold. Synchronous rounds (SYNC_ROUNDS=1, sync_round.py):
eligible iff visible > frozen round base + eps AND hidden >= frozen round base hidden - tol (stage values); winner at round close.
Hidden per-task numbers never leave private/. usage: evald.py <run_dir> <budget_per_agent>"""
import json, math, os, shutil, statistics, subprocess, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import sync_round as SR  # noqa: E402
RUN = Path(sys.argv[1]).resolve(); BUDGET = int(sys.argv[2])
STAGE = json.load(open(RUN / "stage.json")); ED = json.load(open(RUN / "private/eval_data.json"))
FOLDS = ED[STAGE.get("fold_key", "folds")]; VIS, HID = FOLDS[0] + FOLDS[1], FOLDS[2]
TRUTH, BJT = ED["truth"], ED["base_jt"]
# v3.3.2 per-stage formulas: L4 coevolution/evald.py and L5 meta_harness/hevald.py use gain = BJT - jt (no scale), visible
# eps 1e-4, hidden tolerance 1e-9; L7 coevo_x/hevald_x.py divides by the dataset's mean Toto joint error, eps 1e-5, tol 1e-6.
SCALE = (sum(BJT[t] for t in FOLDS[0] + FOLDS[1] + FOLDS[2]) / sum(map(len, FOLDS))) if STAGE["scaled_gain"] else 1.0
PEN, STD_W, EPS, HTOL = 0.5, 0.25, STAGE["eps"], STAGE["hidden_tol"]
MOD = {"numerical": "forecast", "retrieval": "retrieve", "decision": "adjust"}
SYNC = os.environ.get("SYNC_ROUNDS") == "1"


def jt(f, y):
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


def robust(g): return statistics.mean(g) + PEN * statistics.mean(min(0.0, x) for x in g)


def best(role): return RUN / f"shared/best_{MOD[role]}.py"


def score(mods):
    o = RUN / "private/_out.json"
    subprocess.run([sys.executable, str(HERE / "pipeline_runner.py"), str(mods["numerical"]), str(mods["retrieval"]), str(mods["decision"]),
                    str(RUN / "shared/views_train.json"), str(o)], check=True, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    r = json.load(open(o)); per = {t: (BJT[t] - jt(f, TRUTH[t])) / SCALE for t, f in r["forecasts"].items()}
    fl = [robust([per[t] for t in FOLDS[0]]), robust([per[t] for t in FOLDS[1]])]
    vis = {t: round(per[t], 5) for t in VIS}
    pub = dict(visible_fitness=statistics.mean(fl) - STD_W * statistics.pstdev(fl), visible_folds=fl, visible_per_task=vis,
               visible_better=sum(g > 1e-9 for g in vis.values()), visible_worse=sum(g < -1e-9 for g in vis.values()),
               runtime_errors={t: e for t, e in r["errors"].items() if t in vis}, n_runtime_errors=len(r["errors"]))
    return pub, robust([per[t] for t in HID])


def main():
    for s in ("queue", "results", "private", "shared/attempts"): (RUN / s).mkdir(parents=True, exist_ok=True)
    st_f = RUN / "private/state.json"
    if st_f.exists(): st = json.load(open(st_f))
    else:
        pub, hid = score({r: best(r) for r in MOD})
        st = dict(best_visible=pub["visible_fitness"], best_hidden=hid, used={}, n=0, history=[])
        json.dump(dict(pub, attempt=0, agent="seed", accepted=True), open(RUN / "shared/attempts/0000_seed.json", "w"), indent=1)
        json.dump(st, open(st_f, "w"))
    if "--seed-only" in sys.argv: print(json.dumps(dict(seed_visible=st["best_visible"], seed_hidden=st["best_hidden"]))); return
    print("daemon up; seed visible", round(st["best_visible"], 5), "hidden", round(st["best_hidden"], 5), flush=True)
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
            mods = {r: best(r) for r in MOD}; mods[role] = cp
            pub, hid = score(mods)
            ref_h = st["base_hidden"] if SYNC else st["best_hidden"]; ref_v = st["base_visible"] if SYNC else st["best_visible"]
            hid_ok = hid >= ref_h - HTOL; elig = pub["visible_fitness"] > ref_v + EPS and hid_ok
            if SYNC:
                idx = SR.next_index(st, agent); SR.record(st, agent, idx, pub["visible_fitness"], hid, elig, f"shared/{role}/{cp.name}")
                acc = False; out.update(eligible=elig, round=st["round"])
            else:
                acc = elig
                if acc: shutil.copy(cp, best(role)); st.update(best_visible=pub["visible_fitness"], best_hidden=hid)
            out.update(pub, accepted=acc, hidden_check="pass" if hid_ok else "fail", budget_left=BUDGET - st["used"][agent], attempt=st["n"], file=f"{role}/{cp.name}")
            st["history"].append(dict(n=st["n"], agent=agent, role=role, visible=pub["visible_fitness"], hidden=hid, eligible=elig, t=time.time()))
            json.dump(dict(out, note=sub.get("note", "")), open(RUN / f"shared/attempts/{st['n']:04d}_{agent}.json", "w"), indent=1)
        except subprocess.CalledProcessError as e:
            out.update(error="crashed: " + (e.stderr or b"").decode()[-600:])
        except Exception as e:
            out.update(error=repr(e)[:400]); traceback.print_exc()
        json.dump(out, open(RUN / f"results/{rid}.json", "w"), indent=1); json.dump(st, open(st_f, "w"))
        open(RUN / "private/done.log", "a").write(q.stem + "\n"); (RUN / "private/inflight.json").unlink()
        print(f"[{st['n']}] {agent}/{role} -> {'ERR ' + out['error'][:80] if 'error' in out else ('eligible' if out.get('eligible', out.get('accepted')) else 'reject')} "
              f"vis {out.get('visible_fitness', float('nan')):+.5f} hidden {out.get('hidden_check')}", flush=True)
    print("daemon stopped", flush=True)


if __name__ == "__main__":
    main()
