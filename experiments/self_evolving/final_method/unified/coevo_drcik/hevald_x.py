"""Evaluator daemon for cross-dataset Numerical<->Decision co-evolution. Submissions carry a role; only the role of the
current phase (shared/phase.json) is accepted. The submitted code replaces that role's module; the other role is the
current best. Score per dataset: per-task relative gain = (Toto joint error - final joint error) / mean Toto joint error;
dataset fitness = mean over visible folds 0,1 of (mean + 0.5*mean negative) - 0.25*std; overall = mean over datasets.
Accepted iff overall visible fitness improves AND every dataset's hidden-fold robust gain does not drop.
usage: hevald_x.py <run_dir> [budget_per_agent]"""
import json, statistics, subprocess, sys, time, traceback, shutil
from pathlib import Path
sys.path.insert(0, "work/timesx"); from metrics import M
HERE = Path(__file__).resolve().parent; RUN = Path(sys.argv[1]).resolve(); BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ED = json.load(open(RUN / "private/eval_data.json")); DS = list(ED); PEN, STD_W = 0.5, 0.25
def jt(f, y): m = M(y, f, cap=5.0); return m["smae"] + m["srmse"]
def robust(g): return statistics.mean(g) + PEN * statistics.mean(min(0.0, x) for x in g)
def score(fpath, apath):
    res, hid, vis_tasks, errs = {}, {}, {}, {}
    for ds in DS:
        E = ED[ds]; o = RUN / f"private/_out_{ds}.json"
        subprocess.run([sys.executable, str(HERE / "runner_x.py"), str(fpath), str(apath), str(RUN / f"shared/views_train_{ds}.json"), str(o)],
                       check=True, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        r = json.load(open(o)); per = {t: (E["base_jt"][t] - jt(f, E["truth"][t])) / E["scale"] for t, f in r["forecasts"].items()}
        fl = [robust([per[t] for t in E["folds"][0]]), robust([per[t] for t in E["folds"][1]])]
        res[ds] = dict(fitness=statistics.mean(fl) - STD_W * statistics.pstdev(fl), folds=fl,
                       better=sum(per[t] > 1e-9 for t in E["folds"][0] + E["folds"][1]), worse=sum(per[t] < -1e-9 for t in E["folds"][0] + E["folds"][1]))
        hid[ds] = robust([per[t] for t in E["folds"][2]])
        vis_tasks[ds] = {t: round(per[t], 5) for t in E["folds"][0] + E["folds"][1]}
        errs[ds] = {t: e for t, e in r["errors"].items() if t in vis_tasks[ds]}
    return dict(visible_fitness=statistics.mean(res[ds]["fitness"] for ds in DS), per_dataset=res, visible_per_task=vis_tasks, runtime_errors=errs), hid
st_f = RUN / "private/state.json"
if st_f.exists(): st = json.load(open(st_f))
else:
    pub, hid = score(RUN / "shared/best_forecast.py", RUN / "shared/best_adjust.py")
    st = dict(best_visible=pub["visible_fitness"], best_hidden=hid, used={}, n=0, history=[])
    json.dump(dict(pub, attempt=0, agent="seed", accepted=True), open(RUN / "shared/attempts/0000_seed.json", "w"), indent=1); json.dump(st, open(st_f, "w"))
print("daemon up; seed visible", round(st["best_visible"], 5), {k: round(v, 5) for k, v in st["best_hidden"].items()}, flush=True)
while True:
    items = sorted((RUN / "queue").glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not items:
        if (RUN / "STOP").exists(): break
        time.sleep(3); continue
    q = items[0]; sub = json.load(open(q)); q.unlink(); agent, rid, role = sub.get("agent", "?"), sub["request_id"], sub.get("role"); out = dict(request_id=rid, agent=agent, role=role)
    try:
        phase = json.load(open(RUN / "shared/phase.json"))["phase"]
        if role != phase: raise RuntimeError(f"current phase is '{phase}', you submitted role '{role}'")
        if st["used"].get(agent, 0) >= BUDGET: raise RuntimeError(f"budget exhausted ({BUDGET})")
        st["used"][agent] = st["used"].get(agent, 0) + 1; st["n"] += 1
        cp = RUN / f"shared/{role}/{st['n']:04d}_{agent}.py"; cp.write_text(sub["code"])
        fpath = cp if role == "numerical" else RUN / "shared/best_forecast.py"; apath = cp if role == "decision" else RUN / "shared/best_adjust.py"
        pub, hid = score(fpath, apath)
        hid_ok = all(hid[ds] >= st["best_hidden"][ds] - 1e-6 for ds in DS)
        acc = pub["visible_fitness"] > st["best_visible"] + 1e-5 and hid_ok
        out.update(pub, accepted=acc, hidden_check={ds: "pass" if hid[ds] >= st["best_hidden"][ds] - 1e-6 else "fail" for ds in DS}, budget_left=BUDGET - st["used"][agent], attempt=st["n"], file=f"{role}/{cp.name}")
        st["history"].append(dict(n=st["n"], agent=agent, role=role, visible=pub["visible_fitness"], hidden=hid, accepted=acc, t=time.time()))
        if acc:
            st.update(best_visible=pub["visible_fitness"], best_hidden=hid)
            shutil.copy(cp, RUN / ("shared/best_forecast.py" if role == "numerical" else "shared/best_adjust.py"))
        att = dict(out); att["note"] = sub.get("note", ""); json.dump(att, open(RUN / f"shared/attempts/{st['n']:04d}_{agent}.json", "w"), indent=1)
    except subprocess.CalledProcessError as e:
        out.update(error="crashed: " + (e.stderr or b"").decode()[-600:])
    except Exception as e:
        out.update(error=repr(e)[:400]); traceback.print_exc()
    json.dump(out, open(RUN / f"results/{rid}.json", "w"), indent=1); json.dump(st, open(st_f, "w"))
    print(f"[{st['n']}] {agent}/{role} -> {'ERR ' + out['error'][:80] if 'error' in out else ('ACCEPT' if out['accepted'] else 'reject')} vis {out.get('visible_fitness', float('nan')):+.5f} hidden {out.get('hidden_check')}", flush=True)
print("daemon stopped", flush=True)
