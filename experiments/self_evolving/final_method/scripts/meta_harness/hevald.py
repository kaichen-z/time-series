"""Meta-Harness evaluator daemon. Agents submit a correction-function source file; the daemon runs it (subprocess,
timeout) on all Train views and scores it. Returned to agents: fitness + per-task gains on VISIBLE folds 0,1 and
pass/fail from HIDDEN fold 2. Accepted as new best iff visible fitness improves and hidden robust gain does not drop.
usage: hevald.py <run_dir> [budget_per_agent]"""
import json, statistics, subprocess, sys, time, traceback, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import drcik_point_metrics
RUN = Path(sys.argv[1]).resolve(); BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ED = json.load(open(RUN / "private/eval_data.json")); TRUTH, BJT = ED["truth"], ED["base_jt"]; FOLDS = ED["folds"]
VIS, HID = FOLDS[0] + FOLDS[1], FOLDS[2]; PEN, STD_W = 0.5, 0.25


def jt(f, t): x = drcik_point_metrics(t, f, cap=5.0); return x["smae"] + x["srmse"]
def robust(g): return statistics.mean(g) + PEN * statistics.mean(min(0.0, x) for x in g)


def score(src):
    o = RUN / "private/_out.json"
    subprocess.run([sys.executable, str(Path(__file__).parent / "harness_runner.py"), str(src), str(RUN / "shared/views_train.json"), str(o)],
                   check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    r = json.load(open(o)); per = {t: BJT[t] - jt(f, TRUTH[t]) for t, f in r["forecasts"].items()}
    fl = [robust([per[t] for t in FOLDS[0]]), robust([per[t] for t in FOLDS[1]])]
    vis = {t: round(per[t], 4) for t in VIS}
    return dict(visible_fitness=statistics.mean(fl) - STD_W * statistics.pstdev(fl), visible_folds=fl, visible_per_task=vis,
                visible_better=sum(g > 1e-9 for g in vis.values()), visible_worse=sum(g < -1e-9 for g in vis.values()),
                runtime_errors={t: e for t, e in r["errors"].items() if t in VIS}, n_runtime_errors=len(r["errors"])), robust([per[t] for t in HID])


st_f = RUN / "private/state.json"
if st_f.exists(): st = json.load(open(st_f))
else:
    seed = RUN / "shared/harness/0000_seed.py"; pub, hid = score(seed)
    st = dict(best=str(seed), best_visible=pub["visible_fitness"], best_hidden=hid, used={}, n=0, history=[])
    shutil.copy(seed, RUN / "shared/best_harness.py")
    json.dump(dict(pub, attempt=0, agent="seed", accepted=True, harness="harness/0000_seed.py"), open(RUN / "shared/attempts/0000_seed.json", "w"), indent=1)
    json.dump(st, open(st_f, "w"))
print("daemon up; seed visible", round(st["best_visible"], 4), flush=True)
while True:
    items = sorted((RUN / "queue").glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not items:
        if (RUN / "STOP").exists(): break
        time.sleep(3); continue
    q = items[0]; sub = json.load(open(q)); q.unlink(); agent, rid = sub.get("agent", "?"), sub["request_id"]; out = dict(request_id=rid, agent=agent)
    try:
        if st["used"].get(agent, 0) >= BUDGET: raise RuntimeError(f"budget exhausted ({BUDGET})")
        st["used"][agent] = st["used"].get(agent, 0) + 1; st["n"] += 1
        hp = RUN / f"shared/harness/{st['n']:04d}_{agent}.py"; hp.write_text(sub["code"])
        pub, hid = score(hp)
        acc = pub["visible_fitness"] > st["best_visible"] + 1e-4 and hid >= st["best_hidden"] - 1e-9
        out.update(pub, accepted=acc, hidden_check="pass" if hid >= st["best_hidden"] - 1e-9 else "fail", budget_left=BUDGET - st["used"][agent], attempt=st["n"], harness=f"harness/{hp.name}")
        st["history"].append(dict(n=st["n"], agent=agent, visible=pub["visible_fitness"], hidden=hid, accepted=acc, t=time.time()))
        if acc: st.update(best=str(hp), best_visible=pub["visible_fitness"], best_hidden=hid); shutil.copy(hp, RUN / "shared/best_harness.py")
        att = dict(out); att.pop("visible_per_task", None); att["note"] = sub.get("note", ""); att["visible_per_task"] = pub["visible_per_task"]
        json.dump(att, open(RUN / f"shared/attempts/{st['n']:04d}_{agent}.json", "w"), indent=1)
    except subprocess.CalledProcessError as e:
        out.update(error="harness crashed: " + (e.stderr or b"").decode()[-600:])
    except Exception as e:
        out.update(error=repr(e)[:400]); traceback.print_exc()
    json.dump(out, open(RUN / f"results/{rid}.json", "w"), indent=1); json.dump(st, open(st_f, "w"))
    print(f"[{st['n']}] {agent} -> {'ERR ' + out['error'][:80] if 'error' in out else ('ACCEPT' if out['accepted'] else 'reject')} vis {out.get('visible_fitness', float('nan')):+.4f} hidden {out.get('hidden_check')}", flush=True)
print("daemon stopped", flush=True)
