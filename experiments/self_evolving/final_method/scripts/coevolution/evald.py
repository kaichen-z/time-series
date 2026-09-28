"""Step 2 evaluator daemon (evaluator separation, CORAL-style) -- 2026-09-28.

Agents never run the evaluator themselves.  They drop a config into <run>/queue/; this daemon (outside
the agents' sandbox) scores it with e2e.evaluate on all 3 Train folds and writes back only:
  * fitness and per-task gains on the two VISIBLE folds (0, 1),
  * a pass/fail flag from the HIDDEN fold (2): a submission is accepted as the new best only if the
    visible fitness improves AND the hidden-fold robust gain does not drop below the current best's.
Hidden-fold per-task numbers are never written where agents can read them (kept in <run>/private/).
Budget per agent is enforced here.  Usage: evald.py <run_dir> [budget_per_agent]
"""
import copy, json, statistics, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E

RUN = Path(sys.argv[1]).resolve(); BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MAX_INSTR = 2
for sub in ("queue", "results", "shared/attempts", "private"): (RUN / sub).mkdir(parents=True, exist_ok=True)
VIS, HID = [E.FOLDS[0], E.FOLDS[1]], E.FOLDS[2]
VIS_IDS = {d["tid"] for f in VIS for d in f}


def robust(g): return statistics.mean(g) + E.PEN * statistics.mean(min(0.0, x) for x in g)


def score(cfg):
    r = E.evaluate(cfg, VIS)
    hid = robust([r["per_task"][d["tid"]] for d in HID])
    vis = {t: round(g, 4) for t, g in r["per_task"].items() if t in VIS_IDS}
    return dict(visible_fitness=r["fitness"], visible_folds=r["folds"], visible_per_task=vis,
                visible_better=sum(g > 1e-9 for g in vis.values()), visible_worse=sum(g < -1e-9 for g in vis.values())), hid


def load_state():
    f = RUN / "private/state.json"
    if f.exists(): return json.load(open(f))
    cfg = E.main_config(); pub, hid = score(cfg)
    st = dict(best=cfg, best_visible=pub["visible_fitness"], best_hidden=hid, used={}, instr_used={}, n=0, history=[])
    json.dump(cfg, open(RUN / "shared/best_config.json", "w"), indent=1)
    json.dump(dict(pub, attempt=0, agent="seed", note="current main method", accepted=True), open(RUN / "shared/attempts/0000_seed.json", "w"), indent=1)
    json.dump(st, open(f, "w")); return st


def main():
    st = load_state(); print("daemon up; seed visible", round(st["best_visible"], 4), flush=True)
    while True:
        items = sorted((RUN / "queue").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not items:
            if (RUN / "STOP").exists(): break
            time.sleep(3); continue
        q = items[0]; sub = json.load(open(q)); q.unlink()
        agent, rid = sub.get("agent", "?"), sub.get("request_id", q.stem); cfg = sub["config"]
        out = dict(request_id=rid, agent=agent)
        try:
            if st["used"].get(agent, 0) >= BUDGET: raise RuntimeError(f"budget exhausted ({BUDGET} evaluations)")
            if cfg["retrieval"]["instr"] != st["best"]["retrieval"]["instr"]:
                if st["instr_used"].get(agent, 0) >= MAX_INSTR: raise RuntimeError(f"instruction-change budget exhausted ({MAX_INSTR})")
                st["instr_used"][agent] = st["instr_used"].get(agent, 0) + 1
            st["used"][agent] = st["used"].get(agent, 0) + 1; st["n"] += 1
            pub, hid = score(cfg)
            acc = pub["visible_fitness"] > st["best_visible"] + 1e-4 and hid >= st["best_hidden"] - 1e-9
            out.update(pub, accepted=acc, hidden_check="pass" if hid >= st["best_hidden"] - 1e-9 else "fail",
                       budget_left=BUDGET - st["used"][agent], attempt=st["n"])
            st["history"].append(dict(n=st["n"], agent=agent, visible=pub["visible_fitness"], hidden=hid, accepted=acc, t=time.time()))
            if acc:
                st.update(best=cfg, best_visible=pub["visible_fitness"], best_hidden=hid)
                json.dump(cfg, open(RUN / "shared/best_config.json", "w"), indent=1)
            att = dict(out); att["config"] = cfg; att["note"] = sub.get("note", "")
            json.dump(att, open(RUN / f"shared/attempts/{st['n']:04d}_{agent}.json", "w"), indent=1)
        except Exception as e:
            out.update(error=repr(e)[:400]); traceback.print_exc()
        json.dump(out, open(RUN / f"results/{rid}.json", "w"), indent=1)
        json.dump(st, open(RUN / "private/state.json", "w"))
        print(f"[{st['n']}] {agent} {rid} -> {'ERR ' + out['error'][:80] if 'error' in out else ('ACCEPT' if out['accepted'] else 'reject')} "
              f"vis {out.get('visible_fitness', float('nan')):+.4f} hidden {out.get('hidden_check')}", flush=True)
    print("daemon stopped", flush=True)


if __name__ == "__main__":
    main()
