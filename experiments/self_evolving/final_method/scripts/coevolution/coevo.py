"""Step 1: cooperative co-evolution of all three roles under ONE end-to-end fitness (2026-09-28).

Start: the current main method.  Each round, every role proposes candidates that change only that role
(the other roles stay at the current best); a candidate is accepted only if the end-to-end Train fitness
of the whole pipeline (e2e.evaluate) improves, otherwise the round for that role is rolled back.
  Numerical : part-1 program mutation (43-method dictionary) or magnitude-calibrator mutation
  Retrieval : per-correction validator perturbation, and (every INSTR_EVERY rounds) GPT rewrites of the
              extraction instructions, with feedback from evidence errors AND from tasks whose accepted
              repair made the final forecast worse
  Decision  : fill method, repair margin, strength / trust gate
Only Train is used.  Output: coevo/best.json, coevo/log.jsonl.
"""
import copy, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import e2e as E
import tl2
import nrd4 as N4

OUT = E.S / "coevo"; (OUT / "instr").mkdir(exist_ok=True)
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
INSTR_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 2
import os
HOLD = int(os.environ["HOLD"]) if os.environ.get("HOLD") else None   # nested CV: evolve on the other 2 folds
USE = [f for i, f in enumerate(E.FOLDS) if i != HOLD]; HELD = E.FOLDS[HOLD] if HOLD is not None else []
ACC = os.environ.get("ACC", "mean")
if HOLD is not None: OUT = OUT / f"hold{HOLD}{ACC}"; (OUT / "instr").mkdir(parents=True, exist_ok=True)
rng = random.Random(5 + (HOLD or 0))
FILLS = ["phase_median", "linear", "snaive", "truncate"]; MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, None]
TRAIN = sorted(E.TRAIN_IDS)


def mut_numerical(cfg):
    c = copy.deepcopy(cfg)
    if rng.random() < 0.7: c["numerical"]["program"] = E.P1.mutate(c["numerical"]["program"], rng)[0]; what = "program"
    else: c["numerical"]["calib"] = N4.mutate_calib(c["numerical"]["calib"], rng)[0]; what = "calib"
    return c, what


def mut_validator(cfg):
    c = copy.deepcopy(cfg); w = c["retrieval"]["validator"]
    for k in rng.sample(N4.FEATS, rng.randint(1, 3)): w[k] = max(-4.0, min(4.0, w.get(k, 0.0) + rng.gauss(0, 0.6)))
    return c, "validator"


def mut_decision(cfg):
    c = copy.deepcopy(cfg); dg = c["decision"]; op = rng.choice(["fill", "margin", "sd", "sd"])
    if op == "fill": dg["fill"] = rng.choice(FILLS)
    elif op == "margin": dg["margin"] = rng.choice(MARGINS)
    else:
        nd = N4.mutate_decision({"strength": dg["strength"], "trust_gate": dg["trust_gate"]}, rng)[0]; dg.update(nd)
    return c, op


def instr_feedback(cfg, res):
    """evidence errors on Train + tasks where the accepted repair hurt the final forecast"""
    instr = json.load(open(cfg["retrieval"]["instr"]))["instr"]
    ivs, _ = tl2.extract(instr, TRAIN, workers=16); fb = []
    for t in TRAIN:
        r = tl2.evidence_f1(ivs[t], t)
        if r and (r[3] or r[4]): fb.append({"task": t, "missed_annotated_evidence": r[3], "extracted_non_evidence_quotes": r[4]})
    fv = E.fill_variants(cfg["retrieval"]["instr"]); dg = cfg["decision"]; base = copy.deepcopy(cfg); base["decision"]["margin"] = None
    nr = E.evaluate(base, USE)["per_task"]
    for t in TRAIN:
        v = fv.get(t, {}).get(dg["fill"])
        if v and v["val_raw"] is not None and dg["margin"] is not None and v["val_rep"] < v["val_raw"] * (1 - dg["margin"]):
            if res["per_task"][t] < nr[t] - 1e-6:
                fb.append({"task": t, "repair_hurt_final_forecast": True,
                           "removed_intervals": [dict(kind=i["kind"], start=i["start"], end=i["end"], quote=i["quote"][:160]) for i in ivs[t] if i["kind"] in tl2.REPAIR_KINDS and not i["recurs_in_future"]][:4]})
    return instr, fb


def mut_instr(cfg, res):
    instr, fb = instr_feedback(cfg, res)
    child = tl2.mutate(instr, rng.sample(fb, min(12, len(fb))))
    key = E.hashlib.sha1(child.encode()).hexdigest()[:12]; p = OUT / "instr" / f"{key}.json"
    json.dump({"instr": child}, open(p, "w"))
    c = copy.deepcopy(cfg); c["retrieval"]["instr"] = str(p); return c, "instr"


def held(r):
    if not HELD: return ""
    g = [r["per_task"][d["tid"]] for d in HELD]
    return f" | held-out fold mean gain {sum(g) / len(g):+.4f} W/R {sum(x > 1e-9 for x in g)}/{sum(x < -1e-9 for x in g)}"


def log(rec):
    with open(OUT / "log.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")


def main():
    cur = E.main_config(); res = E.evaluate(cur, USE); t0 = time.time()
    print(f"round 0 fitness {res['fitness']:+.4f} folds {[round(x, 3) for x in res['folds']]} W/R {res['better']}/{res['worse']}{held(res)}", flush=True)
    log(dict(round=0, fitness=res["fitness"], folds=res["folds"], better=res["better"], worse=res["worse"]))
    for rd in range(1, ROUNDS + 1):
        for role, mut in (("numerical", mut_numerical), ("retrieval", mut_validator), ("decision", mut_decision)):
            cands = [mut(cur) for _ in range(K)]
            if role == "retrieval" and rd % INSTR_EVERY == 0:
                try: cands.append(mut_instr(cur, res))
                except Exception as e: print("instr mutation failed", repr(e)[:100], flush=True)
            best = None; scored = []
            for c, what in cands:
                try: r = E.evaluate(c, USE)
                except Exception as e: print("eval failed", what, repr(e)[:100], flush=True); continue
                scored.append((c, r, what))
                if best is None or r["fitness"] > best[1]["fitness"]: best = (c, r, what)
            if ACC == "allfolds":   # stricter: better mean AND no fold worse (guards against fitting one fold)
                ok = [(c, r, w) for c, r, w in scored if r["fitness"] > res["fitness"] + 1e-4 and all(a >= b - 1e-9 for a, b in zip(r["folds"], res["folds"]))]
                best = max(ok, key=lambda x: x[1]["fitness"]) if ok else best
                acc = bool(ok)
            else:
                acc = best is not None and best[1]["fitness"] > res["fitness"] + 1e-4
            if acc: cur, res = best[0], best[1]
            print(f"round {rd} {role:9s} {'ACCEPT ' + best[2] if acc else 'reject'} fitness {res['fitness']:+.4f} W/R {res['better']}/{res['worse']} ({round(time.time() - t0)}s){held(res)}", flush=True)
            log(dict(round=rd, role=role, accepted=acc, what=best[2] if best else None, fitness=res["fitness"], folds=res["folds"], better=res["better"], worse=res["worse"]))
            json.dump(cur, open(OUT / "best.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
