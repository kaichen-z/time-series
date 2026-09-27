"""N/R/D with an evolving Retrieval *extraction program* and Numerical magnitude calibration
(2026-09-27).

Retrieval : evolves (document selector, GPT instruction text).  GPT is the mutation operator for the
            instructions; the selector mutates by typed numeric edits.  Output = events with window,
            direction and strength class only.
Numerical : converts strength class -> magnitude = k[strength] * sigma(history); k is calibrated on
            Train per-correction credit (sigma is history-only).  Also supplies the cell trust profile.
Decision  : confidence threshold + trust gate, fitted on Train.
Fitness of a program = stratified Train CV (N and D refit inside each fold).  Dev once at the end;
test99 was opened before, so test numbers are exploratory.
"""
from __future__ import annotations
import argparse, copy, itertools, json, random, statistics, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import nrd_coevolve as C
import nrd3 as R3
import nrd4 as N4
import nrd5_extract as X
from evolving_loop.adjustment.post_adjust import apply_bounded_delta, horizon_window_mask

ANCHOR = "toto_2_0"
KGRID = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5]
CGRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def to_corr(d, events):
    raw = X.load_raw(d["tid"]); fts = [str(x) for x in raw["series"]["future_timestamps"]]
    out = []
    for e in events:
        try: on = [i for i, o in enumerate(horizon_window_mask(fts, str(e["start"]), str(e["end"]))) if o]
        except Exception: on = []
        if on: out.append(dict(s=on[0], e=on[-1] + 1, dir=1.0 if e["direction"] == "up" else -1.0,
                               st=e["strength"], conf=float(e.get("confidence", 0.0)), typ=e.get("type", "")[:60]))
    return out


def sigma(d):
    if "sig0" not in d: d["sig0"] = N4.sigma_of(N4.calib0(), d["history"], d["H"], d["freq"])
    return d["sig0"]


def corr_gain(d, c, k):
    base = d["fc"][ANCHOR]; mult = 1 + c["dir"] * k * sigma(d)
    out = list(apply_bounded_delta(base, R3.apply_ev(base, [(c["s"], c["e"], mult)], 1.0)))
    return d["base_jt"] - C.jt(out, d["truth"])


def fit_nd(ds, trust):
    """Numerical k per strength class + Decision confidence threshold, on Train per-correction credit."""
    items = [(d, c) for d in ds for c in d["ev"]]
    best_k = {}
    for st in ("small", "medium", "large"):
        its = [(d, c) for d, c in items if c["st"] == st]
        best_k[st] = max(KGRID, key=lambda k: sum((lambda g: g if g > 0 else 1.5 * g)(corr_gain(d, c, k)) for d, c in its)) if its else 0.0
    best = None
    for cmin, gate in itertools.product(CGRID, [0.0, 0.3, 0.6, 1.0]):
        sc = 0.0
        for d, c in items:
            if c["conf"] < cmin or trust.get(d["cell"], 1.0) < gate: continue
            g = corr_gain(d, c, best_k[c["st"]]); sc += g if g > 0 else 1.5 * g
        if best is None or sc > best[0]: best = (sc, cmin, gate)
    return {"k": best_k, "cmin": best[1], "gate": best[2]}


def apply_team(d, p, trust):
    base = d["fc"][ANCHOR]
    if trust.get(d["cell"], 1.0) < p["gate"]: return list(base)
    ev = [(c["s"], c["e"], 1 + c["dir"] * p["k"][c["st"]] * sigma(d)) for c in d["ev"] if c["conf"] >= p["cmin"]]
    return list(apply_bounded_delta(base, R3.apply_ev(base, ev, 1.0)))


def summarize(ds, p, trust):
    sb = so = rb = ro = 0.0; w = r = 0
    from common.metrics import drcik_point_metrics
    for d in ds:
        out = apply_team(d, p, trust)
        mb = drcik_point_metrics(d["truth"], d["fc"][ANCHOR], cap=5.0); mo = drcik_point_metrics(d["truth"], out, cap=5.0)
        sb += mb["smae"]; so += mo["smae"]; rb += mb["srmse"]; ro += mo["srmse"]
        dj = (mb["smae"] + mb["srmse"]) - (mo["smae"] + mo["srmse"])
        w += dj > 1e-9; r += dj < -1e-9
    return dict(n=len(ds), smae_gain=(sb - so) / sb, srmse_gain=(rb - ro) / rb, joint_gain=((sb + rb) - (so + ro)) / (sb + rb), wins=w, regressions=r)


def cv_score(train, folds):
    per, feed = [], {}
    for k in range(len(folds)):
        tr = [d for j in range(len(folds)) if j != k for d in folds[j]]
        trust = N4.trust_profile(tr); p = fit_nd(tr, trust)
        s = summarize(folds[k], p, trust); per.append(s)
        for d in folds[k]:                       # per-type feedback (Train only) for the GPT mutator
            for c in d["ev"]:
                g = corr_gain(d, c, p["k"][c["st"]]); t = c["typ"][:40]
                f = feed.setdefault(t, [0, 0]); f[0 if g > 0 else 1] += 1
    js = [s["joint_gain"] for s in per]
    return statistics.mean(js) - 0.25 * statistics.pstdev(js), per, feed


GPT_MUT = """You improve the instruction text given to an LLM that extracts forward-looking events from
documents for a time-series forecast-correction system. Below are the CURRENT INSTRUCTIONS and TRAIN-ONLY
FEEDBACK: for each event type the extractor produced, how many times applying it helped vs hurt the
forecast. Rewrite the instructions so the extractor proposes more of the helpful kinds of events and
fewer of the harmful kinds (e.g. tighten or relax criteria, clarify window/direction/strength rules).
Keep it under 250 words, general (no task ids, no specific dates), and keep the output fields
(window, direction up/down, strength small/medium/large, confidence). Return JSON {"instructions": "..."}."""


def gpt_mutate(instr, feed, model="gpt-6-sol"):
    fb = sorted(feed.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:25]
    prompt = GPT_MUT + "\nCURRENT INSTRUCTIONS:\n" + instr + "\nFEEDBACK (type: helped, hurt):\n" + \
        "\n".join(f"- {t}: {h}, {b}" for t, (h, b) in fb)
    schema = {"type": "object", "properties": {"instructions": {"type": "string"}}, "required": ["instructions"], "additionalProperties": False}
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"; op = Path(td) / "o.json"; sp.write_text(json.dumps(schema))
        subprocess.run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-rules",
                        "-m", model, "--output-schema", str(sp), "-o", str(op), "-"],
                       input=prompt, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        return json.loads(op.read_text())["instructions"]


def mutate_selector(g, rng):
    c = copy.deepcopy(g); op = rng.choice(["top_k", "chars", "pad", "w", "w"])
    if op == "top_k": c["top_k"] = rng.choice([3, 4, 6, 8, 10])
    elif op == "chars": c["chars"] = rng.choice([800, 1200, 1500, 2000, 2500])
    elif op == "pad": c["pad_days"] = rng.choice([0, 1, 3, 7])
    else:
        k = rng.choice(list(c["w"])); c["w"][k] = round(c["w"][k] + rng.gauss(0, 0.7), 2)
    return c


def attach(D, prog, bases):
    ev, key = X.extract(prog, [d["tid"] for d in D], bases)
    for d in D: d["ev"] = to_corr(d, ev[d["tid"]])
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--data", choices=["drcik", "tmmd"], default="drcik")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--open", choices=["cv", "dev", "test"], default="dev")
    a = ap.parse_args(); rng = random.Random(a.seed)
    C.FOLD_MODE = "strat"
    sfx = "" if a.data == "drcik" else "_tmmd"
    if a.data == "tmmd": X.TASKS = Path("external/Time-MMD/tasks")
    TH = json.load(open(f".scratch/self_evolving/toto_hindcast{sfx}.json"))
    D = [R3.prep_task(d, TH) for d in json.load(open(".scratch/self_evolving/nrd_cache.json" if a.data == "drcik" else ".scratch/self_evolving/tmmd_cache.json"))]
    bases = {d["tid"]: d["fc"][ANCHOR] for d in D}
    train = [d for d in D if d["part"] == "train"]; folds = C.gfolds(train, 3)
    cur = {"selector": X.sel0(), "instructions": X.INSTR0}
    attach(train, cur, bases); f, per, feed = cv_score(train, folds)
    hist = [dict(gen=0, fit=f, folds=[s["joint_gain"] for s in per], program=cur, accepted=True)]
    print(f"gen 0 fit {f:+.4f} folds {[round(s['joint_gain'], 4) for s in per]} events {sum(len(d['ev']) for d in train)}", flush=True)
    for g in range(1, a.gens + 1):
        kids = []
        try: kids.append(("gpt", {"selector": cur["selector"], "instructions": gpt_mutate(cur["instructions"], feed)}))
        except Exception as e: print("gpt mutate failed", repr(e)[:80])
        kids.append(("sel", {"selector": mutate_selector(cur["selector"], rng), "instructions": cur["instructions"]}))
        if kids[0][0] == "gpt": kids.append(("both", {"selector": mutate_selector(cur["selector"], rng), "instructions": kids[0][1]["instructions"]}))
        best = None
        for tag, prog in kids:
            attach(train, prog, bases); fk, perk, feedk = cv_score(train, folds)
            print(f"  gen {g} child {tag}: fit {fk:+.4f} folds {[round(s['joint_gain'], 4) for s in perk]} events {sum(len(d['ev']) for d in train)}", flush=True)
            if best is None or fk > best[0]: best = (fk, perk, feedk, prog, tag)
        acc = best[0] > f + 1e-5
        if acc: f, per, feed, cur = best[0], best[1], best[2], best[3]
        hist.append(dict(gen=g, fit=f, folds=[s["joint_gain"] for s in per], accepted=acc, tag=best[4] if acc else None, program=cur))
        print(f"gen {g} {'ACCEPT ' + best[4] if acc else 'reject'} fit {f:+.4f}", flush=True)
    res = dict(history=hist, final_program=cur)
    attach(train, cur, bases); trust = N4.trust_profile(train); p = fit_nd(train, trust)
    res["nd"] = p; res["train"] = summarize(train, p, trust)
    if a.open in ("dev", "test"):
        dev = [d for d in D if d["part"] == "dev"]; attach(dev, cur, bases)
        res["dev"] = summarize(dev, p, trust); res["dev_gate_pass"] = res["dev"]["smae_gain"] >= 0 and res["dev"]["srmse_gain"] >= 0
        print("train", res["train"], "\ndev", res["dev"], "gate", res["dev_gate_pass"], flush=True)
    if a.open == "test":
        te = [d for d in D if d["part"] == "public_test"]; attach(te, cur, bases)
        res["test_exploratory"] = summarize(te, p, trust) if res["dev_gate_pass"] else None
        print("test(exploratory)", res["test_exploratory"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
