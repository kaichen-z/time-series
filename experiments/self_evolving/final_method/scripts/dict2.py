"""Self-evolving dictionary of HISTORY-PROCESSING programs for Toto (2026-09-27).

A dictionary member is a typed program that cleans a task's history before Toto forecasts it:
  src   : where anomalies come from -- Retrieval intervals (evolved tl2 extraction), robust statistics
          (|x - rolling median| > k * MAD), or both
  fill  : phase_median | linear | snaive | truncate (drop everything up to the last anomaly)
  level : re-align the level after the last anomaly (one-off level shifts)
  trim  : drop the earliest fraction of the history
Fitness is HISTORY-ONLY: Toto back-tests the held-out last H history points from the raw vs processed
history; the error is measured on the tail positions not flagged as anomalous (program-independent
target).  MAP-Elites keeps the best program per (frequency x seasonality) cell; the elites are the
dictionary.  Use: for each task pick the elite with the best own back-test; apply it only if it beats
the raw history by `margin`; the final forecast is always Toto on the chosen history.
Runs in the toto2 environment (CPU is fine).
"""
from __future__ import annotations
import copy, json, math, random, statistics, sys
from pathlib import Path
import numpy as np
import torch
from toto2 import Toto2Model
sys.path.insert(0, str(Path(__file__).parent))
import tl2

P = 32; DEV = torch.device("cpu")
MODEL = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(DEV).eval()
_FC = {}


def toto(hv, h):
    key = (hash(tuple(round(float(x), 6) for x in hv)), h)
    if key in _FC: return _FC[key]
    pad = (-len(hv)) % P
    x = torch.tensor([0.0] * pad + list(hv), dtype=torch.float32, device=DEV).reshape(1, 1, -1)
    m = torch.ones_like(x, dtype=torch.bool)
    if pad: m[..., :pad] = False
    with torch.no_grad():
        q = MODEL.forecast({"target": x, "target_mask": m, "series_ids": torch.zeros((1, 1), dtype=torch.long, device=DEV)},
                           horizon=h, decode_block_size=None, has_missing_values=False)
    q = [qq.reshape(-1).tolist() if hasattr(qq, "reshape") else qq for qq in (q if isinstance(q, (list, tuple)) else list(q))]
    _FC[key] = q[4]; return q[4]


def period_of(freq):
    f = (freq or "").lower()
    if "5 min" in f: return 288
    if "min" in f: return 60
    if "hour" in f: return 24
    if "day" in f: return 7
    return 1


def mad_mask(y, k, w):
    y = np.asarray(y, float); n = len(y); out = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1); seg = y[lo:hi]
        med = np.median(seg); mad = np.median(np.abs(seg - med)) * 1.4826 + 1e-9
        out[i] = abs(y[i] - med) > k * mad
    return out


def retrieval_mask(tid, ivs, n_hist):
    d = tl2.raw(tid); stamps = [tl2.ts(x) for x in d["series"]["history_timestamps"]]
    bad = np.zeros(n_hist, bool)
    for iv in ivs:
        if iv["kind"] in tl2.REPAIR_KINDS and not iv["recurs_in_future"]: bad |= tl2.mask(stamps, iv)[:n_hist]
    return bad


def apply(prog, task, ivs, upto=None):
    """return processed history (list) for history[:upto] or None if nothing changes."""
    y = np.array(task["history"][:upto] if upto else task["history"], float); n = len(y)
    bad = np.zeros(n, bool)
    if prog["src"] in ("retrieval", "both"): bad |= retrieval_mask(task["tid"], ivs, n)
    if prog["src"] in ("stat", "both"): bad |= mad_mask(y, prog["k"], prog["w"])
    z = y.copy(); p = period_of(task["freq"])
    if bad.any() and bad.mean() < 0.9:
        good = np.where(~bad)[0]
        if prog["fill"] == "truncate":
            last = np.where(bad)[0][-1]
            if n - last - 1 >= max(8, 2 * task["H"]): z = y[last + 1:]
            else: z = y.copy(); z[bad] = np.interp(np.where(bad)[0], good, y[good])
        else:
            for i in np.where(bad)[0]:
                if prog["fill"] == "phase_median" and p > 1:
                    same = [y[j] for j in range(i % p, n, p) if not bad[j]]
                    z[i] = float(np.median(same)) if len(same) >= 2 else float(np.interp(i, good, y[good]))
                elif prog["fill"] == "snaive" and p > 1 and i - p >= 0 and not bad[i - p]:
                    z[i] = y[i - p]
                else:
                    z[i] = float(np.interp(i, good, y[good]))
        if prog["level"] and prog["fill"] != "truncate":
            last = np.where(bad)[0][-1]
            post = y[last + 1:]; pre = z[:np.where(bad)[0][0]]
            if len(post) >= 4 and len(pre) >= 4:
                shift = float(np.median(post) - np.median(pre)); z[:last + 1] = z[:last + 1] + shift
    if prog["trim"] > 0 and len(z) > 4 * task["H"]:
        z = z[int(len(z) * prog["trim"]):]
    if len(z) == n and np.allclose(z, y): return None
    return z.tolist()


def tail_err(task, ivs, hist_proc_fn):
    """history-only back-test: forecast the last H history points, score on non-anomalous tail positions."""
    y = task["history"]; H = task["H"]
    if len(y) - H < max(H, 16): return None
    tail = np.array(y[-H:], float)
    ok = ~retrieval_mask(task["tid"], ivs, len(y))[-H:]   # only document-stated anomalies are excluded (stat masks would make the check circular)
    if ok.sum() < max(2, H // 3): return None
    base = hist_proc_fn(len(y) - H)
    f = np.array(toto(base, H), float)
    sc = np.mean(np.abs(tail[ok])) + 1e-9
    return float(np.mean(np.abs(f[ok] - tail[ok])) / sc + math.sqrt(np.mean((f[ok] - tail[ok]) ** 2)) / sc)


def raw_err(task, ivs):
    if "_raw" not in task: task["_raw"] = tail_err(task, ivs, lambda u: task["history"][:u])
    return task["_raw"]


def prog_err(prog, task, ivs):
    k = json.dumps(prog, sort_keys=True)
    c = task.setdefault("_pe", {})
    if k not in c:
        def fn(u):
            z = apply(prog, task, ivs, upto=u)
            return z if z is not None else task["history"][:u]
        c[k] = tail_err(task, ivs, fn)
    return c[k]


def rel_gain(prog, task, ivs):
    r = raw_err(task, ivs); e = prog_err(prog, task, ivs)
    if r is None or e is None: return None
    return (r - e) / (r + 1e-9)


def seas(task):
    p = period_of(task["freq"]); y = np.asarray(task["history"], float)
    if p <= 1 or len(y) < 2 * p: return "flat"
    d = y[p:] - y[:-p]; return "seas" if 1 - np.var(d) / (2 * np.var(y) + 1e-12) > 0.4 else "flat"


def cell(task):
    f = (task["freq"] or "").lower()
    fc = next((k for k in ("second", "minute", "hour", "day") if k in f), "other")
    return f"{fc}|{seas(task)}"


def prog0(): return {"src": "retrieval", "k": 5.0, "w": 12, "fill": "phase_median", "level": False, "trim": 0.0}


def mutate(p, rng):
    c = dict(p); op = rng.choice(["src", "k", "w", "fill", "fill", "level", "trim"])
    if op == "src": c["src"] = rng.choice(["retrieval", "stat", "both"])
    elif op == "k": c["k"] = float(np.clip(c["k"] * 2 ** rng.gauss(0, 0.4), 2.0, 12.0))
    elif op == "w": c["w"] = rng.choice([4, 8, 12, 24, 48])
    elif op == "fill": c["fill"] = rng.choice(["phase_median", "linear", "snaive", "truncate"])
    elif op == "level": c["level"] = not c["level"]
    else: c["trim"] = rng.choice([0.0, 0.25, 0.5])
    return c, op


def cell_fitness(prog, tasks, ivs):
    gs = [rel_gain(prog, t, ivs[t["tid"]]) for t in tasks]
    gs = [g for g in gs if g is not None]
    return (statistics.mean(gs) + 0.5 * statistics.mean(min(0.0, g) for g in gs)) if gs else -1.0


def evolve(train, ivs, gens, children, rng, log=print):
    cells = {}
    for t in train: cells.setdefault(cell(t), []).append(t)
    elite = {c: (cell_fitness(prog0(), ts, ivs), prog0()) for c, ts in cells.items()}
    curve = [round(statistics.mean(f for f, _ in elite.values()), 4)]
    log(f"gen 0: mean cell fitness {curve[0]:+.4f} " + str({c: round(f, 3) for c, (f, _) in elite.items()}))
    for g in range(1, gens + 1):
        pool = [p for _, p in elite.values()]; ins = 0
        for _ in range(children):
            ch, _op = mutate(rng.choice(pool), rng)
            for c, ts in cells.items():
                f = cell_fitness(ch, ts, ivs)
                if f > elite[c][0] + 1e-6: elite[c] = (f, ch); ins += 1
        curve.append(round(statistics.mean(f for f, _ in elite.values()), 4))
        log(f"gen {g}: insertions {ins}, mean cell fitness {curve[-1]:+.4f} " + str({c: round(f, 3) for c, (f, _) in elite.items()}))
    return elite, curve


def choose(task, ivs, elite, margin):
    """per task: best elite by its own history back-test; use only if it beats raw by margin."""
    best = None
    seen = set()
    for f, p in elite.values():
        k = json.dumps(p, sort_keys=True)
        if k in seen: continue
        seen.add(k); g = rel_gain(p, task, ivs)
        if g is not None and (best is None or g > best[0]): best = (g, p)
    if best and best[0] > margin:
        z = apply(best[1], task, ivs)
        if z is not None: return best[1], z, best[0]
    return None, None, None


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    C = json.load(open(".scratch/self_evolving/nrd_cache.json")); T = {d["tid"]: d for d in C}
    split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
    instr = json.load(open(".scratch/self_evolving/tl2_best.json"))["instr"]
    ivs = {}
    for part in ("train", "dev", "public_test"):
        r, _ = tl2.extract(instr, split[part]["task_ids"]); ivs.update(r)
    train = [T[t] for t in split["train"]["task_ids"]]
    elite, curve = evolve(train, ivs, gens, 8, random.Random(1))
    out = dict(curve=curve, elite={c: dict(fitness=f, program=p) for c, (f, p) in elite.items()}, chosen={})
    for part in ("train", "dev", "public_test"):
        for t in split[part]["task_ids"]:
            p, z, g = choose(T[t], ivs[t], elite, 0.1)
            if p is not None:
                out["chosen"][t] = dict(part=part, program=p, backtest_gain=g, forecast=toto(z, T[t]["H"]))
    json.dump(out, open(".scratch/self_evolving/dict2_result.json", "w"))
    print("chosen per part:", {pt: sum(1 for v in out["chosen"].values() if v["part"] == pt) for pt in ("train", "dev", "public_test")})


if __name__ == "__main__":
    main()
