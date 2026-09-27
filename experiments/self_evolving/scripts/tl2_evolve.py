"""Evolve the Retrieval extraction instructions (tl2) with gt_evidence F1 on a Train minibatch, then
evaluate history repair on full Train and Dev (2026-09-27)."""
import json, random, statistics, subprocess, sys
sys.path.insert(0, ".scratch/self_evolving")
import tl2
from common.metrics import drcik_point_metrics

split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
D = {d["tid"]: d for d in json.load(open(".scratch/self_evolving/nrd_cache.json"))}
rng = random.Random(7)
train = list(split["train"]["task_ids"]); mb = rng.sample(train, 30)


def score(instr, tids):
    ivs, key = tl2.extract(instr, tids)
    fs, fb = [], []
    for t in tids:
        r = tl2.evidence_f1(ivs[t], t)
        if r is None: continue
        fs.append(r[0])
        if r[3] or r[4]: fb.append({"missed_annotated_evidence": r[3], "extracted_non_evidence_quotes": r[4]})
    return statistics.mean(fs), fb, ivs


cur = tl2.INSTR0; f, fb, _ = score(cur, mb); hist = [round(f, 4)]
print(f"gen 0 evidence-F1 {f:.3f}", flush=True)
for g in range(1, 5):
    best = None
    for _ in range(2):
        try: child = tl2.mutate(cur, rng.sample(fb, min(12, len(fb))))
        except Exception as e: print("mutate failed", repr(e)[:80]); continue
        fc, fbc, _ = score(child, mb); print(f"  gen {g} child F1 {fc:.3f}", flush=True)
        if best is None or fc > best[0]: best = (fc, fbc, child)
    if best and best[0] > f + 1e-3: f, fb, cur = best; print(f"gen {g} ACCEPT F1 {f:.3f}", flush=True)
    else: print(f"gen {g} reject (F1 {f:.3f})", flush=True)
    hist.append(round(f, 4))
json.dump({"instr": cur, "curve": hist}, open(".scratch/self_evolving/tl2_best.json", "w"))

# full Train + Dev with the evolved instructions and with the initial ones
for name, ins in (("initial", tl2.INSTR0), ("evolved", cur)):
    out = {}
    for part in ("train", "dev"):
        ivs, _ = tl2.extract(ins, split[part]["task_ids"])
        ef = [tl2.evidence_f1(ivs[t], t) for t in split["train"]["task_ids"]] if part == "train" else []
        rep = {}
        for t in split[part]["task_ids"]:
            z, fr = tl2.repair(t, ivs[t])
            if z: rep[t] = dict(clean=z, frac=fr)
        out[part] = rep
        if ef: print(f"[{name}] train evidence-F1 {statistics.mean(x[0] for x in ef if x):.3f} recall {statistics.mean(x[1] for x in ef if x):.3f} precision {statistics.mean(x[2] for x in ef if x):.3f}", flush=True)
    json.dump(out, open(f".scratch/self_evolving/tl2_repair_{name}.json", "w"))
print("done", flush=True)
