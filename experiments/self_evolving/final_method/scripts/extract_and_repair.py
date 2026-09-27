"""Final method, step 3: extract timelines with the evolved Retrieval instructions and repair histories
for train/dev/test (extraction uses no labels; gt_evidence is only used during evolution on train)."""
import json, sys
sys.path.insert(0, ".scratch/self_evolving")
import tl2
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
instr = json.load(open(sys.argv[1]))["instr"]
out = {}
for part in ("train", "dev", "public_test"):
    ivs, _ = tl2.extract(instr, split[part]["task_ids"]); out[part] = {}
    for t in split[part]["task_ids"]:
        z, fr = tl2.repair(t, ivs[t])
        if z: out[part][t] = dict(clean=z, frac=fr)
json.dump(out, open(sys.argv[2], "w")); print({p: len(v) for p, v in out.items()})
