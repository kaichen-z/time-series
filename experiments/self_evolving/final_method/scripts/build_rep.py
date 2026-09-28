"""Repaired histories {part: {tid: {clean, frac}}} for an extraction-instruction file (argv[1]) -> argv[2]."""
import json, sys
sys.path.insert(0, ".scratch/self_evolving")
import tl2
split = json.load(open("splits/drcik_public_80_20_99_v3.json"))["partitions"]
instr = json.load(open(sys.argv[1]))["instr"]; out = {}
for part in ("train", "dev", "public_test"):
    ivs, _ = tl2.extract(instr, split[part]["task_ids"], workers=16); rep = {}
    for t in split[part]["task_ids"]:
        z, fr = tl2.repair(t, ivs[t])
        if z: rep[t] = dict(clean=z, frac=fr)
    out[part] = rep
json.dump(out, open(sys.argv[2], "w")); print({k: len(v) for k, v in out.items()})
