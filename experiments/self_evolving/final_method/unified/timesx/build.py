"""Flatten TimesX (haoxin1998/TimesX-project Datasets/) into tasks with the PostTime split (arXiv 2605.29401):
train = 88 ID variables, prediction window after 2023-01-01 and ending before 2025-02-01;
test_id = 88 ID variables, window starting after 2025-01-30; test_ood = 11 held-out variables, same rule.
dev = last train windows (window starting 2024-09-01..2025-01-30) of ID variables, carved out of train."""
import json, glob, collections
ID = set(open("id_vars.txt").read().split()); OOD = set(open("ood_vars.txt").read().split())
out = []; miss = set()
for f in sorted(glob.glob("/tmp/timesx/tx/Datasets/**/*.json", recursive=True)):
    d = json.load(open(f)); v = d["dataset_info"]["dataset_name"]
    if v not in ID and v not in OOD: miss.add(v); continue
    dom = f.split("/Datasets/")[1].split("/")[0]
    for s in d["samples"]:
        ft = s["future_time"]["timestamp"]; st, en = ft[0][:10], ft[-1][:10]
        if st > "2025-01-30": part = "test_id" if v in ID else "test_ood"
        elif v in ID and st > "2023-01-01" and en < "2025-02-01": part = "dev" if st >= "2024-09-01" else "train"
        else: continue
        out.append(dict(tid=f"{v}_{s['idx']}", var=v, domain=dom, part=part, freq=s["freq"], date=s["date"],
                        history=[float(x) for x in s["past_time"]["value"]], truth=[float(x) for x in s["future_time"]["value"]],
                        future_ts=ft, background=s["background"], scenario=s["scenario"], holiday=s["holiday_info"],
                        covariates=s["covariates_info"]))
json.dump(out, open("tasks.json", "w"))
print(len(out), collections.Counter(t["part"] for t in out), "vars", len({t["var"] for t in out}), "unmatched vars", len(miss))
print(collections.Counter((t["part"], t["domain"]) for t in out))
