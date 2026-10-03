"""Build the common task file used by the learned baselines."""
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "work/learned_baselines/tasks.jsonl"
TMMD = {
    "Agriculture": (6, "monthly"), "Climate": (12, "weekly"), "Economy": (6, "monthly"),
    "Energy": (12, "weekly"), "Environment": (48, "daily"), "Health_AFR": (12, "weekly"),
    "Health_US": (12, "weekly"), "Security": (6, "monthly"), "SocialGood": (6, "monthly"),
    "Traffic": (6, "monthly"),
}


def clean(values, last=0.0):
    out = []
    for value in values:
        if value is not None and math.isfinite(float(value)):
            last = float(value)
        out.append(last)
    return out


def text_before(domain, cutoff):
    rows = []
    for kind in ("report", "search"):
        path = ROOT / "external/Time-MMD/textual" / domain / f"{domain}_{kind}.csv"
        if not path.exists():
            continue
        for row in csv.DictReader(open(path)):
            if row.get("end_date", "") < cutoff:
                rows.append((row["end_date"], row.get("fact", ""), row.get("preds", "")))
    rows.sort(reverse=True)
    return "\n".join(f"{date}: {fact} {pred}" for date, fact, pred in rows[:8])


def drcik():
    path = ROOT / ".scratch/self_evolving/nrd_cache_full2.json"
    out = []
    for row in json.load(open(path)):
        out.append(dict(tid=row["tid"], dataset="drcik",
                        part={"public_test": "test"}.get(row["part"], row["part"]),
                        freq=row["freq"], history=clean(row["history"]), truth=clean(row["truth"]),
                        text="\n\n".join(row.get("docs", []))))
    return out


def tmmd():
    split = json.load(open(ROOT / "splits/timemmd_531_181_182_v1.json"))
    parts = {tid: part for part in ("train", "dev", "test") for tid in split[part]}
    out = []
    for domain, (horizon, freq) in TMMD.items():
        path = ROOT / "external/Time-MMD/numerical" / domain / f"{domain}.csv"
        rows = list(csv.DictReader(open(path)))
        date = "date" if "date" in rows[0] else "date.1"
        series = sorted((row[date], float(row["OT"]) if row["OT"] else math.nan) for row in rows)
        dates, values = zip(*series)
        for tid, part in parts.items():
            prefix = f"tmmd_{domain}_"
            if not tid.startswith(prefix):
                continue
            cutoff = tid[len(prefix):]
            start = dates.index(cutoff)
            history = clean(values[:start])
            truth = clean(values[start:start + horizon], history[-1])
            if len(truth) != horizon:
                raise ValueError(tid)
            out.append(dict(tid=tid, dataset="tmmd", part=part, freq=freq,
                            history=history, truth=truth,
                            text=text_before(domain, cutoff)))
    return out


def timesx():
    out = []
    for row in json.load(open(ROOT / "work/timesx/tasks.json")):
        text = "\n".join(filter(None, [row.get("background"), row.get("scenario"),
                                        row.get("holiday"), row.get("covariates")]))
        out.append(dict(tid=row["tid"], dataset="timesx", part=row["part"], freq=row["freq"],
                        history=clean(row["history"]), truth=clean(row["truth"]), text=text))
    return out


def main():
    tasks = drcik() + tmmd() + timesx()
    seen = set()
    for task in tasks:
        key = (task["dataset"], task["tid"])
        if key in seen or not task["history"] or not task["truth"]:
            raise ValueError(key)
        seen.add(key)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    from collections import Counter
    print(OUT, Counter((x["dataset"], x["part"]) for x in tasks))


if __name__ == "__main__":
    main()
