"""Past covariates for Time-MMD tasks: other numeric columns of each domain CSV, aligned to the task's history
timestamps (first row per start date, same as build_tasks.py); only history (no future covariate values)."""
import csv, json, glob, math
from datetime import date
SRC = "external/Time-MMD/numerical"
TASKS = "work/timemmd/tasks"
SKIP = {"OT", "date", "start_date", "end_date", "Date", "Month", "MapDate", "ValidStart", "ValidEnd", "YEAR", "WEEK", "YEAR_WEEK", "CBSA Code", "StatisticFormatID"}
def num(x):
    try: v = float(str(x).replace(",", "")); return v if math.isfinite(v) else None
    except ValueError: return None
cov = {}
for dom in sorted(__import__("os").listdir(SRC)):
    rows = list(csv.DictReader(open(f"{SRC}/{dom}/{dom}.csv", encoding="utf-8")))
    key = "start_date" if "start_date" in rows[0] else "date"
    by = {}
    for r in rows:
        k = str(r[key])[:10]
        if k not in by: by[k] = r
    cols = [c for c in rows[0] if c not in SKIP and sum(num(r[c]) is not None for r in rows) >= 0.8 * len(rows)]
    cov[dom] = dict(cols=cols, by={k: [num(r[c]) for c in cols] for k, r in by.items()})
    print(dom, cols[:6], len(cols))
out = {}
for f in glob.glob(f"{TASKS}/*.json"):
    t = json.load(open(f)); dom = t["domain"]; C = cov.get(dom)
    if not C or not C["cols"]: continue
    ts = [str(x)[:10] for x in t["series"]["history_timestamps"]]
    M = []
    for j in range(len(C["cols"])):
        col, last = [], None
        for k in ts:
            v = C["by"].get(k, [None] * len(C["cols"]))[j]
            if v is None: v = last
            col.append(v); last = v if v is not None else last
        if sum(v is None for v in col) <= 0.1 * len(col):
            first = next(v for v in col if v is not None); col = [first if v is None else v for v in col]
            if max(col) - min(col) > 1e-12: M.append(col)
    if M: out[t["benchmark_id"]] = M
json.dump(out, open("tmmd_cov.json", "w")); print("tasks with covariates", len(out))
