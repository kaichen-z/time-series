"""All statistical methods of the method-evolution portfolio on Time-MMD tasks (same as stat_full.py for Dr-CiK).
cwd = repo. Output: ts_work/unified/tmmd_stat.json {tid: {method: forecast}}."""
import json, time, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(".").resolve(); mr = ROOT / "runs/method_evolution/v001"
OUT = "work/unified/tmmd_stat.json"
FQ = {"monthly": "1 month", "weekly": "1 week", "daily": "1 day"}
def work(chunk):
    from evolving_loop.v2.real.host import _load_screening_policy
    from numerical_agent.evolution.portfolio import read_policy_file
    from numerical_agent.evolution.forecast_store import ForecastStore
    port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
    fs = ForecastStore(ROOT / ".scratch/self_evolving/stat_tmmd_cache", mr / "methods.py", mr / "skills.py", port, None,
                       screening_hash=scr.fingerprint(), runtime_identity={}, statistical_time_budget_s=20.0)
    names = sorted(fs.statistical_names); out = {}
    for tid, h, H, freq in chunk:
        r = {}
        for m in names:
            try: r[m] = list(fs.forecast(m, tuple(h), H, freq))
            except Exception: pass
        out[tid] = r
    fs.close(); return out
if __name__ == "__main__":
    D = json.load(open(".scratch/self_evolving/tmmd_cache.json"))
    items = [(d["tid"], [x for x in d["history"] if x is not None and math.isfinite(x)], d["H"], FQ.get(d["freq"], d["freq"])) for d in D]
    chunks = [items[i::16] for i in range(16)]; res = {}; t = time.time()
    with ProcessPoolExecutor(16) as ex:
        for out in ex.map(work, chunks): res.update(out)
    json.dump(res, open(OUT, "w")); print("done", len(res), "secs", round(time.time() - t))
