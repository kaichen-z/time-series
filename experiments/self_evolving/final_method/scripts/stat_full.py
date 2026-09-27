"""Forecasts of ALL statistical methods in the method-evolution portfolio (runs/method_evolution/v001)
for all 199 Dr-CiK tasks (full history -> future), CPU.  Output: nrd_cache_full.json = nrd_cache with
the fc dict extended.  No labels used to compute forecasts."""
import json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(".").resolve(); mr = ROOT / "runs/method_evolution/v001"
def work(chunk):
    from evolving_loop.v2.real.host import _load_screening_policy
    from numerical_agent.evolution.portfolio import read_policy_file
    from numerical_agent.evolution.forecast_store import ForecastStore
    port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
    fs = ForecastStore(ROOT / ".scratch/self_evolving/stat_full_cache", mr / "methods.py", mr / "skills.py", port, None,
                       screening_hash=scr.fingerprint(), runtime_identity={}, statistical_time_budget_s=20.0)
    names = sorted(fs.statistical_names); out = {}
    for tid, h, H, freq in chunk:
        r = {}
        for m in names:
            try: r[m] = list(fs.forecast(m, tuple(h), H, freq))
            except Exception: pass
        out[tid] = r
    fs.close(); return out, names
if __name__ == "__main__":
    D = json.load(open(".scratch/self_evolving/nrd_cache.json"))
    items = [(d["tid"], d["history"], d["H"], d["freq"]) for d in D]
    chunks = [items[i::12] for i in range(12)]; res = {}; names = None; t = time.time()
    with ProcessPoolExecutor(12) as ex:
        for out, nm in ex.map(work, chunks): res.update(out); names = nm
    for d in D:
        for m, f in res.get(d["tid"], {}).items():
            if m not in d["fc"]: d["fc"][m] = f
    json.dump(D, open(".scratch/self_evolving/nrd_cache_full.json", "w"))
    cov = {m: sum(1 for d in D if m in d["fc"]) for m in set(names)}
    print("statistical methods", len(names), "secs", round(time.time() - t), "methods with >=90% coverage", sum(1 for v in cov.values() if v >= 0.9 * len(D)))
