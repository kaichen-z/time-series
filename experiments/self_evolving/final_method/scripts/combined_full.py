"""Forecasts of the 5 combined portfolio policies (TSFM parent + statistical parent; weighted mean or
signal router) for all Dr-CiK tasks, using the repo's own combination code (ForecastStore._combined).
TSFM parent forecasts come from the cached runs (nrd_cache: toto/chronos; tsfm_full.json; timesfm_full.json).
Then writes nrd_cache_full2.json = nrd_cache_full + the 3 TSFMs + the 5 combined.  No labels used."""
import json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(".").resolve(); mr = ROOT / "runs/method_evolution/v001"; S = ".scratch/self_evolving/"
def tsfm_table():
    C = json.load(open(S + "nrd_cache.json")); T = json.load(open(S + "tsfm_full.json")); F = json.load(open(S + "timesfm_full.json"))
    tab = {}
    for d in C:
        r = {k: d["fc"][k] for k in ("toto_2_0", "chronos_bolt") if k in d["fc"]}; r.update(T.get(d["tid"], {}))
        if d["tid"] in F: r["timesfm_2_5"] = F[d["tid"]]
        tab[d["tid"]] = r
    return tab
def work(chunk):
    from evolving_loop.v2.real.host import _load_screening_policy
    from numerical_agent.evolution.portfolio import read_policy_file
    from numerical_agent.evolution.forecast_store import ForecastStore
    tab = tsfm_table(); cur = {}
    class Store(ForecastStore):
        def _execute(self, name, history, horizon, frequency):
            if name in self.tsfm:
                f = tab[cur["tid"]].get(name)
                if f is None: raise self.not_applicable("tsfm forecast unavailable")
                return tuple(f)
            return super()._execute(name, history, horizon, frequency)
    port = read_policy_file(str(mr / "policies.py")); scr = _load_screening_policy(str(mr / "dictionary.py"))
    fs = Store(ROOT / ".scratch/self_evolving/combined_cache", mr / "methods.py", mr / "skills.py", port, None,
               screening_hash=scr.fingerprint(), runtime_identity={}, statistical_time_budget_s=20.0)
    names = [p.name for p in port.combined]; out = {}
    for tid, h, H, freq in chunk:
        cur["tid"] = tid; r = {}
        for m in names:
            try: r[m] = list(fs._execute(m, tuple(h), H, freq))
            except Exception as e: pass
        out[tid] = r
    fs.close(); return out, names
if __name__ == "__main__":
    D = json.load(open(S + "nrd_cache_full.json")); tab = tsfm_table()
    items = [(d["tid"], d["history"], d["H"], d["freq"]) for d in D]
    res = {}; t = time.time()
    with ProcessPoolExecutor(12) as ex:
        for out, names in ex.map(work, [items[i::12] for i in range(12)]): res.update(out)
    for d in D:
        for k in ("timesfm_2_5", "moirai_2_0", "granite_ttm_r2"):
            if k in tab[d["tid"]] and k not in d["fc"]: d["fc"][k] = tab[d["tid"]][k]
        d["fc"].update(res.get(d["tid"], {}))
    json.dump(D, open(S + "nrd_cache_full2.json", "w"))
    for m in ["timesfm_2_5", "moirai_2_0", "granite_ttm_r2"] + names:
        print(m, "tasks", sum(m in d["fc"] for d in D), "/", len(D))
    print("secs", round(time.time() - t))
