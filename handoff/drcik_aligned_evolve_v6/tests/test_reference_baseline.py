#!/usr/bin/env python3
"""Protocol v6 reference-baseline test (no model): on a run dir created by the orchestrator's dry-run,
  * eval_data base_jt equals the joint error of the frozen reference forecaster on every F0 row (recomputed here),
    and differs from the Toto joint error (toto_jt) on at least one row;
  * stage.json and eval_data record the current reference SHA-256;
  * the seed Numerical module produces exactly the reference forecast on every row (seed == reference);
  * the reference only blends the frozen anchor forecasts and the row's own history (no truth access).
usage: test_reference_baseline.py --run RUN_DIR"""
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import reference as REF, viewstore  # noqa: E402

P = argparse.ArgumentParser(); P.add_argument("--run", type=Path, required=True); A = P.parse_args()
ed = json.load(open(A.run / "private/eval_data.json")); st = json.load(open(A.run / "stage.json"))
S = viewstore.Store(A.run / "shared/store")
recomputed = REF.reference_jt(A.run / "shared/store", ed["truth"])
spec = importlib.util.spec_from_file_location("seed", HERE / "seeds/seed_forecast.py"); seed = importlib.util.module_from_spec(spec); spec.loader.exec_module(seed)
spec = importlib.util.spec_from_file_location("ref", REF.REF); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
same_seed = all(seed.forecast(S.view(k)) == ref.forecast(S.view(k)) for k in ed["truth"])
checks = dict(
    base_jt_is_reference=all(abs(ed["base_jt"][k] - recomputed[k]) < 1e-12 for k in ed["truth"]),
    base_jt_differs_from_toto=any(abs(ed["base_jt"][k] - ed["toto_jt"][k]) > 1e-9 for k in ed["truth"]),
    reference_sha_recorded=(st.get("reference_sha256") == ed.get("reference_sha256") == REF.reference_sha256()),
    seed_equals_reference=same_seed,
    cyclic_seasonal=(ref.seasonal(list(range(10)), 12, 7) == [3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7]
                     and seed.seasonal(list(range(10)), 12, 7) == [3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7]),
    reference_reads_no_truth="truth" not in REF.REF.read_text())
out = dict(rows=len(ed["truth"]), checks=checks, ok=all(checks.values()))
print(json.dumps(out, indent=1)); sys.exit(0 if out["ok"] else 1)
