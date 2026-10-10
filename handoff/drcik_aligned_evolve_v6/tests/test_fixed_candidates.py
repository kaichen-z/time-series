#!/usr/bin/env python3
"""Offline contract checks for host-pinned v6 candidates; no labels or model calls."""
import importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]; ROOT = HERE / "fixed_candidates"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

manifest = json.load(open(ROOT / "manifest.json"))
assert [x["name"] for x in manifest["candidates"]] == ["fixed:raw_toto", "fixed:raw_timesfm", "fixed:raw_moirai",
    "fixed:raw_chronos", "fixed:ensemble_mean4", "fixed:ensemble_median4"]
forecasts = {"toto_2_0": [0.0, 4.0], "timesfm_2_5": [2.0, 2.0], "moirai_2_0": [4.0, 0.0], "chronos_bolt": [10.0, 6.0]}
expected = {"raw_toto.py": [0.0, 4.0], "raw_timesfm.py": [2.0, 2.0], "raw_moirai.py": [4.0, 0.0],
            "raw_chronos.py": [10.0, 6.0], "ensemble_mean.py": [4.0, 3.0], "ensemble_median.py": [3.0, 3.0]}
for i, row in enumerate(manifest["candidates"]): assert load(ROOT / row["forecast"], f"f{i}").forecast({"method_forecasts": forecasts}) == expected[row["forecast"]]
assert load(ROOT / manifest["retrieval"], "r").retrieve({}) == []
assert load(ROOT / manifest["decision"], "a").adjust({"base_forecast": [1.0, 2.0]}) == [1.0, 2.0]
src = (HERE / "final_select.py").read_text()
assert "cands = list(fixed)" in src and "fixed_candidates=fixed_provenance" in src
assert "if fixed_after != fixed_provenance" in src
print("fixed candidate contract PASS (6 host-pinned candidates)")
