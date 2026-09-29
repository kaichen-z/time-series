"""Seed for round 2 = the current main correction step: route each task by cell to one of five evolved
correction functions (their code is readable under FUNC_DIR/<name>/best_correction_function.py).
You may rewrite the routing, edit / merge the functions, or write something new. adjust(view) -> H floats."""
import importlib.util
from pathlib import Path
FUNC_DIR = Path(__file__).resolve().parents[2] / "artifacts/meta_harness"
ROUTE = {"day|flat": "single", "day|seas": "single", "minute|flat": "single", "minute|seas": "single",
         "hour|flat": "indep1", "hour|seas": "indep3", "second|flat": "shared3"}
DEFAULT = "shared3"; _MODS = {}
def _load(name):
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(f"corr_{name}", FUNC_DIR / name / "best_correction_function.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _MODS[name] = m
    return _MODS[name]
def adjust(view):
    return _load(ROUTE.get(view["cell"], DEFAULT)).adjust(view)
