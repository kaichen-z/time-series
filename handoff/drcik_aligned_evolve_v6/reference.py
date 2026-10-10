"""Protocol v6 host-side reference baseline.  The frozen reference forecaster (reference/reference_forecast.py, selected on
official Train only by rolling-origin CV) replaces Toto as the per-task baseline of the evolution score:
    gain_i = jt(reference_i) - jt(program_i)
reference_jt(store_dir, truth_by_key) -> {key: joint error of the reference}.  reference_sha256() pins the file."""
import hashlib, importlib.util, math
from pathlib import Path

import viewstore

HERE = Path(__file__).resolve().parent
REF = HERE / "reference/reference_forecast.py"


def reference_sha256(): return hashlib.sha256(REF.read_bytes()).hexdigest()


def _load():
    s = importlib.util.spec_from_file_location("v6_reference", REF); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def jt(f, y):
    sc = sum(abs(x) for x in y) / len(y) + 1e-12
    mae = sum(abs(a - b) for a, b in zip(f, y)) / len(y); rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(f, y)) / len(y))
    return min(5.0, mae / sc) + min(5.0, rmse / sc)


def reference_jt(store_dir, truth_by_key):
    m = _load(); S = viewstore.Store(store_dir); out = {}
    for k in truth_by_key:
        f = m.forecast(S.view(k)); y = truth_by_key[k]
        assert len(f) == len(y) and all(math.isfinite(x) for x in f), f"reference produced an invalid forecast for {k}"
        out[k] = jt(f, y)
    return out
