"""LLM-as-designer: let an LLM WRITE new rules (regime estimators) as code, run them in
a locked-down sandbox, score them offline, keep the best, and iterate.

This lifts evolution from "turn the knobs I predefined" to "invent new knobs": the LLM
proposes a pure function

    def estimate(history_values, history_timestamps, future_timestamps) -> list[float] | None

returning a per-future-step multiplicative factor (vs the base level), or None when no
regime is observable. The function is AST-audited (no imports/exec/dunder/attributes to
escape the sandbox) and executed with an empty builtins namespace plus a small set of
safe helpers. Whatever it returns is still wrapped by the kernel downstream (grounded +
bounded), so a designed rule can never emit an unsafe forecast. The data is the judge.
"""
from __future__ import annotations

import ast
from statistics import mean, median, pstdev
from typing import Callable, Optional

from .dsl import _parse

# ---- safe helpers the designed function may call (no imports needed) -------------
def _weekday(ts: str):
    d = _parse(str(ts))
    return d.weekday() if d else -1


def _hour(ts: str):
    d = _parse(str(ts))
    return d.hour if d else -1


def _dayofyear(ts: str):
    d = _parse(str(ts))
    return d.timetuple().tm_yday if d else -1


SAFE_HELPERS = {
    "min": min, "max": max, "sum": sum, "len": len, "abs": abs, "range": range,
    "sorted": sorted, "round": round, "float": float, "int": int, "bool": bool,
    "enumerate": enumerate, "zip": zip, "list": list, "tuple": tuple, "map": map,
    "filter": filter, "mean": mean, "median": median, "pstdev": pstdev,
    "weekday": _weekday, "hour": _hour, "dayofyear": _dayofyear,
}

_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.While, ast.With, ast.Try, ast.Global,
    ast.Nonlocal, ast.ClassDef, ast.Delete, ast.Lambda, ast.Attribute,
    getattr(ast, "AsyncFunctionDef", ast.FunctionDef), getattr(ast, "Await", ast.expr),
)
_FORBIDDEN_CALLS = {"exec", "eval", "compile", "open", "__import__", "globals",
                    "locals", "getattr", "setattr", "delattr", "vars", "input", "help"}


def audit(code: str) -> ast.Module:
    """Raise ValueError if the source uses anything outside the safe subset."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name access")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _FORBIDDEN_CALLS:
            raise ValueError(f"forbidden call: {node.func.id}")
    return tree


def safe_compile(code: str, func_name: str = "estimate") -> Callable:
    """AST-audit and compile the source; return the callable in a locked namespace."""
    tree = audit(code)
    ns: dict = dict(SAFE_HELPERS)
    ns["__builtins__"] = {}
    exec(compile(tree, "<llm-designed>", "exec"), ns)  # noqa: S102 (sandboxed)
    fn = ns.get(func_name)
    if not callable(fn):
        raise ValueError(f"no `{func_name}` function defined")
    return fn


def make_regime_estimator(fn: Callable):
    """Wrap an LLM `estimate(hv,hts,fts)` into the _REGIMES estimator signature.

    Returns a callable est(hv, hts, effect, fts, mask) -> per-step factor tuple | None,
    defensively: any error, wrong length, or non-finite output collapses to None (drop).
    """
    def est(hv, hts, effect, fts, mask):
        try:
            out = fn(list(hv), [str(x) for x in hts], [str(x) for x in fts])
        except Exception:
            return None
        if not isinstance(out, (list, tuple)) or len(out) != len(fts):
            return None
        try:
            vals = [float(x) for x in out]
        except (TypeError, ValueError):
            return None
        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
            return None
        return tuple(vals[i] if m else 1.0 for i, m in enumerate(mask))
    return est


def make_window_scaler(fn: Callable):
    """Wrap an LLM `estimate(hv,hts,fts,window_mask)` into a defensive per-step scaler.

    The function is told which future steps are the document-localized event window and
    must estimate the multiplicative scale there. Any error / wrong length / non-finite
    output collapses to None (drop). Downstream still bounds the result.
    """
    def scaler(hv, hts, fts, mask):
        try:
            out = fn(list(hv), [str(x) for x in hts], [str(x) for x in fts], tuple(bool(m) for m in mask))
        except Exception:
            return None
        if not isinstance(out, (list, tuple)) or len(out) != len(fts):
            return None
        try:
            vals = [float(x) for x in out]
        except (TypeError, ValueError):
            return None
        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
            return None
        return tuple(vals)
    return scaler


def design_loop(proposer, evaluate, *, wrap: Callable = make_regime_estimator,
                rounds: int = 8, keep: int = 4):
    """Run the propose→sandbox→score→archive loop.

    proposer(archive) -> source code string (archive = list of (score, code) best-first).
    evaluate(estimator_fn) -> float score (higher is better).
    Returns (archive, log) where log records each round's outcome.
    """
    archive: list[tuple[float, str]] = []
    log: list[dict] = []
    for r in range(rounds):
        code = proposer(list(archive))
        rec: dict = {"round": r}
        try:
            fn = safe_compile(code)
            est = wrap(fn)
            score = float(evaluate(est))
            rec.update(status="ok", score=score)
            archive.append((score, code))
            archive.sort(key=lambda t: t[0], reverse=True)
            del archive[keep:]
        except Exception as exc:  # rejected by sandbox or crashed/scored invalid
            rec.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
        log.append(rec)
    return archive, log
