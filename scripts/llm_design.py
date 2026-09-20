"""LLM proposer for NEW retrieval features (the 'proposer' half of DSL auto-growth).

The human writes the SEED feature vocabulary (scripts/evolve_retrieval.py). This module lets an
LLM ADD new candidate features: it writes each feature as Python source `def f(task, doc) -> float`,
`safe_compile` turns that source into a validated, sandboxed function, and grow_retrieval_dsl.py
runs the SAME keep-if-it-helps loop (add -> re-evolve weights -> keep only if held-out dev F1
improves). So the LLM only PROPOSES; the auditable evolutionary loop decides what survives, and
every kept feature is still readable source. No LLM runs at selection time.

safe_compile is a real sandbox, not decoration: it AST-checks the source (single def, exactly the
(task, doc) signature, no imports, no dunder, no eval/exec/open/getattr..., and crucially no access
to `doc.role`/`d.role` which is the supervision LABEL -> would be leakage), execs it with a
restricted builtins namespace, and smoke-tests it on sample rows before it is allowed near the loop.
"""
from __future__ import annotations
import ast
import math
import re

# helpers a feature may use (same ones the seed features use)
_TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(_TOK.findall((s or "").lower()))


def _extract_items(text: str, needed=("name", "source")) -> list:
    """Pull every {name, source} item out of an LLM reply, tolerant of prose, ```json fences, and a
    malformed OUTER wrapper (models often forget the closing ']' / ','). We scan every balanced
    '{...}' region and keep the ones that parse on their own and carry all `needed` keys -- so the
    well-formed inner objects survive even when the surrounding array is broken. Deduped by name."""
    from common.llm import parse_json_object
    items, seen = [], set()
    for s in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        for e in range(s, len(text)):
            depth += (text[e] == "{") - (text[e] == "}")
            if depth == 0:
                try:
                    o = parse_json_object(text[s:e + 1])
                    if isinstance(o, dict) and all(k in o for k in needed) and o["name"] not in seen:
                        seen.add(o["name"]); items.append(o)
                except Exception:
                    pass
                break
    return items


_SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "abs": abs, "sum": sum, "round": round,
    "float": float, "int": int, "str": str, "bool": bool, "set": set, "list": list,
    "range": range, "any": any, "all": all, "sorted": sorted, "enumerate": enumerate,
    "map": map, "zip": zip, "True": True, "False": False, "None": None,
}
_SAFE_GLOBALS = {"re": re, "math": math, "toks": toks}

# names/attributes a feature source may never contain
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr",
                    "delattr", "globals", "locals", "vars", "input", "exit", "quit", "help"}
_FORBIDDEN_ATTRS = {"role"}          # doc.role is the LABEL -> forbid to prevent leakage


class UnsafeFeatureError(ValueError):
    """Raised when proposed feature source violates the sandbox contract."""


def _check_ast(name: str, src: str, expected_args=("task", "doc")) -> ast.FunctionDef:
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or not funcs:
        raise UnsafeFeatureError("source must be exactly one function definition")
    fn = funcs[0]
    args = [a.arg for a in fn.args.args]
    if args != list(expected_args):
        raise UnsafeFeatureError(f"signature must be {tuple(expected_args)}, got {tuple(args)}")
    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeFeatureError("imports are not allowed")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            raise UnsafeFeatureError(f"attribute '.{node.attr}' is forbidden (label leakage)")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in _FORBIDDEN_CALLS):
            raise UnsafeFeatureError(f"name '{node.id}' is forbidden")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeFeatureError("dunder attribute access is forbidden")
    return fn


def safe_compile(name: str, src: str):
    """Validate + compile feature source into a sandboxed fn(task, doc)->float.
    The returned fn catches its own per-call errors and returns 0.0, so one bad document
    can never crash the evolution loop. Raises UnsafeFeatureError if the source is unsafe."""
    _check_ast(name, src)
    ns: dict = {"__builtins__": _SAFE_BUILTINS, **_SAFE_GLOBALS}
    try:
        exec(compile(src, f"<feat:{name}>", "exec"), ns)          # noqa: S102 (sandboxed ns)
    except Exception as e:                                          # pragma: no cover
        raise UnsafeFeatureError(f"compile/exec failed: {e}")
    raw = ns.get(name) or next((v for v in ns.values() if callable(v) and getattr(v, "__name__", "") not in ("toks",)), None)
    if raw is None:
        raise UnsafeFeatureError("no function defined")

    def wrapped(task, doc, _raw=raw):
        try:
            v = float(_raw(task, doc))
            return v if math.isfinite(v) else 0.0
        except Exception:
            return 0.0
    wrapped.__name__ = name
    wrapped.source = src
    return wrapped


def validate_on(fn, sample_tasks, k_docs: int = 5) -> bool:
    """Smoke-test: the feature must run and produce finite floats on real rows, and must not be
    a constant (a constant feature carries no signal)."""
    vals = []
    for t in sample_tasks:
        for d in list(t.documents)[:k_docs]:
            v = fn(t, d)
            if not isinstance(v, float) or not math.isfinite(v):
                return False
            vals.append(v)
    return len(set(round(v, 6) for v in vals)) > 1     # not constant


# ---------------------------------------------------------------------------
# LLM-authored CORRECTION PRIMITIVES for Loop B (the "LLM writes the code" path).
# A primitive decides HOW to turn document-derived per-window multipliers into a
# corrected forecast. It CANNOT bypass safety: the caller forces every step outside a
# grounded correction window back to the base, then clamps to base +/- 50% (the fixed
# kernel). So an LLM primitive can only reshape WITHIN grounded windows, boundedly --
# a bad one can do no harm, and only CV-validated ones are kept.
# ---------------------------------------------------------------------------

def safe_compile_primitive(name: str, src: str):
    """Validate + compile a Loop-B correction primitive:
        def <name>(base, history, corrections) -> list[float]
      base        : list[float], the base forecast (length H)
      history     : list[float], past observed values (no future -> no leakage possible)
      corrections : list of [start_idx, end_idx, mult] resolved to horizon indices
    Returns a sandboxed fn (own errors -> returns base). Raises UnsafeFeatureError if unsafe."""
    _check_ast(name, src, expected_args=("base", "history", "corrections"))
    ns: dict = {"__builtins__": _SAFE_BUILTINS, **_SAFE_GLOBALS}
    try:
        exec(compile(src, f"<prim:{name}>", "exec"), ns)          # noqa: S102 (sandboxed ns)
    except Exception as e:                                          # pragma: no cover
        raise UnsafeFeatureError(f"compile/exec failed: {e}")
    raw = ns.get(name)
    if not callable(raw):
        raise UnsafeFeatureError("no function defined")

    def wrapped(base, history, corrections, _raw=raw):
        try:
            out = _raw(list(base), list(history), [list(c) for c in corrections])
            out = [float(x) for x in out]
            if len(out) != len(base) or not all(math.isfinite(x) for x in out):
                return list(base)
            return out
        except Exception:
            return list(base)
    wrapped.__name__ = name
    wrapped.source = src
    return wrapped


def safe_compile_clue(name: str, src: str):
    """Validate + compile a recall-gate CLUE (a scalar trust feature):
        def <name>(base, history, corrections, conf) -> float
      base/history : list[float]   corrections : list of [start_idx, end_idx, mult]   conf : float
    Returns a sandboxed fn (own errors / non-finite -> 0.0)."""
    _check_ast(name, src, expected_args=("base", "history", "corrections", "conf"))
    ns: dict = {"__builtins__": _SAFE_BUILTINS, **_SAFE_GLOBALS}
    try:
        exec(compile(src, f"<clue:{name}>", "exec"), ns)          # noqa: S102 (sandboxed ns)
    except Exception as e:                                          # pragma: no cover
        raise UnsafeFeatureError(f"compile/exec failed: {e}")
    raw = ns.get(name)
    if not callable(raw):
        raise UnsafeFeatureError("no function defined")

    def wrapped(base, history, corrections, conf, _raw=raw):
        try:
            v = float(_raw(list(base), list(history), [list(c) for c in corrections], float(conf)))
            return v if math.isfinite(v) else 0.0
        except Exception:
            return 0.0
    wrapped.__name__ = name
    wrapped.source = src
    return wrapped


_SYSTEM_CLUE = (
    "You invent numeric CLUES that decide whether to TRUST a document-derived correction to a "
    "time-series forecast. A strong base forecast exists; a document implied per-window multipliers; "
    "sometimes trusting them helps, sometimes it harms (wrong magnitude / a confounding document). "
    "Each clue is a scalar computed from what is visible at inference. Output ONE JSON object only, "
    "no markdown fences, no prose. Keep each function short."
)

_CLUE_INTERFACE = """Each clue is EXACTLY:
    def <name>(base, history, corrections, conf) -> float
  base        : list[float]  the base forecast (length H)
  history     : list[float]  past observed values (may be empty)
  corrections : list of [start_idx, end_idx, mult]  (the document's per-window multipliers)
  conf        : float        the LLM's self-reported confidence (known to be poorly calibrated)
Return a single finite float (ideally roughly bounded, e.g. [-1,1] or [0,1]). Helpers: math.
Allowed builtins: len,min,max,abs,sum,round,float,int,range,any,all,sorted,enumerate,zip,list.
No imports, no eval/exec/getattr, no dunder. You cannot see the future or the truth. Think of signals
like: correction magnitude vs history volatility, whether the correction direction agrees with the
recent history slope, how localized the windows are, multi-window agreement, etc."""


def propose_clues(client, existing_names, n: int = 6):
    """Ask the LLM for n new recall-gate clues. Returns {name: fn} of the ones that safe_compile."""
    user = (
        f"{_CLUE_INTERFACE}\n\nExisting clues (do NOT duplicate): {sorted(existing_names)}\n\n"
        f"Propose {n} NEW, structurally distinct clues for separating 'trusting the correction helps' "
        f"from 'it harms'. Return JSON:\n"
        '{"clues": [{"name": "snake_case", "source": "def snake_case(base, history, corrections, conf):\\n    ..."}]}'
    )
    resp = client.complete(system=_SYSTEM_CLUE, messages=[{"role": "user", "content": user}], temperature=0.5)
    out = {}
    for item in _extract_items(resp.text):
        name, srcc = item.get("name"), item.get("source")
        if not name or not srcc or name in existing_names or name in out:
            continue
        try:
            out[name] = safe_compile_clue(name, srcc)
        except UnsafeFeatureError:
            continue
    return out


_SYSTEM_PRIM = (
    "You write a small, pure Python CORRECTION PRIMITIVE for a time-series forecasting harness. "
    "A strong base forecast already exists; documents have been read into per-window multipliers "
    "(start_idx, end_idx, mult). Your function decides HOW to apply them -- e.g. flat scaling, a "
    "ramp into the window, decay after an event, partial trust, tapering edges. The harness forces "
    "everything outside the given windows back to base and clamps to base +/-50%, so you only shape "
    "WITHIN windows. Output ONE JSON object and NOTHING else: no markdown fences, no prose before "
    "or after, no explanation. Keep each function short so the reply is never truncated."
)

_PRIM_INTERFACE = """Each primitive is EXACTLY:
    def <name>(base, history, corrections) -> list
  base        : list[float]  the base forecast, length H (index 0..H-1)
  history     : list[float]  past observed values (may be empty)
  corrections : list of [start_idx, end_idx, mult]  (integer horizon indices, inclusive start,
                exclusive end; mult is the document-implied multiplier for that window)
Return a NEW list of length H: the corrected forecast. Typically start from base[:] and modify only
indices inside a window. Helpers in scope: math. Allowed builtins: len,min,max,abs,sum,round,float,
int,range,any,all,sorted,enumerate,zip,list. HARD RULES: no imports; no eval/exec/open/getattr; no
dunder; length must stay H; every value finite. You cannot see the future or the truth."""


def propose_primitives(client, existing_names, n: int = 4):
    """Ask the LLM for n new Loop-B correction primitives. Returns {name: fn} of the ones that
    safe_compile. The keep-if-CV-helps loop (evolve_primitives.py) decides which actually survive."""
    from common.llm import parse_json_object
    user = (
        f"{_PRIM_INTERFACE}\n\nExisting primitives (do NOT duplicate): {sorted(existing_names)}\n\n"
        f"Propose {n} NEW, structurally distinct primitives for applying document multipliers to a "
        f"forecast (think: shapes of how a real event moves a series -- abrupt step, gradual ramp, "
        f"spike then decay, partial/shrunk trust, edge tapering). Return JSON:\n"
        '{"primitives": [{"name": "snake_case", "source": "def snake_case(base, history, corrections):\\n    ..."}]}'
    )
    resp = client.complete(system=_SYSTEM_PRIM, messages=[{"role": "user", "content": user}], temperature=0.5)
    out = {}
    for item in _extract_items(resp.text):
        name, srcp = item.get("name"), item.get("source")
        if not name or not srcp or name in existing_names or name in out:
            continue
        try:
            out[name] = safe_compile_primitive(name, srcp)
        except UnsafeFeatureError:
            continue
    return out


_SYSTEM = (
    "You design numeric FEATURES for an auditable document-relevance classifier in a time-series "
    "forecasting harness. Each feature scores how likely a document is a SUPPORTING document (a "
    "genuine causal driver of the target series) versus a DISTRACTOR (on-topic but causally "
    "irrelevant). You return only compact, pure Python source. Output JSON only."
)

_INTERFACE = """Each feature is EXACTLY:
    def <name>(task, doc) -> float
Available read-only attributes:
    task.numeric.entity_name : str      task.target_name : str      task.target_description : str
    task.future_timestamps   : list     (stringify to read years)   doc.content : str
Helpers in scope: re, math, and toks(s)->set(lowercased word tokens). Allowed builtins: len,min,max,
abs,sum,round,float,int,str,set,list,range,any,all,sorted,enumerate,zip.
HARD RULES: no imports; no eval/exec/open/getattr; no dunder; return one finite float (a bounded
score, e.g. in [-1,1] or [0,1]); recompute purely from task/doc. NEVER read doc.role or doc.subtype
(those are hidden labels -- using them is cheating and will be rejected)."""


def propose_features(client, existing_names, examples, n: int = 4):
    """Ask the LLM for n new feature functions. Returns {name: fn} of the ones that safe_compile.
    `examples` is a short list of (content_snippet, is_supporting) to ground the proposals."""
    ex = "\n".join(f"  - [{'SUPPORTING' if s else 'distractor'}] {c[:220]!r}" for c, s in examples[:12])
    user = (
        f"{_INTERFACE}\n\nExisting features (do NOT duplicate): {sorted(existing_names)}\n\n"
        f"Example documents and their true role (for intuition only -- you cannot read role at "
        f"runtime):\n{ex}\n\nPropose {n} NEW, structurally distinct features that would help "
        f"separate supporting docs from distractors on THIS kind of data. Return JSON:\n"
        '{"features": [{"name": "snake_case", "source": "def snake_case(task, doc):\\n    ..."}]}'
    )
    resp = client.complete(system=_SYSTEM, messages=[{"role": "user", "content": user}], temperature=0.4)
    out = {}
    for item in _extract_items(resp.text):
        name, src = item.get("name"), item.get("source")
        if not name or not src or name in existing_names or name in out:
            continue
        try:
            out[name] = safe_compile(name, src)
        except UnsafeFeatureError:
            continue                                   # silently skip unsafe/invalid proposals
    return out
