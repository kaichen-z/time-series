"""Parse, validate, and restructure the evolving methods module."""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from common.sandbox import ALLOWED_IMPORTS, check_code


# The zero-shot foundation-model adapters, allowed by exact dotted name so the rest of the
# repository stays out of reach.
TSFM_IMPORT_PATH = "common.tsfm"

# The evolving module may use heavier forecasting stacks than evolving_loop's sandbox allows.
ALLOWED_METHOD_IMPORTS = ALLOWED_IMPORTS | frozenset(
    {"torch", "sklearn", "scipy", "lightgbm", "xgboost", "pandas", TSFM_IMPORT_PATH}
)

# torch.nn.Module subclasses cannot be written without super().__init__(); the escape-hatch
# dunders (__globals__, __builtins__, __class__) stay blocked.
ALLOWED_DUNDERS = frozenset({"__init__"})


# No __future__ import: the shared sandbox gate allows only forecasting libraries.
#
# The header is the only module-level text that survives render(), so NotApplicable is defined
# here: a definition written anywhere else is silently dropped on the next write. Every method
# is self-contained, so there is nothing else to import at module level.
METHODS_FILE_HEADER = '''"""Self-contained forecasting methods, one function per method.

Each function takes (history, horizon, frequency) and returns exactly horizon finite floats,
or raises NotApplicable when the series does not meet its stated requirements.
"""


class NotApplicable(Exception):
    """Raised by a method whose stated preconditions the series does not meet."""
'''

SIGNATURE = ("history", "horizon", "frequency")
OPERATIONS = ("delete", "rewrite", "merge")


class ModuleError(ValueError):
    """Raised when a module or a proposed operation violates the method contract."""


@dataclass(frozen=True)
class Method:
    """One forecasting function: its name, its full source, and its docstring."""

    name: str
    source: str
    docstring: str


@dataclass(frozen=True)
class MethodModule:
    """The evolving module as an ordered, name-addressable set of methods."""

    methods: tuple[Method, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(method.name for method in self.methods)

    def get(self, name: str) -> Method | None:
        return next((method for method in self.methods if method.name == name), None)

    def render(self) -> str:
        """Render the complete module text, header first."""
        blocks = [METHODS_FILE_HEADER.rstrip("\n")]
        blocks.extend(method.source.strip("\n") for method in self.methods)
        return "\n\n\n".join(blocks) + "\n"


def parse_module(text: str) -> MethodModule:
    """Read module text into methods, rejecting anything violating the contract."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ModuleError(f"module does not parse: {exc}") from exc

    lines = text.splitlines()
    methods: list[Method] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        _validate_function(node)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        source = "\n".join(lines[start : node.end_lineno])
        methods.append(
            Method(node.name, source, ast.get_docstring(node) or "")
        )

    names = [method.name for method in methods]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ModuleError(f"duplicate method names: {duplicates}")
    return MethodModule(tuple(methods))


def parse_method(source: str, expected_name: str | None = None) -> Method:
    """Read a single function definition, optionally requiring a specific name."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ModuleError(f"method does not parse: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        raise ModuleError(
            "code must be the complete function source starting with 'def', not a placeholder"
        )
    if len(functions) != 1:
        raise ModuleError(f"expected exactly one function definition, found {len(functions)}")
    node = functions[0]
    _validate_function(node)
    if expected_name is not None and node.name != expected_name:
        raise ModuleError(f"function is named {node.name!r}, expected {expected_name!r}")
    return Method(node.name, source.strip("\n"), ast.get_docstring(node) or "")


def apply_operations(
    module: MethodModule, operations: Sequence[Mapping[str, object]]
) -> tuple[MethodModule, tuple[str, ...]]:
    """Apply every operation in order, returning the new module and one summary per operation.

    Raises before changing anything if any operation is malformed, so a bad response from the
    model can never leave the module half-rewritten.
    """
    current = module
    original = frozenset(module.names())
    summaries: list[str] = []
    for index, operation in enumerate(operations, start=1):
        try:
            current, summary = _apply_one(current, operation, original)
        except ModuleError as exc:
            raise ModuleError(f"operation {index}: {exc}") from exc
        summaries.append(summary)
    return current, tuple(summaries)


def _apply_one(
    module: MethodModule, operation: Mapping[str, object], original: frozenset[str] = frozenset()
) -> tuple[MethodModule, str]:
    op = str(operation.get("op", ""))
    if op not in OPERATIONS:
        raise ModuleError(f"unsupported op {op!r}; expected one of {list(OPERATIONS)}")
    reason = str(operation.get("reason", "")).strip()
    if not reason:
        raise ModuleError(f"{op} must state a reason")

    if op == "delete":
        name = str(operation.get("name") or "").strip()
        if not name:
            raise ModuleError("a method name is required")
        if module.get(name) is None:
            # Tolerated only if an earlier operation in this batch consumed it: the requested
            # end state already holds. A name that never existed is still a hallucination.
            if name not in original:
                raise ModuleError(f"unknown method {name!r}")
            return module, f"delete {name}: already removed earlier in this batch; {reason}"
        methods = tuple(m for m in module.methods if m.name != name)
        if not methods:
            raise ModuleError("refusing to delete the last remaining method")
        return replace(module, methods=methods), f"delete {name}: {reason}"

    if op == "rewrite":
        name = _require_existing(module, operation.get("name"))
        method = parse_method(_require_code(operation), expected_name=name)
        methods = tuple(method if m.name == name else m for m in module.methods)
        return replace(module, methods=methods), f"rewrite {name}: {reason}"

    raw_names = operation.get("names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ModuleError("merge requires a list of names")
    requested = tuple(_require_named(module, name, original) for name in raw_names)
    if len(requested) < 2:
        raise ModuleError("merge requires at least two methods")
    # Sources an earlier operation in this batch already consumed are skipped, not fatal.
    sources = tuple(name for name in requested if module.get(name) is not None)
    into = str(operation.get("into", "")).strip()
    if not into:
        raise ModuleError("merge requires an 'into' name")
    if into not in sources and module.get(into) is not None:
        raise ModuleError(f"{into!r} already exists and is not one of the merged methods")
    method = parse_method(_require_code(operation), expected_name=into)
    kept = tuple(m for m in module.methods if m.name not in sources)
    if not kept and not method:
        raise ModuleError("merge would empty the module")
    return (
        replace(module, methods=kept + (method,)),
        f"merge {', '.join(requested)} -> {into}: {reason}",
    )


def _require_existing(module: MethodModule, name: object) -> str:
    text = str(name or "").strip()
    if not text:
        raise ModuleError("a method name is required")
    if module.get(text) is None:
        raise ModuleError(f"unknown method {text!r}")
    return text


def _require_named(module: MethodModule, name: object, original: frozenset[str]) -> str:
    """Accept a name the module still has, or one an earlier operation in this batch consumed."""
    text = str(name or "").strip()
    if not text:
        raise ModuleError("a method name is required")
    if module.get(text) is None and text not in original:
        raise ModuleError(f"unknown method {text!r}")
    return text


def _require_code(operation: Mapping[str, object]) -> str:
    code = operation.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ModuleError("a non-empty 'code' string is required")
    return code


def _validate_function(node: ast.FunctionDef) -> None:
    """Enforce the shared contract: exact signature, a docstring, no decorators."""
    args = node.args
    names = tuple(arg.arg for arg in args.args)
    if names != SIGNATURE or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise ModuleError(
            f"{node.name} must take exactly {SIGNATURE}, got {names or '()'}"
        )
    if node.decorator_list:
        raise ModuleError(f"{node.name} must not be decorated")
    if not (ast.get_docstring(node) or "").strip():
        raise ModuleError(f"{node.name} must have a docstring saying when to use it")
    _reject_silenced_failures(node)


def _reject_silenced_failures(node: ast.FunctionDef) -> None:
    """Forbid the two ways a method can hide its own defects instead of reporting them.

    Converting a caught exception into NotApplicable disguises a bug as inapplicability, and
    returning from a broad handler is the fallback that makes broken code look like it works.
    """
    for handler in (n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)):
        for inner in ast.walk(handler):
            if isinstance(inner, ast.Raise) and _raised_name(inner) == "NotApplicable":
                raise ModuleError(
                    f"{node.name} raises NotApplicable from an except handler; check the "
                    "precondition before running instead of silencing a failure"
                )
            if isinstance(inner, ast.Return) and _is_broad(handler):
                raise ModuleError(
                    f"{node.name} returns a forecast from a broad except handler; no fallbacks"
                )


def _raised_name(node: ast.Raise) -> str:
    exc = node.exc
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id
    return exc.id if isinstance(exc, ast.Name) else ""


def _is_broad(handler: ast.ExceptHandler) -> bool:
    return handler.type is None or (
        isinstance(handler.type, ast.Name) and handler.type.id in ("Exception", "BaseException")
    )


def write_module(path: str | Path, module: MethodModule) -> Path:
    """Write the module only after it re-parses and clears the shared import gate."""
    text = module.render()
    check_code(text, ALLOWED_METHOD_IMPORTS, ALLOWED_DUNDERS)
    reparsed = parse_module(text)
    if reparsed.names() != module.names():
        raise ModuleError("module did not survive a render/parse round trip")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def read_module(path: str | Path) -> MethodModule:
    """Read and validate an existing methods module."""
    return parse_module(Path(path).read_text(encoding="utf-8"))
