"""Static auditing and subprocess execution for a policy source variant."""
from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import tempfile

from common.payload import strict_json_loads

from ..contracts import canonical_v2_bytes
from .contracts import SourceRequestV2, SourceVariantV2


_MAX_MESSAGE_BYTES = 16384
_MAX_TIMEOUT_SECONDS = 2.0
_WORKER = r'''
import json
import sys

payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
audited_source = payload["source"]
request_payload = payload["request"]
scope = {"__builtins__": {}}
exec(compile(audited_source, "policy.py", "exec"), scope)
answer = scope["choose_arm"](request_payload)
sys.stdout.write(json.dumps(answer, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
'''


class _PolicyGrammar(ast.NodeVisitor):
    """Reject everything except the deliberately small policy expression grammar."""

    _binary = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    _unary = (ast.UAdd, ast.USub, ast.Not)
    _compare = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def _reject(self, node: ast.AST, message: str) -> None:
        raise ValueError(f"policy.py:{getattr(node, 'lineno', '?')}: {message}")

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            self._reject(node, "source must define exactly one choose_arm function")
        self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        arguments = node.args
        if (
            node.name != "choose_arm"
            or node.decorator_list
            or getattr(node, "type_params", ())
            or node.returns is not None
            or arguments.posonlyargs
            or len(arguments.args) != 1
            or arguments.args[0].arg != "request"
            or arguments.args[0].annotation is not None
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
            or not node.body
        ):
            self._reject(node, "choose_arm must have exactly one unannotated request argument")
        for statement in node.body:
            self.visit(statement)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self._reject(node, "assignment target must be one local name")
        if node.targets[0].id == "request" or _is_dunder(node.targets[0].id):
            self._reject(node, "assignment cannot target request or a dunder name")
        self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        if not node.body:
            self._reject(node, "if body must not be empty")
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            self._reject(node, "return must have a value")
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load) or _is_dunder(node.id):
            self._reject(node, "only non-dunder names may be read")

    def visit_Constant(self, node: ast.Constant) -> None:
        if type(node.value) not in (str, int, float, bool, type(None)):
            self._reject(node, "constant is not allowed")
        if type(node.value) is float and not math.isfinite(node.value):
            self._reject(node, "non-finite constant is not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.ctx, ast.Load):
            self._reject(node, "subscript must be read-only")
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self._binary):
            self._reject(node, "arithmetic operator is not allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self._unary):
            self._reject(node, "unary operator is not allowed")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, (ast.And, ast.Or)):
            self._reject(node, "boolean operator is not allowed")
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(not isinstance(operator, self._compare) for operator in node.ops):
            self._reject(node, "comparison operator is not allowed")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def generic_visit(self, node: ast.AST) -> None:
        self._reject(node, f"{type(node).__name__} is not allowed")


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def audit_source(variant: SourceVariantV2) -> None:
    """Verify that a variant has precisely the bounded candidate grammar."""
    if not isinstance(variant, SourceVariantV2):
        raise TypeError("variant must be a SourceVariantV2")
    try:
        parsed = ast.parse(variant.source, filename="policy.py", mode="exec")
    except SyntaxError as error:
        raise ValueError(f"policy.py has invalid syntax: {error.msg}") from error
    _PolicyGrammar().visit(parsed)


def _timeout(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("timeout_seconds must be a finite number")
    timeout = float(value)
    if not 0.0 < timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds must be in (0, 2]")
    return timeout


def run_policy(
    variant: SourceVariantV2, request: SourceRequestV2, *, timeout_seconds: float = 2.0
) -> str:
    """Run one audited policy with JSON-only data across a fresh subprocess boundary."""
    if not isinstance(request, SourceRequestV2):
        raise TypeError("request must be a SourceRequestV2")
    audit_source(variant)
    timeout = _timeout(timeout_seconds)
    message = canonical_v2_bytes({"source": variant.source, "request": request.to_payload()})
    if len(message) > _MAX_MESSAGE_BYTES:
        raise ValueError("worker request exceeds 16384 bytes")
    environment = {"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"}
    try:
        with tempfile.TemporaryDirectory(prefix="evolution-v2-policy-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _WORKER],
                input=message,
                capture_output=True,
                timeout=timeout,
                cwd=directory,
                env=environment,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise ValueError("policy worker timed out") from error
    except OSError as error:
        raise ValueError("policy worker could not start") from error
    if completed.returncode != 0:
        raise ValueError("policy worker failed")
    if len(completed.stdout) > _MAX_MESSAGE_BYTES:
        raise ValueError("policy worker response exceeds 16384 bytes")
    try:
        answer = strict_json_loads(completed.stdout.decode("utf-8"), context="policy worker response")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("policy worker returned invalid JSON") from error
    if type(answer) is not str:
        raise ValueError("policy worker must return a JSON string")
    if answer not in request.enabled_arms:
        raise ValueError("policy worker returned an arm outside enabled arms")
    return answer
