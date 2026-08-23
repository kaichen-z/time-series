"""Hard semantic-identity contracts for method-module evolution."""
from __future__ import annotations

import ast
from dataclasses import dataclass


class IdentityError(ValueError):
    """Raised when a repair substitutes a different forecasting algorithm."""


@dataclass(frozen=True)
class IdentityComponent:
    """One required mathematical component and source markers that can evidence it."""

    label: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class IdentityContract:
    """The components a same-name repair must preserve."""

    method: str
    components: tuple[IdentityComponent, ...]

    def payload(self) -> dict[str, object]:
        repair_allowed = bool(self.components and self.method in _STRUCTURAL_VALIDATORS)
        return {
            "method": self.method,
            "repair_allowed": repair_allowed,
            "mode": "repair" if repair_allowed else "fork_only",
            "required_components": [
                {"label": component.label, "markers": list(component.markers)}
                for component in self.components
            ],
            "rule": (
                "A repair must preserve every component and feed the forecast output."
                if repair_allowed
                else "No verified repair contract exists; keep this method unchanged and use fork."
            ),
        }


_CURATED: dict[str, tuple[IdentityComponent, ...]] = {
    "sarima_auto": (
        IdentityComponent("SARIMA model", ("sarima", "sarimax")),
        IdentityComponent("seasonal order", ("seasonal_order",)),
        IdentityComponent("model selection", ("aic", "bic", "information_criterion")),
    ),
    "negative_binomial_dynamic_regression": (
        IdentityComponent("negative-binomial likelihood", ("gammaln", "nbinom", "negativebinomial")),
        IdentityComponent("dispersion parameter", ("dispersion", "alpha", "size")),
        IdentityComponent("parameter fitting", ("minimize", "fit")),
    ),
    "smooth_transition_autoregression": (
        IdentityComponent("smooth transition function", ("expit", "logistic", "transition")),
        IdentityComponent("transition threshold", ("threshold",)),
        IdentityComponent("multiple autoregressive regimes", ("regime", "regime_difference")),
    ),
    "scinet": (
        IdentityComponent("interleaved neural representation", ("interact", "interaction")),
        IdentityComponent("learned convolution", ("conv1d", "convolve", "convolution")),
        IdentityComponent("trainable neural parameters", ("torch", "parameter", "optimizer")),
    ),
    "itransformer": (
        IdentityComponent("attention", ("attention",)),
        IdentityComponent("query representation", ("query", "queries")),
        IdentityComponent("key representation", ("key", "keys")),
    ),
    "ltsf_dlinear": (
        IdentityComponent("trend component", ("trend",)),
        IdentityComponent("remainder component", ("remainder", "residual")),
        IdentityComponent("moving-average decomposition", ("convolve", "moving_average")),
        IdentityComponent("separate component projections", ("weights_trend", "design_trend")),
    ),
}


def identity_contract(method: str, parent_source: str) -> IdentityContract:
    """Return a reviewed contract; unreviewed methods deliberately fail closed."""
    del parent_source
    if method in _CURATED:
        return IdentityContract(method, _CURATED[method])
    return IdentityContract(method, ())


def validate_repair(
    method: str,
    parent_source: str,
    child_source: str,
    contract: IdentityContract,
) -> None:
    """Reject a same-name repair that drops any required mathematical component."""
    if not contract.components:
        raise IdentityError(
            f"{method} has no verified identity contract; keep it unchanged and use fork "
            "with a new honest method name."
        )
    parent_tree = _function_tree(parent_source)
    tree = _function_tree(child_source)
    if _identity_skeleton(parent_tree) != _identity_skeleton(tree):
        raise IdentityError(
            f"{method} repair violates method identity: it changes the verified forecast "
            "output/control-flow structure "
            "or return data flow; use fork with a new honest method name."
        )
    validator = _STRUCTURAL_VALIDATORS.get(method)
    if validator is None:
        raise IdentityError(
            f"{method} has no structural identity validator; use fork with a new name."
        )
    missing = validator(tree)
    if missing:
        raise IdentityError(
            f"{method} repair violates method identity; missing: {', '.join(missing)}. "
            "Use fork with a new honest method name when changing the algorithm."
        )


def _function_tree(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise IdentityError("identity validation requires exactly one method function")
    return functions[0]


class _NormalizeConstants(ast.NodeTransformer):
    """Allow parameter/docstring tuning while freezing executable structure."""

    def visit_Constant(self, node: ast.Constant):  # noqa: N802 - ast visitor API
        kind = type(node.value).__name__
        return ast.copy_location(ast.Constant(value=f"<{kind}>"), node)


def _identity_skeleton(tree: ast.FunctionDef) -> str:
    normalized = _NormalizeConstants().visit(ast.fix_missing_locations(tree))
    assert isinstance(normalized, ast.FunctionDef)
    return ast.dump(normalized, include_attributes=False)


def _call_name(call: ast.Call) -> str:
    return _dotted_name(call.func)


def _calls(node: ast.AST, suffix: str) -> list[ast.Call]:
    wanted = suffix.casefold()
    return [
        call for call in ast.walk(node)
        if isinstance(call, ast.Call) and _call_name(call).casefold().endswith(wanted)
    ]


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _assignments(node: ast.AST, name: str) -> list[ast.AST]:
    values: list[ast.AST] = []
    for item in ast.walk(node):
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                values.append(item.value)
    return values


def _nested_function(node: ast.AST, name: str) -> ast.FunctionDef | None:
    return next(
        (item for item in ast.walk(node) if isinstance(item, ast.FunctionDef) and item.name == name),
        None,
    )


def _has_call_argument(calls: list[ast.Call], argument: str) -> bool:
    return any(
        call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == argument
        for call in calls
    )


def _sarima(tree: ast.FunctionDef) -> list[str]:
    models = _calls(tree, "SARIMAX") + _calls(tree, "SARIMA")
    missing = []
    if not models:
        missing.append("SARIMA model call")
    if not any(any(keyword.arg == "seasonal_order" for keyword in call.keywords) for call in models):
        missing.append("seasonal_order passed to the model")
    criteria = [
        item for item in ast.walk(tree)
        if isinstance(item, ast.Attribute) and item.attr.casefold() in {"aic", "bic"}
    ]
    if not criteria or not any(isinstance(item, (ast.For, ast.While)) for item in ast.walk(tree)):
        missing.append("automatic information-criterion model selection")
    if not _calls(tree, ".fit"):
        missing.append("fitted SARIMA result")
    if not (_calls(tree, ".forecast") or _calls(tree, ".get_forecast")):
        missing.append("SARIMA forecast call")
    return missing


def _negative_binomial(tree: ast.FunctionDef) -> list[str]:
    missing = []
    likelihood = _nested_function(tree, "negative_log_likelihood")
    if likelihood is None or len(_calls(likelihood, "gammaln")) < 2:
        missing.append("negative-binomial log likelihood")
    elif not (_assignments(likelihood, "alpha") and _assignments(likelihood, "size")):
        missing.append("dispersion-to-size parameterization")
    if not _has_call_argument(_calls(tree, "minimize"), "negative_log_likelihood"):
        missing.append("likelihood parameter fitting")
    if not (_assignments(tree, "previous") and _calls(tree, ".append") and any(
        isinstance(item, ast.For) for item in ast.walk(tree)
    )):
        missing.append("recursive dynamic forecast")
    return missing


def _smooth_transition(tree: ast.FunctionDef) -> list[str]:
    missing = []
    residual = _nested_function(tree, "residual_function")
    required = {"base", "regime_difference", "threshold", "transition"}
    if residual is None or not required.issubset(_names(residual)) or not _calls(residual, "expit"):
        missing.append("smooth two-regime transition residual")
    if not _has_call_argument(_calls(tree, "least_squares"), "residual_function"):
        missing.append("nonlinear transition fitting")
    if len(_calls(tree, "expit")) < 2 or not any(isinstance(item, ast.For) for item in ast.walk(tree)):
        missing.append("recursive regime-dependent forecast")
    return missing


def _scinet(tree: ast.FunctionDef) -> list[str]:
    missing = []
    classes = [item for item in ast.walk(tree) if isinstance(item, ast.ClassDef)]
    if not any(any(_dotted_name(base).endswith("torch.nn.Module") for base in item.bases) for item in classes):
        missing.append("torch neural module")
    interact = _nested_function(tree, "interact")
    if interact is None or not _calls(interact, "self.interact") or not _calls(tree, "conv1d"):
        missing.append("recursive interleaved convolution interaction")
    if len(_calls(tree, "Parameter")) < 2 or not _calls(tree, "Adam"):
        missing.append("trainable SCINet parameters")
    if not _calls(tree, ".backward") or not _calls(tree, ".step"):
        missing.append("neural optimization")
    return missing


def _itransformer(tree: ast.FunctionDef) -> list[str]:
    missing = []
    dependency_sets = {
        "queries": {"embedded", "query_matrix"},
        "keys": {"embedded", "key_matrix"},
        "scores": {"queries", "keys"},
        "attended": {"attention", "embedded", "value_matrix"},
    }
    for target, dependencies in dependency_sets.items():
        if not any(dependencies.issubset(_names(value)) for value in _assignments(tree, target)):
            missing.append(f"{target} attention data flow")
    if not any(_calls(value, "exp") for value in _assignments(tree, "attention")):
        missing.append("softmax attention weights")
    if not _calls(tree, "solve") or not any(isinstance(item, ast.For) for item in ast.walk(tree)):
        missing.append("learned attention forecast head")
    return missing


def _dlinear(tree: ast.FunctionDef) -> list[str]:
    missing = []
    if not any(_calls(value, "convolve") for value in _assignments(tree, "trend")):
        missing.append("moving-average trend decomposition")
    if not any({"values", "trend"}.issubset(_names(value)) and isinstance(value, ast.BinOp)
               and isinstance(value.op, ast.Sub) for value in _assignments(tree, "remainder")):
        missing.append("remainder decomposition")
    for target, input_name in (("weights_trend", "trend"), ("weights_remainder", "remainder")):
        if not any(_calls(value, "lstsq") and any(
            input_name in name.casefold() for name in _names(value)
        )
                   for value in _assignments(tree, target)):
            missing.append(f"separate {input_name} projection")
    if not any({"weights_trend", "weights_remainder"}.issubset(_names(value))
               for value in _assignments(tree, "forecast")):
        missing.append("combined trend and remainder forecast")
    return missing


_STRUCTURAL_VALIDATORS = {
    "sarima_auto": _sarima,
    "negative_binomial_dynamic_regression": _negative_binomial,
    "smooth_transition_autoregression": _smooth_transition,
    "scinet": _scinet,
    "itransformer": _itransformer,
    "ltsf_dlinear": _dlinear,
}


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""
