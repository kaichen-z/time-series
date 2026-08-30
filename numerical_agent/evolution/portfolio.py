"""Typed, executable TSFM and Combined policies for method evolution."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pprint
import statistics
import sys
import tempfile
from dataclasses import asdict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

from common.evolution_core.contracts import (
    metric_policy_metadata,
    require_active_metric_policy,
)

from common.metrics import drcik_point_metrics, mae, mase, smape
from common.payload import strict_json_loads

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.foundation import TSFM_IMPLEMENTATION_KIND
from numerical_agent.providers import RuntimeRegistry, RuntimeUnavailableError
from numerical_agent.tsfm.manifests import ManifestRegistry

from .analysis_skills_template import analyze_series
from .cache import (
    CacheMissError,
    OutcomeCache,
    SCALED_METRIC_CAP,
    SCALED_METRIC_SCHEMA,
)
from .execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Outcome,
    Task,
    require_unique_outcome_keys,
    require_unique_task_ids,
)
from .module import MethodModule


FLAGSHIP_METHOD_IDS = (
    "method_tsfm_0031",  # TimesFM 2.5
    "method_tsfm_0017",  # Moirai 2.0
    "method_tsfm_0014",  # Toto 2.0
    "method_tsfm_0018",  # Chronos-Bolt
    "method_tsfm_0006",  # Granite TTM R2
)
FLAGSHIP_TSFM_NAMES = (
    "timesfm_2_5",
    "moirai_2_0",
    "toto_2_0",
    "chronos_bolt",
    "granite_ttm_r2",
)
FLAGSHIP_COMBINED_NAMES = (
    "combined_timesfm_seasonal",
    "combined_chronos_damped_trend",
    "combined_moirai_croston_router",
    "combined_toto_robust_router",
    "combined_granite_regime_profile",
)

Applicability = Literal[
    "all", "periodic", "intermittent", "recent_regime", "trending", "stable"
]
Preprocess = Literal["none", "standardize", "robust_scale", "log1p_shift"]
CombinedOperator = Literal["weighted_mean", "median", "trimmed_mean", "route"]
RouteDirection = Literal["above", "below"]

_APPLICABILITY = frozenset(
    {"all", "periodic", "intermittent", "recent_regime", "trending", "stable"}
)
_PREPROCESS = frozenset({"none", "standardize", "robust_scale", "log1p_shift"})
_COMBINED_OPERATORS = frozenset({"weighted_mean", "median", "trimmed_mean", "route"})
_SIGNALS = frozenset(
    {
        "periodicity_strength",
        "zero_fraction",
        "outlier_fraction",
        "trend_strength",
        "recent_regime_confidence",
        "noise_relative_scale",
        "intermittency_adi",
        "history_length",
        "horizon",
        "horizon_ratio",
    }
)
_ROUTE_DIRECTIONS = frozenset({"above", "below"})


class PolicyError(ValueError):
    """A policy file or requested mutation violates the typed portfolio contract."""


class PolicyNotApplicable(PolicyError):
    """A reviewed policy is inapplicable to this history-only task."""


class InvalidTSFMForecastError(PolicyError):
    """A TSFM runtime returned a structurally invalid forecast."""


@dataclass
class PolicyCacheStats:
    hits: int = 0
    misses: int = 0


class PolicyOutcomeCache:
    """Content-addressed cache for expensive manifest-bound TSFM calls."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = PolicyCacheStats()

    def evaluate(
        self,
        policy: "TSFMPolicy",
        task: Task,
        runtimes: RuntimeRegistry,
    ) -> Outcome:
        key = self._key(policy, task)
        cached = self._read(key, policy.name, task)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        outcome = _run_tsfm(policy, task, runtimes)
        # A worker/checkpoint outage can be transient.  Persist deterministic
        # forecasts and applicability/shape decisions, never transport failures.
        if outcome.status != CRASHED:
            self._write(key, outcome)
        return outcome

    def require_cached(self, policy: "TSFMPolicy", task: Task) -> Outcome:
        """Return an exact cached TSFM outcome without resolving or calling a runtime."""
        outcome = self._read(self._key(policy, task), policy.name, task)
        if outcome is None:
            self.stats.misses += 1
            raise CacheMissError(
                f"cache-only lookup for {policy.name} is missing task {task.task_id}"
            )
        self.stats.hits += 1
        return outcome

    @staticmethod
    def _key(policy: "TSFMPolicy", task: Task) -> str:
        payload = {
            "cache_schema": 3,
            **metric_policy_metadata(),
            "scaled_metric_schema": SCALED_METRIC_SCHEMA,
            "scaled_metric_cap": SCALED_METRIC_CAP,
            "policy": policy.to_payload(),
            "reviewed_candidate": _candidate(policy.method_id).to_payload(),
            "task": asdict(task),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _read(self, key: str, method: str, task: Task) -> Outcome | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = strict_json_loads(
                path.read_text(encoding="utf-8"),
                context="active policy outcome cache row",
            )
        except (OSError, ValueError) as error:
            raise PolicyError("malformed active policy outcome cache row") from error
        try:
            if not isinstance(payload, Mapping):
                raise PolicyError("active policy outcome cache row must be an object")
            require_active_metric_policy(payload, context="active policy outcome cache row")
            if type(payload.get("cache_schema")) is not int or payload["cache_schema"] != 3:
                raise PolicyError("active policy outcome cache row schema mismatch")
            if payload.get("key") != key:
                raise PolicyError("active policy outcome cache row key mismatch")
            if (
                type(payload.get("scaled_metric_schema")) is not int
                or payload["scaled_metric_schema"] != SCALED_METRIC_SCHEMA
                or payload.get("scaled_metric_cap") != SCALED_METRIC_CAP
            ):
                raise PolicyError("active policy outcome cache scaled schema mismatch")
            raw = payload["outcome"]
            if not isinstance(raw, Mapping):
                raise PolicyError("active policy outcome must be an object")
            outcome = OutcomeCache.from_payload(raw)
            if outcome.method != method or outcome.task_id != task.task_id:
                raise PolicyError("active policy outcome cache identity mismatch")
            if outcome.status not in {SUCCESS, NOT_APPLICABLE, CRASHED, INVALID}:
                raise PolicyError("active policy outcome cache status mismatch")
            if outcome.status == SUCCESS and len(outcome.forecast) != task.horizon:
                raise PolicyError("active policy outcome cache forecast horizon mismatch")
            return outcome
        except PolicyError:
            raise
        except ValueError:
            raise
        except (TypeError, KeyError) as error:
            raise PolicyError("malformed active policy outcome cache row") from error

    def _write(self, key: str, outcome: Outcome) -> None:
        payload = json.dumps(
            {
                "cache_schema": 3,
                **metric_policy_metadata(),
                "scaled_metric_schema": SCALED_METRIC_SCHEMA,
                "scaled_metric_cap": SCALED_METRIC_CAP,
                "key": key,
                "outcome": OutcomeCache.to_payload(outcome),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.root, prefix=f".{key}.", delete=False
            ) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.root / f"{key}.json")
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class TSFMPolicy:
    """Evolvable invocation settings around one immutable reviewed model binding."""

    name: str
    method_id: str
    applicability: Applicability = "all"
    context_window: int = 1024
    preprocess: Preprocess = "none"
    shrinkage_to_last: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.name, "TSFM policy name")
        if self.method_id not in FLAGSHIP_METHOD_IDS:
            raise PolicyError("flagship TSFM identities must remain unchanged")
        if self.applicability not in _APPLICABILITY:
            raise PolicyError(f"unsupported applicability {self.applicability!r}")
        if self.preprocess not in _PREPROCESS:
            raise PolicyError(f"unsupported preprocessing {self.preprocess!r}")
        if (
            isinstance(self.context_window, bool)
            or not isinstance(self.context_window, int)
            or not 32 <= self.context_window <= 4096
        ):
            raise PolicyError("context_window must be an integer from 32 to 4096")
        _bounded(self.shrinkage_to_last, "shrinkage_to_last", upper=0.5)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "method_id": self.method_id,
            "applicability": self.applicability,
            "context_window": self.context_window,
            "preprocess": self.preprocess,
            "shrinkage_to_last": self.shrinkage_to_last,
        }


@dataclass(frozen=True)
class CombinedPolicy:
    """One executable history-only operator over an ordered set of leaf parents."""

    name: str
    parents: tuple[str, ...]
    operator: CombinedOperator
    weights: tuple[float, ...] = ()
    signal: str = "periodicity_strength"
    threshold: float = 0.0
    above_parent: str = ""
    below_parent: str = ""
    fallback_parent: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "Combined policy name")
        if not isinstance(self.parents, tuple):
            raise PolicyError("parents must be a tuple")
        if not 2 <= len(self.parents) <= 5:
            raise PolicyError("parents must contain between 2 and 5 entries")
        if not all(isinstance(parent, str) for parent in self.parents):
            raise PolicyError("Combined parents must be strings")
        if len(set(self.parents)) != len(self.parents):
            raise PolicyError("parents must be unique")
        for parent in self.parents:
            _identifier(parent, "Combined parent")
        if not isinstance(self.operator, str) or self.operator not in _COMBINED_OPERATORS:
            raise PolicyError(f"unsupported Combined operator {self.operator!r}")
        if not isinstance(self.weights, tuple):
            raise PolicyError("weights must be a tuple")
        if self.operator == "weighted_mean":
            if len(self.weights) != len(self.parents):
                raise PolicyError("weighted_mean weights must match parents")
            for weight in self.weights:
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(float(weight))
                    or float(weight) < 0.0
                ):
                    raise PolicyError("weights must be finite and non-negative")
            if not math.isclose(
                math.fsum(float(weight) for weight in self.weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise PolicyError("weights must sum to one")
        elif self.weights:
            raise PolicyError(f"{self.operator} does not accept weights")
        if self.operator == "trimmed_mean" and len(self.parents) < 3:
            raise PolicyError("trimmed_mean requires at least three parents")
        if self.operator == "route":
            if len(self.parents) != 2:
                raise PolicyError("route requires exactly two parents")
            if self.above_parent not in self.parents or self.below_parent not in self.parents:
                raise PolicyError("route branch parents must occur in parents")
            if self.above_parent == self.below_parent:
                raise PolicyError("route branch parents must be distinct")
        elif self.above_parent or self.below_parent:
            raise PolicyError("non-route policies require empty route branches")
        if self.fallback_parent not in self.parents:
            raise PolicyError("fallback parent must occur in parents")
        if not isinstance(self.signal, str) or self.signal not in _SIGNALS:
            raise PolicyError(f"unsupported history-only signal {self.signal!r}")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(float(self.threshold))
        ):
            raise PolicyError("threshold must be finite")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parents": self.parents,
            "operator": self.operator,
            "weights": self.weights,
            "signal": self.signal,
            "threshold": self.threshold,
            "above_parent": self.above_parent,
            "below_parent": self.below_parent,
            "fallback_parent": self.fallback_parent,
        }


@dataclass(frozen=True)
class PolicyPortfolio:
    """The Git-tracked non-Python candidates participating beside methods.py."""

    tsfm: tuple[TSFMPolicy, ...]
    combined: tuple[CombinedPolicy, ...]

    def __post_init__(self) -> None:
        if tuple(policy.method_id for policy in self.tsfm) != FLAGSHIP_METHOD_IDS:
            raise PolicyError("flagship TSFM identities and order must remain unchanged")
        if tuple(policy.name for policy in self.tsfm) != FLAGSHIP_TSFM_NAMES:
            raise PolicyError("flagship TSFM policy identities and order must remain unchanged")
        if not isinstance(self.combined, tuple) or not 1 <= len(self.combined) <= 32:
            raise PolicyError("Combined policies must contain between 1 and 32 entries")
        names = tuple(policy.name for policy in self.all_policies)
        if len(names) != len(set(names)):
            raise PolicyError("policy names must be unique")
        tsfm_names = {policy.name for policy in self.tsfm}
        combined_names = {policy.name for policy in self.combined}
        for policy in self.combined:
            if any(parent in combined_names for parent in policy.parents):
                raise PolicyError(
                    f"Combined policy {policy.name!r} cannot use a Combined parent"
                )
            if not any(parent in tsfm_names for parent in policy.parents):
                raise PolicyError(
                    f"Combined policy {policy.name!r} must include a TSFM parent"
                )

    @property
    def all_policies(self) -> tuple[TSFMPolicy | CombinedPolicy, ...]:
        return self.tsfm + self.combined

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(policy.name for policy in self.all_policies)

    def get(self, name: str) -> TSFMPolicy | CombinedPolicy | None:
        return next((policy for policy in self.all_policies if policy.name == name), None)

    def validate_parents(self, method_names: Sequence[str]) -> None:
        known = set(method_names)
        tsfm_names = {policy.name for policy in self.tsfm}
        combined_names = {policy.name for policy in self.combined}
        for policy in self.combined:
            if policy.name in known:
                raise PolicyError(
                    f"Combined policy name {policy.name!r} collides with an executable "
                    "Statistical method"
                )
            if not any(parent in tsfm_names for parent in policy.parents):
                raise PolicyError(
                    f"Combined policy {policy.name!r} must include a TSFM parent"
                )
            for parent in policy.parents:
                if parent in combined_names:
                    raise PolicyError(
                        f"Combined policy {policy.name!r} cannot use a Combined parent"
                    )
                if parent not in tsfm_names and parent not in known:
                    raise PolicyError(
                        f"Combined policy {policy.name!r} has unknown parent {parent!r}"
                    )

    def validate_namespace(self, method_names: Sequence[str]) -> None:
        """Reject any Statistical/TSFM/Combined name collision before evaluation."""
        names = tuple(method_names) + self.names
        duplicates = tuple(sorted({name for name in names if names.count(name) > 1}))
        if duplicates:
            raise PolicyError(
                "candidate namespace collision: " + ", ".join(duplicates)
            )
        self.validate_parents(method_names)

    def replace(
        self, name: str, replacement: TSFMPolicy | CombinedPolicy
    ) -> "PolicyPortfolio":
        parent = self.get(name)
        if parent is None:
            raise PolicyError(f"unknown policy {name!r}")
        if type(parent) is not type(replacement) or replacement.name != name:
            raise PolicyError("policy repair must preserve its name and family")
        if isinstance(parent, TSFMPolicy):
            assert isinstance(replacement, TSFMPolicy)
            if replacement.method_id != parent.method_id:
                raise PolicyError("TSFM model identity cannot change during repair")
            return replace(
                self,
                tsfm=tuple(replacement if policy.name == name else policy for policy in self.tsfm),
            )
        assert isinstance(parent, CombinedPolicy) and isinstance(replacement, CombinedPolicy)
        return replace(
            self,
            combined=tuple(
                replacement if policy.name == name else policy for policy in self.combined
            ),
        )

    def add_combined(self, policy: CombinedPolicy) -> "PolicyPortfolio":
        """Return a portfolio with one new Combined policy."""
        if not isinstance(policy, CombinedPolicy):
            raise PolicyError("Combined mutation requires a CombinedPolicy")
        if self.get(policy.name) is not None:
            raise PolicyError(f"policy name {policy.name!r} already exists")
        return replace(self, combined=self.combined + (policy,))

    def remove_combined(self, name: str) -> "PolicyPortfolio":
        """Return a portfolio without one Combined policy."""
        if not any(policy.name == name for policy in self.combined):
            raise PolicyError(f"unknown Combined policy {name!r}")
        if len(self.combined) == 1:
            raise PolicyError("cannot remove the final Combined policy")
        return replace(
            self,
            combined=tuple(policy for policy in self.combined if policy.name != name),
        )

    def fork_combined(
        self, source: str, child: CombinedPolicy
    ) -> "PolicyPortfolio":
        """Return a portfolio with a new child Combined policy."""
        source_policy = self.get(source)
        if not isinstance(source_policy, CombinedPolicy):
            raise PolicyError(f"unknown Combined source {source!r}")
        if not isinstance(child, CombinedPolicy):
            raise PolicyError("Combined fork requires a CombinedPolicy child")
        return self.add_combined(child)

    @classmethod
    def flagship5(cls) -> "PolicyPortfolio":
        """Return five reviewed models plus five executable statistical combinations."""
        return cls(
            tsfm=(
                TSFMPolicy("timesfm_2_5", "method_tsfm_0031", context_window=1024),
                TSFMPolicy("moirai_2_0", "method_tsfm_0017", context_window=512),
                TSFMPolicy("toto_2_0", "method_tsfm_0014", context_window=512),
                TSFMPolicy("chronos_bolt", "method_tsfm_0018", context_window=512),
                TSFMPolicy("granite_ttm_r2", "method_tsfm_0006", context_window=512),
            ),
            combined=(
                CombinedPolicy(
                    "combined_timesfm_seasonal",
                    ("timesfm_2_5", "seasonal_naive"),
                    "weighted_mean",
                    (0.65, 0.35),
                    "periodicity_strength",
                    0.45,
                    fallback_parent="timesfm_2_5",
                ),
                CombinedPolicy(
                    "combined_chronos_damped_trend",
                    ("chronos_bolt", "holt_damped_trend"),
                    "weighted_mean",
                    (0.65, 0.35),
                    "trend_strength",
                    0.45,
                    fallback_parent="chronos_bolt",
                ),
                CombinedPolicy(
                    "combined_moirai_croston_router",
                    ("moirai_2_0", "croston_sba"),
                    "route",
                    (),
                    "zero_fraction",
                    0.30,
                    above_parent="croston_sba",
                    below_parent="moirai_2_0",
                    fallback_parent="moirai_2_0",
                ),
                CombinedPolicy(
                    "combined_toto_robust_router",
                    ("toto_2_0", "robust_loess_trend"),
                    "route",
                    (),
                    "outlier_fraction",
                    0.05,
                    above_parent="robust_loess_trend",
                    below_parent="toto_2_0",
                    fallback_parent="toto_2_0",
                ),
                CombinedPolicy(
                    "combined_granite_regime_profile",
                    ("granite_ttm_r2", "median_seasonal_profile_forecast"),
                    "weighted_mean",
                    (0.60, 0.40),
                    "recent_regime_confidence",
                    0.50,
                    fallback_parent="granite_ttm_r2",
                ),
            ),
        )


def render_policy_source(portfolio: PolicyPortfolio) -> str:
    """Render data-only Python suitable for a small auditable Git repository."""
    tsfm = tuple(policy.to_payload() for policy in portfolio.tsfm)
    combined = tuple(policy.to_payload() for policy in portfolio.combined)
    return (
        '"""Evolvable TSFM invocation and Combined forecast policies."""\n\n'
        f"TSFM_POLICIES = {pprint.pformat(tsfm, sort_dicts=False, width=100)}\n\n"
        f"COMBINED_POLICIES = {pprint.pformat(combined, sort_dicts=False, width=100)}\n"
    )


def parse_policy_source(source: str) -> PolicyPortfolio:
    """Parse only two literal assignments; never import or execute policy source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PolicyError(f"invalid policy source: {error}") from error
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            continue
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id not in {"TSFM_POLICIES", "COMBINED_POLICIES"}
        ):
            raise PolicyError("policy source may contain only the two literal assignments")
        name = node.targets[0].id
        if name in values:
            raise PolicyError(f"duplicate policy assignment {name}")
        try:
            values[name] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise PolicyError(f"{name} must be a Python literal") from error
    if set(values) != {"TSFM_POLICIES", "COMBINED_POLICIES"}:
        raise PolicyError("policy source must define TSFM_POLICIES and COMBINED_POLICIES")
    tsfm_payloads = _payload_sequence(values["TSFM_POLICIES"], "TSFM_POLICIES")
    combined_payloads = _payload_sequence(values["COMBINED_POLICIES"], "COMBINED_POLICIES")
    try:
        tsfm = tuple(TSFMPolicy(**_exact_payload(payload, TSFMPolicy)) for payload in tsfm_payloads)
        combined = tuple(
            _combined_from_payload(payload)
            for payload in combined_payloads
        )
    except TypeError as error:
        raise PolicyError(f"invalid policy fields: {error}") from error
    return PolicyPortfolio(tsfm, combined)


def read_policy_file(path: str | Path) -> PolicyPortfolio:
    return parse_policy_source(Path(path).read_text(encoding="utf-8"))


def write_policy_file(path: str | Path, portfolio: PolicyPortfolio) -> None:
    Path(path).write_text(render_policy_source(portfolio), encoding="utf-8")


def evaluate_portfolio(
    module: MethodModule,
    portfolio: PolicyPortfolio,
    tasks: Sequence[Task],
    *,
    outcome_cache: OutcomeCache,
    runtimes: RuntimeRegistry,
    isolated_methods: bool,
    policy_cache: PolicyOutcomeCache | None = None,
) -> tuple[Outcome, ...]:
    """Evaluate Python, TSFM, and Combined candidates on one trusted task sequence."""
    require_unique_task_ids(tasks)
    portfolio.validate_namespace(module.names())
    python_outcomes = tuple(
        outcome
        for method in module.methods
        for outcome in outcome_cache.evaluate_method(
            method, tasks, isolated=isolated_methods, require_forecasts=True
        )
    )
    tsfm_outcomes = tuple(
        (
            policy_cache.evaluate(policy, task, runtimes)
            if policy_cache is not None
            else _run_tsfm(policy, task, runtimes)
        )
        for policy in portfolio.tsfm
        for task in tasks
    )
    leaf_outcomes = python_outcomes + tsfm_outcomes
    require_unique_outcome_keys(leaf_outcomes)
    by_key = {
        (outcome.method, outcome.task_id): outcome
        for outcome in leaf_outcomes
    }
    combined_outcomes = tuple(
        _run_combined(policy, task, by_key)
        for policy in portfolio.combined
        for task in tasks
    )
    return python_outcomes + tsfm_outcomes + combined_outcomes


def require_flagship_runtimes(
    portfolio: PolicyPortfolio, runtimes: RuntimeRegistry
) -> None:
    """Fail before evaluation unless all five reviewed bindings have a provider."""
    missing = []
    for policy in portfolio.tsfm:
        resolution = runtimes.resolve(_candidate(policy.method_id))
        if not resolution.available or resolution.runtime is None:
            missing.append(f"{policy.name}: {resolution.reason}")
    if missing:
        raise PolicyError("flagship5 runtime preflight failed: " + "; ".join(missing))


def _run_tsfm(
    policy: TSFMPolicy, task: Task, runtimes: RuntimeRegistry
) -> Outcome:
    try:
        calibrated = forecast_tsfm(
            policy,
            history=task.history,
            horizon=task.horizon,
            frequency=task.frequency,
            runtimes=runtimes,
        )
    except PolicyNotApplicable:
        return Outcome(
            policy.name,
            task.task_id,
            NOT_APPLICABLE,
            detail=f"history does not satisfy {policy.applicability} applicability",
        )
    except RuntimeUnavailableError as error:
        return Outcome(policy.name, task.task_id, CRASHED, detail=str(error)[:200])
    except InvalidTSFMForecastError as error:
        return Outcome(policy.name, task.task_id, INVALID, detail=str(error)[:200])
    except Exception as error:
        return Outcome(
            policy.name,
            task.task_id,
            CRASHED,
            detail=f"{type(error).__name__}: {error}"[:200],
        )
    return _scored(policy.name, task, calibrated)


def forecast_tsfm(
    policy: TSFMPolicy,
    *,
    history: Sequence[float],
    horizon: int,
    frequency: str,
    runtimes: RuntimeRegistry,
) -> tuple[float, ...]:
    """Run one manifest-bound TSFM without constructing labels or invoking a scorer."""
    profile = analyze_series(history, frequency)
    if not _applicable(policy.applicability, profile):
        raise PolicyNotApplicable(
            f"history does not satisfy {policy.applicability} applicability"
        )
    candidate = _candidate(policy.method_id)
    resolution = runtimes.resolve(candidate)
    if not resolution.available or resolution.runtime is None:
        raise RuntimeUnavailableError(resolution.reason[:200])
    context = tuple(float(value) for value in history[-policy.context_window :])
    transformed, inverse = _transform(context, policy.preprocess)
    raw = resolution.runtime.forecast(candidate, transformed, horizon, frequency)
    forecast = tuple(float(inverse(float(value))) for value in raw)
    if len(forecast) != horizon or not all(math.isfinite(value) for value in forecast):
        raise InvalidTSFMForecastError("TSFM returned invalid forecast")
    shrinkage = policy.shrinkage_to_last
    last = float(history[-1])
    return tuple((1.0 - shrinkage) * value + shrinkage * last for value in forecast)


def _run_combined(
    policy: CombinedPolicy,
    task: Task,
    outcomes: Mapping[tuple[str, str], Outcome],
) -> Outcome:
    composed = combine_materialized_outcome(
        policy,
        {
            parent: outcomes.get(
                (parent, task.task_id),
                Outcome(
                    parent,
                    task.task_id,
                    CRASHED,
                    detail="missing materialized parent outcome",
                ),
            )
            for parent in policy.parents
        },
        task_id=task.task_id,
        history=task.history,
        horizon=task.horizon,
        frequency=task.frequency,
    )
    if composed.status != SUCCESS:
        return composed
    try:
        scored = _scored(policy.name, task, composed.forecast)
    except (ArithmeticError, TypeError, ValueError):
        return Outcome(policy.name, task.task_id, INVALID, detail="combined score is invalid")
    return replace(scored, detail=composed.detail) if composed.detail else scored


def combine_materialized_outcome(
    policy: CombinedPolicy,
    parent_outcomes: Mapping[str, Outcome],
    *,
    task_id: str,
    history: Sequence[float],
    horizon: int,
    frequency: str,
) -> Outcome:
    """Compose materialized leaf outcomes without reading future labels or scoring."""
    parent_outcomes = tuple(
        parent_outcomes.get(
            parent,
            Outcome(
                parent,
                task_id,
                CRASHED,
                detail="missing materialized parent outcome",
            ),
        )
        for parent in policy.parents
    )
    failed = tuple(
        outcome
        for outcome in parent_outcomes
        if not _is_successful_parent(outcome, horizon)
    )
    if failed:
        detail = "; ".join(
            f"{parent.method}={_parent_failure_status(parent, horizon)}"
            for parent in failed
        )
        return _combined_fallback_or_failure(
            policy,
            parent_outcomes,
            task_id=task_id,
            horizon=horizon,
            detail=detail,
            failed=failed,
        )
    try:
        forecast = combine_materialized_forecast(
            policy,
            {outcome.method: outcome for outcome in parent_outcomes},
            history=history,
            horizon=horizon,
            frequency=frequency,
        )
    except (ArithmeticError, TypeError, ValueError):
        return _combined_fallback_or_failure(
            policy,
            parent_outcomes,
            task_id=task_id,
            horizon=horizon,
            detail="combined composition is invalid",
            failed=(),
        )
    if not _forecast_is_valid(forecast, horizon):
        return _combined_fallback_or_failure(
            policy,
            parent_outcomes,
            task_id=task_id,
            horizon=horizon,
            detail="combined composition is invalid",
            failed=(),
        )
    return Outcome(policy.name, task_id, SUCCESS, forecast=forecast)


def _combined_fallback_or_failure(
    policy: CombinedPolicy,
    parent_outcomes: Sequence[Outcome],
    *,
    task_id: str,
    horizon: int,
    detail: str,
    failed: Sequence[Outcome],
) -> Outcome:
    """Use the reviewed fallback or return the strongest sanitized failure."""
    fallback = next(
        outcome
        for outcome in parent_outcomes
        if outcome.method == policy.fallback_parent
    )
    if _is_successful_parent(fallback, horizon):
        return Outcome(
            policy.name,
            task_id,
            SUCCESS,
            detail=f"fallback={policy.fallback_parent}; {detail}"[:200],
            forecast=fallback.forecast,
        )
    statuses = [_parent_failure_status(parent, horizon) for parent in failed]
    if not statuses:
        statuses.append(INVALID)
    status = max(statuses, key=_failure_precedence)
    return Outcome(policy.name, task_id, status, detail=detail[:200])


def combine_materialized_forecast(
    policy: CombinedPolicy,
    parent_outcomes: Mapping[str, Outcome],
    *,
    history: Sequence[float],
    horizon: int,
    frequency: str,
) -> tuple[float, ...]:
    """Compose materialized successful forecasts using history-only policy inputs."""
    parents = tuple(parent_outcomes[parent] for parent in policy.parents)
    if policy.operator == "weighted_mean":
        forecast = tuple(
            _stable_weighted_mean(
                tuple(
                    (weight, outcome.forecast[index])
                    for weight, outcome in zip(policy.weights, parents, strict=True)
                )
            )
            for index in range(horizon)
        )
    elif policy.operator == "route":
        signal = _history_signal(policy.signal, history, horizon, frequency)
        selected = policy.above_parent if signal >= policy.threshold else policy.below_parent
        forecast = parent_outcomes[selected].forecast
    elif policy.operator == "median":
        forecast = tuple(
            _overflow_stable_median(
                tuple(outcome.forecast[index] for outcome in parents)
            )
            for index in range(horizon)
        )
    elif policy.operator == "trimmed_mean":
        forecast = tuple(
            _overflow_stable_mean(
                sorted(outcome.forecast[index] for outcome in parents)[1:-1]
            )
            for index in range(horizon)
        )
    else:  # pragma: no cover - CombinedPolicy validates operators
        raise PolicyError(f"unsupported Combined operator {policy.operator!r}")
    return tuple(float(value) for value in forecast)


def _stable_weighted_mean(values: Sequence[tuple[float, float]]) -> float:
    """Return a weighted mean without overflowing intermediate additions."""
    terms = tuple((float(weight), float(value)) for weight, value in values)
    maximum = max((abs(value) for _, value in terms), default=0.0)
    weight_sum = math.fsum(weight for weight, _ in terms)
    if maximum == 0.0:
        return 0.0
    if maximum <= sys.float_info.max / max(weight_sum, 1.0):
        return sum(weight * value for weight, value in terms)
    return float(
        sum(
            Fraction.from_float(weight) * Fraction.from_float(value)
            for weight, value in terms
        )
    )


def _overflow_stable_mean(values: Sequence[float]) -> float:
    """Return an arithmetic mean while keeping intermediate sums finite."""
    numbers = tuple(float(value) for value in values)
    maximum = max((abs(value) for value in numbers), default=0.0)
    if not numbers:
        raise ValueError("mean requires at least one value")
    if maximum == 0.0:
        return 0.0
    if maximum <= sys.float_info.max / len(numbers):
        return statistics.fmean(numbers)
    return float(sum(Fraction.from_float(value) for value in numbers) / len(numbers))


def _overflow_stable_median(values: Sequence[float]) -> float:
    """Return the median while averaging an even pair without overflow."""
    numbers = tuple(sorted(float(value) for value in values))
    if not numbers:
        raise ValueError("median requires at least one value")
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return _overflow_stable_mean(numbers[middle - 1 : middle + 1])


def _combine_forecasts(
    policy: CombinedPolicy,
    parent_outcomes: Mapping[str, Outcome],
    task: Task,
) -> tuple[float, ...]:
    """Compatibility wrapper for the task-shaped canonical composition contract."""
    return combine_materialized_forecast(
        policy,
        parent_outcomes,
        history=task.history,
        horizon=task.horizon,
        frequency=task.frequency,
    )


def _is_successful_parent(outcome: Outcome, horizon: int) -> bool:
    return outcome.status == SUCCESS and _forecast_is_valid(outcome.forecast, horizon)


def _forecast_is_valid(forecast: Sequence[float], horizon: int) -> bool:
    try:
        return len(forecast) == horizon and all(
            math.isfinite(float(value)) for value in forecast
        )
    except (ArithmeticError, TypeError, ValueError):
        return False


def _parent_failure_status(outcome: Outcome, horizon: int) -> str:
    if outcome.status == CRASHED:
        return CRASHED
    if outcome.status == INVALID or not _forecast_is_valid(outcome.forecast, horizon):
        return INVALID
    return NOT_APPLICABLE


def _failure_precedence(status: str) -> int:
    return {NOT_APPLICABLE: 0, INVALID: 1, CRASHED: 2}[status]


def _candidate(method_id: str) -> MethodCandidate:
    manifest = ManifestRegistry.load_default().require(method_id)
    provider = manifest.adapter if manifest.status == "direct" else "tsfm_worker"
    return MethodCandidate(
        method_id=method_id,
        provider=provider,
        implementation_kind=TSFM_IMPLEMENTATION_KIND,
        implementation=manifest.candidate_binding(),
    )


def _scored(name: str, task: Task, forecast: Sequence[float]) -> Outcome:
    values = tuple(float(value) for value in forecast)
    if len(values) != task.horizon or not all(math.isfinite(value) for value in values):
        return Outcome(name, task.task_id, INVALID, detail="combined forecast is invalid")
    point = drcik_point_metrics(task.future, values)
    return Outcome(
        name,
        task.task_id,
        SUCCESS,
        smae=float(point["smae"]),
        srmse=float(point["srmse"]),
        smae_raw=float(point["smae_raw"]),
        srmse_raw=float(point["srmse_raw"]),
        smae_clipped=bool(point["smae_clipped"]),
        srmse_clipped=bool(point["srmse_clipped"]),
        smape=smape(task.future, values),
        mae=mae(task.future, values),
        mase=mase(task.future, values, task.history),
        forecast=values,
    )


def _applicable(applicability: str, profile: Mapping[str, object]) -> bool:
    if applicability == "all":
        return True
    periodicity = cast(Mapping[str, object], profile["periodicity"])
    intermittency = cast(Mapping[str, object], profile["intermittency"])
    recent = cast(Mapping[str, object], profile["recent_regime"])
    trend = cast(Mapping[str, object], profile["trend"])
    stationarity = cast(Mapping[str, object], profile["stationarity"])
    return {
        "periodic": float(periodicity["confidence"]) >= 0.5,
        "intermittent": bool(intermittency["is_intermittent"]),
        "recent_regime": float(recent["confidence"]) >= 0.5,
        "trending": float(trend["strength"]) >= 0.5,
        "stable": bool(stationarity["likely_stationary"]),
    }[applicability]


def _signal(name: str, task: Task) -> float:
    return _history_signal(name, task.history, task.horizon, task.frequency)


def _history_signal(
    name: str, history: Sequence[float], horizon: int, frequency: str
) -> float:
    if name == "history_length":
        return float(len(history))
    if name == "horizon":
        return float(horizon)
    if name == "horizon_ratio":
        return float(horizon / len(history))
    profile = analyze_series(history, frequency)
    if name == "periodicity_strength":
        return float(cast(Mapping[str, object], profile["periodicity"])["strength"])
    if name == "zero_fraction":
        return float(cast(Mapping[str, object], profile["intermittency"])["zero_fraction"])
    if name == "outlier_fraction":
        outliers = cast(Mapping[str, object], profile["outliers"])
        return len(cast(Sequence[int], outliers["indices"])) / len(history)
    if name == "trend_strength":
        return float(cast(Mapping[str, object], profile["trend"])["strength"])
    if name == "recent_regime_confidence":
        return float(cast(Mapping[str, object], profile["recent_regime"])["confidence"])
    if name == "noise_relative_scale":
        return float(cast(Mapping[str, object], profile["noise"])["relative_scale"])
    if name == "intermittency_adi":
        return float(
            cast(Mapping[str, object], profile["intermittency"])["average_nonzero_gap"]
        )
    raise PolicyError(f"unsupported signal {name!r}")


def _transform(history: Sequence[float], mode: str):
    values = tuple(float(value) for value in history)
    if mode == "none":
        return values, float
    if mode == "standardize":
        center = statistics.fmean(values)
        scale = statistics.pstdev(values) or 1.0
        return tuple((value - center) / scale for value in values), lambda value: value * scale + center
    if mode == "robust_scale":
        center = statistics.median(values)
        deviations = tuple(abs(value - center) for value in values)
        scale = statistics.median(deviations) / 0.67448975 or 1.0
        return tuple((value - center) / scale for value in values), lambda value: value * scale + center
    if mode == "log1p_shift":
        shift = max(0.0, 1.0 - min(values))
        return tuple(math.log1p(value + shift) for value in values), lambda value: math.expm1(value) - shift
    raise PolicyError(f"unsupported preprocessing {mode!r}")


def _payload_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise PolicyError(f"{name} must be a tuple or list")
    if not all(isinstance(item, Mapping) for item in value):
        raise PolicyError(f"{name} entries must be dictionaries")
    return tuple(cast(Mapping[str, object], item) for item in value)


def _exact_payload(payload: Mapping[str, object], cls: type) -> dict[str, object]:
    fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(payload) - fields
    if unknown:
        raise PolicyError(f"unknown {cls.__name__} fields: {sorted(unknown)!r}")
    return dict(payload)


_COMBINED_CANONICAL_FIELDS = frozenset(
    {
        "name",
        "parents",
        "operator",
        "weights",
        "signal",
        "threshold",
        "above_parent",
        "below_parent",
        "fallback_parent",
    }
)
_COMBINED_LEGACY_FIELDS = frozenset(
    {
        "name",
        "tsfm_parent",
        "statistical_parent",
        "mode",
        "weight",
        "signal",
        "threshold",
        "tsfm_when",
    }
)


def _combined_from_payload(payload: Mapping[str, object]) -> CombinedPolicy:
    """Build a canonical CombinedPolicy, migrating one exact legacy payload shape."""
    keys = set(payload)
    known = _COMBINED_CANONICAL_FIELDS | _COMBINED_LEGACY_FIELDS
    unknown = keys - known
    if unknown:
        raise PolicyError(f"unknown CombinedPolicy fields: {sorted(unknown)!r}")
    if keys == _COMBINED_CANONICAL_FIELDS:
        return CombinedPolicy(**dict(payload))
    if keys != _COMBINED_LEGACY_FIELDS:
        raise PolicyError(
            "CombinedPolicy fields must match exactly the canonical or legacy schema"
        )

    mode = payload["mode"]
    tsfm_parent = payload["tsfm_parent"]
    statistical_parent = payload["statistical_parent"]
    tsfm_when = payload["tsfm_when"]
    weight = payload["weight"]
    _bounded(weight, "weight", lower=0.05, upper=0.95)
    if mode == "blend":
        return CombinedPolicy(
            name=payload["name"],  # type: ignore[arg-type]
            parents=(tsfm_parent, statistical_parent),  # type: ignore[assignment]
            operator="weighted_mean",
            weights=(weight, 1.0 - weight),  # type: ignore[operator]
            signal=payload["signal"],  # type: ignore[arg-type]
            threshold=payload["threshold"],  # type: ignore[arg-type]
            fallback_parent=tsfm_parent,  # type: ignore[arg-type]
        )
    if mode == "route":
        if tsfm_when == "above":
            above_parent, below_parent = tsfm_parent, statistical_parent
        elif tsfm_when == "below":
            above_parent, below_parent = statistical_parent, tsfm_parent
        else:
            raise PolicyError(f"unsupported route direction {tsfm_when!r}")
        return CombinedPolicy(
            name=payload["name"],  # type: ignore[arg-type]
            parents=(tsfm_parent, statistical_parent),  # type: ignore[assignment]
            operator="route",
            signal=payload["signal"],  # type: ignore[arg-type]
            threshold=payload["threshold"],  # type: ignore[arg-type]
            above_parent=above_parent,  # type: ignore[arg-type]
            below_parent=below_parent,  # type: ignore[arg-type]
            fallback_parent=tsfm_parent,  # type: ignore[arg-type]
        )
    raise PolicyError(f"unsupported Combined mode {mode!r}")


def _identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.isidentifier() or value.startswith("_"):
        raise PolicyError(f"{field} must be a public Python identifier")


def _bounded(
    value: float, field: str, *, lower: float = 0.0, upper: float = 1.0
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not lower <= float(value) <= upper
    ):
        raise PolicyError(f"{field} must be between {lower:g} and {upper:g}")
