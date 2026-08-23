"""Typed, executable TSFM and Combined policies for method evolution."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pprint
import statistics
import tempfile
from dataclasses import asdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

from common.metrics import mae, mase, smape

from numerical_agent.dictionary import MethodCandidate
from numerical_agent.foundation import TSFM_IMPLEMENTATION_KIND
from numerical_agent.providers import RuntimeRegistry, RuntimeUnavailableError
from numerical_agent.tsfm.manifests import ManifestRegistry

from .analysis_skills_template import analyze_series
from .cache import CacheMissError, OutcomeCache
from .execution import (
    CRASHED,
    INVALID,
    NOT_APPLICABLE,
    SUCCESS,
    Outcome,
    Task,
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
CombinedMode = Literal["blend", "route"]
RouteDirection = Literal["above", "below"]

_APPLICABILITY = frozenset(
    {"all", "periodic", "intermittent", "recent_regime", "trending", "stable"}
)
_PREPROCESS = frozenset({"none", "standardize", "robust_scale", "log1p_shift"})
_COMBINED_MODES = frozenset({"blend", "route"})
_SIGNALS = frozenset(
    {
        "periodicity_strength",
        "zero_fraction",
        "outlier_fraction",
        "trend_strength",
        "recent_regime_confidence",
    }
)
_ROUTE_DIRECTIONS = frozenset({"above", "below"})


class PolicyError(ValueError):
    """A policy file or requested mutation violates the typed portfolio contract."""


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
            "schema": 1,
            "policy": policy.to_payload(),
            "reviewed_candidate": _candidate(policy.method_id).to_payload(),
            "task": asdict(task),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _read(self, key: str, method: str, task: Task) -> Outcome | None:
        try:
            payload = json.loads((self.root / f"{key}.json").read_text(encoding="utf-8"))
            if payload.get("schema") != 1 or payload.get("key") != key:
                return None
            raw = payload["outcome"]
            outcome = Outcome(
                method=str(raw["method"]),
                task_id=str(raw["task_id"]),
                status=str(raw["status"]),
                smape=_optional_metric(raw.get("smape")),
                mae=_optional_metric(raw.get("mae")),
                mase=_optional_metric(raw.get("mase")),
                detail=str(raw.get("detail", "")),
                forecast=tuple(float(value) for value in raw.get("forecast", ())),
            )
            if outcome.method != method or outcome.task_id != task.task_id:
                return None
            if outcome.status not in {SUCCESS, NOT_APPLICABLE, CRASHED, INVALID}:
                return None
            if outcome.status == SUCCESS and (
                len(outcome.forecast) != task.horizon
                or any(value is None for value in (outcome.smape, outcome.mae, outcome.mase))
            ):
                return None
            return outcome
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _write(self, key: str, outcome: Outcome) -> None:
        payload = json.dumps(
            {"schema": 1, "key": key, "outcome": asdict(outcome)},
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
    """One executable history-only blend or router over fixed parent identities."""

    name: str
    tsfm_parent: str
    statistical_parent: str
    mode: CombinedMode
    weight: float
    signal: str
    threshold: float
    tsfm_when: RouteDirection = "above"

    def __post_init__(self) -> None:
        _identifier(self.name, "Combined policy name")
        _identifier(self.tsfm_parent, "TSFM parent")
        _identifier(self.statistical_parent, "statistical parent")
        if self.mode not in _COMBINED_MODES:
            raise PolicyError(f"unsupported Combined mode {self.mode!r}")
        _bounded(self.weight, "weight", lower=0.05, upper=0.95)
        if self.signal not in _SIGNALS:
            raise PolicyError(f"unsupported history-only signal {self.signal!r}")
        if not isinstance(self.threshold, (int, float)) or not math.isfinite(self.threshold):
            raise PolicyError("threshold must be finite")
        if self.tsfm_when not in _ROUTE_DIRECTIONS:
            raise PolicyError(f"unsupported route direction {self.tsfm_when!r}")

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tsfm_parent": self.tsfm_parent,
            "statistical_parent": self.statistical_parent,
            "mode": self.mode,
            "weight": self.weight,
            "signal": self.signal,
            "threshold": self.threshold,
            "tsfm_when": self.tsfm_when,
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
        if tuple(policy.name for policy in self.combined) != FLAGSHIP_COMBINED_NAMES:
            raise PolicyError("Combined identities and order must remain unchanged")
        names = tuple(policy.name for policy in self.all_policies)
        if len(names) != len(set(names)):
            raise PolicyError("policy names must be unique")
        tsfm_names = {policy.name for policy in self.tsfm}
        for policy in self.combined:
            if policy.tsfm_parent not in tsfm_names:
                raise PolicyError(
                    f"Combined policy {policy.name!r} has unknown TSFM parent"
                )

    @property
    def all_policies(self) -> tuple[TSFMPolicy | CombinedPolicy, ...]:
        return self.tsfm + self.combined

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(policy.name for policy in self.all_policies)

    def get(self, name: str) -> TSFMPolicy | CombinedPolicy | None:
        return next((policy for policy in self.all_policies if policy.name == name), None)

    def validate_statistical_parents(self, method_names: Sequence[str]) -> None:
        known = set(method_names)
        for policy in self.combined:
            if policy.statistical_parent not in known:
                raise PolicyError(
                    f"Combined policy {policy.name!r} has unknown statistical parent "
                    f"{policy.statistical_parent!r}"
                )

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
        if (
            replacement.tsfm_parent != parent.tsfm_parent
            or replacement.statistical_parent != parent.statistical_parent
        ):
            raise PolicyError("Combined parent identities cannot change during repair")
        return replace(
            self,
            combined=tuple(
                replacement if policy.name == name else policy for policy in self.combined
            ),
        )

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
                    "timesfm_2_5",
                    "seasonal_naive",
                    "blend",
                    0.65,
                    "periodicity_strength",
                    0.45,
                ),
                CombinedPolicy(
                    "combined_chronos_damped_trend",
                    "chronos_bolt",
                    "holt_damped_trend",
                    "blend",
                    0.65,
                    "trend_strength",
                    0.45,
                ),
                CombinedPolicy(
                    "combined_moirai_croston_router",
                    "moirai_2_0",
                    "croston_sba",
                    "route",
                    0.65,
                    "zero_fraction",
                    0.30,
                    "below",
                ),
                CombinedPolicy(
                    "combined_toto_robust_router",
                    "toto_2_0",
                    "robust_loess_trend",
                    "route",
                    0.65,
                    "outlier_fraction",
                    0.05,
                    "below",
                ),
                CombinedPolicy(
                    "combined_granite_regime_profile",
                    "granite_ttm_r2",
                    "median_seasonal_profile_forecast",
                    "blend",
                    0.60,
                    "recent_regime_confidence",
                    0.50,
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
            CombinedPolicy(**_exact_payload(payload, CombinedPolicy))
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
    portfolio.validate_statistical_parents(module.names())
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
    by_key = {
        (outcome.method, outcome.task_id): outcome
        for outcome in python_outcomes + tsfm_outcomes
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
    profile = analyze_series(task.history, task.frequency)
    if not _applicable(policy.applicability, profile):
        return Outcome(
            policy.name,
            task.task_id,
            NOT_APPLICABLE,
            detail=f"history does not satisfy {policy.applicability} applicability",
        )
    candidate = _candidate(policy.method_id)
    resolution = runtimes.resolve(candidate)
    if not resolution.available or resolution.runtime is None:
        return Outcome(policy.name, task.task_id, CRASHED, detail=resolution.reason[:200])
    history = tuple(task.history[-policy.context_window :])
    transformed, inverse = _transform(history, policy.preprocess)
    try:
        raw = resolution.runtime.forecast(
            candidate, transformed, task.horizon, task.frequency
        )
        forecast = tuple(float(inverse(float(value))) for value in raw)
    except RuntimeUnavailableError as error:
        return Outcome(policy.name, task.task_id, CRASHED, detail=str(error)[:200])
    except Exception as error:
        return Outcome(
            policy.name,
            task.task_id,
            CRASHED,
            detail=f"{type(error).__name__}: {error}"[:200],
        )
    if len(forecast) != task.horizon or not all(math.isfinite(value) for value in forecast):
        return Outcome(
            policy.name, task.task_id, INVALID, detail="TSFM returned invalid forecast"
        )
    shrinkage = policy.shrinkage_to_last
    last = float(task.history[-1])
    calibrated = tuple((1.0 - shrinkage) * value + shrinkage * last for value in forecast)
    return _scored(policy.name, task, calibrated)


def _run_combined(
    policy: CombinedPolicy,
    task: Task,
    outcomes: Mapping[tuple[str, str], Outcome],
) -> Outcome:
    tsfm = outcomes[(policy.tsfm_parent, task.task_id)]
    statistical = outcomes[(policy.statistical_parent, task.task_id)]
    failed = [parent for parent in (tsfm, statistical) if parent.status != SUCCESS]
    if failed:
        status = CRASHED if any(parent.status == CRASHED for parent in failed) else NOT_APPLICABLE
        detail = "; ".join(
            f"{parent.method}={parent.status}" for parent in failed
        )
        return Outcome(policy.name, task.task_id, status, detail=detail[:200])
    if policy.mode == "blend":
        forecast = tuple(
            policy.weight * left + (1.0 - policy.weight) * right
            for left, right in zip(tsfm.forecast, statistical.forecast, strict=True)
        )
    else:
        signal = _signal(policy.signal, task)
        choose_tsfm = signal >= policy.threshold
        if policy.tsfm_when == "below":
            choose_tsfm = not choose_tsfm
        forecast = tsfm.forecast if choose_tsfm else statistical.forecast
    return _scored(policy.name, task, forecast)


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
    return Outcome(
        name,
        task.task_id,
        SUCCESS,
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
    profile = analyze_series(task.history, task.frequency)
    if name == "periodicity_strength":
        return float(cast(Mapping[str, object], profile["periodicity"])["strength"])
    if name == "zero_fraction":
        return float(cast(Mapping[str, object], profile["intermittency"])["zero_fraction"])
    if name == "outlier_fraction":
        outliers = cast(Mapping[str, object], profile["outliers"])
        return len(cast(Sequence[int], outliers["indices"])) / len(task.history)
    if name == "trend_strength":
        return float(cast(Mapping[str, object], profile["trend"])["strength"])
    if name == "recent_regime_confidence":
        return float(cast(Mapping[str, object], profile["recent_regime"])["confidence"])
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


def _optional_metric(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cached policy metric must be finite")
    return number
