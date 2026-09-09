"""Numbers-only, falsifiable evolution of executable forecasting skills."""
from __future__ import annotations

import json
import math
import statistics
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol

from evolving_loop.coding_agent.skill_library import Skill, SkillLibrary
from evolving_loop.data import Task
from evolving_loop.knowledge_base import (
    DiagnosticProfile,
    KnowledgeSelection,
    TimeSeriesKnowledgeBase,
    diagnose,
)
from common.sandbox import SandboxResult, run_forecast_code
from common.llm import LLMClient, TransientLLMError, parse_json_object
from common.metrics import drcik_point_metrics

CodingSetting = Literal["llm_only", "statistics", "tsfm", "combined"]

_MIN_NORMAL_FLOAT = 2.2250738585072014e-308
_PROFILE_UNSET = object()


class NumericForecaster(Protocol):
    def forecast(
        self, history: tuple[float, ...], horizon: int, frequency: str
    ) -> tuple[float, ...]: ...

STATISTICAL_SKILL_DICTIONARY = """Available statistical ideas (choose only when justified):
- robust local level: median/recent trimmed level for outlier resistance
- local linear trend: recent slope with damping for non-stationary series
- seasonal naive: repeat an empirically supported lag
- seasonal trend: combine a repeated seasonal profile with a damped level drift
- moving-average residual: forecast a smoothed level and decay the last residual
- Fourier/harmonic extrapolation: use only when repeated cycles survive holdouts
Every method is a hypothesis, not a guaranteed rule. State how historical holdouts could falsify it.
"""

GENERATION_PROMPT = """You are the numbers-only single-agent forecaster in a time-series harness.
You may see historical numbers, horizon, frequency, and optionally a reusable skill summary.
You must not request or infer documents, retrieved evidence, ground-truth evidence, or future values.

First read `numeric_evidence`, then inspect the raw history. Decide which structure
is actually supported: persistence/local level, trend, seasonality/repeated motifs, a recent regime,
intermittency, or weak/noisy structure. The profile is diagnostic evidence, not a label and not a
command. Resolve disagreements in favor of the raw history and conservative behavior.
The trailing `numeric_profile` object contains legacy coarse fields for compatibility only; never
let it override a more specific v4 evidence card.

Every host assessment is a falsifiable hypothesis. `supported` or `persistent_regime` permits at
most one specialist from that family; it is never a selection command. Ambiguous, absent, or
insufficient seasonal evidence must not trigger an LLM seasonal/state-cycle candidate. Likewise,
for `no_regime` or other `insufficient` families, do not invent a specialist to fill a quota. If
evidence cards conflict, prefer abstention. Do not paste the host's current window lengths,
assessment, n, or H into code as exact branches.

A seasonality card with `support_mode=causal_replay_plateau` is positive causal replay evidence
even when its ACF maximum is broad rather than sharp; flat ACF alone is never support. For smooth
cycles, one robust phase specialist may be generated, but it must re-estimate lag and replay quality
from every history prefix and return repeat-last unless the strict card conditions still hold. The
host separately evaluates one fixed, self-gating structural state-cycle candidate for repeated
plateaus or discrete states. Do not duplicate, rename, or mutate that host candidate. If no card is
supported, do not turn a weaker ambiguous screen into an active deployment forecast.

Every LLM-supplied specialist is an untrusted conditional proposal. Its specialized branch may
depart from persistence only while its applicability checks pass on the supplied history prefix.
Every inapplicable, ambiguous, insufficient-data, or failed-check branch must repeat the latest
finite history value for every horizon step exactly. Do not substitute a recent median, local mean,
fitted level, or any other non-anchor fallback: a changed fallback level is not evidence that the
specialist activated.

Return compact, structurally distinct, falsifiable single-method programs. Give priority to:
- exact or robust phase continuation when multiple cycles support the same period;
- a damped robust trend only when its horizon-scale effect is material and stable;
- a recent-regime method only when the suffix contains enough observations;
- conservative local level for weak, noisy, intermittent, or ambiguous structure.
Do not average unrelated forecasters or create arbitrary weighted ensembles: method combination is
handled separately after single-method reliability is established. Do not copy one anomalous point,
extrapolate an undamped slope over a long horizon, infer periodicity from a single coincidence, or
hard-code this task's values/identity. Each implementation must recompute its logic from the
`history` argument so causal hindcasts remain valid. Never branch on a validation horizon, fold
index/role/count, an exact train/history length, or equality to likely validation constants. Horizon
adaptation must be smooth or derived from scale-free structure re-estimated from each supplied
history prefix; it must not make short validation calls copy the anchor while activating a different
method only at deployment.

Put the simplest complete program first; valid JSON and executable code take priority over
sophistication. A `declared_period_steps` value is only a hypothesis; include a phase-aligned
candidate only when its seasonality evidence is supported. Treat `raw_seasonality_metadata` and
legacy diagnostic `candidate_lags` the same way. Every program must define exactly:
    def forecast(history: list[float], horizon: int, frequency: str) -> list[float]
It must return exactly horizon finite numbers. Allowed imports are numpy, math, statistics,
itertools, functools, and collections. Do not access files/network, use randomness, eval/exec,
or hard-code the supplied series.

Return exactly one JSON object:
{"programs": [{"name": "short_snake_case", "description": "when to use it",
"assumption": "a falsifiable statement about the numeric process",
"failure_condition": "an observable condition under which it should fail",
"knowledge_ids": ["cited external knowledge entry IDs"],
"prior_confidence": 0.0,
"code": "def forecast(...): ..."}]}

When external knowledge is supplied, treat it as a falsifiable prior and cite only its exact IDs.
The host reruns every program on multiple hidden historical cutoffs, including deployment-scale
audits when history permits; their exact geometry is intentionally withheld. A specialized program
normally replaces the numeric anchor only when its current forecast materially differs from the
anchor, both sMAE and sRMSE are non-worse on every fold, at least two distinct cutoffs are Pareto
wins, and the aggregate gain is at least 10%. If the anchor has a catastrophic historical fold, a
bounded rescue is possible only when every causal fold is non-worse in both metrics and both mean and
worst-fold errors fall sharply. Separately, a tightly bounded trust-region refinement may tolerate
one small historical regression only when at least 80% of folds are dual-metric safe, at least two
folds are Pareto wins, every regression is at most 15%, both mean metrics improve by at least 10%,
and the deployment forecast stays close to the anchor. A stricter zero-regression micro-refinement
may use a 5% dual-metric mean-gain threshold only when every causal fold is non-worse, at least two
folds are Pareto wins, and the same 5% mean/10% point deployment trust region holds. A specialist
whose assumptions are
conditional should implement a causal applicability check and fall back exactly to repeat-last
when that check fails; this is abstention, not an arbitrary ensemble. Learned near-constant level
relocations are rejected when the numeric evidence does not independently support a persistent
regime, even if their historical errors happen to be small. Never branch on this
exact supplied series length or copy supplied values into code. If no supported specialist exists,
return no new specialist rather than inventing methods to fill a quota. Optimize for reliable
switching rather than one spectacular but brittle fit.

An unbounded learned specialist must also have departed from exact repeat-last persistence on an
exact deployment-horizon historical audit. A specialist that reproduced repeat-last on every such audit
has not validated its active branch and may only be used as a tightly bounded trust-region
refinement. Host declared-period seasonal candidates are separately admitted only when the numeric
evidence contains at least three complete cycles, support from multiple historical windows, and
stable cycle amplitude.
"""

REVISION_PROMPT = """You are the numbers-only single-agent revision engineer.
The parent numerical framework was evaluated on historical cutoffs. You may rewrite the entire
algorithm or generate multiple competing single-method descendants. Use only the anonymous,
unordered fold cards, execution errors, deterministic numeric evidence, and historical numbers; do
not use future labels or textual context. Positive normalized gain means the parent beat the anchor;
signed bias/change/amplitude diagnostics describe the parent's error, not a forecast target. Diagnose
whether the failure is phase, trend, regime, scale, outlier sensitivity, or horizon mismatch. Change
only the mechanism implicated by that diagnosis. Prefer a causal applicability check with an exact
repeat-last fallback when a specialist is valid only in some regimes. Every inapplicable,
ambiguous, insufficient-data, or failed-check path must repeat the latest finite history value;
it must not substitute a recent median, local mean, or fitted level and call that specialist
activation. Do not hide instability
inside a weighted ensemble. Never branch on a fold ID/order, a validation horizon/role/count, an
exact train/history length, or equality to likely validation constants. Any horizon adaptation must
be smooth or causally re-estimated from the supplied history rather than inferred from validation
geometry. Each descendant must state a falsifiable assumption and why it addresses the observed
failure. Preserve
the required forecast(history, horizon, frequency) signature and return exactly one JSON object:
{"programs": [{"name": "short_snake_case", "description": "when to use it",
"assumption": "falsifiable numeric assumption", "failure_condition": "when it fails",
"knowledge_ids": ["cited external knowledge entry IDs"], "prior_confidence": 0.0,
"code": "def forecast(...): ..."}]}

Every host evidence assessment remains a falsifiable hypothesis during revision. A descendant may
use a `supported`/`persistent_regime` family at most once. Do not turn an ambiguous seasonal card
or a failed structural check into a descendant; return no descendant for that family. Do not create
an `absent`/`no_regime`/`insufficient` family, and resolve conflicting cards by abstaining. Never
encode the current evidence window lengths, assessment, n, or H as exact program branches. The
trailing `numeric_profile` contains legacy coarse compatibility fields and must never override a
more specific v4 evidence card. The host already evaluates a fixed structural state-cycle
candidate; do not duplicate, rename, or mutate it.

Learned near-constant level relocations are rejected unless the independent regime evidence is
`persistent_regime`. Forecast difference from the anchor is not, by itself, proof that the
specialized branch activated.

For an unbounded learned descendant, the host additionally requires its non-persistence branch to have
activated on an exact deployment-horizon historical audit. Do not rely on having more observations
at the current endpoint to activate a branch that reproduced repeat-last on every same-horizon
audit. Such a descendant must abstain or remain inside the bounded trust region.

When the parent cites external knowledge, revise it as a falsifiable prior: preserve only relevant
allowed IDs, lower confidence when hindcasts contradict it, and never cite an ID not supplied. The
host accepts a normal switch only when the deployment forecast is genuinely different from the
anchor, all folds are dual-metric safe, at least two distinct folds are Pareto wins, and aggregate
gain is at least 10%; do not manufacture anchor-equal folds to game this rule. A catastrophic rescue
cannot excuse even one worse fold. The only bounded-regret exception is a host-checked trust-region
refinement whose current forecast remains close to the anchor. A zero-regression micro-refinement
may instead use a 5% dual-metric gain only when every fold is non-worse, at least two folds win,
and the deployment forecast stays within the same 5% mean/10% point trust region.
"""


@dataclass(frozen=True)
class ForecastProgram:
    name: str
    description: str
    assumption: str
    failure_condition: str
    code: str
    knowledge_ids: tuple[str, ...] = ()
    prior_confidence: float | None = None
    generation: int = 0
    source: str = "generated"


@dataclass(frozen=True)
class ValidatedProgram:
    program: ForecastProgram
    forecast: tuple[float, ...]
    hindcast_smae: float
    hindcast_srmse: float
    fold_smae: tuple[float, ...]
    fold_srmse: tuple[float, ...]
    fold_errors: tuple[str, ...]
    sandbox_result: SandboxResult
    fold_smae_raw: tuple[float, ...] = ()
    fold_srmse_raw: tuple[float, ...] = ()


@dataclass(frozen=True)
class CodingEvolutionResult:
    candidates: tuple[ValidatedProgram, ...]
    selected: ValidatedProgram
    initial_best: ValidatedProgram
    repeat_last_hindcast_smae: float
    repeat_last_hindcast_srmse: float
    improvement_smae: float
    improvement_srmse: float
    saved_skill_name: str | None
    knowledge_base_version: str | None = None
    retrieved_knowledge_ids: tuple[str, ...] = ()
    selected_knowledge_ids: tuple[str, ...] = ()
    diagnostic_profile: DiagnosticProfile | None = None
    reference_anchor_name: str = "repeat_last"
    selection_mode: str = "preserve_anchor"
    selection_fold_wins: int = 0
    selection_fold_count: int = 0
    validation_has_representative_fold: bool = False

    @property
    def improvement(self) -> float:
        """Legacy alias; current selection is governed by the metric pair."""
        return self.improvement_smae


@dataclass(frozen=True)
class CodingEvolutionConfig:
    setting: CodingSetting = "statistics"
    initial_programs: int = 3
    mutations: int = 1
    mutation_children: int = 1
    validation_folds: int = 3
    validation_horizon: int = 8
    minimum_validation_history: int = 16
    minimum_library_improvement: float = 0.0
    use_external_knowledge: bool = False


def _metric_key(program: ValidatedProgram) -> tuple[float, float, float, float]:
    """Rank by official capped metrics, then recover information hidden by the cap."""
    raw_srmse = program.fold_srmse_raw or program.fold_srmse
    raw_smae = program.fold_smae_raw or program.fold_smae
    return (
        program.hindcast_srmse,
        program.hindcast_smae,
        _overflow_safe_mean(tuple(raw_srmse)),
        _overflow_safe_mean(tuple(raw_smae)),
    )


def _finite_history(values: tuple[float, ...]) -> tuple[float, ...]:
    """Causally forward-fill numeric gaps so the safety fallback cannot fail."""
    parsed: list[float] = []
    latest = 0.0
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError, OverflowError):
            candidate = float("nan")
        if math.isfinite(candidate):
            latest = candidate
        parsed.append(latest)
    if not parsed:
        return (0.0,)
    return tuple(parsed)


def _overflow_safe_median(values: tuple[float, ...]) -> float:
    """Compute a finite median without summing two near-limit floats."""
    if not values:
        return 0.0
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    result = statistics.median(value / scale for value in values) * scale
    return result if math.isfinite(result) else values[-1]


def _overflow_safe_mean(values: tuple[float, ...]) -> float:
    """Average finite metrics without overflowing their intermediate sum."""
    if not values:
        return math.inf
    if not all(math.isfinite(value) for value in values):
        return math.inf
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    result = statistics.fmean(value / scale for value in values) * scale
    return result if math.isfinite(result) else math.inf


def _finite_task(task: Task) -> Task:
    history = _finite_history(task.history_values)
    return task if history == task.history_values else replace(task, history_values=history)


def _seasonal_period(task: Task) -> int | None:
    try:
        value = float(task.seasonal_period)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not value.is_integer():
        return None
    period = int(value)
    return period if 2 <= period and 2 * period <= len(task.history_values) else None


def _pareto_better(
    candidate: ValidatedProgram,
    reference: ValidatedProgram | tuple[float, float],
    *,
    margin: float = 0.0,
) -> bool:
    reference_smae, reference_srmse = (
        (reference.hindcast_smae, reference.hindcast_srmse)
        if isinstance(reference, ValidatedProgram)
        else reference
    )
    return (
        candidate.hindcast_smae <= reference_smae
        and candidate.hindcast_srmse <= reference_srmse
        and (
            candidate.hindcast_smae + margin < reference_smae
            or candidate.hindcast_srmse + margin < reference_srmse
        )
    )


def _forecast_materially_differs(
    candidate: tuple[float, ...],
    anchor: tuple[float, ...],
    *,
    relative_tolerance: float = 1e-6,
) -> bool:
    """Distinguish a deployed method change from an anchor-equivalent forecast."""
    if not candidate or len(candidate) != len(anchor):
        return False
    if not all(math.isfinite(value) for value in (*candidate, *anchor)):
        return False
    normalized_differences = tuple(
        abs(candidate_value / scale - anchor_value / scale)
        for candidate_value, anchor_value in zip(candidate, anchor, strict=True)
        for scale in (max(1.0, abs(candidate_value), abs(anchor_value)),)
    )
    return _overflow_safe_mean(normalized_differences) > relative_tolerance


def _unsupported_large_flat_level_relocation(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
    profile: DiagnosticProfile | None,
    *,
    maximum_flatness: float = 0.02,
    minimum_mean_deviation: float = 0.10,
    minimum_point_deviation: float = 0.15,
) -> bool:
    """Reject an ungrounded learned level rewrite masquerading as activation.

    Learned programs can implement their own fallback.  A nearly constant forecast
    far from persistence is therefore accepted only when independent, label-free
    diagnostics identify a persistent regime.  Host-authored candidates retain
    their ordinary strict gates, including non-flat seasonal/state trajectories.
    """
    if candidate.program.source not in {
        "generated",
        "knowledge",
        "library",
        "mutation",
        "knowledge_mutation",
    }:
        return False
    if (
        not candidate.forecast
        or len(candidate.forecast) != len(anchor.forecast)
        or not all(
            math.isfinite(value)
            for value in (*candidate.forecast, *anchor.forecast)
        )
    ):
        return True
    deviations = tuple(
        abs(candidate_value / scale - anchor_value / scale)
        for candidate_value, anchor_value in zip(
            candidate.forecast, anchor.forecast, strict=True
        )
        for scale in (max(1.0, abs(candidate_value), abs(anchor_value)),)
    )
    candidate_scale = max(1.0, *(abs(value) for value in candidate.forecast))
    normalized_candidate = tuple(
        value / candidate_scale for value in candidate.forecast
    )
    flatness = max(normalized_candidate) - min(normalized_candidate)
    persistent_regime = bool(
        profile is not None
        and profile.regime_ood_evidence.assessment == "persistent_regime"
    )
    return (
        flatness <= maximum_flatness
        and _overflow_safe_mean(deviations) > minimum_mean_deviation
        and max(deviations) > minimum_point_deviation
        and not persistent_regime
    )


def _fold_safe_better(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
    *,
    minimum_relative_gain: float = 0.10,
) -> bool:
    """Replace a safe anchor only after foldwise safety and a material gain."""
    if candidate.fold_errors:
        return False
    candidate_smae = candidate.fold_smae_raw or candidate.fold_smae
    candidate_srmse = candidate.fold_srmse_raw or candidate.fold_srmse
    anchor_smae = anchor.fold_smae_raw or anchor.fold_smae
    anchor_srmse = anchor.fold_srmse_raw or anchor.fold_srmse
    if not all((candidate_smae, candidate_srmse, anchor_smae, anchor_srmse)):
        return False
    if not all(
        math.isfinite(value)
        for values in (candidate_smae, candidate_srmse, anchor_smae, anchor_srmse)
        for value in values
    ):
        return False
    if not (
        len(candidate_smae)
        == len(candidate_srmse)
        == len(anchor_smae)
        == len(anchor_srmse)
    ):
        return False
    foldwise_safe = (
        all(
            candidate_value <= anchor_value
            for candidate_value, anchor_value in zip(
                candidate_smae, anchor_smae, strict=True
            )
        )
        and all(
            candidate_value <= anchor_value
            for candidate_value, anchor_value in zip(
                candidate_srmse, anchor_srmse, strict=True
            )
        )
    )
    if not foldwise_safe:
        return False
    pareto_wins = sum(
        candidate_fold_smae <= anchor_fold_smae
        and candidate_fold_srmse <= anchor_fold_srmse
        and (
            candidate_fold_smae < anchor_fold_smae
            or candidate_fold_srmse < anchor_fold_srmse
        )
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in zip(
            candidate_smae,
            candidate_srmse,
            anchor_smae,
            anchor_srmse,
            strict=True,
        )
    )
    if pareto_wins < 2:
        return False
    candidate_mean_smae = _overflow_safe_mean(tuple(candidate_smae))
    candidate_mean_srmse = _overflow_safe_mean(tuple(candidate_srmse))
    anchor_mean_smae = _overflow_safe_mean(tuple(anchor_smae))
    anchor_mean_srmse = _overflow_safe_mean(tuple(anchor_srmse))
    return (
        (
            candidate_mean_smae < anchor_mean_smae
            and candidate_mean_smae
            <= (1.0 - minimum_relative_gain) * anchor_mean_smae
        )
        or (
            candidate_mean_srmse < anchor_mean_srmse
            and candidate_mean_srmse
            <= (1.0 - minimum_relative_gain) * anchor_mean_srmse
        )
    )


def _catastrophic_anchor_rescue(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
    *,
    fold_horizons: tuple[int, ...] = (),
    representative_horizon: int = 0,
    minimum_relative_gain: float = 0.20,
    minimum_safe_fraction: float = 1.00,
    maximum_fold_regret: float = 0.25,
    finite_fold_bound: float = 5.0,
) -> bool:
    """Permit a tightly bounded rescue when the anchor itself is catastrophic."""
    if candidate.fold_errors:
        return False
    candidate_smae = tuple(candidate.fold_smae_raw or candidate.fold_smae)
    candidate_srmse = tuple(candidate.fold_srmse_raw or candidate.fold_srmse)
    anchor_smae = tuple(anchor.fold_smae_raw or anchor.fold_smae)
    anchor_srmse = tuple(anchor.fold_srmse_raw or anchor.fold_srmse)
    if not (
        candidate_smae
        and len(candidate_smae)
        == len(candidate_srmse)
        == len(anchor_smae)
        == len(anchor_srmse)
        == len(fold_horizons)
    ):
        return False
    if not all(
        math.isfinite(value)
        for values in (candidate_smae, candidate_srmse, anchor_smae, anchor_srmse)
        for value in values
    ):
        return False
    if max((*anchor_smae, *anchor_srmse)) <= finite_fold_bound:
        return False
    if max((*candidate_smae, *candidate_srmse)) > finite_fold_bound:
        return False
    deployment_folds = tuple(
        index
        for index, horizon in enumerate(fold_horizons)
        if representative_horizon > 0 and horizon >= representative_horizon
    )
    if not deployment_folds or any(
        candidate_smae[index] > anchor_smae[index]
        or candidate_srmse[index] > anchor_srmse[index]
        for index in deployment_folds
    ):
        return False
    safe_folds = sum(
        candidate_fold_smae <= anchor_fold_smae
        and candidate_fold_srmse <= anchor_fold_srmse
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in zip(
            candidate_smae,
            candidate_srmse,
            anchor_smae,
            anchor_srmse,
            strict=True,
        )
    )
    required_safe_folds = math.ceil(minimum_safe_fraction * len(candidate_smae))
    if safe_folds < required_safe_folds:
        return False
    regret_multiplier = 1.0 + maximum_fold_regret
    if any(
        candidate_fold_smae > regret_multiplier * anchor_fold_smae
        or candidate_fold_srmse > regret_multiplier * anchor_fold_srmse
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in zip(
            candidate_smae,
            candidate_srmse,
            anchor_smae,
            anchor_srmse,
            strict=True,
        )
    ):
        return False
    multiplier = 1.0 - minimum_relative_gain
    return (
        _overflow_safe_mean(candidate_smae)
        <= multiplier * _overflow_safe_mean(anchor_smae)
        and _overflow_safe_mean(candidate_srmse)
        <= multiplier * _overflow_safe_mean(anchor_srmse)
        and max(candidate_smae) <= multiplier * max(anchor_smae)
        and max(candidate_srmse) <= multiplier * max(anchor_srmse)
    )


def _bounded_trust_region_better(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
    *,
    minimum_relative_gain: float = 0.10,
    minimum_safe_fraction: float = 0.80,
    minimum_pareto_wins: int = 2,
    maximum_fold_regret: float = 0.15,
    maximum_mean_deviation: float = 0.05,
    maximum_point_deviation: float = 0.10,
) -> bool:
    """Allow only a small, well-supported forecast refinement around the anchor."""
    if candidate.fold_errors:
        return False
    candidate_smae = tuple(candidate.fold_smae_raw or candidate.fold_smae)
    candidate_srmse = tuple(candidate.fold_srmse_raw or candidate.fold_srmse)
    anchor_smae = tuple(anchor.fold_smae_raw or anchor.fold_smae)
    anchor_srmse = tuple(anchor.fold_srmse_raw or anchor.fold_srmse)
    if not (
        candidate_smae
        and len(candidate_smae)
        == len(candidate_srmse)
        == len(anchor_smae)
        == len(anchor_srmse)
    ):
        return False
    if not all(
        math.isfinite(value)
        for values in (candidate_smae, candidate_srmse, anchor_smae, anchor_srmse)
        for value in values
    ):
        return False
    fold_rows = tuple(
        zip(
            candidate_smae,
            candidate_srmse,
            anchor_smae,
            anchor_srmse,
            strict=True,
        )
    )
    safe_folds = sum(
        candidate_fold_smae <= anchor_fold_smae
        and candidate_fold_srmse <= anchor_fold_srmse
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in fold_rows
    )
    pareto_wins = sum(
        candidate_fold_smae <= anchor_fold_smae
        and candidate_fold_srmse <= anchor_fold_srmse
        and (
            candidate_fold_smae < anchor_fold_smae
            or candidate_fold_srmse < anchor_fold_srmse
        )
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in fold_rows
    )
    if safe_folds < math.ceil(minimum_safe_fraction * len(fold_rows)):
        return False
    if pareto_wins < minimum_pareto_wins:
        return False
    regret_multiplier = 1.0 + maximum_fold_regret
    if any(
        candidate_fold_smae > regret_multiplier * anchor_fold_smae
        or candidate_fold_srmse > regret_multiplier * anchor_fold_srmse
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in fold_rows
    ):
        return False
    gain_multiplier = 1.0 - minimum_relative_gain
    if not (
        _overflow_safe_mean(candidate_smae)
        <= gain_multiplier * _overflow_safe_mean(anchor_smae)
        and _overflow_safe_mean(candidate_srmse)
        <= gain_multiplier * _overflow_safe_mean(anchor_srmse)
    ):
        return False
    if (
        not candidate.forecast
        or len(candidate.forecast) != len(anchor.forecast)
        or not all(
            math.isfinite(value)
            for value in (*candidate.forecast, *anchor.forecast)
        )
    ):
        return False
    deviations = tuple(
        abs(candidate_value / scale - anchor_value / scale)
        for candidate_value, anchor_value in zip(
            candidate.forecast, anchor.forecast, strict=True
        )
        for scale in (max(1.0, abs(candidate_value), abs(anchor_value)),)
    )
    return (
        _overflow_safe_mean(deviations) <= maximum_mean_deviation
        and max(deviations) <= maximum_point_deviation
    )


def _zero_regression_micro_trust_better(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
) -> bool:
    """Allow a smaller gain only when every fold is safe and movement is tiny."""
    if not _forecast_materially_differs(candidate.forecast, anchor.forecast):
        return False
    return _bounded_trust_region_better(
        candidate,
        anchor,
        minimum_relative_gain=0.05,
        minimum_safe_fraction=1.00,
        minimum_pareto_wins=2,
        maximum_fold_regret=0.00,
        maximum_mean_deviation=0.05,
        maximum_point_deviation=0.10,
    )


def _fold_non_worse(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
) -> bool:
    """Validate a host-triggered hedge without requiring historical trigger frequency."""
    if candidate.fold_errors:
        return False
    candidate_smae = candidate.fold_smae_raw or candidate.fold_smae
    candidate_srmse = candidate.fold_srmse_raw or candidate.fold_srmse
    anchor_smae = anchor.fold_smae_raw or anchor.fold_smae
    anchor_srmse = anchor.fold_srmse_raw or anchor.fold_srmse
    values = (candidate_smae, candidate_srmse, anchor_smae, anchor_srmse)
    return (
        all(values)
        and len(candidate_smae) == len(anchor_smae)
        and len(candidate_srmse) == len(anchor_srmse)
        and all(math.isfinite(value) for group in values for value in group)
        and all(
            candidate_value <= anchor_value
            for candidate_value, anchor_value in zip(
                candidate_smae, anchor_smae, strict=True
            )
        )
        and all(
            candidate_value <= anchor_value
            for candidate_value, anchor_value in zip(
                candidate_srmse, anchor_srmse, strict=True
            )
        )
    )


def _uniform_fold_relative_gain(
    candidate: ValidatedProgram,
    anchor: ValidatedProgram,
    *,
    minimum_relative_gain: float = 0.10,
) -> bool:
    """Require every causal fold and both metrics to improve materially."""
    if candidate.fold_errors:
        return False
    candidate_smae = tuple(candidate.fold_smae_raw or candidate.fold_smae)
    candidate_srmse = tuple(candidate.fold_srmse_raw or candidate.fold_srmse)
    anchor_smae = tuple(anchor.fold_smae_raw or anchor.fold_smae)
    anchor_srmse = tuple(anchor.fold_srmse_raw or anchor.fold_srmse)
    if not (
        candidate_smae
        and len(candidate_smae)
        == len(candidate_srmse)
        == len(anchor_smae)
        == len(anchor_srmse)
    ):
        return False
    if not all(
        math.isfinite(value) and value >= 0.0
        for values in (candidate_smae, candidate_srmse, anchor_smae, anchor_srmse)
        for value in values
    ):
        return False
    multiplier = 1.0 - minimum_relative_gain
    return all(
        anchor_fold_smae > 0.0
        and anchor_fold_srmse > 0.0
        and candidate_fold_smae <= multiplier * anchor_fold_smae
        and candidate_fold_srmse <= multiplier * anchor_fold_srmse
        for candidate_fold_smae, candidate_fold_srmse, anchor_fold_smae, anchor_fold_srmse
        in zip(
            candidate_smae,
            candidate_srmse,
            anchor_smae,
            anchor_srmse,
            strict=True,
        )
    )


class CodingEvolutionAgent:
    """Generate, hindcast, revise, and retain only validated numbers-only skills."""

    def __init__(
        self,
        llm: LLMClient,
        library: SkillLibrary | None = None,
        config: CodingEvolutionConfig | None = None,
        tsfm_forecaster: NumericForecaster | None = None,
        *,
        generation_prompt: str = GENERATION_PROMPT,
        revision_prompt: str = REVISION_PROMPT,
        knowledge_base: TimeSeriesKnowledgeBase | None = None,
    ) -> None:
        self.llm = llm
        self.library = library
        self.config = config or CodingEvolutionConfig()
        self.tsfm_forecaster = tsfm_forecaster
        self.generation_prompt = generation_prompt
        self.revision_prompt = revision_prompt
        self.knowledge_base = knowledge_base
        if self.config.use_external_knowledge and self.knowledge_base is None:
            try:
                self.knowledge_base = TimeSeriesKnowledgeBase()
            except Exception:
                self.knowledge_base = None

    def run_task(
        self, task: Task, *, allow_skill_writes: bool = True
    ) -> CodingEvolutionResult:
        validation_task = task
        task = _finite_task(task)
        numeric_profile = self._numeric_profile(task)
        try:
            knowledge = (
                self._knowledge(task, numeric_profile)
                if numeric_profile is not None
                else None
            )
        except Exception:
            # External diagnostics are optional.  A finite numeric task must
            # still reach the deterministic anchors when diagnostics overflow
            # or a knowledge artifact is unusable.
            knowledge = None
        anchors = self._anchor_programs(task)
        reserved_program_names = (
            {"tsfm_backbone"}
            if self.config.setting in {"tsfm", "combined"}
            else set()
        )
        try:
            library_programs = [
                replace(program, source="library")
                for program in self._library_programs()
            ]
        except Exception:
            library_programs = []
        programs = [*anchors] if self.config.setting == "tsfm" else [
            *anchors, *library_programs
        ]
        if self.config.setting != "tsfm":
            try:
                generated_programs = self._generate(
                    task,
                    knowledge,
                    numeric_profile,
                )
            except TransientLLMError:
                raise
            except Exception:
                generated_programs = []
            programs.extend(
                replace(
                    program,
                    source=("knowledge" if program.knowledge_ids else "generated"),
                )
                for program in generated_programs
            )
            programs = self._unique_program_names(
                programs,
                existing_names=reserved_program_names,
            )
        used_program_names = reserved_program_names | {
            program.name for program in programs
        }
        validated = [
            candidate
            for program in programs
            if (
                candidate := self._safe_validate(
                    task, program, validation_task=validation_task
                )
            )
        ]
        if self.config.setting in {"tsfm", "combined"}:
            if self.tsfm_forecaster is not None:
                tsfm_candidate = self._validate_tsfm(
                    task, validation_task=validation_task
                )
                if tsfm_candidate is not None:
                    validated.append(tsfm_candidate)
        if not validated:
            validated = [
                candidate
                for program in anchors
                if (
                    candidate := self._safe_validate(
                        task, program, validation_task=validation_task
                    )
                ) is not None
            ]
        validated = [
            item
            for item in validated
            if item is not None and self._current_input_stable(task, item)
        ]
        if not validated:
            validated = [
                candidate
                for program in anchors
                if (
                    candidate := self._safe_validate(
                        task, program, validation_task=validation_task
                    )
                ) is not None
                and self._current_input_stable(task, candidate)
            ]
        if not validated:
            validated = [self._host_native_fallback(task)]
        initial_best = self._safe_anchor_selection(
            validation_task,
            validated,
            numeric_profile=numeric_profile,
        )
        reference_anchor, _terminal_hedge, _terminal_projection = (
            self._reference_anchor(validation_task, validated)
        )
        all_candidates = list(validated)

        generated_candidates = [
            item
            for item in validated
            if item.program.source
            in {"generated", "knowledge", "library", "mutation", "knowledge_mutation"}
        ]
        plain_candidates = [
            item for item in generated_candidates if not self._knowledge_lineage(item.program)
        ]
        knowledge_candidates = [
            item for item in generated_candidates if self._knowledge_lineage(item.program)
        ]
        parents = {
            lineage: min(items, key=_metric_key)
            for lineage, items in (
                ("plain", plain_candidates),
                ("knowledge", knowledge_candidates),
            )
            if items
        }
        for generation in range(1, self.config.mutations + 1):
            if not parents:
                break
            next_parents = dict(parents)
            for lineage, parent in parents.items():
                try:
                    mutations = self._mutate(
                        task,
                        parent,
                        generation,
                        knowledge if lineage == "knowledge" else None,
                        validation_task=validation_task,
                        reference_anchor=reference_anchor,
                        numeric_profile=numeric_profile,
                    )
                except TransientLLMError:
                    raise
                except Exception:
                    mutations = []
                mutation_source = (
                    "knowledge_mutation"
                    if lineage == "knowledge"
                    else "mutation"
                )
                mutations = [
                    replace(program, source=mutation_source)
                    for program in mutations
                ]
                mutations = self._unique_program_names(
                    mutations,
                    existing_names=used_program_names,
                )
                used_program_names.update(program.name for program in mutations)
                children = [
                    candidate
                    for program in mutations
                    if (
                        candidate := self._safe_validate(
                            task, program, validation_task=validation_task
                        )
                    )
                    and self._current_input_stable(task, candidate)
                ]
                all_candidates.extend(children)
                if children:
                    child = min(children, key=_metric_key)
                    if _pareto_better(child, parent) or (
                        child.hindcast_smae == parent.hindcast_smae
                        and child.hindcast_srmse == parent.hindcast_srmse
                        and _metric_key(child) < _metric_key(parent)
                    ):
                        next_parents[lineage] = child
            parents = next_parents

        selected = self._safe_anchor_selection(
            validation_task,
            all_candidates,
            numeric_profile=numeric_profile,
        )
        selection_fold_wins, selection_fold_count = self._fold_win_count(
            selected, reference_anchor
        )
        baseline_smae, baseline_srmse = self._repeat_last_hindcast(validation_task)
        saved_name = None
        if (
            allow_skill_writes
            and self.library is not None
            and selected.program.source
            in {"generated", "knowledge", "mutation", "knowledge_mutation"}
            and _pareto_better(
                selected,
                (baseline_smae, baseline_srmse),
                margin=self.config.minimum_library_improvement,
            )
        ):
            try:
                self.library.add(
                    Skill(
                        skill_id=str(uuid.uuid4()),
                        name=selected.program.name,
                        description=selected.program.description,
                        code=selected.program.code,
                        created_from_task=task.task_id,
                        assumption=selected.program.assumption,
                        failure_condition=selected.program.failure_condition,
                        validation_smae=selected.hindcast_smae,
                        validation_srmse=selected.hindcast_srmse,
                    )
                )
            except Exception:
                pass
            else:
                saved_name = selected.program.name
        knowledge_base_version = None
        if knowledge is not None and self.knowledge_base is not None:
            try:
                knowledge_base_version = self.knowledge_base.version
            except Exception:
                pass
        return CodingEvolutionResult(
            candidates=tuple(sorted(all_candidates, key=_metric_key)),
            selected=selected,
            initial_best=initial_best,
            repeat_last_hindcast_smae=baseline_smae,
            repeat_last_hindcast_srmse=baseline_srmse,
            improvement_smae=initial_best.hindcast_smae - selected.hindcast_smae,
            improvement_srmse=initial_best.hindcast_srmse - selected.hindcast_srmse,
            saved_skill_name=saved_name,
            knowledge_base_version=knowledge_base_version,
            retrieved_knowledge_ids=(knowledge.entry_ids if knowledge is not None else ()),
            selected_knowledge_ids=selected.program.knowledge_ids,
            diagnostic_profile=(
                knowledge.profile if knowledge is not None else numeric_profile
            ),
            reference_anchor_name=reference_anchor.program.name,
            selection_mode=(
                "preserve_anchor"
                if selected.program.name == reference_anchor.program.name
                else (
                    "strict_safe_specialized_switch"
                    if _fold_safe_better(selected, reference_anchor)
                    else (
                        "bounded_trust_region_refinement"
                        if _bounded_trust_region_better(
                            selected, reference_anchor
                        )
                        else (
                            "zero_regression_micro_trust_refinement"
                            if _zero_regression_micro_trust_better(
                                selected, reference_anchor
                            )
                            else "catastrophic_anchor_rescue"
                        )
                    )
                )
            ),
            selection_fold_wins=selection_fold_wins,
            selection_fold_count=selection_fold_count,
            validation_has_representative_fold=(
                self._has_representative_horizon_fold(validation_task)
            ),
        )

    def _library_programs(self) -> list[ForecastProgram]:
        if self.library is None:
            return []
        return [
            ForecastProgram(
                name=skill.name,
                description=skill.description,
                assumption=skill.assumption or skill.description,
                failure_condition=skill.failure_condition or "Historical holdout performance degrades.",
                code=skill.code,
                source="library",
            )
            for skill in self.library.all()
        ]

    def _knowledge(
        self,
        task: Task,
        profile: DiagnosticProfile | None = None,
    ) -> KnowledgeSelection | None:
        if self.knowledge_base is None or self.config.setting not in {"statistics", "combined"}:
            return None
        selection = self.knowledge_base.retrieve(
            task,
            include_tsfm=self.config.setting == "combined",
            profile=profile,
        )
        return selection if isinstance(selection, KnowledgeSelection) else None

    @staticmethod
    def _numeric_profile(task: Task) -> DiagnosticProfile | None:
        """Expose the same label-free diagnostics with or without Setting 2."""
        try:
            profile = diagnose(task)
        except Exception:
            return None
        return profile if isinstance(profile, DiagnosticProfile) else None

    def _generate(
        self,
        task: Task,
        knowledge: KnowledgeSelection | None,
        numeric_profile: DiagnosticProfile | None = None,
    ) -> list[ForecastProgram]:
        setting = self.config.setting
        guidance = ""
        if setting in {"statistics", "combined"}:
            guidance += "\n" + STATISTICAL_SKILL_DICTIONARY
        if setting in {"tsfm", "combined"}:
            guidance += (
                "\nA separately executed TSFM candidate may be supplied by the harness. "
                "Your code should add transparent alternatives, not imitate neural weights."
            )
        profile = numeric_profile or (knowledge.profile if knowledge is not None else None)
        user = self._numeric_payload(task, profile)
        if self.library is not None:
            user += "\nReusable skill summaries:\n" + self.library.list_for_prompt()
        programs = self._call_programs(
            self.generation_prompt + guidance,
            user,
            generation=0,
            limit=self.config.initial_programs,
        )
        if knowledge is None or self.knowledge_base is None:
            return programs
        try:
            knowledge_prompt = knowledge.prompt_text(
                self.knowledge_base.sources, include_profile=False
            )
        except TransientLLMError:
            raise
        except Exception:
            return programs
        conditioned = self._call_programs(
            self.generation_prompt
            + guidance
            + "\n\n"
            + knowledge_prompt,
            user,
            generation=0,
            limit=self.config.initial_programs,
            allowed_knowledge_ids=knowledge.entry_ids,
        )
        return self._unique_program_names(
            [*programs, *(replace(item, source="knowledge") for item in conditioned)]
        )

    def _mutate(
        self,
        task: Task,
        parent: ValidatedProgram,
        generation: int,
        knowledge: KnowledgeSelection | None = None,
        *,
        validation_task: Task | None = None,
        reference_anchor: ValidatedProgram | None = None,
        numeric_profile: DiagnosticProfile | None = None,
    ) -> list[ForecastProgram]:
        cited_ids = tuple(
            entry_id
            for entry_id in parent.program.knowledge_ids
            if knowledge is not None and entry_id in knowledge.entry_ids
        )
        cited_entries = (
            tuple(item for item in knowledge.entries if item.entry_id in cited_ids)
            if knowledge is not None
            else ()
        )
        fold_task = validation_task or task
        folds = self._folds(fold_task)
        parent_smae_raw = parent.fold_smae_raw or parent.fold_smae
        parent_srmse_raw = parent.fold_srmse_raw or parent.fold_srmse
        anchor_smae_raw = (
            reference_anchor.fold_smae_raw or reference_anchor.fold_smae
            if reference_anchor is not None
            else ()
        )
        anchor_srmse_raw = (
            reference_anchor.fold_srmse_raw or reference_anchor.fold_srmse
            if reference_anchor is not None
            else ()
        )

        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(value) else None

        def normalized_gain(anchor_value: float, parent_value: float) -> float | None:
            if not math.isfinite(anchor_value) or not math.isfinite(parent_value):
                return None
            scale = max(abs(anchor_value), abs(parent_value), _MIN_NORMAL_FLOAT)
            return anchor_value / scale - parent_value / scale

        def fold_forecast(
            candidate: ValidatedProgram | None,
            train: tuple[float, ...],
            horizon: int,
        ) -> tuple[float, ...] | None:
            if candidate is None:
                return None
            if candidate.program.source == "tsfm":
                if self.tsfm_forecaster is None:
                    return None
                try:
                    prediction = tuple(
                        float(value)
                        for value in self.tsfm_forecaster.forecast(
                            train,
                            horizon,
                            fold_task.frequency,
                        )
                    )
                except Exception:
                    return None
            else:
                try:
                    execution = run_forecast_code(
                        candidate.program.code,
                        list(train),
                        horizon,
                        fold_task.frequency,
                    )
                except Exception:
                    return None
                if not execution.ok or execution.forecast is None:
                    return None
                prediction = execution.forecast
            if len(prediction) != horizon or not all(
                math.isfinite(value) for value in prediction
            ):
                return None
            return prediction

        validation_folds: list[dict[str, object]] = []
        for index, (train, target) in enumerate(folds):
            parent_raw_smae = (
                parent_smae_raw[index] if index < len(parent_smae_raw) else float("inf")
            )
            parent_raw_srmse = (
                parent_srmse_raw[index] if index < len(parent_srmse_raw) else float("inf")
            )
            anchor_raw_smae = (
                anchor_smae_raw[index] if index < len(anchor_smae_raw) else float("inf")
            )
            anchor_raw_srmse = (
                anchor_srmse_raw[index] if index < len(anchor_srmse_raw) else float("inf")
            )
            metrics_are_finite = all(
                math.isfinite(value)
                for value in (
                    parent_raw_smae,
                    parent_raw_srmse,
                    anchor_raw_smae,
                    anchor_raw_srmse,
                )
            )
            pareto_safe = (
                metrics_are_finite
                and parent_raw_smae <= anchor_raw_smae
                and parent_raw_srmse <= anchor_raw_srmse
            )
            pareto_win = pareto_safe and (
                parent_raw_smae < anchor_raw_smae
                or parent_raw_srmse < anchor_raw_srmse
            )
            if not metrics_are_finite:
                status = "invalid"
            elif pareto_win:
                status = "pareto_win"
            elif pareto_safe:
                status = "tie"
            elif (
                parent_raw_smae > anchor_raw_smae
                and parent_raw_srmse > anchor_raw_srmse
            ):
                status = "regression"
            else:
                status = "mixed"

            parent_prediction = fold_forecast(parent, train, len(target))
            anchor_prediction = fold_forecast(
                reference_anchor,
                train,
                len(target),
            )
            diagnostics: dict[str, float | None] = {
                "parent_vs_anchor_forecast_distance": None,
                "parent_mean_bias": None,
                "parent_endpoint_bias": None,
                "parent_change_error": None,
                "parent_log_amplitude_ratio": None,
            }
            if parent_prediction is not None and anchor_prediction is not None:
                scale = max(
                    1.0,
                    *(abs(value) for value in target),
                    *(abs(value) for value in parent_prediction),
                    *(abs(value) for value in anchor_prediction),
                )
                normalized_parent = tuple(value / scale for value in parent_prediction)
                normalized_anchor = tuple(value / scale for value in anchor_prediction)
                normalized_target = tuple(value / scale for value in target)
                parent_std = statistics.pstdev(normalized_parent)
                target_std = statistics.pstdev(normalized_target)
                diagnostics = {
                    "parent_vs_anchor_forecast_distance": finite_or_none(
                        statistics.fmean(
                            abs(parent_value - anchor_value)
                            for parent_value, anchor_value in zip(
                                normalized_parent,
                                normalized_anchor,
                                strict=True,
                            )
                        )
                    ),
                    "parent_mean_bias": finite_or_none(
                        statistics.fmean(
                            parent_value - target_value
                            for parent_value, target_value in zip(
                                normalized_parent,
                                normalized_target,
                                strict=True,
                            )
                        )
                    ),
                    "parent_endpoint_bias": finite_or_none(
                        normalized_parent[-1] - normalized_target[-1]
                    ),
                    "parent_change_error": finite_or_none(
                        (normalized_parent[-1] - normalized_parent[0])
                        - (normalized_target[-1] - normalized_target[0])
                    ),
                    "parent_log_amplitude_ratio": finite_or_none(
                        max(
                            -20.0,
                            min(
                                20.0,
                                math.log(
                                    (parent_std + 1e-12)
                                    / (target_std + 1e-12)
                                ),
                            ),
                        )
                    ),
                }
            validation_folds.append(
                {
                    "status": status,
                    "normalized_gain_smae": normalized_gain(
                        anchor_raw_smae,
                        parent_raw_smae,
                    ),
                    "normalized_gain_srmse": normalized_gain(
                        anchor_raw_srmse,
                        parent_raw_srmse,
                    ),
                    **diagnostics,
                }
            )
        status_order = {
            "regression": 0,
            "mixed": 1,
            "invalid": 2,
            "tie": 3,
            "pareto_win": 4,
        }
        validation_folds.sort(
            key=lambda item: (
                status_order[str(item["status"])],
                item["normalized_gain_smae"]
                if item["normalized_gain_smae"] is not None
                else -math.inf,
                item["normalized_gain_srmse"]
                if item["normalized_gain_srmse"] is not None
                else -math.inf,
            )
        )
        for index, fold_card in enumerate(validation_folds, start=1):
            fold_card["fold_id"] = f"anonymous_{index}"
        pareto_win_count = sum(
            fold_card["status"] == "pareto_win"
            for fold_card in validation_folds
        )
        all_fold_dual_metric_safe = bool(validation_folds) and all(
            fold_card["status"] in {"pareto_win", "tie"}
            for fold_card in validation_folds
        )
        current_forecast_differs_from_anchor = (
            reference_anchor is not None
            and _forecast_materially_differs(
                parent.forecast,
                reference_anchor.forecast,
            )
        )
        representative_fold_present = self._has_representative_horizon_fold(
            fold_task
        )
        payload = {
            "task": json.loads(self._numeric_payload(task, numeric_profile)),
            "parent": {
                "name": parent.program.name,
                "description": parent.program.description,
                "assumption": parent.program.assumption,
                "failure_condition": parent.program.failure_condition,
                "code": parent.program.code,
                "source": parent.program.source,
                "knowledge_ids": list(cited_ids),
                "prior_confidence": parent.program.prior_confidence,
            },
            "historical_validation": {
                "mean_smae": parent.hindcast_smae,
                "mean_srmse": parent.hindcast_srmse,
                "execution_errors": list(parent.fold_errors),
                "reference_anchor": (
                    {
                        "name": reference_anchor.program.name,
                        "mean_smae": reference_anchor.hindcast_smae,
                        "mean_srmse": reference_anchor.hindcast_srmse,
                    }
                    if reference_anchor is not None
                    else None
                ),
                "protocol_summary": {
                    "cards_are_anonymous_and_unordered": True,
                    "representative_fold_present": representative_fold_present,
                    "current_forecast_differs_from_anchor": (
                        current_forecast_differs_from_anchor
                    ),
                    "forecast_difference_is_not_activation_proof": True,
                    "all_fold_dual_metric_safe": all_fold_dual_metric_safe,
                    "pareto_win_count": pareto_win_count,
                    "minimum_required_pareto_wins": 2,
                    "minimum_required_aggregate_gain": 0.10,
                },
                "strict_anchor_gate_passed": (
                    representative_fold_present
                    and current_forecast_differs_from_anchor
                    and _fold_safe_better(parent, reference_anchor)
                    if reference_anchor is not None
                    else False
                ),
                "validation_folds": validation_folds,
            },
        }
        guidance = self.revision_prompt
        if knowledge is not None and cited_entries and self.knowledge_base is not None:
            try:
                guidance += "\n\n" + KnowledgeSelection(
                    profile=knowledge.profile,
                    entries=cited_entries,
                ).prompt_text(self.knowledge_base.sources, include_profile=False)
            except TransientLLMError:
                raise
            except Exception:
                return []
        programs = self._call_programs(
            guidance,
            json.dumps(payload, ensure_ascii=False),
            generation=generation,
            limit=self.config.mutation_children,
            allowed_knowledge_ids=cited_ids,
        )
        source = "knowledge_mutation" if self._knowledge_lineage(parent.program) else "mutation"
        return [replace(program, source=source) for program in programs]

    @staticmethod
    def _knowledge_lineage(program: ForecastProgram) -> bool:
        return program.source in {"knowledge", "knowledge_mutation"}

    @staticmethod
    def _numeric_payload(
        task: Task,
        profile: DiagnosticProfile | None | object = _PROFILE_UNSET,
    ) -> str:
        if profile is _PROFILE_UNSET:
            profile = CodingEvolutionAgent._numeric_profile(task)
        if profile is not None and not isinstance(profile, DiagnosticProfile):
            profile = None
        numeric_evidence = (
            {
                "profile_version": profile.profile_version,
                "trend": asdict(profile.trend_evidence),
                "seasonality": [
                    asdict(card) for card in profile.seasonality_evidence
                ],
                "regime_ood": asdict(profile.regime_ood_evidence),
                "risk_flags": list(profile.risk_flags),
            }
            if profile is not None
            else {
                "profile_version": "v4.1-causal-phase-replay",
                "trend": {"assessment": "insufficient"},
                "seasonality": [],
                "regime_ood": {"assessment": "insufficient"},
                "risk_flags": ["diagnostics_unavailable"],
            }
        )
        legacy_profile = (
            {
                "history_length": profile.history_length,
                "horizon": profile.horizon,
                "horizon_ratio": profile.horizon_ratio,
                "seasonal_period": profile.seasonal_period,
                "lag1_autocorrelation": profile.lag1_autocorrelation,
                "seasonal_autocorrelation": profile.seasonal_autocorrelation,
                "trend_effect_over_horizon": profile.trend_effect_over_horizon,
                "recent_level_shift": profile.recent_level_shift,
                "recent_trend_change": profile.recent_trend_change,
                "outlier_fraction": profile.outlier_fraction,
                "zero_fraction": profile.zero_fraction,
                "variance_ratio_recent_to_early": profile.variance_ratio_recent_to_early,
                "candidate_lags": profile.candidate_lags,
                "tags": profile.tags,
            }
            if profile is not None
            else None
        )
        payload = {
            "numeric_evidence": numeric_evidence,
            "evidence_contract": {
                "host_assessments_are_hypotheses": True,
                "specialist_requires_supported_or_recomputed_evidence": True,
                "ambiguous_requires_anchor_fallback": True,
                "inactive_specialist_policy": "exact_repeat_last",
                "non_anchor_fallback_is_contract_violation": True,
                "code_must_recompute_from_each_history_prefix": True,
                "no_exact_window_or_horizon_branching": True,
            },
            "history_values": list(_finite_history(task.history_values)),
            "horizon": task.prediction_length,
            "frequency": task.frequency,
            "declared_period_steps": _seasonal_period(task),
            "raw_seasonality_metadata": task.seasonal_period,
            "selection_contract": {
                "metrics": ["sMAE", "sRMSE"],
                "validation_protocol": (
                    "multiple hidden historical cutoffs, including a "
                    "deployment-scale audit when history permits; exact fold "
                    "geometry is intentionally withheld"
                ),
                "deployment_forecast_difference_required": True,
                "forecast_difference_is_not_activation_proof": True,
                "unbounded_learned_switch_requires_exact_horizon_activation": True,
                "host_seasonal_requires_three_cycles_multiwindow_amplitude_stability": True,
                "minimum_distinct_pareto_wins": 2,
                "minimum_aggregate_gain": 0.10,
                "bounded_trust_region": {
                    "minimum_safe_fraction": 0.80,
                    "minimum_pareto_wins": 2,
                    "maximum_fold_regret": 0.15,
                    "minimum_dual_metric_mean_gain": 0.10,
                    "maximum_mean_forecast_deviation": 0.05,
                    "maximum_point_forecast_deviation": 0.10,
                },
                "zero_regression_micro_trust": {
                    "all_folds_dual_metric_non_worse": True,
                    "minimum_pareto_wins": 2,
                    "minimum_dual_metric_mean_gain": 0.05,
                    "maximum_mean_forecast_deviation": 0.05,
                    "maximum_point_forecast_deviation": 0.10,
                },
                "catastrophic_rescue_requires_all_folds_safe": True,
                "switch_rule": (
                    "non-worse on every available causal fold for both metrics "
                    "with at least two distinct Pareto-winning cutoffs, a "
                    "materially different deployment forecast, and at least "
                    "10% aggregate gain over the numeric anchor; a "
                    "catastrophic-anchor rescue requires every fold to be "
                    "non-worse; only a forecast-close trust-region refinement "
                    "may tolerate one bounded historical regression; an all-fold "
                    "zero-regression micro-refinement may use a 5% dual-metric "
                    "gain while staying in the same deployment trust region; a learned "
                    "near-constant level relocation also requires independent "
                    "persistent-regime evidence; every other unbounded learned "
                    "switch must have activated on an exact-horizon historical "
                    "audit; host seasonal switches require a stable multi-cycle "
                    "numeric evidence card"
                ),
                "diagnostic_note": (
                    "candidate_lags are hypotheses, not validated periods; raw "
                    "seasonality metadata is not a step count unless "
                    "declared_period_steps is non-null"
                ),
            },
            # Keep the v3 flat fields for external/custom prompt compatibility,
            # but place them after the evidence-first inference contract.
            "numeric_profile": legacy_profile,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        )

    def _call_programs(
        self,
        system: str,
        user: str,
        *,
        generation: int,
        limit: int,
        allowed_knowledge_ids: tuple[str, ...] = (),
    ) -> list[ForecastProgram]:
        try:
            response = self.llm.complete(
                system=(
                    f"{system.rstrip()}\n\n"
                    f"Host execution budget: return at most {limit} program records. "
                    "Cover each distinct supported evidence family once, budget permitting, "
                    "before proposing variants. The host already supplies persistence, "
                    "robust-level, damped-trend, declared-period, and structural state-cycle "
                    "candidates. An active specialist must be structurally distinct, but every "
                    "inactive or failed-check branch must reproduce repeat-last exactly. A "
                    "data-adaptive specialist in a supported family is allowed. Return an empty "
                    "list only when no supported family remains beyond the host anchors."
                ),
                messages=[{"role": "user", "content": user}],
                # Candidate diversity comes from the required multi-program output.
                # Deterministic decoding keeps parent/child policy evaluation comparable.
                temperature=0.0,
            )
            payload = parse_json_object(response.text)
        except TransientLLMError:
            raise
        except Exception:
            return []
        records = payload.get("programs", [])
        if not isinstance(records, list):
            return []
        programs = []
        for index, record in enumerate(records[:limit]):
            if not isinstance(record, dict):
                continue
            fields = ("name", "description", "assumption", "failure_condition", "code")
            if not all(isinstance(record.get(field), str) and record[field].strip() for field in fields):
                continue
            raw_ids = record.get("knowledge_ids", ())
            knowledge_ids = (
                tuple(
                    dict.fromkeys(
                        str(value)
                        for value in raw_ids
                        if str(value) in allowed_knowledge_ids
                    )
                )
                if isinstance(raw_ids, list)
                else ()
            )
            raw_confidence = record.get("prior_confidence")
            prior_confidence = None
            if isinstance(raw_confidence, (int, float)) and not isinstance(
                raw_confidence, bool
            ):
                try:
                    parsed_confidence = float(raw_confidence)
                except (TypeError, ValueError, OverflowError):
                    pass
                else:
                    if math.isfinite(parsed_confidence):
                        prior_confidence = min(1.0, max(0.0, parsed_confidence))
            programs.append(
                ForecastProgram(
                    name=record["name"],
                    description=record["description"],
                    assumption=record["assumption"],
                    failure_condition=record["failure_condition"],
                    code=record["code"],
                    knowledge_ids=knowledge_ids,
                    prior_confidence=prior_confidence,
                    generation=generation,
                )
            )
        return programs

    @staticmethod
    def _unique_program_names(
        programs: list[ForecastProgram],
        *,
        existing_names: set[str] | None = None,
    ) -> list[ForecastProgram]:
        used = set(existing_names or ())
        unique = []
        for program in programs:
            name = program.name
            suffix = 2
            while name in used:
                name = f"{program.name}__{suffix}"
                suffix += 1
            used.add(name)
            unique.append(program if name == program.name else replace(program, name=name))
        return unique

    def _validate(
        self,
        task: Task,
        program: ForecastProgram,
        *,
        validation_task: Task | None = None,
    ) -> ValidatedProgram | None:
        validation_task = validation_task or task
        task = _finite_task(task)
        result = run_forecast_code(
            program.code,
            list(task.history_values),
            task.prediction_length,
            task.frequency,
        )
        if not result.ok or result.forecast is None:
            return None
        fold_smae = []
        fold_srmse = []
        fold_smae_raw = []
        fold_srmse_raw = []
        errors = []
        for train, target in self._folds(validation_task):
            fold = run_forecast_code(
                program.code, list(train), len(target), task.frequency
            )
            if not fold.ok or fold.forecast is None:
                errors.append(fold.error or "unknown sandbox failure")
                fold_smae.append(5.0)
                fold_srmse.append(5.0)
                fold_smae_raw.append(float("inf"))
                fold_srmse_raw.append(float("inf"))
            else:
                try:
                    scores = drcik_point_metrics(target, [fold.forecast])
                except ValueError as exc:
                    errors.append(str(exc))
                    scores = {
                        "smae": 5.0,
                        "srmse": 5.0,
                        "smae_raw": float("inf"),
                        "srmse_raw": float("inf"),
                    }
                fold_smae.append(scores["smae"])
                fold_srmse.append(scores["srmse"])
                fold_smae_raw.append(scores["smae_raw"])
                fold_srmse_raw.append(scores["srmse_raw"])
        if not fold_smae:
            fold_smae = [5.0]
            fold_srmse = [5.0]
            fold_smae_raw = [float("inf")]
            fold_srmse_raw = [float("inf")]
            errors.append("insufficient_history_for_hindcast")
        return ValidatedProgram(
            program=program,
            forecast=result.forecast,
            hindcast_smae=statistics.fmean(fold_smae),
            hindcast_srmse=statistics.fmean(fold_srmse),
            fold_smae=tuple(fold_smae),
            fold_srmse=tuple(fold_srmse),
            fold_errors=tuple(errors),
            sandbox_result=result,
            fold_smae_raw=tuple(fold_smae_raw),
            fold_srmse_raw=tuple(fold_srmse_raw),
        )

    def _safe_validate(
        self,
        task: Task,
        program: ForecastProgram,
        *,
        validation_task: Task | None = None,
    ) -> ValidatedProgram | None:
        """Treat a broken sandbox boundary as a failed optional challenger."""
        try:
            return self._validate(
                task,
                program,
                validation_task=validation_task,
            )
        except Exception:
            return None

    def _host_native_fallback(self, task: Task) -> ValidatedProgram:
        """Return one finite forecast even when every sandbox call failed."""
        history = _finite_history(task.history_values)
        projection = self._terminal_numeric_ood_projection(task)
        if projection is None:
            value = history[-1]
            program = self._fallback_program()
        else:
            width, _cleaned = projection
            start = len(history) - width
            prior = _overflow_safe_median(history[start - 7:start])
            suffix = _overflow_safe_median(history[start:])
            value = 0.5 * prior + 0.5 * suffix
            if not math.isfinite(value):
                value = history[-1]
            program = ForecastProgram(
                name="host_native_terminal_ood_hedge",
                description=(
                    "Host-native midpoint between the prior robust level and a "
                    "detected terminal regime candidate."
                ),
                assumption=(
                    "The terminal shift is ambiguous between a genuine regime and numeric OOD."
                ),
                failure_condition=(
                    "The terminal suffix is known to be entirely signal or entirely corruption."
                ),
                code=(
                    "import math\n"
                    "import statistics\n"
                    "def forecast(history, horizon, frequency):\n"
                    f"    width = {width}\n"
                    "    start = len(history) - width\n"
                    "    prior_values = history[start - 7:start]\n"
                    "    suffix_values = history[start:]\n"
                    "    prior_scale = max(abs(x) for x in prior_values) or 1.0\n"
                    "    suffix_scale = max(abs(x) for x in suffix_values) or 1.0\n"
                    "    prior = statistics.median([x / prior_scale for x in prior_values]) * prior_scale\n"
                    "    suffix = statistics.median([x / suffix_scale for x in suffix_values]) * suffix_scale\n"
                    "    value = 0.5 * prior + 0.5 * suffix\n"
                    "    if not math.isfinite(value):\n"
                    "        value = history[-1]\n"
                    "    return [value for _ in range(horizon)]\n"
                ),
                source="fallback",
            )
        forecast = tuple(value for _ in range(task.prediction_length))
        result = SandboxResult(
            ok=True,
            forecast=forecast,
            error=None,
            duration_ms=0.0,
        )
        return ValidatedProgram(
            program=program,
            forecast=forecast,
            hindcast_smae=5.0,
            hindcast_srmse=5.0,
            fold_smae=(5.0,),
            fold_srmse=(5.0,),
            fold_errors=("host_native_fallback_after_sandbox_failure",),
            sandbox_result=result,
            fold_smae_raw=(float("inf"),),
            fold_srmse_raw=(float("inf"),),
        )

    def validate_median(
        self,
        task: Task,
        candidates: tuple[ValidatedProgram, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None:
        """Replay a pointwise-median candidate on the same causal folds."""
        if not candidates:
            return None
        forecast = tuple(
            _overflow_safe_median(tuple(values))
            for values in zip(*(item.forecast for item in candidates), strict=True)
        )
        fold_smae = []
        fold_srmse = []
        for train, target in self._folds(task):
            member_forecasts = []
            for item in candidates:
                result = run_forecast_code(
                    item.program.code,
                    list(train),
                    len(target),
                    task.frequency,
                )
                if not result.ok or result.forecast is None:
                    return None
                member_forecasts.append(result.forecast)
            median_forecast = tuple(
                _overflow_safe_median(tuple(values))
                for values in zip(*member_forecasts, strict=True)
            )
            try:
                scores = drcik_point_metrics(target, [median_forecast])
            except ValueError:
                scores = {"smae": 5.0, "srmse": 5.0}
            fold_smae.append(scores["smae"])
            fold_srmse.append(scores["srmse"])
        return (
            (forecast, tuple(fold_smae), tuple(fold_srmse))
            if fold_smae
            else None
        )

    def score_program_folds(
        self,
        program: ForecastProgram,
        folds: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
        frequency: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """Replay one raw program against caller-supplied causal fold targets."""
        fold_smae = []
        fold_srmse = []
        for train, target in folds:
            result = run_forecast_code(
                program.code,
                list(train),
                len(target),
                frequency,
            )
            if not result.ok or result.forecast is None:
                return None
            try:
                scores = drcik_point_metrics(target, [result.forecast])
            except ValueError:
                scores = {"smae": 5.0, "srmse": 5.0}
            fold_smae.append(scores["smae"])
            fold_srmse.append(scores["srmse"])
        return (tuple(fold_smae), tuple(fold_srmse)) if fold_smae else None

    def _validate_tsfm(
        self, task: Task, *, validation_task: Task | None = None
    ) -> ValidatedProgram | None:
        validation_task = validation_task or task
        task = _finite_task(task)
        try:
            forecast = self.tsfm_forecaster.forecast(
                task.history_values, task.prediction_length, task.frequency
            )
            fold_smae = []
            fold_srmse = []
            fold_smae_raw = []
            fold_srmse_raw = []
            fold_errors = []
            for train, target in self._folds(validation_task):
                prediction = self.tsfm_forecaster.forecast(train, len(target), task.frequency)
                try:
                    scores = drcik_point_metrics(target, [prediction])
                except ValueError as exc:
                    fold_errors.append(str(exc))
                    scores = {
                        "smae": 5.0,
                        "srmse": 5.0,
                        "smae_raw": float("inf"),
                        "srmse_raw": float("inf"),
                    }
                fold_smae.append(scores["smae"])
                fold_srmse.append(scores["srmse"])
                fold_smae_raw.append(scores["smae_raw"])
                fold_srmse_raw.append(scores["srmse_raw"])
        except Exception:
            return None
        if (
            not fold_smae
            or fold_errors
            or not all(
                math.isfinite(value)
                for values in (fold_smae_raw, fold_srmse_raw)
                for value in values
            )
        ):
            return None
        try:
            forecast = tuple(float(value) for value in forecast)
        except (TypeError, ValueError, OverflowError):
            return None
        if len(forecast) != task.prediction_length or not all(
            math.isfinite(value) for value in forecast
        ):
            return None
        result = SandboxResult(ok=True, forecast=forecast, error=None, duration_ms=0.0)
        return ValidatedProgram(
            program=ForecastProgram(
                name="tsfm_backbone",
                description="Zero-shot time-series foundation-model forecast.",
                assumption="Patterns learned across many series transfer to this numeric history.",
                failure_condition="The task contains a regime or mechanism absent from the numeric input.",
                code="# External TSFM adapter; not a generated reusable Python skill.",
                source="tsfm",
            ),
            forecast=forecast,
            hindcast_smae=statistics.fmean(fold_smae) if fold_smae else 5.0,
            hindcast_srmse=statistics.fmean(fold_srmse) if fold_srmse else 5.0,
            fold_smae=tuple(fold_smae),
            fold_srmse=tuple(fold_srmse),
            fold_errors=(
                tuple(fold_errors)
                if fold_smae
                else ("insufficient_history_for_hindcast",)
            ),
            sandbox_result=result,
            fold_smae_raw=(
                tuple(fold_smae_raw) if fold_smae_raw else (float("inf"),)
            ),
            fold_srmse_raw=(
                tuple(fold_srmse_raw) if fold_srmse_raw else (float("inf"),)
            ),
        )

    def _folds(self, task: Task) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        history = task.history_values
        short_horizon = max(
            1, min(self.config.validation_horizon, task.prediction_length)
        )
        minimum = max(
            self.config.minimum_validation_history,
            short_horizon * 2,
        )
        possible = max(0, (len(history) - minimum) // short_horizon)
        folds = []
        # Search backward until enough *observed* targets are found.  Missing
        # recent labels should not erase otherwise valid causal validation.
        for offset in range(1, possible + 1):
            if len(folds) >= self.config.validation_folds:
                break
            cutoff = len(history) - offset * short_horizon
            if cutoff < minimum:
                continue
            target = self._finite_validation_target(
                history[cutoff : cutoff + short_horizon]
            )
            if target is not None:
                folds.append((_finite_history(history[:cutoff]), target))
        folds.reverse()

        # Short folds detect local instability; these two audits expose methods
        # whose apparently good short forecast drifts or explodes over the real
        # deployment horizon.  Each remains strictly causal.
        representative_horizon = min(
            task.prediction_length,
            max(short_horizon, math.ceil(task.prediction_length / 2)),
        )
        for audit_horizon in dict.fromkeys((
            task.prediction_length,
            representative_horizon,
        )):
            cutoff = len(history) - audit_horizon
            if (
                audit_horizon <= short_horizon
                or cutoff < max(self.config.minimum_validation_history, audit_horizon)
            ):
                continue
            target = self._finite_validation_target(history[cutoff:])
            if target is not None:
                audit_fold = (_finite_history(history[:cutoff]), target)
                if audit_fold not in folds:
                    folds.insert(0, audit_fold)
        return folds

    @staticmethod
    def _finite_validation_target(
        values: tuple[float, ...],
    ) -> tuple[float, ...] | None:
        """Never turn an absent observation into a hindcast label."""
        try:
            target = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            return None
        return target if target and all(math.isfinite(value) for value in target) else None

    def _repeat_last_hindcast(self, task: Task) -> tuple[float, float]:
        smae = []
        srmse = []
        for train, target in self._folds(task):
            prediction = [train[-1]] * len(target)
            try:
                scores = drcik_point_metrics(target, [prediction])
            except ValueError:
                scores = {"smae": 5.0, "srmse": 5.0}
            smae.append(scores["smae"])
            srmse.append(scores["srmse"])
        return (
            statistics.fmean(smae) if smae else 5.0,
            statistics.fmean(srmse) if srmse else 5.0,
        )

    @staticmethod
    def _fallback_program() -> ForecastProgram:
        return ForecastProgram(
            name="repeat_last",
            description="Repeat the latest observed value when no generated program is valid.",
            assumption="The local level persists over the forecast horizon.",
            failure_condition="The process changes level, trend, or regime after the cutoff.",
            code=(
                "def forecast(history, horizon, frequency):\n"
                "    value = history[-1] if history else 0.0\n"
                "    return [value for _ in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _structural_state_cycle() -> ForecastProgram:
        """A development-promoted, self-gating structural state-cycle prior."""
        return ForecastProgram(
            name="robust_state_cycle",
            description=(
                "Forecast recurring discrete states only when recent onset intervals, "
                "run durations, and state levels are all robustly stable."
            ),
            assumption=(
                "At least four recent state onsets and three complete non-boundary "
                "runs define a stable recurring two-state process."
            ),
            failure_condition=(
                "Transition timing, duration, or state separation is unstable; the "
                "program then returns repeat-last persistence exactly."
            ),
            code="""def forecast(history: list[float], horizon: int, frequency: str) -> list[float]:
    import math
    import statistics
    if horizon <= 0:
        return []
    x = []
    for v in history:
        try:
            z = float(v)
            if math.isfinite(z):
                x.append(z)
        except (TypeError, ValueError, OverflowError):
            pass
    if not x:
        return [0.0] * horizon
    fallback = float(x[-1])
    normalizer = max(abs(v) for v in x) or 1.0
    if not math.isfinite(normalizer):
        return [fallback] * horizon
    y = [v / normalizer for v in x]
    n = len(y)
    width = min(n, max(24, int(round(9.0 * math.sqrt(n)))))
    tail = y[-width:]
    ordered = sorted(tail)
    low = float(statistics.median(ordered[:max(3, len(ordered) // 2)]))
    upper = float(ordered[min(len(ordered) - 1, int(0.75 * (len(ordered) - 1)))])
    low_dev = float(statistics.median([abs(v - low) for v in tail]))
    scale = max(1e-9, abs(low) * 0.01, low_dev)
    if upper - low <= 6.0 * scale:
        return [fallback] * horizon
    threshold = low + 0.5 * (upper - low)
    active = [v > threshold for v in tail]
    starts = []
    ends = []
    for i, flag in enumerate(active):
        if flag and (i == 0 or not active[i - 1]):
            starts.append(i)
        if flag and (i == len(active) - 1 or not active[i + 1]):
            ends.append(i + 1)
    complete = []
    for s in starts:
        e = next((q for q in ends if q > s), None)
        if e is not None and s > 0 and e < len(tail):
            complete.append((s, e))
    if len(starts) < 4 or len(complete) < 3:
        return [fallback] * horizon
    recent_starts = starts[-4:]
    intervals = [recent_starts[i] - recent_starts[i - 1] for i in range(1, len(recent_starts))]
    period = float(statistics.median(intervals))
    period_dev = float(statistics.median([abs(v - period) for v in intervals]))
    period_max_dev = max(abs(v - period) for v in intervals)
    recent_complete = complete[-3:]
    durations = [e - s for s, e in recent_complete]
    duration = float(statistics.median(durations))
    duration_dev = max(abs(v - duration) for v in durations)
    high_values = [tail[i] for s, e in recent_complete for i in range(s, e)]
    high = float(statistics.median(high_values))
    high_dev = float(statistics.median([abs(v - high) for v in high_values]))
    if (
        period < 2.0
        or period_dev > 0.12 * period
        or period_max_dev > 0.25 * period
    ):
        return [fallback] * horizon
    if duration < 1.0 or duration >= period or duration_dev > 0.25 * duration:
        return [fallback] * horizon
    if high - low <= 6.0 * scale or high_dev > 0.15 * max(abs(high - low), scale):
        return [fallback] * horizon
    p = max(2, int(math.floor(period + 0.5)))
    d = max(1, min(p - 1, int(math.floor(duration + 0.5))))
    last_start_global = n - width + starts[-1]
    phase_at_next = (n - last_start_global) % p
    result = []
    for k in range(horizon):
        phase = (phase_at_next + k) % p
        state = high if phase < d else low
        value = state * normalizer
        result.append(value if math.isfinite(value) else fallback)
    return [float(v) for v in result]
""",
            # This is a host structural hypothesis, not a conservative anchor.
            # It must still pass the same strict/rescue selector as LLM programs.
            source="structural",
        )

    def _anchor_programs(self, task: Task) -> list[ForecastProgram]:
        programs = [
            self._fallback_program(),
            self._robust_recent_level(),
            self._robust_preterminal_level(),
            self._terminal_numeric_ood_hedge(task),
            self._damped_median_trend(),
            self._structural_state_cycle(),
        ]
        for seasonal in (
            self._seasonal_last_cycle(task),
            self._seasonal_naive_fallback(task),
            self._seasonal_cycle_drift(task),
            self._seasonal_last_cycle_half_shrink(task),
            self._seasonal_phase_median_half_shrink(task),
        ):
            if seasonal is not None:
                programs.append(seasonal)
        return programs

    def _has_representative_horizon_fold(self, task: Task) -> bool:
        short_horizon = max(
            1, min(self.config.validation_horizon, task.prediction_length)
        )
        representative_horizon = min(
            task.prediction_length,
            max(short_horizon, math.ceil(task.prediction_length / 2)),
        )
        return any(
            len(target) >= representative_horizon
            for _train, target in self._folds(task)
        )

    def _has_exact_horizon_activation(
        self,
        task: Task,
        candidate: ValidatedProgram,
        _anchor: ValidatedProgram,
    ) -> bool:
        """Require departure from repeat-last on a deployment-scale audit."""
        for train, target in self._folds(task):
            if len(target) != task.prediction_length or not train:
                continue
            candidate_replay = self._replay_forecast(task, candidate, train)
            repeat_last_replay = tuple(
                train[-1] for _ in range(task.prediction_length)
            )
            if (
                candidate_replay is not None
                and _forecast_materially_differs(
                    candidate_replay,
                    repeat_last_replay,
                )
            ):
                return True
        return False

    @staticmethod
    def _seasonal_deployment_certified(
        task: Task,
        candidate: ValidatedProgram,
        anchor: ValidatedProgram,
        profile: DiagnosticProfile | None,
        *,
        maximum_amplitude_dispersion: float = 0.75,
        maximum_full_amplitude_dispersion: float = 0.50,
    ) -> bool:
        """Bind full and shrunken host seasonal switches to calibrated evidence."""
        host_seasonal_names = {
            "seasonal_last_cycle",
            "seasonal_phase_median_fallback",
            "seasonal_cycle_drift",
            "seasonal_last_cycle_half_shrink",
            "seasonal_phase_median_half_shrink",
        }
        if not (
            candidate.program.source == "fallback"
            and candidate.program.name in host_seasonal_names
        ):
            return True
        period = _seasonal_period(task)
        if profile is None or period is None or profile.seasonal_period != period:
            return False
        exact_period_cards = tuple(
            card for card in profile.seasonality_evidence if card.lag == period
        )
        certified_cards = tuple(
            card
            for card in exact_period_cards
            if (
                card.complete_cycles >= 3
                and card.supporting_windows >= 2
                and math.isfinite(card.cycle_amplitude_dispersion)
                and card.cycle_amplitude_dispersion
                <= maximum_amplitude_dispersion
            )
        )
        if not certified_cards:
            return False
        full_strength_names = {
            "seasonal_last_cycle",
            "seasonal_phase_median_fallback",
            "seasonal_cycle_drift",
        }
        if candidate.program.name in full_strength_names and not any(
            card.cycle_amplitude_dispersion <= maximum_full_amplitude_dispersion
            for card in certified_cards
        ):
            return False
        if (
            candidate.program.name == "seasonal_cycle_drift"
            and not _uniform_fold_relative_gain(candidate, anchor)
        ):
            return False
        return True

    def _candidate_safety_certified(
        self,
        task: Task,
        candidate: ValidatedProgram,
        anchor: ValidatedProgram,
        profile: DiagnosticProfile | None,
    ) -> bool:
        """Apply deployment certificates after metric gates but before selection."""
        if not self._seasonal_deployment_certified(
            task,
            candidate,
            anchor,
            profile,
        ):
            return False
        learned_sources = {
            "generated",
            "knowledge",
            "library",
            "mutation",
            "knowledge_mutation",
        }
        if (
            candidate.program.source in learned_sources
            and not _bounded_trust_region_better(candidate, anchor)
            and not _zero_regression_micro_trust_better(candidate, anchor)
            and not self._has_exact_horizon_activation(task, candidate, anchor)
        ):
            return False
        return True

    def _safe_anchor_selection(
        self,
        task: Task,
        candidates: list[ValidatedProgram],
        *,
        numeric_profile: DiagnosticProfile | None = None,
    ) -> ValidatedProgram:
        anchor, terminal_hedge, terminal_projection = self._reference_anchor(
            task, candidates
        )
        representative_horizon = min(
            task.prediction_length,
            max(
                min(self.config.validation_horizon, task.prediction_length),
                math.ceil(task.prediction_length / 2),
            ),
        )
        folds = self._folds(task)
        fold_horizons = tuple(len(target) for _, target in folds)
        has_representative_horizon_fold = self._has_representative_horizon_fold(task)
        # The reference anchor is unconditional.  OOD sensitivity can reject a
        # challenger, but it must never filter the anchor and force an unvalidated
        # hedge through an empty candidate pool.
        safe = [anchor]
        if not has_representative_horizon_fold:
            return anchor
        for candidate in candidates:
            if candidate is anchor:
                continue
            if not _forecast_materially_differs(candidate.forecast, anchor.forecast):
                continue
            if _unsupported_large_flat_level_relocation(
                candidate,
                anchor,
                numeric_profile,
            ):
                continue
            strict_switch = _fold_safe_better(candidate, anchor)
            bounded_refinement = _bounded_trust_region_better(candidate, anchor)
            zero_regression_micro_refinement = (
                _zero_regression_micro_trust_better(candidate, anchor)
            )
            bounded_rescue = (
                candidate.program.source != "fallback"
                and _catastrophic_anchor_rescue(
                    candidate,
                    anchor,
                    fold_horizons=fold_horizons,
                    representative_horizon=representative_horizon,
                )
            )
            if not (
                strict_switch
                or bounded_refinement
                or zero_regression_micro_refinement
                or bounded_rescue
            ):
                continue
            if not self._candidate_safety_certified(
                task,
                candidate,
                anchor,
                numeric_profile,
            ):
                continue
            if candidate is terminal_hedge and terminal_projection is None:
                continue
            if (
                terminal_projection is not None
                and terminal_hedge is not None
                and candidate is not terminal_hedge
                and not self._terminal_suffix_stable(
                    task,
                    candidate,
                    terminal_hedge,
                    terminal_projection,
                )
            ):
                continue
            safe.append(candidate)
        return min(safe, key=_metric_key)

    def _reference_anchor(
        self,
        task: Task,
        candidates: list[ValidatedProgram],
    ) -> tuple[
        ValidatedProgram,
        ValidatedProgram | None,
        tuple[int, tuple[float, ...]] | None,
    ]:
        """Resolve the immutable numeric baseline before testing specialists."""
        terminal_projection = self._terminal_numeric_ood_projection(task)
        terminal_hedge = next(
            (
                candidate
                for candidate in candidates
                if candidate.program.name == "terminal_numeric_ood_hedge"
            ),
            None,
        )
        tsfm_anchors = [
            candidate
            for candidate in candidates
            if candidate.program.source == "tsfm"
        ]
        if self.config.setting in {"tsfm", "combined"} and tsfm_anchors:
            anchor = min(tsfm_anchors, key=_metric_key)
        else:
            fallback_anchors = [
                candidate
                for candidate in candidates
                if candidate.program.source == "fallback"
            ]
            if not fallback_anchors:
                return (
                    min(candidates, key=_metric_key),
                    terminal_hedge,
                    terminal_projection,
                )
            anchor = next(
                (
                    candidate
                    for candidate in fallback_anchors
                    if candidate.program.name == "repeat_last"
                ),
                fallback_anchors[0],
            )
        if terminal_projection is not None:
            repeat_last = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.program.name == "repeat_last"
                ),
                None,
            )
            if (
                terminal_hedge is not None
                and repeat_last is not None
                and self._has_representative_horizon_fold(task)
                and _forecast_materially_differs(
                    terminal_hedge.forecast,
                    repeat_last.forecast,
                )
                and _fold_safe_better(terminal_hedge, repeat_last)
            ):
                anchor = terminal_hedge
        return anchor, terminal_hedge, terminal_projection

    @staticmethod
    def _fold_win_count(
        selected: ValidatedProgram,
        anchor: ValidatedProgram,
    ) -> tuple[int, int]:
        selected_smae = selected.fold_smae_raw or selected.fold_smae
        selected_srmse = selected.fold_srmse_raw or selected.fold_srmse
        anchor_smae = anchor.fold_smae_raw or anchor.fold_smae
        anchor_srmse = anchor.fold_srmse_raw or anchor.fold_srmse
        if not (
            len(selected_smae)
            == len(selected_srmse)
            == len(anchor_smae)
            == len(anchor_srmse)
        ):
            return 0, 0
        wins = sum(
            candidate_smae <= baseline_smae
            and candidate_srmse <= baseline_srmse
            and (
                candidate_smae < baseline_smae
                or candidate_srmse < baseline_srmse
            )
            for candidate_smae, candidate_srmse, baseline_smae, baseline_srmse
            in zip(
                selected_smae,
                selected_srmse,
                anchor_smae,
                anchor_srmse,
                strict=True,
            )
        )
        return wins, len(selected_smae)

    def _replay_forecast(
        self,
        task: Task,
        candidate: ValidatedProgram,
        history: tuple[float, ...],
    ) -> tuple[float, ...] | None:
        if candidate.program.source == "tsfm":
            if self.tsfm_forecaster is None:
                return None
            try:
                replay = tuple(
                    float(value)
                    for value in self.tsfm_forecaster.forecast(
                        history,
                        task.prediction_length,
                        task.frequency,
                    )
                )
            except Exception:
                return None
        else:
            try:
                result = run_forecast_code(
                    candidate.program.code,
                    list(history),
                    task.prediction_length,
                    task.frequency,
                )
            except Exception:
                return None
            if not result.ok or result.forecast is None:
                return None
            replay = result.forecast
        if len(replay) != len(candidate.forecast) or not all(
            math.isfinite(value) for value in replay
        ):
            return None
        return replay

    def _terminal_suffix_stable(
        self,
        task: Task,
        candidate: ValidatedProgram,
        hedge: ValidatedProgram,
        projection: tuple[int, tuple[float, ...]],
    ) -> bool:
        """Reject a challenger that reacts more strongly to the suffix than the hedge."""
        _, cleaned = projection
        candidate_replay = self._replay_forecast(task, candidate, cleaned)
        hedge_replay = self._replay_forecast(task, hedge, cleaned)
        if candidate_replay is None or hedge_replay is None:
            return False
        comparison_scale = max(
            *(abs(value) for value in candidate.forecast),
            *(abs(value) for value in candidate_replay),
            *(abs(value) for value in hedge.forecast),
            *(abs(value) for value in hedge_replay),
        )
        if comparison_scale == 0.0:
            comparison_scale = 1.0

        def exposure(
            original: tuple[float, ...], replay: tuple[float, ...]
        ) -> float:
            return max(
                abs(value / comparison_scale - baseline / comparison_scale)
                for value, baseline in zip(original, replay, strict=True)
            )

        candidate_exposure = exposure(candidate.forecast, candidate_replay)
        hedge_exposure = exposure(hedge.forecast, hedge_replay)
        if candidate.forecast != candidate_replay and (
            candidate_exposure < _MIN_NORMAL_FLOAT
            or hedge_exposure < _MIN_NORMAL_FLOAT
        ):
            return False
        return candidate_exposure <= hedge_exposure

    @staticmethod
    def _internal_ood_projection(
        task: Task,
    ) -> tuple[tuple[float, ...], float, float] | None:
        """Replace only isolated, non-seasonal interior innovations for sensitivity replay."""
        raw_values = _finite_history(task.history_values)
        if len(raw_values) < 7:
            return None
        period = _seasonal_period(task)
        cleaned = list(raw_values)
        changed = False

        def robust_scale(points: tuple[float, ...]) -> float:
            local_level = statistics.median(abs(value) for value in points)
            local_differences = tuple(
                points[index] - points[index - 1]
                for index in range(1, len(points))
            )
            difference_center = (
                statistics.median(local_differences) if local_differences else 0.0
            )
            difference_mad = (
                statistics.median(
                    abs(value - difference_center) for value in local_differences
                )
                if local_differences
                else 0.0
            )
            return max(
                1.4826 * difference_mad,
                0.01 * local_level,
            )

        def neighborhood_scale(start: int, stop: int) -> tuple[float, float]:
            raw_neighbors = (
                raw_values[max(0, start - 4):start]
                + raw_values[stop:min(len(raw_values), stop + 4)]
            )
            normalizer = max(
                *(abs(value) for value in raw_neighbors),
                *(abs(value) for value in raw_values[start:stop]),
            ) or 1.0
            return normalizer, robust_scale(
                tuple(value / normalizer for value in raw_neighbors)
            )

        def matches_seasonal_phase(index: int) -> bool:
            if period is None:
                return False
            references = [
                raw_values[position]
                for position in range(index % period, len(raw_values), period)
                if position != index
            ][-3:]
            if not references:
                return False
            normalizer = max(
                abs(raw_values[index]),
                *(abs(value) for value in references),
            ) or 1.0
            scaled_references = tuple(value / normalizer for value in references)
            phase_center = statistics.median(scaled_references)
            phase_mad = statistics.median(
                abs(value - phase_center) for value in scaled_references
            )
            return abs(raw_values[index] / normalizer - phase_center) <= 3.0 * max(
                1.4826 * phase_mad,
                robust_scale(scaled_references),
            )

        for index in range(1, len(raw_values) - 1):
            normalizer, scale = neighborhood_scale(index, index + 1)
            previous = raw_values[index - 1] / normalizer
            following = raw_values[index + 1] / normalizer
            value = raw_values[index] / normalizer
            neighbor_level = statistics.median((previous, following))
            if (
                abs(previous - following) > 3.0 * scale
                or abs(value - neighbor_level) <= 6.0 * scale
            ):
                continue
            if matches_seasonal_phase(index):
                continue
            cleaned[index] = neighbor_level * normalizer
            changed = True
        for width in (2, 3):
            for start in range(1, len(raw_values) - width):
                stop = start + width
                normalizer, scale = neighborhood_scale(start, stop)
                previous = raw_values[start - 1] / normalizer
                following = raw_values[stop] / normalizer
                boundary_level = statistics.median((previous, following))
                if (
                    abs(previous - following) > 3.0 * scale
                    or not all(
                        abs(raw_values[index] / normalizer - boundary_level)
                        > 6.0 * scale
                        for index in range(start, stop)
                    )
                    or any(matches_seasonal_phase(index) for index in range(start, stop))
                ):
                    continue
                cleaned[start:stop] = [boundary_level * normalizer] * width
                changed = True
        if not changed:
            return None
        recent_raw = raw_values[-min(len(raw_values), 8):]
        normalizer = max(abs(value) for value in recent_raw) or 1.0
        recent = tuple(value / normalizer for value in recent_raw)
        recent_level = statistics.median(abs(value) for value in recent)
        tolerance = max(
            6.0 * robust_scale(recent),
            0.05 * recent_level,
        )
        return (
            tuple(cleaned),
            tolerance,
            normalizer,
        )

    def _current_input_stable(
        self,
        task: Task,
        candidate: ValidatedProgram,
    ) -> bool:
        projection = self._internal_ood_projection(task)
        if projection is None:
            return True
        cleaned, tolerance, history_scale = projection
        replay = self._replay_forecast(task, candidate, cleaned)
        if replay is None:
            return False
        comparison_scale = max(
            history_scale,
            *(abs(value) for value in (*replay, *candidate.forecast)),
        )
        allowed = tolerance * (history_scale / comparison_scale)
        sensitivity = max(
            abs(value / comparison_scale - baseline / comparison_scale)
            for value, baseline in zip(replay, candidate.forecast, strict=True)
        )
        if replay != candidate.forecast and sensitivity < _MIN_NORMAL_FLOAT:
            return False
        return sensitivity <= allowed

    @staticmethod
    def _terminal_numeric_ood_projection(
        task: Task,
    ) -> tuple[int, tuple[float, ...]] | None:
        """Detect a coherent endpoint shift and replace it with the prior robust level."""
        raw_values = _finite_history(task.history_values)
        if len(raw_values) < 8:
            return None
        period = _seasonal_period(task) or 0

        def matches_seasonal_phase(
            index: int,
            normalizer: float,
            tolerance: float,
        ) -> bool:
            reference = raw_values[index - period]
            pair_scale = max(normalizer, abs(reference)) or 1.0
            return abs(
                raw_values[index] / pair_scale - reference / pair_scale
            ) <= tolerance * (normalizer / pair_scale)

        for width in range(min(3, len(raw_values) - 7), 0, -1):
            start = len(raw_values) - width
            local_values = raw_values[start - 7:]
            normalizer = max(abs(value) for value in local_values) or 1.0
            recent_prior = tuple(
                value / normalizer for value in raw_values[start - 7:start]
            )
            prior_differences = tuple(
                recent_prior[index] - recent_prior[index - 1]
                for index in range(1, len(recent_prior))
            )
            difference_center = statistics.median(prior_differences)
            difference_mad = statistics.median(
                abs(value - difference_center) for value in prior_differences
            )
            level_center = statistics.median(recent_prior)
            level_mad = statistics.median(
                abs(value - level_center) for value in recent_prior
            )
            scale_floor = 0.01 * abs(level_center)
            difference_scale = max(1.4826 * difference_mad, scale_floor)
            level_scale = max(1.4826 * level_mad, scale_floor)
            suffix = tuple(value / normalizer for value in raw_values[start:])
            suffix_center = statistics.median(suffix)
            shift = abs(suffix_center - level_center)
            if max(abs(value - suffix_center) for value in suffix) > max(
                6.0 * max(difference_scale, level_scale),
                0.1 * shift,
            ):
                continue
            seasonal_tolerance = 3.0 * max(difference_scale, level_scale)
            if period >= 2 and all(
                index >= period
                and matches_seasonal_phase(
                    index,
                    normalizer,
                    seasonal_tolerance,
                )
                for index in range(start, len(raw_values))
            ):
                continue
            boundary_innovation = (
                raw_values[start] / normalizer
                - raw_values[start - 1] / normalizer
            )
            if (
                abs(boundary_innovation - difference_center)
                > 6.0 * difference_scale
                and shift > 6.0 * level_scale
            ):
                cleaned = list(raw_values)
                cleaned[start:] = [level_center * normalizer] * width
                return width, tuple(cleaned)
        return None

    @staticmethod
    def _terminal_numeric_ood_width(task: Task) -> int | None:
        projection = CodingEvolutionAgent._terminal_numeric_ood_projection(task)
        return projection[0] if projection is not None else None

    @staticmethod
    def _terminal_numeric_ood(task: Task) -> bool:
        return CodingEvolutionAgent._terminal_numeric_ood_width(task) is not None

    @staticmethod
    def _robust_recent_level() -> ForecastProgram:
        return ForecastProgram(
            name="robust_recent_level",
            description="Forecast the median of a horizon-aware recent window.",
            assumption="A noisy series has a persistent recent level without a clean trend.",
            failure_condition="The cutoff starts a trend, seasonal phase, or persistent level shift.",
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                "    if not history:\n"
                "        return [0.0 for _ in range(horizon)]\n"
                "    width = min(len(history), max(3, min(12, horizon)))\n"
                "    value = statistics.median(history[-width:])\n"
                "    return [value for _ in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _robust_preterminal_level() -> ForecastProgram:
        return ForecastProgram(
            name="robust_preterminal_level",
            description="Use a recent median that cannot copy an isolated terminal spike.",
            assumption="The terminal observation may be numeric OOD while the prior local level persists.",
            failure_condition="The terminal jump begins a genuine persistent regime.",
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                "    if len(history) <= 1:\n"
                "        value = history[-1] if history else 0.0\n"
                "    else:\n"
                "        value = statistics.median(history[max(0, len(history) - 8):-1])\n"
                "    return [value for _ in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _terminal_numeric_ood_hedge(task: Task) -> ForecastProgram:
        period = _seasonal_period(task) or 0
        return ForecastProgram(
            name="terminal_numeric_ood_hedge",
            description="Shrink a host-detectable terminal innovation halfway toward a robust bound.",
            assumption="An extreme endpoint is uncertain between a real shift and numeric OOD.",
            failure_condition="The endpoint is certainly either a full persistent jump or pure noise.",
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                "    if len(history) < 8:\n"
                "        value = history[-1] if history else 0.0\n"
                "    else:\n"
                f"        period = {period}\n"
                "        selected = None\n"
                "        for width in range(min(3, len(history) - 7), 0, -1):\n"
                "            start = len(history) - width\n"
                "            local = history[start - 7:]\n"
                "            normalizer = max(abs(x) for x in local) or 1.0\n"
                "            prior = [x / normalizer for x in history[start - 7:start]]\n"
                "            center = statistics.median(prior)\n"
                "            level_mad = statistics.median([abs(x - center) for x in prior])\n"
                "            differences = [prior[i] - prior[i - 1] "
                "for i in range(1, len(prior))]\n"
                "            diff_center = statistics.median(differences)\n"
                "            diff_mad = statistics.median([abs(x - diff_center) "
                "for x in differences])\n"
                "            floor = 0.01 * abs(center)\n"
                "            level_scale = max(1.4826 * level_mad, floor)\n"
                "            diff_scale = max(1.4826 * diff_mad, floor)\n"
                "            suffix = [x / normalizer for x in history[start:]]\n"
                "            suffix_center = statistics.median(suffix)\n"
                "            shift = abs(suffix_center - center)\n"
                "            coherent = max(abs(x - suffix_center) for x in suffix) "
                "<= max(6.0 * max(level_scale, diff_scale), 0.1 * shift)\n"
                "            seasonal_tolerance = 3.0 * max(level_scale, diff_scale)\n"
                "            seasonal = period >= 2 and all("
                "i >= period and "
                "abs(history[i] / (max(normalizer, abs(history[i - period])) or 1.0) "
                "- history[i - period] / "
                "(max(normalizer, abs(history[i - period])) or 1.0)) <= "
                "seasonal_tolerance * (normalizer / "
                "(max(normalizer, abs(history[i - period])) or 1.0)) "
                "for i in range(start, len(history)))\n"
                "            boundary = (history[start] / normalizer "
                "- history[start - 1] / normalizer)\n"
                "            if coherent and not seasonal and "
                "abs(boundary - diff_center) > 6.0 * diff_scale and "
                "shift > 6.0 * level_scale:\n"
                "                selected = (center, level_scale, suffix_center, normalizer)\n"
                "                break\n"
                "        if selected is None:\n"
                "            value = history[-1]\n"
                "        else:\n"
                "            center, level_scale, suffix_center, normalizer = selected\n"
                "            clipped = min(max(suffix_center, center - 6.0 * level_scale), "
                "center + 6.0 * level_scale)\n"
                "            value = 0.5 * (suffix_center + clipped) * normalizer\n"
                "    return [value for _ in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _damped_median_trend() -> ForecastProgram:
        return ForecastProgram(
            name="damped_median_trend",
            description="Extrapolate a robust recent slope with geometric damping.",
            assumption="A recent trend persists briefly but weakens with horizon.",
            failure_condition="The series is level, seasonal, or reverses at the cutoff.",
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                "    if len(history) < 2:\n"
                "        value = history[-1] if history else 0.0\n"
                "        return [value for _ in range(horizon)]\n"
                "    start = max(1, len(history) - 8)\n"
                "    slope = statistics.median([history[i] - history[i - 1] "
                "for i in range(start, len(history))])\n"
                "    values = []\n"
                "    value = history[-1]\n"
                "    for step in range(horizon):\n"
                "        value += slope * (0.8 ** step)\n"
                "        values.append(value)\n"
                "    return values\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _seasonal_last_cycle(task: Task) -> ForecastProgram | None:
        period = _seasonal_period(task)
        if period is None:
            return None
        return ForecastProgram(
            name="seasonal_last_cycle",
            description="Repeat the latest complete declared seasonal cycle exactly.",
            assumption="Recent phase shape repeats and is more relevant than older cycles.",
            failure_condition="The last cycle is anomalous or the seasonal phase/regime changes.",
            code=(
                "def forecast(history, horizon, frequency):\n"
                f"    period = {period}\n"
                "    if len(history) < period:\n"
                "        value = history[-1] if history else 0.0\n"
                "        return [value for _ in range(horizon)]\n"
                "    cycle = history[-period:]\n"
                "    return [cycle[index % period] for index in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _seasonal_naive_fallback(task: Task) -> ForecastProgram | None:
        period = _seasonal_period(task)
        if period is None:
            return None
        return ForecastProgram(
            name="seasonal_phase_median_fallback",
            description="Repeat a phase-wise median from the latest three complete cycles.",
            assumption="The declared seasonal phase recurs without trusting one isolated cycle value.",
            failure_condition="The seasonal phase shifts or the latest cycle starts a new regime.",
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                f"    period = {period}\n"
                "    if len(history) < period:\n"
                "        value = history[-1] if history else 0.0\n"
                "        return [value for _ in range(horizon)]\n"
                "    cycles = min(3, len(history) // period)\n"
                "    start = len(history) - cycles * period\n"
                "    profile = [statistics.median(history[start + phase::period]) "
                "for phase in range(period)]\n"
                "    return [profile[index % period] for index in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _seasonal_cycle_drift(task: Task) -> ForecastProgram | None:
        period = _seasonal_period(task)
        if period is None:
            return None
        return ForecastProgram(
            name="seasonal_cycle_drift",
            description="Repeat the last cycle with a robust, damped cycle-level drift.",
            assumption="Seasonal phase persists while the cycle level changes gradually.",
            failure_condition="Cycle-to-cycle drift is unstable, absent, or reverses after cutoff.",
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                f"    period = {period}\n"
                "    if len(history) < 2 * period:\n"
                "        value = history[-1] if history else 0.0\n"
                "        return [value for _ in range(horizon)]\n"
                "    cycle = history[-period:]\n"
                "    prior = history[-2 * period:-period]\n"
                "    cycle_delta = statistics.median([a - b for a, b in zip(cycle, prior)])\n"
                "    values = []\n"
                "    for index in range(horizon):\n"
                "        cycle_ahead = index // period + 1\n"
                "        drift = cycle_delta * sum(0.7 ** step for step in range(cycle_ahead))\n"
                "        values.append(cycle[index % period] + drift)\n"
                "    return values\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _seasonal_last_cycle_half_shrink(task: Task) -> ForecastProgram | None:
        period = _seasonal_period(task)
        if period is None:
            return None
        return ForecastProgram(
            name="seasonal_last_cycle_half_shrink",
            description=(
                "Repeat half of the latest declared-cycle displacement from "
                "the persistence anchor."
            ),
            assumption=(
                "Seasonal phase is useful but a full-amplitude replay is too uncertain."
            ),
            failure_condition=(
                "Seasonality disappears, changes phase, or requires its full historical amplitude."
            ),
            code=(
                "def forecast(history, horizon, frequency):\n"
                f"    period = {period}\n"
                "    anchor = history[-1] if history else 0.0\n"
                "    if len(history) < period:\n"
                "        return [anchor for _ in range(horizon)]\n"
                "    cycle = history[-period:]\n"
                "    return [anchor + 0.5 * (cycle[index % period] - anchor) "
                "for index in range(horizon)]\n"
            ),
            source="fallback",
        )

    @staticmethod
    def _seasonal_phase_median_half_shrink(task: Task) -> ForecastProgram | None:
        period = _seasonal_period(task)
        if period is None:
            return None
        return ForecastProgram(
            name="seasonal_phase_median_half_shrink",
            description=(
                "Use half of a robust three-cycle phase profile relative to persistence."
            ),
            assumption=(
                "Repeated phase is informative but historical amplitude should be shrunk."
            ),
            failure_condition=(
                "Seasonal phase shifts, or shrinkage removes a stable full-amplitude cycle."
            ),
            code=(
                "import statistics\n"
                "def forecast(history, horizon, frequency):\n"
                f"    period = {period}\n"
                "    anchor = history[-1] if history else 0.0\n"
                "    if len(history) < period:\n"
                "        return [anchor for _ in range(horizon)]\n"
                "    cycles = min(3, len(history) // period)\n"
                "    start = len(history) - cycles * period\n"
                "    profile = [statistics.median(history[start + phase::period]) "
                "for phase in range(period)]\n"
                "    return [anchor + 0.5 * (profile[index % period] - anchor) "
                "for index in range(horizon)]\n"
            ),
            source="fallback",
        )
