"""Closed confidence evidence for hierarchical task-local routing."""

from __future__ import annotations

import hashlib
import math
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from common.payload import canonical_json_bytes

from .screening import TaskProfile


_LEVELS = frozenset({"exact", "coarse", "global"})
_REGIONS = frozenset({"full", "early", "late"})


def _canonical_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")
    return value


def beta_win_probability(wins: int, losses: int) -> float:
    """Return P(p > 0.5) for a neutral Beta-Binomial win posterior."""
    if (
        type(wins) is not int
        or type(losses) is not int
        or wins < 0
        or losses < 0
        or wins + losses > 10_000
    ):
        raise ValueError("win and loss counts must be bounded nonnegative integers")
    trials = wins + losses + 1
    numerator = sum(math.comb(trials, index) for index in range(losses + 1, trials + 1))
    return float(numerator / (2**trials))


def robust_effect_margin(values: Sequence[float], *, multiplier: float) -> float:
    """Return median improvement minus a deterministic median-deviation margin."""
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(float(multiplier))
        or float(multiplier) < 0.0
    ):
        raise ValueError("robust effect multiplier must be finite and nonnegative")
    try:
        supplied = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("robust effects require nonempty finite values") from error
    if not supplied or any(not math.isfinite(value) for value in supplied):
        raise ValueError("robust effects require nonempty finite values")
    center = float(statistics.median(supplied))
    deviation = float(statistics.median(abs(value - center) for value in supplied))
    return center - float(multiplier) * deviation


def _morphology_payload(profile: TaskProfile, *, coarse: bool) -> dict[str, object]:
    if type(profile) is not TaskProfile:
        raise TypeError("confidence morphology requires an exact TaskProfile")
    trend = (
        profile.trend_direction
        if profile.trend_strength >= 0.35 and profile.trend_direction != "flat"
        else "flat"
    )
    payload: dict[str, object] = {
        "trend": trend,
        "periodic": bool(
            profile.periodicity_periods and profile.periodicity_confidence >= 0.5
        ),
        "intermittent": bool(
            profile.zero_fraction >= 0.5 or profile.intermittency_adi >= 1.32
        ),
        "recent_regime": bool(
            profile.recent_regime_start is not None
            and profile.recent_regime_confidence >= 0.5
        ),
        "signed": profile.signed,
    }
    if not coarse:
        history_bucket = (
            "short"
            if profile.history_length < 64
            else "medium"
            if profile.history_length < 256
            else "long"
        )
        ratio = profile.horizon / profile.history_length
        horizon_bucket = "short" if ratio <= 0.1 else "medium" if ratio <= 0.3 else "long"
        payload = {
            "frequency": _canonical_name(profile.frequency).strip(),
            "history": history_bucket,
            "horizon": horizon_bucket,
            **payload,
        }
    return payload


def exact_morphology_key(profile: TaskProfile) -> str:
    return _sha256(_morphology_payload(profile, coarse=False))


def coarse_morphology_key(profile: TaskProfile) -> str:
    return _sha256(_morphology_payload(profile, coarse=True))


@dataclass(frozen=True)
class ConfidencePolicy:
    """Host-owned evidence and confidence thresholds."""

    schema_version: int = 2
    exact_minimum_support: int = 8
    coarse_minimum_support: int = 12
    global_minimum_support: int = 20
    posterior_win_probability: float = 0.80
    minimum_paired_origins: int = 3
    robust_mad_multiplier: float = 1.0
    maximum_prior_metric_regression: float = 0.05
    regional_minimum_horizon: int = 4

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("confidence policy schema must be exactly two")
        supports = (
            self.exact_minimum_support,
            self.coarse_minimum_support,
            self.global_minimum_support,
        )
        if any(type(value) is not int or not 2 <= value <= 10_000 for value in supports):
            raise ValueError("confidence support thresholds are outside their closed range")
        if tuple(sorted(supports)) != supports:
            raise ValueError("confidence support thresholds must become more conservative")
        if (
            type(self.minimum_paired_origins) is not int
            or not 2 <= self.minimum_paired_origins <= 10
        ):
            raise ValueError("paired-origin threshold is outside its closed range")
        if (
            type(self.regional_minimum_horizon) is not int
            or not 4 <= self.regional_minimum_horizon <= 10_000
        ):
            raise ValueError("regional horizon threshold is outside its closed range")
        if (
            type(self.posterior_win_probability) is not float
            or not math.isfinite(self.posterior_win_probability)
            or not 0.5 < self.posterior_win_probability < 1.0
        ):
            raise ValueError("posterior threshold must be a float within (0.5, 1)")
        if (
            type(self.robust_mad_multiplier) is not float
            or not math.isfinite(self.robust_mad_multiplier)
            or self.robust_mad_multiplier < 0.0
        ):
            raise ValueError("robust MAD multiplier must be finite and nonnegative")
        if (
            type(self.maximum_prior_metric_regression) is not float
            or not math.isfinite(self.maximum_prior_metric_regression)
            or not 0.0 <= self.maximum_prior_metric_regression <= 0.25
        ):
            raise ValueError("prior metric regression must be within [0, 0.25]")


@dataclass(frozen=True)
class WeightRecipe:
    """One closed anchor-heavy recipe for a fixed horizon region."""

    region: str
    names: tuple[str, ...]
    weight_units: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.region) is not str or self.region not in _REGIONS:
            raise ValueError("weight recipe region is unsupported")
        if (
            type(self.names) is not tuple
            or not 2 <= len(self.names) <= 3
            or any(type(name) is not str or not name.isidentifier() for name in self.names)
            or len({_canonical_name(name) for name in self.names}) != len(self.names)
        ):
            raise ValueError("weight recipe requires two or three unique identifiers")
        if (
            type(self.weight_units) is not tuple
            or len(self.weight_units) != len(self.names)
            or any(type(unit) is not int or unit <= 0 for unit in self.weight_units)
        ):
            raise ValueError("weight recipe units must be positive exact integers")
        if sum(self.weight_units) != 10:
            raise ValueError("weight recipe units must sum to ten")
        if self.weight_units[0] < 5:
            raise ValueError("weight recipe anchor weight must be at least one half")

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(float(unit / 10.0) for unit in self.weight_units)

    @property
    def fingerprint(self) -> str:
        return _sha256(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "region": self.region,
            "names": list(self.names),
            "weight_units": list(self.weight_units),
        }


@dataclass(frozen=True)
class ConfidenceEvidenceRecord:
    """Anonymous aggregate evidence for one recipe at one hierarchy level."""

    level: str
    group_key: str
    recipe: WeightRecipe
    independent_groups: int
    task_support: int
    wins: int
    ties: int
    losses: int
    posterior_win_probability: float
    robust_margin_joint: float
    robust_margin_smae: float
    robust_margin_srmse: float
    p90_regret_smae_raw: float
    p90_regret_srmse_raw: float
    failure_count: int
    clipped_smae_count: int
    clipped_srmse_count: int

    def __post_init__(self) -> None:
        if type(self.level) is not str or self.level not in _LEVELS:
            raise ValueError("confidence evidence level is unsupported")
        _require_sha256(self.group_key, "confidence evidence group key")
        if type(self.recipe) is not WeightRecipe:
            raise ValueError("confidence evidence requires an exact weight recipe")
        WeightRecipe.__post_init__(self.recipe)
        counts = (
            self.independent_groups,
            self.task_support,
            self.wins,
            self.ties,
            self.losses,
            self.failure_count,
            self.clipped_smae_count,
            self.clipped_srmse_count,
        )
        if any(type(value) is not int or not 0 <= value <= 10_000 for value in counts):
            raise ValueError("confidence evidence counts must be bounded integers")
        if self.independent_groups <= 0 or self.task_support <= 0:
            raise ValueError("confidence evidence requires positive support")
        if self.wins + self.ties + self.losses + self.failure_count != self.task_support:
            raise ValueError("confidence evidence task counts do not conserve support")
        if self.independent_groups > self.task_support:
            raise ValueError("independent group support exceeds task support")
        if (
            self.clipped_smae_count > self.task_support
            or self.clipped_srmse_count > self.task_support
        ):
            raise ValueError("clipped counts exceed task support")
        expected_probability = beta_win_probability(self.wins, self.losses)
        if (
            type(self.posterior_win_probability) is not float
            or not math.isclose(
                self.posterior_win_probability,
                expected_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("confidence evidence posterior probability is forged")
        for name in (
            "robust_margin_joint",
            "robust_margin_smae",
            "robust_margin_srmse",
            "p90_regret_smae_raw",
            "p90_regret_srmse_raw",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if self.p90_regret_smae_raw < 0.0 or self.p90_regret_srmse_raw < 0.0:
            raise ValueError("confidence evidence regret must be nonnegative")

    def to_payload(self) -> dict[str, object]:
        return {
            "level": self.level,
            "group_key": self.group_key,
            "recipe": self.recipe.to_payload(),
            "independent_groups": self.independent_groups,
            "task_support": self.task_support,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "posterior_win_probability": self.posterior_win_probability,
            "robust_margin_joint": self.robust_margin_joint,
            "robust_margin_smae": self.robust_margin_smae,
            "robust_margin_srmse": self.robust_margin_srmse,
            "p90_regret_smae_raw": self.p90_regret_smae_raw,
            "p90_regret_srmse_raw": self.p90_regret_srmse_raw,
            "failure_count": self.failure_count,
            "clipped_smae_count": self.clipped_smae_count,
            "clipped_srmse_count": self.clipped_srmse_count,
        }


def _record_key(record: ConfidenceEvidenceRecord) -> tuple[str, str, str]:
    return (record.level, record.group_key, record.recipe.fingerprint)


@dataclass(frozen=True)
class HierarchicalEvidenceBank:
    """Canonical cross-fitted recipe evidence with no task identities."""

    schema_version: int
    policy: ConfidencePolicy
    records: tuple[ConfidenceEvidenceRecord, ...]
    fit_group_ids: tuple[str, ...]
    fit_group_fingerprint: str
    evidence_fingerprint: str

    @classmethod
    def build(
        cls,
        policy: ConfidencePolicy,
        *,
        records: Sequence[ConfidenceEvidenceRecord],
        fit_group_ids: Sequence[str],
    ) -> "HierarchicalEvidenceBank":
        if type(policy) is not ConfidencePolicy:
            raise ValueError("confidence bank requires an exact policy")
        ConfidencePolicy.__post_init__(policy)
        normalized_records = tuple(sorted(records, key=_record_key))
        normalized_groups = tuple(sorted(fit_group_ids))
        fit_fingerprint = _sha256({"fit_group_ids": list(normalized_groups)})
        base = _bank_payload(
            schema_version=2,
            policy=policy,
            records=normalized_records,
            fit_group_ids=normalized_groups,
            fit_group_fingerprint=fit_fingerprint,
        )
        return cls(
            schema_version=2,
            policy=policy,
            records=normalized_records,
            fit_group_ids=normalized_groups,
            fit_group_fingerprint=fit_fingerprint,
            evidence_fingerprint=_sha256(base),
        )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("confidence evidence bank schema must be exactly two")
        if type(self.policy) is not ConfidencePolicy:
            raise ValueError("confidence evidence bank requires an exact policy")
        ConfidencePolicy.__post_init__(self.policy)
        if (
            type(self.records) is not tuple
            or any(type(record) is not ConfidenceEvidenceRecord for record in self.records)
            or tuple(sorted(self.records, key=_record_key)) != self.records
            or len({_record_key(record) for record in self.records}) != len(self.records)
        ):
            raise ValueError("confidence evidence records must be unique and sorted")
        for record in self.records:
            ConfidenceEvidenceRecord.__post_init__(record)
            if record.level == "global" and record.group_key != self.global_group_key():
                raise ValueError("global confidence evidence key is noncanonical")
        if (
            type(self.fit_group_ids) is not tuple
            or not self.fit_group_ids
            or tuple(sorted(self.fit_group_ids)) != self.fit_group_ids
            or len(set(self.fit_group_ids)) != len(self.fit_group_ids)
        ):
            raise ValueError("confidence fit groups must be nonempty unique and sorted")
        for value in self.fit_group_ids:
            _require_sha256(value, "confidence fit group")
        expected_fit = _sha256({"fit_group_ids": list(self.fit_group_ids)})
        if self.fit_group_fingerprint != expected_fit:
            raise ValueError("confidence fit-group fingerprint mismatch")
        expected_evidence = _sha256(
            _bank_payload(
                schema_version=self.schema_version,
                policy=self.policy,
                records=self.records,
                fit_group_ids=self.fit_group_ids,
                fit_group_fingerprint=self.fit_group_fingerprint,
            )
        )
        if self.evidence_fingerprint != expected_evidence:
            raise ValueError("confidence evidence fingerprint mismatch")

    @staticmethod
    def global_group_key() -> str:
        return _sha256({"scope": "global"})

    def resolve(
        self, profile: TaskProfile, recipe: WeightRecipe
    ) -> ConfidenceEvidenceRecord | None:
        return self.resolve_many(profile, (recipe,))[0]

    def resolve_many(
        self,
        profile: TaskProfile,
        recipes: Sequence[WeightRecipe],
    ) -> tuple[ConfidenceEvidenceRecord | None, ...]:
        """Resolve many recipes with one evidence index construction."""
        if type(profile) is not TaskProfile:
            raise TypeError("confidence resolution requires an exact profile")
        supplied = tuple(recipes)
        if any(type(recipe) is not WeightRecipe for recipe in supplied):
            raise TypeError("confidence resolution requires exact recipe values")
        keys = (
            ("exact", exact_morphology_key(profile), self.policy.exact_minimum_support),
            ("coarse", coarse_morphology_key(profile), self.policy.coarse_minimum_support),
            ("global", self.global_group_key(), self.policy.global_minimum_support),
        )
        by_key = {_record_key(record): record for record in self.records}
        resolved: list[ConfidenceEvidenceRecord | None] = []
        for recipe in supplied:
            selected = None
            for level, group_key, minimum_support in keys:
                record = by_key.get((level, group_key, recipe.fingerprint))
                if record is not None and record.task_support >= minimum_support:
                    selected = record
                    break
            resolved.append(selected)
        return tuple(resolved)

    def to_payload(self) -> dict[str, object]:
        payload = _bank_payload(
            schema_version=self.schema_version,
            policy=self.policy,
            records=self.records,
            fit_group_ids=self.fit_group_ids,
            fit_group_fingerprint=self.fit_group_fingerprint,
        )
        payload["evidence_fingerprint"] = self.evidence_fingerprint
        return payload


def _bank_payload(
    *,
    schema_version: int,
    policy: ConfidencePolicy,
    records: Sequence[ConfidenceEvidenceRecord],
    fit_group_ids: Sequence[str],
    fit_group_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "policy": asdict(policy),
        "records": [record.to_payload() for record in records],
        "fit_group_ids": list(fit_group_ids),
        "fit_group_fingerprint": fit_group_fingerprint,
    }


def _parse_recipe(payload: object) -> WeightRecipe:
    if type(payload) is not dict or set(payload) != {"region", "names", "weight_units"}:
        raise ValueError("weight recipe schema is malformed")
    if type(payload["names"]) is not list or type(payload["weight_units"]) is not list:
        raise ValueError("weight recipe arrays are malformed")
    try:
        return WeightRecipe(
            region=payload["region"],
            names=tuple(payload["names"]),
            weight_units=tuple(payload["weight_units"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("weight recipe schema is malformed") from error


def _parse_record(payload: object) -> ConfidenceEvidenceRecord:
    expected = {
        "level",
        "group_key",
        "recipe",
        "independent_groups",
        "task_support",
        "wins",
        "ties",
        "losses",
        "posterior_win_probability",
        "robust_margin_joint",
        "robust_margin_smae",
        "robust_margin_srmse",
        "p90_regret_smae_raw",
        "p90_regret_srmse_raw",
        "failure_count",
        "clipped_smae_count",
        "clipped_srmse_count",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("confidence evidence record schema is malformed")
    try:
        return ConfidenceEvidenceRecord(
            level=payload["level"],
            group_key=payload["group_key"],
            recipe=_parse_recipe(payload["recipe"]),
            independent_groups=payload["independent_groups"],
            task_support=payload["task_support"],
            wins=payload["wins"],
            ties=payload["ties"],
            losses=payload["losses"],
            posterior_win_probability=payload["posterior_win_probability"],
            robust_margin_joint=payload["robust_margin_joint"],
            robust_margin_smae=payload["robust_margin_smae"],
            robust_margin_srmse=payload["robust_margin_srmse"],
            p90_regret_smae_raw=payload["p90_regret_smae_raw"],
            p90_regret_srmse_raw=payload["p90_regret_srmse_raw"],
            failure_count=payload["failure_count"],
            clipped_smae_count=payload["clipped_smae_count"],
            clipped_srmse_count=payload["clipped_srmse_count"],
        )
    except ValueError as error:
        if "posterior probability" in str(error):
            raise
        raise ValueError("confidence evidence record schema is malformed") from error
    except (KeyError, TypeError) as error:
        raise ValueError("confidence evidence record schema is malformed") from error


def parse_hierarchical_evidence(payload: object) -> HierarchicalEvidenceBank:
    """Parse the exact canonical confidence evidence schema."""
    expected = {
        "schema_version",
        "policy",
        "records",
        "fit_group_ids",
        "fit_group_fingerprint",
        "evidence_fingerprint",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("confidence evidence bank schema is malformed")
    policy_payload = payload["policy"]
    policy_fields = set(asdict(ConfidencePolicy()))
    if type(policy_payload) is not dict or set(policy_payload) != policy_fields:
        raise ValueError("confidence policy schema is malformed")
    if type(payload["records"]) is not list or type(payload["fit_group_ids"]) is not list:
        raise ValueError("confidence evidence arrays are malformed")
    try:
        return HierarchicalEvidenceBank(
            schema_version=payload["schema_version"],
            policy=ConfidencePolicy(**policy_payload),
            records=tuple(_parse_record(record) for record in payload["records"]),
            fit_group_ids=tuple(payload["fit_group_ids"]),
            fit_group_fingerprint=payload["fit_group_fingerprint"],
            evidence_fingerprint=payload["evidence_fingerprint"],
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and (
            "evidence record schema" in str(error)
            or "posterior probability" in str(error)
        ):
            raise
        raise ValueError("confidence evidence bank schema is malformed") from error
