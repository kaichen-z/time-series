"""Bounded, history-only morphology observations and grounded assumptions."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from common.llm import LLMClient

from . import analysis_skills_template as _reviewed_skills


class MorphologyError(ValueError):
    """A morphology action or artifact violated the bounded protocol."""


class MorphologyInputError(MorphologyError):
    """History-only inputs cannot satisfy the morphology protocol."""


_MAX_TURNS = 4
_MAX_TOOL_CALLS = 8
_MAX_TOOL_CALLS_PER_TURN = 3
_ASSUMPTION_KINDS = frozenset(
    {"seasonality", "trend", "intermittency", "regime", "noise", "level"}
)
_TOOL_ACTION_KEYS = frozenset({"action", "call_id", "tool", "window"})
_FINAL_ACTION_KEYS = frozenset({"action", "short_term", "long_term", "assumptions"})
_ASSUMPTION_KEYS = frozenset(
    {
        "assumption_id",
        "kind",
        "claim",
        "failure_condition",
        "supporting_call_ids",
        "candidate_names",
        "prior_confidence",
    }
)


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MorphologyError("artifact must be canonical JSON with finite values") from exc


def _finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))


def _freeze_json(value: object) -> object:
    if not _finite_json(value):
        raise MorphologyError("reviewed tool returned non-finite or non-JSON output")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MorphologyError(f"{field} must be a nonempty string")
    return value.strip()


def _required_string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MorphologyError(f"{field} must be a nonempty list of strings")
    return _normalized_string_collection(value, field)


def _normalized_string_collection(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise MorphologyError(f"{field} must be a nonempty collection of strings")
    result = tuple(_required_string(item, field) for item in value)
    if len(result) != len(set(result)):
        if field == "supporting_call_ids":
            raise MorphologyError("duplicate supporting call id")
        raise MorphologyError(f"{field} must not contain duplicate values")
    return result


@dataclass(frozen=True)
class MorphologyToolCall:
    """One exact, host-executed call against a reviewed history-only tool."""

    call_id: str
    tool: str
    start: int
    end: int

    def __post_init__(self) -> None:
        call_id = _required_string(self.call_id, "call_id")
        tool = _required_string(self.tool, "tool")
        if tool not in _reviewed_skills.ANALYSIS_SKILL_NAMES:
            raise MorphologyError(f"unknown reviewed tool {tool!r}")
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or not 0 <= self.start < self.end
        ):
            raise MorphologyError("invalid tool window")
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "tool", tool)

    @property
    def window(self) -> Mapping[str, int]:
        return MappingProxyType({"start": self.start, "end": self.end})

    def to_payload(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "window": {"start": self.start, "end": self.end},
        }


@dataclass(frozen=True)
class MorphologyObservation:
    """A tool call paired with its finite, Python-produced observation."""

    call: MorphologyToolCall
    output: object

    def __post_init__(self) -> None:
        if not isinstance(self.call, MorphologyToolCall):
            raise MorphologyError("observation call must be a MorphologyToolCall")
        object.__setattr__(self, "output", _freeze_json(self.output))

    @property
    def call_id(self) -> str:
        return self.call.call_id

    def to_payload(self) -> dict[str, object]:
        return {"call": self.call.to_payload(), "output": _thaw_json(self.output)}


@dataclass(frozen=True)
class AssumptionGrounding:
    """One falsifiable morphology claim grounded in exact executed call IDs."""

    assumption_id: str
    kind: str
    claim: str
    failure_condition: str
    supporting_call_ids: tuple[str, ...]
    candidate_names: tuple[str, ...]
    prior_confidence: float

    def __post_init__(self) -> None:
        assumption_id = _required_string(self.assumption_id, "assumption_id")
        kind = _required_string(self.kind, "kind")
        if kind not in _ASSUMPTION_KINDS:
            raise MorphologyError(f"unsupported assumption kind {kind!r}")
        confidence = self.prior_confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise MorphologyError("prior_confidence must be finite and within [0, 1]")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise MorphologyError("prior_confidence must be finite and within [0, 1]")
        object.__setattr__(self, "assumption_id", assumption_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "claim", _required_string(self.claim, "claim"))
        object.__setattr__(
            self, "failure_condition", _required_string(self.failure_condition, "failure_condition")
        )
        object.__setattr__(
            self,
            "supporting_call_ids",
            _normalized_string_collection(self.supporting_call_ids, "supporting_call_ids"),
        )
        object.__setattr__(
            self,
            "candidate_names",
            _normalized_string_collection(self.candidate_names, "candidate_names"),
        )
        object.__setattr__(self, "prior_confidence", confidence)

    def to_payload(self) -> dict[str, object]:
        return {
            "assumption_id": self.assumption_id,
            "kind": self.kind,
            "claim": self.claim,
            "failure_condition": self.failure_condition,
            "supporting_call_ids": list(self.supporting_call_ids),
            "candidate_names": list(self.candidate_names),
            "prior_confidence": self.prior_confidence,
        }


@dataclass(frozen=True)
class MorphologyCard:
    """Immutable artifact containing only historical, reviewed observations."""

    short_term: str
    long_term: str
    tool_calls: tuple[MorphologyToolCall, ...]
    observations: tuple[MorphologyObservation, ...]
    assumptions: tuple[AssumptionGrounding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.short_term, str) or not isinstance(self.long_term, str):
            raise MorphologyError("card descriptions must be strings")
        try:
            tool_calls = tuple(self.tool_calls)
            observations = tuple(self.observations)
            assumptions = tuple(self.assumptions)
        except TypeError as exc:
            raise MorphologyError("card collections must be iterable") from exc
        if any(not isinstance(item, MorphologyToolCall) for item in tool_calls):
            raise MorphologyError("tool_calls must contain MorphologyToolCall artifacts")
        if any(not isinstance(item, MorphologyObservation) for item in observations):
            raise MorphologyError("observations must contain MorphologyObservation artifacts")
        if any(not isinstance(item, AssumptionGrounding) for item in assumptions):
            raise MorphologyError("assumptions must contain AssumptionGrounding artifacts")
        object.__setattr__(self, "short_term", _required_string(self.short_term, "short_term"))
        object.__setattr__(self, "long_term", _required_string(self.long_term, "long_term"))
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "assumptions", assumptions)
        if len(self.tool_calls) != len(self.observations):
            raise MorphologyError("every tool call must have exactly one observation")
        if tuple(item.call for item in self.observations) != self.tool_calls:
            raise MorphologyError("observation calls must match the executed tool calls")
        if not 1 <= len(self.assumptions) <= 7:
            raise MorphologyError("morphology card must contain one to seven assumptions")
        ids = tuple(item.assumption_id for item in self.assumptions)
        if len(ids) != len(set(ids)):
            raise MorphologyError("morphology card contains duplicate assumption ids")
        call_ids = tuple(item.call_id for item in self.tool_calls)
        if len(call_ids) != len(set(call_ids)):
            raise MorphologyError("morphology card contains duplicate tool call ids")
        executed = set(call_ids)
        if any(set(item.supporting_call_ids) - executed for item in self.assumptions):
            raise MorphologyError("assumption cites unknown call id")
        _canonical_bytes(self.to_payload())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

    def assumption_call_ids(self, assumption_id: str) -> tuple[str, ...]:
        for assumption in self.assumptions:
            if assumption.assumption_id == assumption_id:
                return assumption.supporting_call_ids
        raise KeyError(assumption_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "short_term": self.short_term,
            "long_term": self.long_term,
            "tool_calls": [item.to_payload() for item in self.tool_calls],
            "observations": [item.to_payload() for item in self.observations],
            "assumptions": [item.to_payload() for item in self.assumptions],
        }


class MorphologyReasoner:
    """Execute a fixed-budget LLM tool loop over reviewed historical analysis only."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_turns: int = _MAX_TURNS,
        max_tool_calls: int = _MAX_TOOL_CALLS,
        max_tool_calls_per_turn: int = _MAX_TOOL_CALLS_PER_TURN,
    ) -> None:
        if not 1 <= max_turns <= _MAX_TURNS:
            raise MorphologyInputError(f"max_turns must be within [1, {_MAX_TURNS}]")
        if not 1 <= max_tool_calls <= _MAX_TOOL_CALLS:
            raise MorphologyInputError(
                f"max_tool_calls must be within [1, {_MAX_TOOL_CALLS}]"
            )
        if not 1 <= max_tool_calls_per_turn <= _MAX_TOOL_CALLS_PER_TURN:
            raise MorphologyInputError(
                "max_tool_calls_per_turn must be within "
                f"[1, {_MAX_TOOL_CALLS_PER_TURN}]"
            )
        self._llm = llm
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        self._max_tool_calls_per_turn = max_tool_calls_per_turn

    def reason(
        self,
        *,
        history: Sequence[float],
        frequency: str,
        horizon: int,
        active_names: Sequence[str],
        families: Mapping[str, str],
    ) -> MorphologyCard:
        values, normalized_frequency, active, normalized_families = self._validate_inputs(
            history, frequency, horizon, active_names, families
        )
        tool_calls: list[MorphologyToolCall] = []
        observations: list[MorphologyObservation] = []
        messages = [{"role": "user", "content": self._initial_prompt(values, normalized_frequency, horizon, active, normalized_families)}]

        for _turn in range(self._max_turns):
            response = self._llm.complete(
                system=self._system_prompt(), messages=messages, temperature=0.0
            )
            action = self._parse_action(response.text)
            if action["action"] == "final":
                return self._finalize(action, values, active, tool_calls, observations)
            if len(tool_calls) >= self._max_tool_calls:
                raise MorphologyError("tool-call budget exceeded")
            call = self._parse_tool_action(action, len(values), {item.call_id for item in tool_calls})
            output = self._execute_tool(call, values, normalized_frequency)
            tool_calls.append(call)
            observation = MorphologyObservation(call, output)
            observations.append(observation)
            messages.append({"role": "assistant", "content": _canonical_bytes(action).decode("utf-8")})
            messages.append(
                {
                    "role": "user",
                    "content": _canonical_bytes(
                        {"tool_result": observation.to_payload()}
                    ).decode("utf-8"),
                }
            )
        raise MorphologyError("turn budget exhausted before final action")

    @staticmethod
    def _validate_inputs(
        history: Sequence[float],
        frequency: str,
        horizon: int,
        active_names: Sequence[str],
        families: Mapping[str, str],
    ) -> tuple[tuple[float, ...], str, tuple[str, ...], Mapping[str, str]]:
        if isinstance(history, (str, bytes)):
            raise MorphologyInputError("history must be a sequence of finite values")
        try:
            values = tuple(float(value) for value in history)
        except (TypeError, ValueError) as exc:
            raise MorphologyInputError("history must be a sequence of finite values") from exc
        if len(values) < 2:
            raise MorphologyInputError("history must contain at least two values")
        if not all(math.isfinite(value) for value in values):
            raise MorphologyInputError("history must contain only finite values")
        try:
            normalized_frequency = _required_string(frequency, "frequency")
        except MorphologyError as exc:
            raise MorphologyInputError("frequency must be a nonempty string") from exc
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise MorphologyInputError("horizon must be a positive integer")
        if isinstance(active_names, (str, bytes)):
            raise MorphologyInputError("active_names must be a sequence of candidate names")
        try:
            raw_active = tuple(active_names)
            active = tuple(_required_string(name, "active_names") for name in raw_active)
        except MorphologyError as exc:
            raise MorphologyInputError("active_names must be a sequence of candidate names") from exc
        if any(raw != normalized for raw, normalized in zip(raw_active, active, strict=True)):
            raise MorphologyInputError(
                "active candidate names and families must not contain surrounding whitespace"
            )
        if not active or len(active) != len(set(active)):
            raise MorphologyInputError("active_names must be nonempty and unique")
        if not isinstance(families, Mapping):
            raise MorphologyInputError("families must define exactly the active candidate names")
        try:
            normalized_families: dict[str, str] = {}
            for name, family in families.items():
                normalized_name = _required_string(name, "families")
                if name != normalized_name:
                    raise MorphologyInputError(
                        "active candidate names and families must not contain surrounding whitespace"
                    )
                normalized_families[normalized_name] = _required_string(
                    family, "candidate family"
                )
        except MorphologyError as exc:
            raise MorphologyInputError("candidate families must be nonempty strings") from exc
        if len(normalized_families) != len(families) or set(normalized_families) != set(active):
            raise MorphologyInputError("families must define exactly the active candidate names")
        return values, normalized_frequency, active, MappingProxyType(normalized_families)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a bounded history-only morphology reasoner. Return exactly one JSON object "
            "with action 'tool' or 'final'. Use only listed reviewed tools. Never forecast, use "
            "future labels, documents, code, or unlisted candidates. Tool action keys exactly are "
            "action, call_id, tool, window; final action keys exactly are action, short_term, "
            "long_term, assumptions. Final assumptions must be a list of one to seven objects with "
            "keys exactly assumption_id, kind, claim, failure_condition, supporting_call_ids, "
            "candidate_names, prior_confidence. kind must be one of seasonality, trend, "
            "intermittency, regime, noise, level. prior_confidence must be finite and within "
            "[0, 1]. candidate_names must be active candidate names. supporting_call_ids must be "
            "unique executed call IDs. Final assumptions must cite both a full-history inspection "
            "and a distinct recent inspection ending at the history boundary."
        )

    @staticmethod
    def _initial_prompt(
        history: tuple[float, ...],
        frequency: str,
        horizon: int,
        active_names: tuple[str, ...],
        families: Mapping[str, str],
    ) -> str:
        return _canonical_bytes(
            {
                "history": list(history),
                "frequency": frequency,
                "horizon": horizon,
                "active_candidates": [
                    {"name": name, "family": families[name]} for name in active_names
                ],
                "reviewed_tools": list(_reviewed_skills.ANALYSIS_SKILL_NAMES),
                "window_contract": {
                    "start_inclusive": 0,
                    "end_exclusive": len(history),
                    "requires_full_history_and_distinct_recent": True,
                },
            }
        ).decode("utf-8")

    @staticmethod
    def _parse_action(text: str) -> dict[str, object]:
        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            action: dict[str, object] = {}
            for key, value in pairs:
                if key in action:
                    raise MorphologyError(f"duplicate JSON key {key!r}")
                action[key] = value
            return action

        try:
            action = json.loads(text, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise MorphologyError("action must be an exact JSON object") from exc
        if not isinstance(action, dict):
            raise MorphologyError("action must be an exact JSON object")
        kind = action.get("action")
        if kind == "tool":
            if set(action) != _TOOL_ACTION_KEYS:
                raise MorphologyError("tool action schema drift")
        elif kind == "final":
            if set(action) != _FINAL_ACTION_KEYS:
                raise MorphologyError("final action schema drift")
        else:
            raise MorphologyError("action must be 'tool' or 'final'")
        return action

    @staticmethod
    def _parse_tool_action(
        action: Mapping[str, object], history_length: int, call_ids: set[str]
    ) -> MorphologyToolCall:
        call_id = _required_string(action["call_id"], "call_id")
        if call_id in call_ids:
            raise MorphologyError("duplicate tool call id")
        tool = _required_string(action["tool"], "tool")
        if tool not in _reviewed_skills.ANALYSIS_SKILL_NAMES:
            raise MorphologyError(f"unknown reviewed tool {tool!r}")
        window = action["window"]
        if not isinstance(window, Mapping) or set(window) != {"start", "end"}:
            raise MorphologyError("tool window schema drift")
        start, end = window["start"], window["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= history_length
        ):
            raise MorphologyError("invalid tool window")
        return MorphologyToolCall(call_id, tool, start, end)

    @staticmethod
    def _execute_tool(
        call: MorphologyToolCall, history: tuple[float, ...], frequency: str
    ) -> object:
        tool = getattr(_reviewed_skills, call.tool, None)
        if not callable(tool):
            raise MorphologyError(f"reviewed tool {call.tool!r} is unavailable")
        window = history[call.start : call.end]
        parameters = inspect.signature(tool).parameters
        if tuple(parameters) == ("history", "frequency"):
            output = tool(window, frequency)
        elif tuple(parameters) == ("history",):
            output = tool(window)
        else:
            raise MorphologyError(f"reviewed tool {call.tool!r} has an unsupported signature")
        if not _finite_json(output):
            raise MorphologyError("reviewed tool returned non-finite or non-JSON output")
        return output

    @staticmethod
    def _finalize(
        action: Mapping[str, object],
        history: tuple[float, ...],
        active_names: tuple[str, ...],
        tool_calls: Sequence[MorphologyToolCall],
        observations: Sequence[MorphologyObservation],
    ) -> MorphologyCard:
        if not tool_calls:
            raise MorphologyError("finalization requires distinct broad and recent inspections")
        assumptions_value = action["assumptions"]
        if not isinstance(assumptions_value, list) or not 1 <= len(assumptions_value) <= 7:
            raise MorphologyError("final assumptions must contain one to seven objects")
        full = [item.call_id for item in tool_calls if item.start == 0 and item.end == len(history)]
        recent = [item.call_id for item in tool_calls if item.start > 0 and item.end == len(history)]
        if not full or not recent:
            raise MorphologyError("finalization requires distinct broad and recent inspections")
        executed = {item.call_id for item in tool_calls}
        assumptions = tuple(
            MorphologyReasoner._parse_assumption(item, executed, set(active_names))
            for item in assumptions_value
        )
        assumption_ids = tuple(item.assumption_id for item in assumptions)
        if len(assumption_ids) != len(set(assumption_ids)):
            raise MorphologyError("duplicate assumption id")
        cited = {call_id for item in assumptions for call_id in item.supporting_call_ids}
        if not (set(full) & cited) or not (set(recent) & cited):
            raise MorphologyError("finalization requires distinct broad and recent inspections")
        return MorphologyCard(
            short_term=_required_string(action["short_term"], "short_term"),
            long_term=_required_string(action["long_term"], "long_term"),
            tool_calls=tuple(tool_calls),
            observations=tuple(observations),
            assumptions=assumptions,
        )

    @staticmethod
    def _parse_assumption(
        value: object, executed: set[str], active_names: set[str]
    ) -> AssumptionGrounding:
        if not isinstance(value, Mapping) or set(value) != _ASSUMPTION_KEYS:
            raise MorphologyError("assumption schema drift")
        supporting_call_ids = _required_string_list(
            value["supporting_call_ids"], "supporting_call_ids"
        )
        missing = set(supporting_call_ids) - executed
        if missing:
            raise MorphologyError(f"assumption cites unknown call id {sorted(missing)!r}")
        candidate_names = _required_string_list(value["candidate_names"], "candidate_names")
        inactive = set(candidate_names) - active_names
        if inactive:
            raise MorphologyError(f"assumption cites inactive candidate {sorted(inactive)!r}")
        kind = _required_string(value["kind"], "kind")
        if kind not in _ASSUMPTION_KINDS:
            raise MorphologyError(f"unsupported assumption kind {kind!r}")
        confidence = value["prior_confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise MorphologyError("prior_confidence must be finite and within [0, 1]")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise MorphologyError("prior_confidence must be finite and within [0, 1]")
        return AssumptionGrounding(
            assumption_id=_required_string(value["assumption_id"], "assumption_id"),
            kind=kind,
            claim=_required_string(value["claim"], "claim"),
            failure_condition=_required_string(value["failure_condition"], "failure_condition"),
            supporting_call_ids=supporting_call_ids,
            candidate_names=candidate_names,
            prior_confidence=confidence,
        )


__all__ = [
    "AssumptionGrounding",
    "MorphologyCard",
    "MorphologyError",
    "MorphologyInputError",
    "MorphologyObservation",
    "MorphologyReasoner",
    "MorphologyToolCall",
]
