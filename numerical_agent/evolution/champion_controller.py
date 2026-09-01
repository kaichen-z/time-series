"""Build evolution plus the durable 64/16/20 Champion lifecycle."""
from __future__ import annotations

import hashlib
import dis
import inspect
import math
import os
import re
import secrets
import stat
import sys
import unicodedata
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import BuiltinFunctionType, CodeType, FunctionType, ModuleType
from typing import Callable, Literal, NoReturn, cast

from common.data import Task
from common.payload import canonical_json_bytes, strict_json_loads

from .champion import (
    ChampionRecipe,
    ChampionRelease,
    EvolutionAssumption,
    FittedChampionPolicy,
    _parse_fitted_policy,
    champion_fingerprint,
    parse_champion_recipe,
    parse_champion_release,
)
from .champion_evidence import (
    _InvalidAttemptAggregate,
    ChampionComparison,
    ChampionEvidenceError,
    ChampionGateConfig,
    ChampionHistoryDiagnostic,
    ChampionScore,
    ChampionTaskRow,
    MorphologyAggregate,
    ProposerEvidence,
    _ProposerComparison,
    _assert_sanitized,
    compare_champion,
    sanitize_build_evidence,
    score_policy,
)
from .champion_proposal import ChampionProposalError, expand_recipe
from .champion_runtime import ChampionExecution, execute_champion
from .numerical_selector import CandidateDiagnostics
from .screening import TaskProfile


_FORMAL_EXPAND_RECIPE = expand_recipe
_FORMAL_EXECUTE_CHAMPION = execute_champion
_FORMAL_SCORE_POLICY = score_policy
_FORMAL_COMPARE_CHAMPION = compare_champion


_FORMAL_SIZES = (64, 16, (8, 32, 64))
_SMOKE_SIZES = (8, 2, (4, 8))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_STAGE = re.compile(r"build_generation_([1-9][0-9]*)\Z")
_PROFILE_FEATURES = frozenset({
    "history_length",
    "horizon",
    "horizon_ratio",
    "zero_fraction",
    "trend_strength",
    "periodicity_strength",
    "periodicity_confidence",
    "outlier_fraction",
    "noise_relative_scale",
    "stationarity_score",
    "recent_regime_confidence",
    "intermittency_adi",
    "intermittency_cv2",
})


class ChampionControllerError(ValueError):
    """Build evolution input or evidence violates its fixed authority."""


class ChampionLifecycleError(ChampionControllerError):
    """Formal Train/Dev lifecycle state violates its fixed authority."""


def _fail(message: str) -> NoReturn:
    raise ChampionControllerError(message)


def _canonical_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _lifecycle_fail(message: str) -> NoReturn:
    raise ChampionLifecycleError(message)


_CALLABLE_KINDS = frozenset(
    {"proposer", "row_provider", "expander", "executor", "scorer", "comparator"}
)
_BEHAVIOR_SOURCE_CACHE: ContextVar[dict[str, dict[str, object]] | None] = (
    ContextVar("champion_behavior_source_cache", default=None)
)
_BEHAVIOR_IDENTITY_CACHE: ContextVar[dict[int, dict[str, object]] | None] = (
    ContextVar("champion_behavior_identity_cache", default=None)
)
_BEHAVIOR_STRICT_GRAPH: ContextVar[bool] = ContextVar(
    "champion_behavior_strict_graph", default=False
)
_HOST_SOURCE_DIGEST_CACHE: dict[
    tuple[str, int, int, int, int, int], str
] = {}
_HOST_CODE_FINGERPRINT_CACHE: dict[CodeType, str] = {}


def _host_constant_identity(value: object) -> object:
    if isinstance(value, CodeType):
        return {"nested_code": _host_code_fingerprint(value)}
    if type(value) is float and not math.isfinite(cast(float, value)):
        number = cast(float, value)
        return {"nonfinite_float": "nan" if math.isnan(number) else repr(number)}
    if type(value) in {str, int, float, bool, type(None)}:
        return value
    if type(value) is bytes:
        return {"bytes_sha256": hashlib.sha256(cast(bytes, value)).hexdigest()}
    if type(value) is tuple:
        return [
            _host_constant_identity(item) for item in cast(tuple[object, ...], value)
        ]
    if type(value) is frozenset:
        members = [
            _host_constant_identity(item)
            for item in cast(frozenset[object], value)
        ]
        return {
            "frozenset": sorted(members, key=lambda item: champion_fingerprint(item))
        }
    if type(value) is complex:
        complex_number = cast(complex, value)
        return {
            "complex": {
                "real": complex_number.real,
                "imag": complex_number.imag,
            }
        }
    if value is Ellipsis:
        return {"constant": "Ellipsis"}
    _lifecycle_fail("callable implementation contains an unsupported host constant")


def _host_code_fingerprint(code: CodeType) -> str:
    """Fingerprint immutable code fields without CPython quickening state."""
    cached = _HOST_CODE_FINGERPRINT_CACHE.get(code)
    if cached is not None:
        return cached
    constants = [_host_constant_identity(value) for value in code.co_consts]
    fingerprint = champion_fingerprint(
        {
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
            "constants": constants,
            "names": code.co_names,
            "varnames": code.co_varnames,
            "freevars": code.co_freevars,
            "cellvars": code.co_cellvars,
            "argcount": code.co_argcount,
            "posonlyargcount": code.co_posonlyargcount,
            "kwonlyargcount": code.co_kwonlyargcount,
            "flags": code.co_flags,
        }
    )
    _HOST_CODE_FINGERPRINT_CACHE[code] = fingerprint
    return fingerprint


def _module_source_identity(module: ModuleType) -> dict[str, object]:
    module_name = getattr(module, "__name__", None)
    if type(module_name) is not str or not module_name:
        _lifecycle_fail("behavior module has no exact host identity")
    cache = _BEHAVIOR_SOURCE_CACHE.get()
    if cache is not None and module_name in cache:
        return cache[module_name]
    try:
        source = inspect.getsourcefile(module)
    except TypeError:
        source = None
    if source is None:
        raw_file = getattr(module, "__file__", None)
        source = raw_file if type(raw_file) is str else None
    if source is None:
        identity: dict[str, object] = {
            "module": module_name,
            "origin": "builtin",
            "python_runtime": sys.version,
        }
        if cache is not None:
            cache[module_name] = identity
        return identity
    try:
        path = Path(source).resolve(strict=True)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            _lifecycle_fail("runtime module source must be an exact file")
        digest_key = (
            str(path),
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
        )
        source_digest = _HOST_SOURCE_DIGEST_CACHE.get(digest_key)
        if source_digest is None:
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            source_digest = digest.hexdigest()
            _HOST_SOURCE_DIGEST_CACHE[digest_key] = source_digest
    except OSError as error:
        raise ChampionLifecycleError("runtime module source cannot be read") from error
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    identity = {
        "module": module_name,
        "source_path": str(path),
        "source_sha256": source_digest,
    }
    if cache is not None:
        cache[module_name] = identity
    return identity


def _code_global_names(code: CodeType) -> frozenset[str]:
    names = {
        cast(str, instruction.argval)
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and type(instruction.argval) is str
    }
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_code_global_names(constant))
    return frozenset(names)


def _module_attribute_names(code: CodeType, global_name: str) -> frozenset[str]:
    attributes: set[str] = set()
    instructions = tuple(dis.get_instructions(code))
    for index, instruction in enumerate(instructions[:-1]):
        if (
            instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
            and instruction.argval == global_name
        ):
            following = instructions[index + 1]
            if following.opname in {"LOAD_ATTR", "LOAD_METHOD"} and type(
                following.argval
            ) is str:
                attributes.add(cast(str, following.argval))
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            attributes.update(_module_attribute_names(constant, global_name))
    return frozenset(attributes)


def _host_type_identity(value: type, seen: set[int]) -> dict[str, object]:
    cache = _BEHAVIOR_IDENTITY_CACHE.get()
    if cache is not None and id(value) in cache:
        return cache[id(value)]
    module = sys.modules.get(value.__module__)
    identity: dict[str, object] = {
        "module": value.__module__,
        "qualname": value.__qualname__,
    }
    if isinstance(module, ModuleType):
        identity["module_source"] = _module_source_identity(module)
    if cache is not None:
        cache[id(value)] = identity
    return identity


def _is_generated_dataclass_method(name: str, value: FunctionType) -> bool:
    if value.__code__.co_filename == "<string>":
        return True
    return (
        name == "__repr__"
        and value.__code__.co_name == "wrapper"
        and Path(value.__code__.co_filename).name == "dataclasses.py"
    )


def _host_callable_type_identity(
    value: type, seen: set[int]
) -> dict[str, object]:
    methods = {
        name: _host_function_identity(member, seen)
        for name, member in sorted(vars(value).items())
        if isinstance(member, FunctionType)
        and not _is_generated_dataclass_method(name, member)
    }
    if "__call__" not in methods:
        _lifecycle_fail("formal callable has no exact host implementation")
    return {
        "type": _host_type_identity(value, seen),
        "methods": methods,
    }


def _resolve_loaded_name(
    callback: FunctionType, name: str
) -> tuple[str, object]:
    if name in callback.__globals__:
        return "global", callback.__globals__[name]
    builtins = getattr(callback, "__builtins__", None)
    if isinstance(builtins, ModuleType):
        if hasattr(builtins, name):
            return "builtin", getattr(builtins, name)
    elif type(builtins) is dict and name in builtins:
        return "builtin", cast(dict[str, object], builtins)[name]
    _lifecycle_fail(f"formal callable has unresolved loaded name: {name}")


def _shallow_function_identity(
    value: FunctionType, seen: set[int], *, include_globals: bool
) -> dict[str, object]:
    reference = id(value)
    stable_name = f"{value.__module__}.{value.__qualname__}"
    if reference in seen:
        return {"function_reference": stable_name}
    module = sys.modules.get(value.__module__)
    if not isinstance(module, ModuleType):
        _lifecycle_fail("behavior dependency module is not loaded")
    seen.add(reference)
    try:
        identity: dict[str, object] = {
            "function": stable_name,
            "module_source": _module_source_identity(module),
            "code_fingerprint": _host_code_fingerprint(value.__code__),
            "defaults": _host_behavior_value(value.__defaults__ or (), seen),
            "kwdefaults": _host_behavior_value(
                tuple(sorted((value.__kwdefaults__ or {}).items())), seen
            ),
            "closure": tuple(
                _host_behavior_value(cell.cell_contents, seen)
                for cell in (value.__closure__ or ())
            ),
        }
        if include_globals:
            globals_identity: dict[str, object] = {}
            for name in sorted(_code_global_names(value.__code__)):
                scope, global_value = _resolve_loaded_name(value, name)
                dependency: object
                if isinstance(global_value, FunctionType):
                    dependency = _shallow_function_identity(
                        global_value, seen, include_globals=False
                    )
                else:
                    dependency = _shallow_dependency_identity(global_value, seen)
                globals_identity[name] = {
                    "scope": scope,
                    "identity": dependency,
                }
            identity["globals"] = globals_identity
        return identity
    finally:
        seen.remove(reference)


def _shallow_dependency_identity(value: object, seen: set[int]) -> object:
    if isinstance(value, FunctionType):
        return _shallow_function_identity(
            value,
            seen,
            include_globals=_BEHAVIOR_STRICT_GRAPH.get(),
        )
    if isinstance(value, BuiltinFunctionType):
        return {
            "builtin_module": value.__module__,
            "builtin_name": value.__qualname__,
        }
    if isinstance(value, type):
        return _host_type_identity(value, seen)
    if isinstance(value, ModuleType):
        return _module_source_identity(value)
    return _host_behavior_value(value, seen, allow_mutable_globals=True)


def _host_behavior_value(
    value: object, seen: set[int], *, allow_mutable_globals: bool = False
) -> object:
    if type(value) is float and not math.isfinite(cast(float, value)):
        number = cast(float, value)
        return {"nonfinite_float": "nan" if math.isnan(number) else repr(number)}
    if type(value) in {str, int, float, bool, type(None)}:
        return value
    if type(value) is bytes:
        return {"bytes_sha256": hashlib.sha256(cast(bytes, value)).hexdigest()}
    if isinstance(value, Path):
        return {"path": str(value.absolute()), "type": type(value).__name__}
    if isinstance(value, re.Pattern):
        return {"regex_pattern": value.pattern, "regex_flags": value.flags}
    if isinstance(value, range):
        return {"range": (value.start, value.stop, value.step)}
    if isinstance(value, slice):
        return {"slice": (value.start, value.stop, value.step)}
    if type(value).__module__ == "typing":
        typing_module = sys.modules.get("typing")
        if not isinstance(typing_module, ModuleType):
            _lifecycle_fail("typing behavior dependency is not loaded")
        return {
            "typing_value": repr(value),
            "module_source": _module_source_identity(typing_module),
        }
    if type(value) is tuple:
        return [
            _host_behavior_value(
                item, seen, allow_mutable_globals=allow_mutable_globals
            )
            for item in cast(tuple[object, ...], value)
        ]
    if type(value) is frozenset:
        members = [
            _host_behavior_value(
                item, seen, allow_mutable_globals=allow_mutable_globals
            )
            for item in cast(frozenset[object], value)
        ]
        return {
            "frozenset": sorted(members, key=lambda item: champion_fingerprint(item))
        }
    if type(value) in {list, dict, set, bytearray}:
        if not allow_mutable_globals:
            _lifecycle_fail("formal callable behavior state must be immutable")
        if type(value) is list:
            return {
                "mutable_global_list": [
                    _host_behavior_value(item, seen, allow_mutable_globals=True)
                    for item in cast(list[object], value)
                ]
            }
        if type(value) is dict:
            dict_items = [
                (
                    _host_behavior_value(key, seen, allow_mutable_globals=True),
                    _host_behavior_value(item, seen, allow_mutable_globals=True),
                )
                for key, item in cast(dict[object, object], value).items()
            ]
            return {
                "mutable_global_dict": sorted(
                    dict_items, key=lambda item: champion_fingerprint(item[0])
                )
            }
        if type(value) is set:
            set_items = [
                _host_behavior_value(item, seen, allow_mutable_globals=True)
                for item in cast(set[object], value)
            ]
            return {
                "mutable_global_set": sorted(
                    set_items, key=lambda item: champion_fingerprint(item)
                )
            }
        return {
            "mutable_global_bytes": hashlib.sha256(
                bytes(cast(bytearray, value))
            ).hexdigest()
        }
    if isinstance(value, FunctionType):
        return _host_function_identity(value, seen)
    if isinstance(value, BuiltinFunctionType):
        return {
            "builtin_module": value.__module__,
            "builtin_name": value.__qualname__,
        }
    if isinstance(value, ModuleType):
        return _module_source_identity(value)
    if isinstance(value, type):
        return _host_type_identity(value, seen)
    if is_dataclass(value):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or parameters.frozen is not True:
            _lifecycle_fail("formal callable behavior state must be frozen")
        reference = id(value)
        if reference in seen:
            return {
                "frozen_reference": f"{type(value).__module__}.{type(value).__qualname__}"
            }
        seen.add(reference)
        try:
            identity = {
                "frozen_type": _host_type_identity(type(value), seen),
                "fields": {
                    item.name: _host_behavior_value(getattr(value, item.name), seen)
                    for item in fields(value)
                },
            }
            if callable(value):
                identity["callable_implementation"] = _host_callable_type_identity(
                    type(value), seen
                )
            return identity
        finally:
            seen.remove(reference)
    _lifecycle_fail(
        "formal callable has unsupported mutable behavior state: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _host_function_identity(
    callback: FunctionType, seen: set[int]
) -> dict[str, object]:
    reference = id(callback)
    cache = _BEHAVIOR_IDENTITY_CACHE.get()
    if cache is not None and reference in cache:
        return cache[reference]
    stable_name = f"{callback.__module__}.{callback.__qualname__}"
    if reference in seen:
        return {"function_reference": stable_name}
    seen.add(reference)
    try:
        module = sys.modules.get(callback.__module__)
        if not isinstance(module, ModuleType):
            _lifecycle_fail("callable function module is not loaded")
        globals_identity: dict[str, object] = {}
        for name in sorted(_code_global_names(callback.__code__)):
            scope, global_value = _resolve_loaded_name(callback, name)
            if isinstance(global_value, ModuleType):
                attributes: dict[str, object] = {}
                for attribute in sorted(
                    _module_attribute_names(callback.__code__, name)
                ):
                    if not hasattr(global_value, attribute):
                        _lifecycle_fail(
                            "formal callable has unresolved module attribute: "
                            f"{name}.{attribute}"
                        )
                    attributes[attribute] = _shallow_dependency_identity(
                        getattr(global_value, attribute), seen
                    )
                dependency: object = {
                    "module": _module_source_identity(global_value),
                    "attributes": attributes,
                }
            elif isinstance(global_value, FunctionType) and (
                global_value.__module__ == callback.__module__
                or global_value.__module__.startswith("numerical_agent.")
                or global_value.__module__.startswith("common.")
            ):
                dependency = _host_function_identity(global_value, seen)
            elif isinstance(global_value, (FunctionType, BuiltinFunctionType, type)):
                dependency = _shallow_dependency_identity(global_value, seen)
            else:
                dependency = _host_behavior_value(
                    global_value, seen, allow_mutable_globals=True
                )
            globals_identity[name] = {
                "scope": scope,
                "identity": dependency,
            }
        closure = callback.__closure__ or ()
        identity = {
            "owner_module": callback.__module__,
            "owner_qualname": callback.__qualname__,
            "module_source": _module_source_identity(module),
            "code_fingerprint": _host_code_fingerprint(callback.__code__),
            "defaults": _host_behavior_value(callback.__defaults__ or (), seen),
            "kwdefaults": _host_behavior_value(
                tuple(sorted((callback.__kwdefaults__ or {}).items())), seen
            ),
            "closure": tuple(
                _host_behavior_value(cell.cell_contents, seen) for cell in closure
            ),
            "globals": globals_identity,
        }
        if cache is not None:
            cache[reference] = identity
        return identity
    finally:
        seen.remove(reference)


def _host_callable_fingerprint(callback: object) -> str:
    """Derive an explicit bounded manifest of actually loaded dependencies."""
    source_token = _BEHAVIOR_SOURCE_CACHE.set({})
    identity_token = _BEHAVIOR_IDENTITY_CACHE.set({})
    strict_token = _BEHAVIOR_STRICT_GRAPH.set(
        not isinstance(callback, FunctionType)
    )
    try:
        if isinstance(callback, FunctionType):
            module = sys.modules.get(callback.__module__)
            if not isinstance(module, ModuleType):
                _lifecycle_fail("formal callable module is not loaded")
            identity = {
                "callable": _host_function_identity(callback, set()),
            }
        else:
            if not callable(callback) or not is_dataclass(callback):
                _lifecycle_fail("formal callable requires a closed frozen host adapter")
            parameters = getattr(type(callback), "__dataclass_params__", None)
            if parameters is None or parameters.frozen is not True:
                _lifecycle_fail("formal callable requires immutable behavior state")
            implementation = vars(type(callback)).get("__call__")
            if not isinstance(implementation, FunctionType):
                _lifecycle_fail("formal callable has no exact host implementation")
            module = sys.modules.get(implementation.__module__)
            if not isinstance(module, ModuleType):
                _lifecycle_fail("formal callable module is not loaded")
            identity = {
                "callable_type": _host_callable_type_identity(
                    type(callback), set()
                ),
                "state": {
                    item.name: _host_behavior_value(
                        getattr(callback, item.name), set()
                    )
                    for item in fields(callback)
                },
            }
        return champion_fingerprint(identity)
    finally:
        _BEHAVIOR_STRICT_GRAPH.reset(strict_token)
        _BEHAVIOR_IDENTITY_CACHE.reset(identity_token)
        _BEHAVIOR_SOURCE_CACHE.reset(source_token)


@dataclass(frozen=True, init=False)
class ChampionCallableBinding:
    """Host-derived executable plus canonical behavior config for one boundary."""

    kind: str
    identity: str
    callback: object = field(repr=False, compare=False)
    implementation_fingerprint: str
    config_fingerprint: str

    @classmethod
    def bind(
        cls,
        *,
        kind: str,
        identity: str,
        callback: object,
        config: object,
    ) -> "ChampionCallableBinding":
        if kind in {"proposer", "row_provider"}:
            _lifecycle_fail("formal external boundaries require closed typed adapters")
        expected = {
            "expander": _FORMAL_EXPAND_RECIPE,
            "executor": _FORMAL_EXECUTE_CHAMPION,
            "scorer": _FORMAL_SCORE_POLICY,
            "comparator": _FORMAL_COMPARE_CHAMPION,
        }
        if kind not in expected or callback is not expected[kind]:
            _lifecycle_fail("formal runtime binding is not a closed host executable")
        return cls._bind(
            kind=kind,
            identity=identity,
            callback=callback,
            config=config,
        )

    @classmethod
    def _bind(
        cls,
        *,
        kind: str,
        identity: str,
        callback: object,
        config: object,
    ) -> "ChampionCallableBinding":
        if type(kind) is not str or kind not in _CALLABLE_KINDS:
            _lifecycle_fail("callable binding kind is not registered")
        if type(identity) is not str or not identity.strip():
            _lifecycle_fail("callable binding identity must be nonempty")
        if not callable(callback):
            _lifecycle_fail("callable binding requires a live callable")
        try:
            config_fingerprint = champion_fingerprint(config)
        except Exception as error:
            raise ChampionLifecycleError(
                "callable binding config must be canonical"
            ) from error
        instance = object.__new__(cls)
        object.__setattr__(instance, "kind", kind)
        object.__setattr__(instance, "identity", identity)
        object.__setattr__(instance, "callback", callback)
        object.__setattr__(
            instance,
            "implementation_fingerprint",
            _host_callable_fingerprint(callback),
        )
        object.__setattr__(instance, "config_fingerprint", config_fingerprint)
        instance.verify()
        return instance

    @property
    def fingerprint(self) -> str:
        return champion_fingerprint(
            {
                "kind": self.kind,
                "identity": self.identity,
                "implementation_fingerprint": self.implementation_fingerprint,
                "config_fingerprint": self.config_fingerprint,
            }
        )

    def verify(self) -> None:
        if type(self.kind) is not str or self.kind not in _CALLABLE_KINDS:
            _lifecycle_fail("callable binding kind drifted")
        if type(self.identity) is not str or not self.identity.strip():
            _lifecycle_fail("callable binding identity drifted")
        if not callable(self.callback):
            _lifecycle_fail("callable binding lost its executable")
        if _host_callable_fingerprint(self.callback) != self.implementation_fingerprint:
            _lifecycle_fail("callable executable implementation drifted")

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.verify()
        try:
            return cast(Callable[..., object], self.callback)(*args, **kwargs)
        finally:
            self.verify()


@dataclass(frozen=True, init=False)
class ChampionProposerAdapter:
    """Exact host-owned formal proposer adapter with derived behavior identity."""

    binding: ChampionCallableBinding

    @classmethod
    def bind(
        cls, *, identity: str, callback: object, config: object
    ) -> "ChampionProposerAdapter":
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "binding",
            ChampionCallableBinding._bind(
                kind="proposer",
                identity=identity,
                callback=callback,
                config=config,
            ),
        )
        instance.verify()
        return instance

    @property
    def identity(self) -> str:
        return self.binding.identity

    @property
    def implementation_fingerprint(self) -> str:
        return self.binding.implementation_fingerprint

    @property
    def config_fingerprint(self) -> str:
        return self.binding.config_fingerprint

    @property
    def fingerprint(self) -> str:
        return self.binding.fingerprint

    def verify(self) -> None:
        if (
            type(self.binding) is not ChampionCallableBinding
            or self.binding.kind != "proposer"
        ):
            _lifecycle_fail("formal proposer adapter is malformed")
        self.binding.verify()

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.binding(*args, **kwargs)


@dataclass(frozen=True, init=False)
class ChampionRowProviderAdapter:
    """Exact host-owned formal row-provider adapter with derived behavior identity."""

    binding: ChampionCallableBinding

    @classmethod
    def bind(
        cls, *, identity: str, callback: object, config: object
    ) -> "ChampionRowProviderAdapter":
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "binding",
            ChampionCallableBinding._bind(
                kind="row_provider",
                identity=identity,
                callback=callback,
                config=config,
            ),
        )
        instance.verify()
        return instance

    @property
    def identity(self) -> str:
        return self.binding.identity

    @property
    def implementation_fingerprint(self) -> str:
        return self.binding.implementation_fingerprint

    @property
    def config_fingerprint(self) -> str:
        return self.binding.config_fingerprint

    @property
    def fingerprint(self) -> str:
        return self.binding.fingerprint

    def verify(self) -> None:
        if (
            type(self.binding) is not ChampionCallableBinding
            or self.binding.kind != "row_provider"
        ):
            _lifecycle_fail("formal row-provider adapter is malformed")
        self.binding.verify()

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.binding(*args, **kwargs)


@dataclass(frozen=True, init=False)
class ChampionRuntimeBindings:
    """Closed formal numerical execution boundary set."""

    expander: ChampionCallableBinding
    executor: ChampionCallableBinding
    scorer: ChampionCallableBinding
    comparator: ChampionCallableBinding

    @classmethod
    def formal(cls) -> "ChampionRuntimeBindings":
        instance = object.__new__(cls)
        bindings = (
            (
                "expander",
                ChampionCallableBinding.bind(
                    kind="expander",
                    identity="champion_recipe_expander",
                    callback=_FORMAL_EXPAND_RECIPE,
                    config={"contract": "champion_recipe_expander_v1"},
                ),
            ),
            (
                "executor",
                ChampionCallableBinding.bind(
                    kind="executor",
                    identity="champion_runtime_executor",
                    callback=_FORMAL_EXECUTE_CHAMPION,
                    config={"contract": "champion_runtime_executor_v1"},
                ),
            ),
            (
                "scorer",
                ChampionCallableBinding.bind(
                    kind="scorer",
                    identity="champion_evidence_scorer",
                    callback=_FORMAL_SCORE_POLICY,
                    config={"contract": "champion_evidence_scorer_v1"},
                ),
            ),
            (
                "comparator",
                ChampionCallableBinding.bind(
                    kind="comparator",
                    identity="champion_gate_comparator",
                    callback=_FORMAL_COMPARE_CHAMPION,
                    config={"contract": "champion_gate_comparator_v1"},
                ),
            ),
        )
        for name, binding in bindings:
            object.__setattr__(instance, name, binding)
        instance.verify()
        return instance

    @property
    def fingerprint(self) -> str:
        return champion_fingerprint(
            {
                "expander": self.expander.fingerprint,
                "executor": self.executor.fingerprint,
                "scorer": self.scorer.fingerprint,
                "comparator": self.comparator.fingerprint,
            }
        )

    def verify(self) -> None:
        expected_kinds = (
            (self.expander, "expander"),
            (self.executor, "executor"),
            (self.scorer, "scorer"),
            (self.comparator, "comparator"),
        )
        for binding, kind in expected_kinds:
            if type(binding) is not ChampionCallableBinding or binding.kind != kind:
                _lifecycle_fail("formal runtime binding set is malformed")
            binding.verify()

    def verify_live_globals(self) -> None:
        self.verify()
        live = (
            (expand_recipe, self.expander.callback, "expander"),
            (execute_champion, self.executor.callback, "executor"),
            (score_policy, self.scorer.callback, "scorer"),
            (compare_champion, self.comparator.callback, "comparator"),
        )
        for current, bound, label in live:
            if current is not bound:
                _lifecycle_fail(f"formal runtime {label} executable drifted")


@dataclass(frozen=True)
class TrainPartitions:
    """Exact entity-disjoint internal partitions of the formal Train set."""

    build: tuple[Task, ...]
    calibration: tuple[Task, ...]


def partition_train_tasks(
    tasks: tuple[Task, ...] | list[Task],
    *,
    build_size: int = 64,
    calibration_size: int = 16,
    seed: int,
) -> TrainPartitions:
    """Choose an exact SHA-seeded entity-group subset for Calibration."""
    if type(build_size) is not int or type(calibration_size) is not int:
        _lifecycle_fail("partition sizes must be exact integers")
    if build_size <= 0 or calibration_size <= 0:
        _lifecycle_fail("partition sizes must be positive")
    if type(seed) is not int:
        _lifecycle_fail("partition seed must be an exact integer")
    if type(tasks) not in {tuple, list}:
        _lifecycle_fail("Train tasks must be an exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], tasks))
    if len(snapshot) != build_size + calibration_size:
        _lifecycle_fail("Train tasks do not match the exact partition sizes")

    task_ids: dict[str, str] = {}
    entity_ids: dict[str, str] = {}
    groups: dict[str, list[Task]] = {}
    for raw_task in snapshot:
        if type(raw_task) is not Task:
            _lifecycle_fail("Train partitions require exact Task values")
        task = cast(Task, raw_task)
        if type(task.task_id) is not str or not task.task_id:
            _lifecycle_fail("Train task IDs must be nonempty exact strings")
        canonical_task_id = _canonical_identity(task.task_id)
        if canonical_task_id in task_ids:
            _lifecycle_fail("Train tasks contain duplicate task IDs")
        task_ids[canonical_task_id] = task.task_id
        if (
            type(task.entity_name) is not str
            or not task.entity_name.strip()
            or _canonical_identity(task.entity_name) == "unknown"
        ):
            _lifecycle_fail("Train tasks contain an unknown entity")
        canonical_entity = _canonical_identity(task.entity_name)
        prior_entity = entity_ids.setdefault(canonical_entity, task.entity_name)
        if prior_entity != task.entity_name:
            _lifecycle_fail("Train tasks contain duplicate entity identities")
        groups.setdefault(task.entity_name, []).append(task)

    ordered_groups = tuple(
        sorted(
            (
                (
                    hashlib.sha256(
                        f"{seed}{entity_name}".encode("utf-8")
                    ).hexdigest(),
                    entity_name,
                    tuple(sorted(group, key=lambda item: item.task_id)),
                )
                for entity_name, group in groups.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
    )
    subsets: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, _, group) in enumerate(ordered_groups):
        next_subsets = dict(subsets)
        for size, selected in subsets.items():
            candidate_size = size + len(group)
            if candidate_size <= calibration_size and candidate_size not in next_subsets:
                next_subsets[candidate_size] = selected + (index,)
        subsets = next_subsets
    selected_groups = subsets.get(calibration_size)
    if selected_groups is None:
        _lifecycle_fail(
            "Train entity groups cannot form the exact entity-disjoint partition"
        )
    calibration_indexes = set(selected_groups)
    build = tuple(
        task
        for index, (_, _, group) in enumerate(ordered_groups)
        if index not in calibration_indexes
        for task in group
    )
    calibration = tuple(
        task
        for index, (_, _, group) in enumerate(ordered_groups)
        if index in calibration_indexes
        for task in group
    )
    if len(build) != build_size or len(calibration) != calibration_size:
        _lifecycle_fail("entity-disjoint partition produced the wrong exact sizes")
    if {task.entity_name for task in build} & {
        task.entity_name for task in calibration
    }:
        _lifecycle_fail("entity-disjoint partition contains entity overlap")
    return TrainPartitions(build=build, calibration=calibration)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _lifecycle_fail(f"{label} must be a canonical SHA-256 digest")
    return cast(str, value)


def _validated_hash_inventory(
    values: object, label: str
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or not values:
        _lifecycle_fail(f"{label} must be a nonempty exact tuple")
    inventory = cast(tuple[object, ...], values)
    parsed: list[tuple[str, str]] = []
    canonical_names: set[str] = set()
    for raw_pair in inventory:
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            _lifecycle_fail(f"{label} must contain exact name/hash pairs")
        name, digest = cast(tuple[object, object], raw_pair)
        if type(name) is not str or not name or not name.isidentifier():
            _lifecycle_fail(f"{label} names must be public identifiers")
        canonical_name = _canonical_identity(name)
        if canonical_name in canonical_names:
            _lifecycle_fail(f"{label} contains a duplicate identity")
        canonical_names.add(canonical_name)
        parsed.append((name, _require_sha256(digest, f"{label} hash")))
    result = tuple(parsed)
    if result != tuple(sorted(result)):
        _lifecycle_fail(f"{label} must use canonical sorted order")
    return result


def _validated_membership(
    values: object,
    label: str,
    *,
    expected_size: int,
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or len(values) != expected_size:
        _lifecycle_fail(f"{label} must contain exactly {expected_size} tasks")
    membership = cast(tuple[object, ...], values)
    parsed: list[tuple[str, str]] = []
    task_ids: set[str] = set()
    entity_aliases: dict[str, str] = {}
    for raw_pair in membership:
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            _lifecycle_fail(f"{label} must contain exact task/entity pairs")
        task_id, entity_name = cast(tuple[object, object], raw_pair)
        if type(task_id) is not str or not task_id:
            _lifecycle_fail(f"{label} contains an invalid task ID")
        canonical_task = _canonical_identity(task_id)
        if canonical_task in task_ids:
            _lifecycle_fail(f"{label} contains duplicate task IDs")
        task_ids.add(canonical_task)
        if (
            type(entity_name) is not str
            or not entity_name.strip()
            or _canonical_identity(entity_name) == "unknown"
        ):
            _lifecycle_fail(f"{label} contains an unknown entity")
        canonical_entity = _canonical_identity(entity_name)
        prior = entity_aliases.setdefault(canonical_entity, entity_name)
        if prior != entity_name:
            _lifecycle_fail(f"{label} contains duplicate entity identities")
        parsed.append((task_id, entity_name))
    return tuple(parsed)


def task_content_fingerprint(task: Task) -> str:
    """Bind every numeric task field, including labels and temporal metadata."""
    if type(task) is not Task:
        _lifecycle_fail("task content fingerprint requires an exact Task")
    return champion_fingerprint(
        {
            "task_id": task.task_id,
            "history_values": task.history_values,
            "future_values": task.future_values,
            "prediction_length": task.prediction_length,
            "frequency": task.frequency,
            "seasonal_period": task.seasonal_period,
            "entity_name": task.entity_name,
        }
    )


def _validated_task_hashes(
    values: object,
    label: str,
    membership: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or len(values) != len(membership):
        _lifecycle_fail(f"{label} must bind every registered task")
    expected_ids = tuple(task_id for task_id, _ in membership)
    parsed: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(cast(tuple[object, ...], values)):
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            _lifecycle_fail(f"{label} must contain exact task/hash pairs")
        task_id, digest = cast(tuple[object, object], raw_pair)
        if task_id != expected_ids[index]:
            _lifecycle_fail(f"{label} task order drifted from membership")
        parsed.append(
            (cast(str, task_id), _require_sha256(digest, f"{label} hash"))
        )
    return tuple(parsed)


@dataclass(frozen=True)
class ChampionRunManifest:
    """Complete pre-execution identity of one formal 64/16/20 run."""

    schema_version: int
    partition_seed: int
    source_hashes: tuple[tuple[str, str], ...]
    train_tasks: tuple[tuple[str, str], ...]
    train_task_hashes: tuple[tuple[str, str], ...]
    dev_tasks: tuple[tuple[str, str], ...]
    dev_task_hashes: tuple[tuple[str, str], ...]
    split_manifest_fingerprint: str
    build_tasks: tuple[tuple[str, str], ...]
    calibration_tasks: tuple[tuple[str, str], ...]
    dictionary_hashes: tuple[tuple[str, str], ...]
    forecast_store_fingerprint: str
    metric_policy_fingerprint: str
    proposal_model: str
    proposal_config_fingerprint: str
    proposal_implementation_fingerprint: str
    row_provider_fingerprint: str
    schedule_fingerprint: str
    numeric_grid_fingerprint: str
    candidate_minimum_gain: float
    research_target_gain: float
    runtime_fingerprint: str
    runtime_implementation_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _lifecycle_fail("run manifest schema_version must be exactly one")
        if type(self.partition_seed) is not int:
            _lifecycle_fail("run manifest partition seed must be an exact integer")
        _validated_hash_inventory(self.source_hashes, "source hashes")
        _validated_hash_inventory(self.dictionary_hashes, "Dictionary hashes")
        train = _validated_membership(
            self.train_tasks, "Train membership", expected_size=80
        )
        dev = _validated_membership(self.dev_tasks, "Dev membership", expected_size=20)
        _validated_task_hashes(
            self.train_task_hashes, "Train task hashes", train
        )
        _validated_task_hashes(self.dev_task_hashes, "Dev task hashes", dev)
        build = _validated_membership(
            self.build_tasks, "Build membership", expected_size=64
        )
        calibration = _validated_membership(
            self.calibration_tasks,
            "Calibration membership",
            expected_size=16,
        )
        train_map = dict(train)
        if set(build) & set(calibration):
            _lifecycle_fail("Build and Calibration membership overlap")
        if dict(build) | dict(calibration) != train_map:
            _lifecycle_fail("Build and Calibration must exactly partition Train")
        if {entity for _, entity in build} & {
            entity for _, entity in calibration
        }:
            _lifecycle_fail("Build and Calibration must be entity-disjoint")
        if {_canonical_identity(task_id) for task_id, _ in train} & {
            _canonical_identity(task_id) for task_id, _ in dev
        }:
            _lifecycle_fail("Train and Dev contain duplicate task IDs")
        if {_canonical_identity(entity) for _, entity in train} & {
            _canonical_identity(entity) for _, entity in dev
        }:
            _lifecycle_fail("Train and Dev contain duplicate entities")
        for name in (
            "split_manifest_fingerprint",
            "forecast_store_fingerprint",
            "metric_policy_fingerprint",
            "proposal_config_fingerprint",
            "proposal_implementation_fingerprint",
            "row_provider_fingerprint",
            "schedule_fingerprint",
            "numeric_grid_fingerprint",
            "runtime_fingerprint",
            "runtime_implementation_fingerprint",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.proposal_model) is not str or not self.proposal_model.strip():
            _lifecycle_fail("proposal model identity must be a nonempty exact string")
        for name in ("candidate_minimum_gain", "research_target_gain"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                _lifecycle_fail(f"{name} must be a finite nonnegative float")
        if self.research_target_gain < self.candidate_minimum_gain:
            _lifecycle_fail("research target cannot be below candidate threshold")

    def _identity_payload(self) -> dict[str, object]:
        return {
            entry.name: getattr(self, entry.name)
            for entry in fields(self)
        }

    @property
    def input_fingerprint(self) -> str:
        return champion_fingerprint(self._identity_payload())

    def to_payload(self) -> dict[str, object]:
        payload = self._identity_payload()
        payload["input_fingerprint"] = self.input_fingerprint
        return payload


def _validated_feedback_list(
    value: object,
    expected_fields: frozenset[str],
    label: str,
) -> tuple[dict[str, object], ...]:
    if type(value) is not list:
        _lifecycle_fail(f"checkpoint {label} feedback must be an exact JSON array")
    result: list[dict[str, object]] = []
    for raw_item in cast(list[object], value):
        if type(raw_item) is not dict or set(raw_item) != expected_fields:
            _lifecycle_fail(f"checkpoint {label} feedback has a malformed schema")
        result.append(cast(dict[str, object], raw_item))
    return tuple(result)


def _parse_sanitized_feedback(payload: dict[str, object]) -> ProposerEvidence:
    if set(payload) != {
        "label",
        "independent_generalization_claim",
        "morphology",
        "comparisons",
        "invalid_attempts",
    }:
        _lifecycle_fail("checkpoint feedback has a malformed schema")
    if (
        payload["label"] != "adaptive_train_build_diagnostic"
        or payload["independent_generalization_claim"] is not False
    ):
        _lifecycle_fail("checkpoint feedback violates Build-only authority")
    morphology_payloads = _validated_feedback_list(
        payload["morphology"],
        frozenset(MorphologyAggregate.__dataclass_fields__),
        "morphology",
    )
    comparison_payloads = _validated_feedback_list(
        payload["comparisons"],
        frozenset(_ProposerComparison.__dataclass_fields__),
        "comparison",
    )
    invalid_payloads = _validated_feedback_list(
        payload["invalid_attempts"],
        frozenset(_InvalidAttemptAggregate.__dataclass_fields__),
        "invalid-attempt",
    )
    try:
        morphology = tuple(
            MorphologyAggregate(**item)  # type: ignore[arg-type]
            for item in morphology_payloads
        )
        comparisons: list[_ProposerComparison] = []
        count_fields = {
            "smae_clipped_count",
            "srmse_clipped_count",
            "improved_folds",
            "total_folds",
            "wins",
            "ties",
            "losses",
        }
        raw_fields = {
            "p90_smae_raw",
            "p95_smae_raw",
            "p90_srmse_raw",
            "p95_srmse_raw",
        }
        for item in comparison_payloads:
            if (
                type(item["candidate_name"]) is not str
                or not cast(str, item["candidate_name"]).isidentifier()
            ):
                _lifecycle_fail("checkpoint comparison candidate is malformed")
            for name, value in item.items():
                if name == "candidate_name":
                    continue
                if name in count_fields:
                    if type(value) is not int or value < 0:
                        _lifecycle_fail("checkpoint comparison count is malformed")
                elif name in raw_fields and value == "positive_infinity":
                    continue
                elif type(value) is not float or not math.isfinite(value):
                    _lifecycle_fail("checkpoint comparison metric is malformed")
            comparisons.append(
                _ProposerComparison(**item)  # type: ignore[arg-type]
            )
        invalid_attempts = tuple(
            _InvalidAttemptAggregate(**item)  # type: ignore[arg-type]
            for item in invalid_payloads
        )
        evidence = ProposerEvidence(
            label="adaptive_train_build_diagnostic",
            independent_generalization_claim=False,
            morphology=morphology,
            comparisons=tuple(comparisons),
            invalid_attempts=invalid_attempts,
        )
        ProposerEvidence.__post_init__(evidence)
        _assert_sanitized(payload)
    except ChampionLifecycleError:
        raise
    except Exception as error:
        raise ChampionLifecycleError("checkpoint feedback is malformed") from error
    champion_fingerprint(payload)
    return evidence


@dataclass(frozen=True)
class ChampionCheckpoint:
    """Immutable post-Build state from which Calibration may run once."""

    schema_version: int
    input_fingerprint: str
    completed_stage: str
    active_parent: ChampionRelease
    proposal_archive: tuple[FittedChampionPolicy, ...]
    sanitized_feedback: ProposerEvidence
    build_evidence_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _lifecycle_fail("checkpoint schema_version must be exactly one")
        _require_sha256(self.input_fingerprint, "checkpoint input fingerprint")
        if type(self.completed_stage) is not str or _CHECKPOINT_STAGE.fullmatch(
            self.completed_stage
        ) is None:
            _lifecycle_fail("checkpoint completed stage is invalid")
        if type(self.active_parent) is not ChampionRelease:
            _lifecycle_fail("checkpoint active Parent must be an exact release")
        _validated_parent(self.active_parent)
        if (
            type(self.proposal_archive) is not tuple
            or len(self.proposal_archive) > 3
        ):
            _lifecycle_fail("checkpoint proposal archive cannot exceed three policies")
        policy_ids: list[str] = []
        for policy in self.proposal_archive:
            if type(policy) is not FittedChampionPolicy:
                _lifecycle_fail("checkpoint archive contains an invalid policy")
            FittedChampionPolicy.__post_init__(policy)
            policy_ids.append(champion_fingerprint(policy))
        if len(policy_ids) != len(set(policy_ids)):
            _lifecycle_fail("checkpoint archive contains duplicate policies")
        if type(self.sanitized_feedback) is not ProposerEvidence:
            _lifecycle_fail("checkpoint feedback must be exact sanitized evidence")
        try:
            ProposerEvidence.__post_init__(self.sanitized_feedback)
            _assert_sanitized(self.sanitized_feedback.to_payload())
        except Exception as error:
            raise ChampionLifecycleError("checkpoint feedback is malformed") from error
        _require_sha256(
            self.build_evidence_fingerprint,
            "checkpoint Build evidence fingerprint",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_fingerprint": self.input_fingerprint,
            "completed_stage": self.completed_stage,
            "active_parent": self.active_parent.to_payload(),
            "proposal_archive": [
                policy.to_payload() for policy in self.proposal_archive
            ],
            "sanitized_feedback": self.sanitized_feedback.to_payload(),
            "build_evidence_fingerprint": self.build_evidence_fingerprint,
        }


@dataclass(frozen=True)
class ChampionStageCandidateReport:
    policy_sha256: str
    evidence_fingerprint: str
    comparison: ChampionComparison | None
    invalid_reason: Literal["unscorable_child"] | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.policy_sha256, "reported policy fingerprint")
        _require_sha256(self.evidence_fingerprint, "reported evidence fingerprint")
        if self.comparison is None:
            if self.invalid_reason != "unscorable_child":
                _lifecycle_fail("invalid stage evidence requires a typed reason")
        else:
            if type(self.comparison) is not ChampionComparison:
                _lifecycle_fail("stage report contains an invalid comparison")
            ChampionComparison.__post_init__(self.comparison)
            if self.invalid_reason is not None:
                _lifecycle_fail("scored stage evidence cannot carry an invalid reason")

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_sha256": self.policy_sha256,
            "evidence_fingerprint": self.evidence_fingerprint,
            "comparison": (
                asdict(self.comparison) if self.comparison is not None else None
            ),
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True)
class ChampionEvaluationReport:
    schema_version: int
    input_fingerprint: str
    split: Literal["calibration", "dev"]
    parent_sha256: str
    candidates: tuple[ChampionStageCandidateReport, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _lifecycle_fail("evaluation report schema_version must be exactly one")
        _require_sha256(self.input_fingerprint, "report input fingerprint")
        _require_sha256(self.parent_sha256, "report Parent fingerprint")
        if type(self.split) is not str or self.split not in {"calibration", "dev"}:
            _lifecycle_fail("evaluation report has an invalid split")
        if type(self.candidates) is not tuple or not self.candidates:
            _lifecycle_fail("evaluation report requires candidate evidence")
        for candidate in self.candidates:
            if type(candidate) is not ChampionStageCandidateReport:
                _lifecycle_fail("evaluation report candidate is malformed")
            ChampionStageCandidateReport.__post_init__(candidate)
        fingerprints = tuple(item.policy_sha256 for item in self.candidates)
        if len(fingerprints) != len(set(fingerprints)):
            _lifecycle_fail("evaluation report contains duplicate candidates")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_fingerprint": self.input_fingerprint,
            "split": self.split,
            "parent_sha256": self.parent_sha256,
            "candidates": [item.to_payload() for item in self.candidates],
        }


@dataclass(frozen=True)
class ChampionEvolutionOutcome:
    manifest: ChampionRunManifest
    checkpoint: ChampionCheckpoint
    release: ChampionRelease
    build_result: BuildEvolutionResult | None
    calibration_report: ChampionEvaluationReport | None
    dev_report: ChampionEvaluationReport | None


def canonical_release_bytes(release: ChampionRelease) -> bytes:
    """Return the exact canonical bytes used by release publication."""
    if type(release) is not ChampionRelease:
        _lifecycle_fail("release must be an exact ChampionRelease")
    _validated_parent(release)
    return canonical_json_bytes(release.to_payload())


def _reject_nonfinite_json(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        _lifecycle_fail("stored JSON contains a nonfinite number")
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                _lifecycle_fail("stored JSON contains a non-string key")
            _reject_nonfinite_json(item)
    elif type(value) is list:
        for item in cast(list[object], value):
            _reject_nonfinite_json(item)


def _attested_file_digest(path: Path, label: str) -> str:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            _lifecycle_fail(f"{label} cannot use a symlink or path alias")
    try:
        descriptor = os.open(
            absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as error:
        raise ChampionLifecycleError(f"{label} is missing or unreadable") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            _lifecycle_fail(f"{label} must be an exact unaliased file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _attested_path_digest(path: Path, label: str) -> str:
    if path.is_symlink():
        _lifecycle_fail(f"{label} cannot use a symlink or path alias")
    if path.is_file():
        return _attested_file_digest(path, label)
    if not path.is_dir():
        _lifecycle_fail(f"{label} must be an exact file or directory")
    inventory: list[tuple[str, str]] = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            _lifecycle_fail(f"{label} cannot contain a symlink or path alias")
        if child.is_dir():
            continue
        if not child.is_file():
            _lifecycle_fail(f"{label} contains a non-file entry")
        inventory.append(
            (child.relative_to(path).as_posix(), _attested_file_digest(child, label))
        )
    return champion_fingerprint({"files": tuple(inventory)})


def _validated_attestation_files(
    values: object, label: str
) -> tuple[tuple[str, Path], ...]:
    if type(values) is not tuple or not values:
        _lifecycle_fail(f"{label} must be a nonempty exact tuple")
    parsed: list[tuple[str, Path]] = []
    identities: set[str] = set()
    for raw_pair in cast(tuple[object, ...], values):
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            _lifecycle_fail(f"{label} must contain exact name/path pairs")
        name, raw_path = cast(tuple[object, object], raw_pair)
        if type(name) is not str or not name or not name.isidentifier():
            _lifecycle_fail(f"{label} contains an invalid name")
        identity = _canonical_identity(name)
        if identity in identities:
            _lifecycle_fail(f"{label} contains a duplicate identity")
        identities.add(identity)
        if not isinstance(raw_path, Path):
            _lifecycle_fail(f"{label} paths must be exact Path values")
        path = cast(Path, raw_path).absolute()
        _attested_file_digest(path, label)
        parsed.append((name, path))
    result = tuple(parsed)
    if tuple(name for name, _ in result) != tuple(
        sorted(name for name, _ in result)
    ):
        _lifecycle_fail(f"{label} must use canonical sorted order")
    return result


@dataclass(frozen=True)
class ChampionRunAttestations:
    """Typed actual inputs from which manifest declarations are rederived."""

    source_files: tuple[tuple[str, Path], ...]
    split_manifest_file: Path
    dictionary_files: tuple[tuple[str, Path], ...]
    forecast_store: Path
    proposal_model: str
    proposal_config: object
    proposer_binding: ChampionProposerAdapter
    row_provider_binding: ChampionRowProviderAdapter
    runtime_bindings: ChampionRuntimeBindings
    numeric_grid: object
    runtime_files: tuple[tuple[str, Path], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_files",
            _validated_attestation_files(self.source_files, "source attestations"),
        )
        object.__setattr__(
            self,
            "dictionary_files",
            _validated_attestation_files(
                self.dictionary_files, "Dictionary attestations"
            ),
        )
        object.__setattr__(
            self,
            "runtime_files",
            _validated_attestation_files(self.runtime_files, "runtime attestations"),
        )
        for name in ("split_manifest_file", "forecast_store"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                _lifecycle_fail(f"{name} attestation must be an exact Path")
            absolute = cast(Path, value).absolute()
            _attested_path_digest(absolute, name)
            object.__setattr__(self, name, absolute)
        if type(self.proposal_model) is not str or not self.proposal_model.strip():
            _lifecycle_fail("actual proposal model identity must be nonempty")
        if (
            type(self.proposer_binding) is not ChampionProposerAdapter
            or type(self.row_provider_binding) is not ChampionRowProviderAdapter
            or type(self.runtime_bindings) is not ChampionRuntimeBindings
        ):
            _lifecycle_fail("actual formal callable bindings are malformed")
        self.proposer_binding.verify()
        self.row_provider_binding.verify()
        self.runtime_bindings.verify_live_globals()
        if self.proposer_binding.identity != self.proposal_model:
            _lifecycle_fail("actual proposer identity drifted from proposal model")
        if self.proposer_binding.config_fingerprint != champion_fingerprint(
            self.proposal_config
        ):
            _lifecycle_fail("actual proposer config drifted")
        try:
            champion_fingerprint(self.proposal_config)
            champion_fingerprint(self.numeric_grid)
        except Exception as error:
            raise ChampionLifecycleError(
                "proposal config and numeric grid must be canonical inputs"
            ) from error

    @staticmethod
    def _inventory(values: tuple[tuple[str, Path], ...]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, _attested_file_digest(path, f"{name} attestation"))
            for name, path in values
        )

    @property
    def source_hashes(self) -> tuple[tuple[str, str], ...]:
        return self._inventory(self.source_files)

    @property
    def dictionary_hashes(self) -> tuple[tuple[str, str], ...]:
        return self._inventory(self.dictionary_files)

    @property
    def split_manifest_fingerprint(self) -> str:
        return _attested_file_digest(
            self.split_manifest_file, "split manifest attestation"
        )

    @property
    def forecast_store_fingerprint(self) -> str:
        return _attested_path_digest(
            self.forecast_store, "forecast store attestation"
        )

    @property
    def proposal_config_fingerprint(self) -> str:
        return champion_fingerprint(self.proposal_config)

    @property
    def proposal_implementation_fingerprint(self) -> str:
        return self.proposer_binding.implementation_fingerprint

    @property
    def row_provider_fingerprint(self) -> str:
        return self.row_provider_binding.fingerprint

    @property
    def runtime_implementation_fingerprint(self) -> str:
        return self.runtime_bindings.fingerprint

    @property
    def numeric_grid_fingerprint(self) -> str:
        return champion_fingerprint(self.numeric_grid)

    @property
    def runtime_fingerprint(self) -> str:
        return champion_fingerprint({"files": self._inventory(self.runtime_files)})

    def verify(self, manifest: ChampionRunManifest) -> None:
        """Recompute every actual identity; no manifest digest authenticates itself."""
        ChampionRunAttestations.__post_init__(self)
        comparisons = {
            "source attestation": (self.source_hashes, manifest.source_hashes),
            "Dictionary attestation": (
                self.dictionary_hashes,
                manifest.dictionary_hashes,
            ),
            "split manifest attestation": (
                self.split_manifest_fingerprint,
                manifest.split_manifest_fingerprint,
            ),
            "forecast store attestation": (
                self.forecast_store_fingerprint,
                manifest.forecast_store_fingerprint,
            ),
            "proposal config attestation": (
                self.proposal_config_fingerprint,
                manifest.proposal_config_fingerprint,
            ),
            "proposal implementation attestation": (
                self.proposal_implementation_fingerprint,
                manifest.proposal_implementation_fingerprint,
            ),
            "row provider attestation": (
                self.row_provider_fingerprint,
                manifest.row_provider_fingerprint,
            ),
            "numeric grid attestation": (
                self.numeric_grid_fingerprint,
                manifest.numeric_grid_fingerprint,
            ),
            "runtime attestation": (
                self.runtime_fingerprint,
                manifest.runtime_fingerprint,
            ),
            "runtime implementation attestation": (
                self.runtime_implementation_fingerprint,
                manifest.runtime_implementation_fingerprint,
            ),
            "proposal model attestation": (
                self.proposal_model,
                manifest.proposal_model,
            ),
        }
        for label, (actual, expected) in comparisons.items():
            if actual != expected:
                _lifecycle_fail(f"{label} drifted from the run manifest")


def _open_pinned_directory(
    root: str | Path,
    label: str,
    *,
    create: bool,
) -> tuple[Path, tuple[tuple[str, int, int, int], ...]]:
    """Open every path component no-follow and pin the full ancestry chain."""
    absolute_root = Path(root).absolute()
    components = absolute_root.parts[1:]
    if any(component in {"", ".", ".."} for component in components):
        _lifecycle_fail(f"{label} cannot use a relative path component")
    chain: list[tuple[str, int, int, int]] = []
    anchor_descriptor: int | None = None
    try:
        anchor_descriptor = os.open(
            absolute_root.anchor,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        anchor_details = os.fstat(anchor_descriptor)
    except OSError as error:
        if anchor_descriptor is not None:
            os.close(anchor_descriptor)
        raise ChampionLifecycleError(f"{label} must be an exact directory") from error
    if not stat.S_ISDIR(anchor_details.st_mode):
        os.close(anchor_descriptor)
        _lifecycle_fail(f"{label} must be an exact directory")
    chain.append(
        (
            absolute_root.anchor,
            anchor_descriptor,
            anchor_details.st_dev,
            anchor_details.st_ino,
        )
    )
    try:
        for component in components:
            parent_descriptor = chain[-1][1]
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError as error:
                if not create:
                    raise ChampionLifecycleError(
                        f"{label} is missing and must be explicitly provisioned"
                    ) from error
                try:
                    os.mkdir(component, mode=0o700, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                    descriptor = os.open(
                        component,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_descriptor,
                    )
                except OSError as creation_error:
                    raise ChampionLifecycleError(
                        f"{label} cannot be safely provisioned"
                    ) from creation_error
            except OSError as error:
                raise ChampionLifecycleError(
                    f"{label} cannot use a symlink or path alias"
                ) from error
            try:
                details = os.fstat(descriptor)
                linked = os.stat(
                    component, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (
                    not stat.S_ISDIR(details.st_mode)
                    or stat.S_ISLNK(linked.st_mode)
                    or (details.st_dev, details.st_ino)
                    != (linked.st_dev, linked.st_ino)
                ):
                    _lifecycle_fail(f"{label} cannot use a symlink or path alias")
                chain.append((component, descriptor, details.st_dev, details.st_ino))
                descriptor = None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        return absolute_root, tuple(chain)
    except BaseException:
        for _, descriptor, _, _ in reversed(chain):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


class _PinnedJsonDirectory:
    """Canonical JSON primitives relative to a verified no-follow directory FD."""

    def __init__(
        self, root: str | Path, label: str, *, create: bool = True
    ) -> None:
        self._closed = False
        self.root, self._root_chain = _open_pinned_directory(
            root, label, create=create
        )
        _, self._root_fd, self._root_dev, self._root_ino = self._root_chain[-1]
        self._root_label = label

    def close(self) -> None:
        """Idempotently release every pinned ancestry descriptor."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for _, descriptor, _, _ in reversed(getattr(self, "_root_chain", ())):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> "_PinnedJsonDirectory":
        self._verify_root()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _verify_root(self) -> None:
        if self._closed:
            _lifecycle_fail(f"{self._root_label} is closed")
        for index, (component, descriptor, device, inode) in enumerate(
            self._root_chain
        ):
            try:
                pinned = os.fstat(descriptor)
                if index == 0:
                    linked = os.lstat(component)
                else:
                    linked = os.stat(
                        component,
                        dir_fd=self._root_chain[index - 1][1],
                        follow_symlinks=False,
                    )
            except OSError as error:
                raise ChampionLifecycleError(
                    f"{self._root_label} path drifted from its pinned authority"
                ) from error
            if (
                stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISDIR(linked.st_mode)
                or (linked.st_dev, linked.st_ino) != (device, inode)
                or (pinned.st_dev, pinned.st_ino) != (device, inode)
            ):
                _lifecycle_fail(
                    f"{self._root_label} path drifted from its pinned authority"
                )

    def _simple_name(self, name: str) -> str:
        if (
            type(name) is not str
            or not name
            or Path(name).name != name
            or not name.endswith(".json")
        ):
            _lifecycle_fail("artifact name must be a simple JSON filename")
        return name

    def _name_exists(self, name: str) -> bool:
        self._verify_root()
        safe_name = self._simple_name(name)
        try:
            details = os.stat(safe_name, dir_fd=self._root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ChampionLifecycleError(
                f"artifact cannot be inspected: {safe_name}"
            ) from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            _lifecycle_fail(f"artifact is missing or aliased: {safe_name}")
        if details.st_nlink != 1:
            _lifecycle_fail(f"artifact is an immutable hardlink alias: {safe_name}")
        return True

    def _sync_root(self) -> None:
        self._verify_root()
        os.fsync(self._root_fd)

    def _write_temporary(self, name: str, content: bytes) -> str:
        self._verify_root()
        for _ in range(32):
            temporary = f".{name}.{secrets.token_hex(12)}.tmp"
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=self._root_fd,
                )
                break
            except FileExistsError:
                continue
            except OSError as error:
                raise ChampionLifecycleError(
                    f"cannot create temporary artifact for {name}"
                ) from error
        else:
            _lifecycle_fail(f"cannot allocate temporary artifact for {name}")
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    _lifecycle_fail(f"cannot write temporary artifact for {name}")
                view = view[written:]
            os.fsync(descriptor)
        except Exception:
            try:
                os.unlink(temporary, dir_fd=self._root_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        self._verify_root()
        return temporary

    def _atomic_create_name(self, name: str, payload: dict[str, object]) -> None:
        safe_name = self._simple_name(name)
        if self._name_exists(safe_name):
            _lifecycle_fail(f"immutable artifact already exists: {safe_name}")
        temporary = self._write_temporary(safe_name, canonical_json_bytes(payload))
        try:
            try:
                os.link(
                    temporary,
                    safe_name,
                    src_dir_fd=self._root_fd,
                    dst_dir_fd=self._root_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise ChampionLifecycleError(
                    f"immutable artifact already exists: {safe_name}"
                ) from error
            self._sync_root()
        finally:
            removed_temporary = False
            try:
                os.unlink(temporary, dir_fd=self._root_fd)
                removed_temporary = True
            except FileNotFoundError:
                pass
            if removed_temporary:
                self._sync_root()

    def _atomic_replace_name(self, name: str, content: bytes) -> None:
        safe_name = self._simple_name(name)
        if self._name_exists(safe_name):
            # _name_exists also rejects symlinks, hardlinks, and non-files.
            pass
        temporary = self._write_temporary(safe_name, content)
        try:
            self._verify_root()
            os.replace(
                temporary,
                safe_name,
                src_dir_fd=self._root_fd,
                dst_dir_fd=self._root_fd,
            )
            temporary = ""
            self._sync_root()
        finally:
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=self._root_fd)
                except FileNotFoundError:
                    pass

    def _force_replace_name_bytes(self, name: str, content: bytes) -> None:
        """Replace one exact leaf through the pinned dirfd without following it."""
        safe_name = self._simple_name(name)
        temporary = self._write_temporary(safe_name, content)
        try:
            self._verify_root()
            os.replace(
                temporary,
                safe_name,
                src_dir_fd=self._root_fd,
                dst_dir_fd=self._root_fd,
            )
            temporary = ""
            self._sync_root()
        finally:
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=self._root_fd)
                except OSError:
                    pass
        if self._read_name_bytes(safe_name) != content:
            _lifecycle_fail(f"artifact restoration could not be verified: {safe_name}")

    def _read_name_bytes(self, name: str) -> bytes:
        self._verify_root()
        safe_name = self._simple_name(name)
        try:
            descriptor = os.open(
                safe_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._root_fd,
            )
        except OSError as error:
            raise ChampionLifecycleError(
                f"artifact is missing or aliased: {safe_name}"
            ) from error
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                _lifecycle_fail(f"artifact is an immutable alias: {safe_name}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        self._verify_root()
        return b"".join(chunks)

    def _read_name_payload(self, name: str) -> dict[str, object]:
        safe_name = self._simple_name(name)
        raw = self._read_name_bytes(safe_name)
        try:
            payload = strict_json_loads(raw.decode("utf-8"), context=safe_name)
        except (UnicodeError, ValueError) as error:
            raise ChampionLifecycleError(
                f"artifact contains malformed JSON: {safe_name}"
            ) from error
        if type(payload) is not dict:
            _lifecycle_fail(f"artifact must contain an exact JSON object: {safe_name}")
        result = cast(dict[str, object], payload)
        _reject_nonfinite_json(result)
        try:
            canonical = canonical_json_bytes(result)
        except (TypeError, ValueError) as error:
            raise ChampionLifecycleError(
                f"artifact is not canonical JSON: {safe_name}"
            ) from error
        if raw != canonical:
            _lifecycle_fail(f"artifact is not canonical JSON: {safe_name}")
        return result


@dataclass(frozen=True)
class ChampionSplitLease:
    """Exclusive operator-owned authority to consume one holdout split."""

    role: Literal["calibration", "dev"]
    split_key: str
    run_input_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in {"calibration", "dev"}:
            _lifecycle_fail("split lease has an invalid role")
        _require_sha256(self.split_key, "split lease key")
        _require_sha256(
            self.run_input_fingerprint, "split lease run input fingerprint"
        )


@dataclass(frozen=True)
class _AuthorityProviderBundle:
    lease: ChampionSplitLease
    candidate_fingerprint: str
    runtime_fingerprint: str
    rows: tuple[ChampionTaskRow, ...]


def _row_payload(row: ChampionTaskRow) -> dict[str, object]:
    return cast(dict[str, object], asdict(row))


def _parse_tuple_numbers(value: object, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if type(value) is not list:
        _lifecycle_fail(f"authority {label} must be an exact JSON array")
    values = cast(list[object], value)
    if any(type(item) not in {int, float} for item in values):
        _lifecycle_fail(f"authority {label} contains a nonnumeric value")
    result = tuple(float(cast(int | float, item)) for item in values)
    if any(not math.isfinite(item) for item in result):
        _lifecycle_fail(f"authority {label} contains a nonfinite value")
    return result


def _parse_authority_row(value: object) -> ChampionTaskRow:
    if type(value) is not dict or set(cast(dict[object, object], value)) != {
        entry.name for entry in fields(ChampionTaskRow)
    }:
        _lifecycle_fail("authority provider row has a malformed schema")
    payload = cast(dict[str, object], value)
    profile_payload = payload["profile"]
    if type(profile_payload) is not dict or set(
        cast(dict[object, object], profile_payload)
    ) != {entry.name for entry in fields(TaskProfile)}:
        _lifecycle_fail("authority provider profile has a malformed schema")
    profile_values = dict(cast(dict[str, object], profile_payload))
    periods = profile_values["periodicity_periods"]
    if type(periods) is not list:
        _lifecycle_fail("authority provider periods have a malformed schema")
    profile_values["periodicity_periods"] = tuple(cast(list[object], periods))
    diagnostic_payload = payload["diagnostic"]
    diagnostic: ChampionHistoryDiagnostic | None
    if diagnostic_payload is None:
        diagnostic = None
    else:
        if type(diagnostic_payload) is not dict or set(
            cast(dict[object, object], diagnostic_payload)
        ) != {entry.name for entry in fields(ChampionHistoryDiagnostic)}:
            _lifecycle_fail("authority provider diagnostic has a malformed schema")
        try:
            diagnostic = ChampionHistoryDiagnostic(
                **cast(dict[str, object], diagnostic_payload)  # type: ignore[arg-type]
            )
        except Exception as error:
            raise ChampionLifecycleError(
                "authority provider diagnostic is malformed"
            ) from error
    try:
        row = ChampionTaskRow(
            task_id=payload["task_id"],  # type: ignore[arg-type]
            candidate_name=payload["candidate_name"],  # type: ignore[arg-type]
            profile=TaskProfile(**profile_values),  # type: ignore[arg-type]
            truth=_parse_tuple_numbers(payload["truth"], "truth"),
            forecast=_parse_tuple_numbers(payload["forecast"], "forecast"),
            failure_reason=payload["failure_reason"],  # type: ignore[arg-type]
            fold=payload["fold"],  # type: ignore[arg-type]
            split=payload["split"],  # type: ignore[arg-type]
            history=_parse_tuple_numbers(payload["history"], "history"),
            diagnostic=diagnostic,
        )
        ChampionTaskRow.__post_init__(row)
    except ChampionLifecycleError:
        raise
    except Exception as error:
        raise ChampionLifecycleError("authority provider row is malformed") from error
    return row


class ChampionAuthorityStore(_PinnedJsonDirectory):
    """Operator-owned, run-directory-independent one-shot split authority."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_authority_identity: str,
    ) -> None:
        expected = _require_sha256(
            expected_authority_identity, "expected authority identity"
        )
        super().__init__(root, "authority root", create=False)
        try:
            identity_name = "authority_identity.json"
            if not self._name_exists(identity_name):
                _lifecycle_fail("authority root is not explicitly provisioned")
            payload = self._read_name_payload(identity_name)
            if set(payload) != {"schema_version", "authority_identity"} or (
                payload["schema_version"] != 1
            ):
                _lifecycle_fail("authority identity anchor is malformed")
            if payload["authority_identity"] != expected:
                _lifecycle_fail("authority identity does not match operator authority")
            self.authority_identity = expected
        except BaseException:
            self.close()
            raise

    @classmethod
    def provision(
        cls,
        root: str | Path,
        *,
        authority_identity: str,
    ) -> "ChampionAuthorityStore":
        """Explicitly provision once; normal construction never creates authority."""
        expected = _require_sha256(authority_identity, "authority identity")
        path = Path(root).absolute()
        if path.exists() or path.is_symlink():
            _lifecycle_fail("authority root is already provisioned")
        instance = object.__new__(cls)
        _PinnedJsonDirectory.__init__(
            instance, path, "authority root", create=True
        )
        try:
            instance._atomic_create_name(
                "authority_identity.json",
                {"schema_version": 1, "authority_identity": expected},
            )
            instance.authority_identity = expected
        except BaseException:
            instance.close()
            raise
        return instance

    @staticmethod
    def _split_key(
        role: Literal["calibration", "dev"],
        split_manifest_fingerprint: str,
        task_content_fingerprint: str,
    ) -> str:
        if type(role) is not str or role not in {"calibration", "dev"}:
            _lifecycle_fail("split authority has an invalid role")
        _require_sha256(
            split_manifest_fingerprint, "split authority manifest fingerprint"
        )
        _require_sha256(
            task_content_fingerprint, "split authority task-content fingerprint"
        )
        return champion_fingerprint(
            {
                "role": role,
                "task_content_fingerprint": task_content_fingerprint,
            }
        )

    def acquire_split(
        self,
        *,
        role: Literal["calibration", "dev"],
        split_manifest_fingerprint: str,
        task_content_fingerprint: str,
        run_input_fingerprint: str,
    ) -> ChampionSplitLease:
        """Irreversibly acquire a split before any provider sees its tasks."""
        split_key = self._split_key(
            role, split_manifest_fingerprint, task_content_fingerprint
        )
        _require_sha256(run_input_fingerprint, "split authority run fingerprint")
        lease = ChampionSplitLease(role, split_key, run_input_fingerprint)
        name = f"{split_key}.lease.json"
        payload: dict[str, object] = {
            "schema_version": 1,
            "role": role,
            "split_key": split_key,
            "run_input_fingerprint": run_input_fingerprint,
        }
        try:
            self._atomic_create_name(name, payload)
        except ChampionLifecycleError as error:
            if self._name_exists(name):
                _lifecycle_fail(f"{role} split has already been consumed")
            raise error
        return lease

    def bind_build_evidence(
        self,
        *,
        run_authority_fingerprint: str,
        input_fingerprint: str,
        build_evidence_fingerprint: str,
    ) -> None:
        """Anchor Build provenance outside every replaceable run directory."""
        for value, label in (
            (run_authority_fingerprint, "Build run authority fingerprint"),
            (input_fingerprint, "Build input fingerprint"),
            (build_evidence_fingerprint, "Build evidence fingerprint"),
        ):
            _require_sha256(value, label)
        payload: dict[str, object] = {
            "schema_version": 1,
            "run_authority_fingerprint": run_authority_fingerprint,
            "input_fingerprint": input_fingerprint,
            "build_evidence_fingerprint": build_evidence_fingerprint,
        }
        name = f"{run_authority_fingerprint}.build.json"
        if not self._name_exists(name):
            self._atomic_create_name(name, payload)
        elif canonical_json_bytes(self._read_name_payload(name)) != canonical_json_bytes(
            payload
        ):
            _lifecycle_fail("operator-owned Build evidence authority drifted")

    def require_build_evidence(
        self,
        *,
        run_authority_fingerprint: str,
        input_fingerprint: str,
        build_evidence_fingerprint: str,
    ) -> None:
        for value, label in (
            (run_authority_fingerprint, "Build run authority fingerprint"),
            (input_fingerprint, "Build input fingerprint"),
            (build_evidence_fingerprint, "Build evidence fingerprint"),
        ):
            _require_sha256(value, label)
        name = f"{run_authority_fingerprint}.build.json"
        if not self._name_exists(name):
            _lifecycle_fail("operator-owned Build evidence authority disappeared")
        expected: dict[str, object] = {
            "schema_version": 1,
            "run_authority_fingerprint": run_authority_fingerprint,
            "input_fingerprint": input_fingerprint,
            "build_evidence_fingerprint": build_evidence_fingerprint,
        }
        if canonical_json_bytes(self._read_name_payload(name)) != canonical_json_bytes(
            expected
        ):
            _lifecycle_fail("operator-owned Build evidence authority drifted")

    def _load_lease(self, split_key: str) -> ChampionSplitLease:
        payload = self._read_name_payload(f"{split_key}.lease.json")
        if set(payload) != {
            "schema_version",
            "role",
            "split_key",
            "run_input_fingerprint",
        } or payload["schema_version"] != 1:
            _lifecycle_fail("split lease has a malformed schema")
        try:
            lease = ChampionSplitLease(
                role=payload["role"],  # type: ignore[arg-type]
                split_key=payload["split_key"],  # type: ignore[arg-type]
                run_input_fingerprint=payload["run_input_fingerprint"],  # type: ignore[arg-type]
            )
        except Exception as error:
            raise ChampionLifecycleError("split lease is malformed") from error
        if lease.split_key != split_key:
            _lifecycle_fail("split lease key drifted from its authority name")
        return lease

    def claim_or_resume_split(
        self,
        *,
        role: Literal["calibration", "dev"],
        split_manifest_fingerprint: str,
        task_content_fingerprint: str,
        run_input_fingerprint: str,
        candidate_fingerprint: str,
        runtime_fingerprint: str,
    ) -> tuple[ChampionSplitLease, _AuthorityProviderBundle | None]:
        """Acquire once, or resume only from a complete committed result bundle."""
        split_key = self._split_key(
            role, split_manifest_fingerprint, task_content_fingerprint
        )
        for value, label in (
            (run_input_fingerprint, "split authority run fingerprint"),
            (candidate_fingerprint, "split authority candidate fingerprint"),
            (runtime_fingerprint, "split authority runtime fingerprint"),
        ):
            _require_sha256(value, label)
        lease_name = f"{split_key}.lease.json"
        if not self._name_exists(lease_name):
            return (
                self.acquire_split(
                    role=role,
                    split_manifest_fingerprint=split_manifest_fingerprint,
                    task_content_fingerprint=task_content_fingerprint,
                    run_input_fingerprint=run_input_fingerprint,
                ),
                None,
            )
        lease = self._load_lease(split_key)
        bundle_name = f"{split_key}.provider.json"
        if (
            lease.role != role
            or lease.run_input_fingerprint != run_input_fingerprint
            or not self._name_exists(bundle_name)
        ):
            _lifecycle_fail(f"{role} split has already been consumed")
        payload = self._read_name_payload(bundle_name)
        if set(payload) != {
            "schema_version",
            "role",
            "split_key",
            "run_input_fingerprint",
            "candidate_fingerprint",
            "runtime_fingerprint",
            "rows_fingerprint",
            "rows",
        } or payload["schema_version"] != 1:
            _lifecycle_fail("split provider bundle has a malformed schema")
        if (
            payload["role"] != role
            or payload["split_key"] != split_key
            or payload["run_input_fingerprint"] != run_input_fingerprint
            or payload["candidate_fingerprint"] != candidate_fingerprint
            or payload["runtime_fingerprint"] != runtime_fingerprint
        ):
            _lifecycle_fail(f"{role} split provider bundle is stale or foreign")
        raw_rows = payload["rows"]
        if type(raw_rows) is not list or not raw_rows:
            _lifecycle_fail("split provider bundle has malformed rows")
        rows = tuple(_parse_authority_row(item) for item in cast(list[object], raw_rows))
        if champion_fingerprint(rows) != payload["rows_fingerprint"]:
            _lifecycle_fail("split provider bundle rows drifted")
        return (
            lease,
            _AuthorityProviderBundle(
                lease=lease,
                candidate_fingerprint=candidate_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
                rows=rows,
            ),
        )

    def commit_provider_bundle(
        self,
        lease: ChampionSplitLease,
        *,
        candidate_fingerprint: str,
        runtime_fingerprint: str,
        rows: tuple[ChampionTaskRow, ...],
    ) -> None:
        ChampionSplitLease.__post_init__(lease)
        _require_sha256(candidate_fingerprint, "provider candidate fingerprint")
        _require_sha256(runtime_fingerprint, "provider runtime fingerprint")
        if type(rows) is not tuple or not rows:
            _lifecycle_fail("provider bundle requires exact nonempty rows")
        stored_lease = self._load_lease(lease.split_key)
        if stored_lease != lease:
            _lifecycle_fail("provider bundle lease drifted")
        self._atomic_create_name(
            f"{lease.split_key}.provider.json",
            {
                "schema_version": 1,
                "role": lease.role,
                "split_key": lease.split_key,
                "run_input_fingerprint": lease.run_input_fingerprint,
                "candidate_fingerprint": candidate_fingerprint,
                "runtime_fingerprint": runtime_fingerprint,
                "rows_fingerprint": champion_fingerprint(rows),
                "rows": [_row_payload(row) for row in rows],
            },
        )

    def commit_evaluation(
        self, lease: ChampionSplitLease, report: ChampionEvaluationReport
    ) -> None:
        ChampionSplitLease.__post_init__(lease)
        ChampionEvaluationReport.__post_init__(report)
        if report.split != lease.role:
            _lifecycle_fail("evaluation report crossed its split lease")
        self._atomic_create_name(
            f"{lease.split_key}.evaluation.json",
            {
                "schema_version": 1,
                "role": lease.role,
                "split_key": lease.split_key,
                "run_input_fingerprint": lease.run_input_fingerprint,
                "report_fingerprint": champion_fingerprint(report.to_payload()),
                "report": report.to_payload(),
            },
        )

    def verify_evaluation(
        self, lease: ChampionSplitLease, report: ChampionEvaluationReport
    ) -> bool:
        name = f"{lease.split_key}.evaluation.json"
        if not self._name_exists(name):
            return False
        payload = self._read_name_payload(name)
        expected = {
            "schema_version": 1,
            "role": lease.role,
            "split_key": lease.split_key,
            "run_input_fingerprint": lease.run_input_fingerprint,
            "report_fingerprint": champion_fingerprint(report.to_payload()),
            "report": report.to_payload(),
        }
        if canonical_json_bytes(payload) != canonical_json_bytes(expected):
            _lifecycle_fail("committed split evaluation drifted from bound evidence")
        return True


class ChampionArtifactStore(_PinnedJsonDirectory):
    """Canonical, atomic, immutable lifecycle artifact storage."""

    def __init__(self, root: str | Path) -> None:
        super().__init__(root, "artifact root")
        try:
            self._release_archive = _PinnedJsonDirectory(
                self.root / "releases", "release archive root"
            )
            self._recover_publication()
        except BaseException:
            release_archive = getattr(self, "_release_archive", None)
            if release_archive is not None:
                release_archive.close()
            super().close()
            raise

    def close(self) -> None:
        release_archive = getattr(self, "_release_archive", None)
        if release_archive is not None:
            release_archive.close()
        super().close()

    def _publication_hook(self, phase: str) -> None:
        """Deterministic failure-injection seam for transaction tests."""

    def _unlink_root_name(self, name: str) -> None:
        self._verify_root()
        try:
            os.unlink(name, dir_fd=self._root_fd)
        except FileNotFoundError:
            return

    def _archive_release_bytes(self, release_id: str) -> bytes:
        _require_sha256(release_id, "archived release fingerprint")
        payload = self._release_archive._read_name_payload(f"{release_id}.json")
        try:
            release = parse_champion_release(payload)
        except Exception as error:
            raise ChampionLifecycleError("archived release is malformed") from error
        if champion_fingerprint(release) != release_id:
            _lifecycle_fail("archived release fingerprint drifted")
        return canonical_release_bytes(release)

    def _parse_publication_record(
        self, payload: dict[str, object], label: str
    ) -> tuple[str, str | None, str, bool]:
        if set(payload) != {
            "schema_version",
            "transaction_id",
            "prior_release_id",
            "next_release_id",
            "accepted",
        } or payload["schema_version"] != 1:
            _lifecycle_fail(f"{label} publication record is malformed")
        transaction_id = _require_sha256(
            payload["transaction_id"], f"{label} transaction ID"
        )
        prior = payload["prior_release_id"]
        if prior is not None:
            prior = _require_sha256(prior, f"{label} prior release ID")
        next_release = _require_sha256(
            payload["next_release_id"], f"{label} next release ID"
        )
        accepted = payload["accepted"]
        if type(accepted) is not bool:
            _lifecycle_fail(f"{label} publication accepted flag is malformed")
        expected_transaction = champion_fingerprint(
            {
                "prior_release_id": prior,
                "next_release_id": next_release,
                "accepted": accepted,
            }
        )
        if transaction_id != expected_transaction:
            _lifecycle_fail(f"{label} publication transaction drifted")
        return transaction_id, cast(str | None, prior), next_release, accepted

    def _bind_accepted_release(self, transaction_id: str, release_id: str) -> None:
        payload: dict[str, object] = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "release_id": release_id,
        }
        name = "last_accepted_release.json"
        if not self._name_exists(name):
            self._atomic_create_name(name, payload)
        elif canonical_json_bytes(self._read_name_payload(name)) != canonical_json_bytes(
            payload
        ):
            _lifecycle_fail("last accepted release authority cannot be overwritten")

    def _read_accepted_release_authority(self) -> tuple[str, str, bytes] | None:
        name = "last_accepted_release.json"
        if not self._name_exists(name):
            return None
        payload = self._read_name_payload(name)
        if set(payload) != {"schema_version", "transaction_id", "release_id"} or (
            payload["schema_version"] != 1
        ):
            _lifecycle_fail("last accepted release authority is malformed")
        transaction_id = _require_sha256(
            payload["transaction_id"], "accepted transaction ID"
        )
        release_id = _require_sha256(
            payload["release_id"], "last accepted release fingerprint"
        )
        return transaction_id, release_id, self._archive_release_bytes(release_id)

    def has_accepted_release(self) -> bool:
        authority = self._read_accepted_release_authority()
        if authority is None:
            return False
        _, _, expected = authority
        if (
            not self._name_exists("champion_release.json")
            or self._read_name_bytes("champion_release.json") != expected
        ):
            _lifecycle_fail("last accepted release pointer drifted")
        return True

    def _restore_transaction_prior(self, prior_release_id: str | None) -> None:
        current = self.root / "champion_release.json"
        if prior_release_id is None:
            self._unlink_root_name("champion_release.json")
            self._sync_root()
            return
        prior_bytes = self._archive_release_bytes(prior_release_id)
        self._atomic_replace(current, prior_bytes)
        if self._read_name_bytes("champion_release.json") != prior_bytes:
            _lifecycle_fail("publication rollback did not restore the exact Parent")

    def _recover_publication(self) -> None:
        """Resolve an interrupted pointer transaction on every fresh-process open."""
        prepare_name = "release_prepare.json"
        commit_name = "release_commit.json"
        has_prepare = self._name_exists(prepare_name)
        has_commit = self._name_exists(commit_name)
        accepted_authority = self._read_accepted_release_authority()
        if accepted_authority is not None:
            accepted_transaction, accepted_release, accepted_bytes = accepted_authority
            for marker_name, label in (
                (prepare_name, "prepared"),
                (commit_name, "committed"),
            ):
                if not self._name_exists(marker_name):
                    continue
                marker = self._parse_publication_record(
                    self._read_name_payload(marker_name), label
                )
                marker_transaction, _, marker_release, marker_accepted = marker
                if (
                    marker_transaction != accepted_transaction
                    or marker_release != accepted_release
                    or marker_accepted is not True
                ):
                    _lifecycle_fail(
                        "publication marker drifted from accepted transaction authority"
                    )
            try:
                pointer_matches = self._name_exists(
                    "champion_release.json"
                ) and self._read_name_bytes("champion_release.json") == accepted_bytes
            except ChampionLifecycleError:
                pointer_matches = False
            if not pointer_matches:
                self._force_replace_name_bytes(
                    "champion_release.json", accepted_bytes
                )
            if has_commit:
                self._unlink_root_name(commit_name)
                self._sync_root()
            if has_prepare:
                self._unlink_root_name(prepare_name)
                self._sync_root()
            return
        if not has_prepare:
            if has_commit:
                _lifecycle_fail("orphan publication commit marker is malformed state")
            return
        prepare = self._read_name_payload(prepare_name)
        transaction_id, prior_release_id, next_release_id, accepted = (
            self._parse_publication_record(prepare, "prepared")
        )
        if has_commit:
            commit = self._read_name_payload(commit_name)
            committed_id, committed_prior, committed_next, committed_accepted = (
                self._parse_publication_record(commit, "committed")
            )
            if (
                committed_id != transaction_id
                or committed_prior != prior_release_id
                or committed_next != next_release_id
                or committed_accepted != accepted
            ):
                _lifecycle_fail("publication commit marker is stale or foreign")
            next_bytes = self._archive_release_bytes(next_release_id)
            current = self.root / "champion_release.json"
            if (
                not self._name_exists("champion_release.json")
                or self._read_name_bytes("champion_release.json") != next_bytes
            ):
                self._atomic_replace(current, next_bytes)
            if accepted:
                self._bind_accepted_release(transaction_id, next_release_id)
        else:
            self._restore_transaction_prior(prior_release_id)
            if self._name_exists("last_accepted_release.json"):
                accepted_payload = self._read_name_payload(
                    "last_accepted_release.json"
                )
                if accepted_payload.get("transaction_id") == transaction_id:
                    self._unlink_root_name("last_accepted_release.json")
        self._unlink_root_name(commit_name)
        self._unlink_root_name(prepare_name)
        self._sync_root()

    def _path(self, name: str) -> Path:
        self._verify_root()
        if (
            type(name) is not str
            or not name
            or Path(name).name != name
            or not name.endswith(".json")
        ):
            _lifecycle_fail("artifact name must be a simple JSON filename")
        path = self.root / name
        if self._name_exists(name):
            pass
        return path

    def _sync_directory(self, directory: Path) -> None:
        if directory == self.root:
            self._sync_root()
            return
        if directory == self._release_archive.root:
            self._release_archive._sync_root()
            return
        _lifecycle_fail("artifact directory escaped its pinned root")

    def _atomic_create(self, path: Path, payload: dict[str, object]) -> None:
        if path.parent == self.root:
            self._atomic_create_name(path.name, payload)
            return
        if path.parent == self._release_archive.root:
            self._release_archive._atomic_create_name(path.name, payload)
            return
        _lifecycle_fail("artifact creation escaped its pinned root")

    def _atomic_replace(self, path: Path, content: bytes) -> None:
        if path.parent != self.root:
            _lifecycle_fail("release replacement escaped its pinned root")
        self._atomic_replace_name(path.name, content)

    def _read_payload(self, path: Path) -> dict[str, object]:
        if path.parent == self.root:
            return self._read_name_payload(path.name)
        if path.parent == self._release_archive.root:
            return self._release_archive._read_name_payload(path.name)
        _lifecycle_fail("artifact read escaped its pinned root")

    def bind_manifest(self, manifest: ChampionRunManifest) -> Path:
        if type(manifest) is not ChampionRunManifest:
            _lifecycle_fail("manifest must be an exact ChampionRunManifest")
        ChampionRunManifest.__post_init__(manifest)
        path = self._path("run_manifest.json")
        if not path.exists():
            self._atomic_create(path, manifest.to_payload())
            return path
        if canonical_json_bytes(self._read_payload(path)) != canonical_json_bytes(
            manifest.to_payload()
        ):
            _lifecycle_fail("run manifest input drifted from durable authority")
        return path

    def bind_run_instance(self, manifest: ChampionRunManifest) -> str:
        """Bootstrap once per durable run root; reopening preserves this identity."""
        if type(manifest) is not ChampionRunManifest:
            _lifecycle_fail("run instance requires an exact manifest")
        name = "run_instance.json"
        if not self._name_exists(name):
            instance_fingerprint = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
            self._atomic_create_name(
                name,
                {
                    "schema_version": 1,
                    "input_fingerprint": manifest.input_fingerprint,
                    "instance_fingerprint": instance_fingerprint,
                },
            )
            return instance_fingerprint
        payload = self._read_name_payload(name)
        if set(payload) != {
            "schema_version",
            "input_fingerprint",
            "instance_fingerprint",
        } or payload["schema_version"] != 1:
            _lifecycle_fail("run instance authority is malformed")
        if payload["input_fingerprint"] != manifest.input_fingerprint:
            _lifecycle_fail("run instance belongs to a stale or foreign manifest")
        return _require_sha256(
            payload["instance_fingerprint"], "run instance fingerprint"
        )

    def require_open_lifecycle(self, run_authority_fingerprint: str) -> None:
        _require_sha256(run_authority_fingerprint, "run authority fingerprint")
        name = "lifecycle_complete.json"
        if not self._name_exists(name):
            return
        payload = self._read_name_payload(name)
        if set(payload) != {
            "schema_version",
            "run_authority_fingerprint",
            "status",
            "release_fingerprint",
        } or payload["schema_version"] != 1:
            _lifecycle_fail("lifecycle completion authority is malformed")
        if payload["run_authority_fingerprint"] != run_authority_fingerprint:
            _lifecycle_fail("lifecycle completion belongs to a stale or foreign run")
        _require_sha256(payload["release_fingerprint"], "completed release fingerprint")
        _lifecycle_fail("one-shot lifecycle report cannot be overwritten or replayed")

    def complete_lifecycle(
        self,
        *,
        run_authority_fingerprint: str,
        status: Literal[
            "no_finalist", "calibration_rejected", "dev_rejected", "accepted"
        ],
        release: ChampionRelease,
    ) -> None:
        _require_sha256(run_authority_fingerprint, "run authority fingerprint")
        if status not in {
            "no_finalist",
            "calibration_rejected",
            "dev_rejected",
            "accepted",
        }:
            _lifecycle_fail("lifecycle completion status is invalid")
        self._atomic_create_name(
            "lifecycle_complete.json",
            {
                "schema_version": 1,
                "run_authority_fingerprint": run_authority_fingerprint,
                "status": status,
                "release_fingerprint": champion_fingerprint(release),
            },
        )

    def write_checkpoint(self, checkpoint: ChampionCheckpoint) -> Path:
        if type(checkpoint) is not ChampionCheckpoint:
            _lifecycle_fail("checkpoint must be an exact ChampionCheckpoint")
        ChampionCheckpoint.__post_init__(checkpoint)
        path = self._path("checkpoint.json")
        self._atomic_create(path, checkpoint.to_payload())
        return path

    def write_build_evidence(
        self,
        *,
        manifest: ChampionRunManifest,
        parent: ChampionRelease,
        result: BuildEvolutionResult,
        rows: tuple[ChampionTaskRow, ...],
    ) -> str:
        """Persist the complete immutable Build attempt provenance."""
        combined_feedback = _combined_feedback(result)
        combined_feedback_payload = combined_feedback.to_payload()
        payload: dict[str, object] = {
            "schema_version": 1,
            "input_fingerprint": manifest.input_fingerprint,
            "parent_fingerprint": champion_fingerprint(parent),
            "config_fingerprint": result.config_fingerprint,
            "build_rows_fingerprint": champion_fingerprint(rows),
            "generations": [
                {
                    "number": generation.number,
                    "mutation_parent_sha256": generation.mutation_parent_sha256,
                    "config_fingerprint": generation.config_fingerprint,
                    "stage_counts": generation.stage_counts,
                    "full_build_children": generation.full_build_children,
                    "attempts": [
                        {
                            "fitted_id": attempt.fitted_id,
                            "policy": attempt.policy.to_payload(),
                            "stage_task_counts": attempt.stage_task_counts,
                            "comparison": (
                                asdict(attempt.comparison)
                                if attempt.comparison is not None
                                else None
                            ),
                            "status": attempt.status,
                            "invalid_reason": attempt.invalid_reason,
                        }
                        for attempt in generation.attempts
                    ],
                    "finalist_fingerprints": tuple(
                        champion_fingerprint(policy)
                        for policy in generation.finalists
                    ),
                    "feedback_fingerprint": champion_fingerprint(
                        generation.feedback.to_payload()
                    ),
                }
                for generation in result.generations
            ],
            "shortlist": [policy.to_payload() for policy in result.shortlist],
            "combined_feedback": combined_feedback_payload,
            "combined_feedback_fingerprint": champion_fingerprint(
                combined_feedback_payload
            ),
        }
        fingerprint = champion_fingerprint(payload)
        path = self._path("build_evidence.json")
        if not path.exists():
            self._atomic_create(path, payload)
        elif canonical_json_bytes(self._read_payload(path)) != canonical_json_bytes(
            payload
        ):
            _lifecycle_fail("immutable Build evidence drifted")
        return fingerprint

    def require_build_evidence(
        self,
        checkpoint: ChampionCheckpoint,
        *,
        manifest: ChampionRunManifest,
        parent: ChampionRelease,
        config: ChampionEvolutionConfig,
    ) -> None:
        payload = self._read_payload(self._path("build_evidence.json"))
        if set(payload) != {
            "schema_version",
            "input_fingerprint",
            "parent_fingerprint",
            "config_fingerprint",
            "build_rows_fingerprint",
            "generations",
            "shortlist",
            "combined_feedback",
            "combined_feedback_fingerprint",
        } or payload["schema_version"] != 1:
            _lifecycle_fail("immutable Build evidence has a malformed schema")
        if champion_fingerprint(payload) != checkpoint.build_evidence_fingerprint:
            _lifecycle_fail("checkpoint drifted from immutable Build evidence")
        if (
            payload["input_fingerprint"] != manifest.input_fingerprint
            or payload["parent_fingerprint"] != champion_fingerprint(parent)
            or payload["config_fingerprint"] != config.fingerprint
        ):
            _lifecycle_fail("Build evidence is stale or foreign")
        _require_sha256(payload["build_rows_fingerprint"], "Build rows fingerprint")
        raw_generations = payload["generations"]
        if type(raw_generations) is not list or len(raw_generations) != config.generations:
            _lifecycle_fail("Build evidence generation schedule drifted")
        finalist_ids: set[str] = set()
        for expected_number, raw_generation in enumerate(
            cast(list[object], raw_generations), start=1
        ):
            if type(raw_generation) is not dict or set(
                cast(dict[object, object], raw_generation)
            ) != {
                "number",
                "mutation_parent_sha256",
                "config_fingerprint",
                "stage_counts",
                "full_build_children",
                "attempts",
                "finalist_fingerprints",
                "feedback_fingerprint",
            }:
                _lifecycle_fail("Build attempt evidence has a malformed schema")
            generation = cast(dict[str, object], raw_generation)
            if (
                generation["number"] != expected_number
                or generation["mutation_parent_sha256"]
                != champion_fingerprint(parent)
                or generation["config_fingerprint"] != config.fingerprint
                or generation["stage_counts"] != list(config.screen_sizes)
            ):
                _lifecycle_fail("Build attempt evidence drifted from its schedule")
            raw_attempts = generation["attempts"]
            if type(raw_attempts) is not list:
                _lifecycle_fail("Build attempt evidence must be an exact JSON array")
            generation_finalists: list[str] = []
            for raw_attempt in cast(list[object], raw_attempts):
                if type(raw_attempt) is not dict or set(
                    cast(dict[object, object], raw_attempt)
                ) != {
                    "fitted_id",
                    "policy",
                    "stage_task_counts",
                    "comparison",
                    "status",
                    "invalid_reason",
                }:
                    _lifecycle_fail("Build attempt evidence has a malformed attempt")
                attempt = cast(dict[str, object], raw_attempt)
                try:
                    policy = _parse_fitted_policy(attempt["policy"])
                except Exception as error:
                    raise ChampionLifecycleError(
                        "Build attempt evidence contains a malformed policy"
                    ) from error
                fitted_id = champion_fingerprint(policy)
                if attempt["fitted_id"] != fitted_id:
                    _lifecycle_fail("Build attempt fitted fingerprint drifted")
                if attempt["status"] == "finalist":
                    comparison = attempt["comparison"]
                    if type(comparison) is not dict:
                        _lifecycle_fail("Build finalist lacks exact score evidence")
                    joint = cast(dict[str, object], comparison).get(
                        "joint_improvement"
                    )
                    if (
                        type(joint) is not float
                        or not math.isfinite(joint)
                        or joint < config.candidate_minimum_gain
                    ):
                        _lifecycle_fail("Build finalist violates the bound threshold")
                    generation_finalists.append(fitted_id)
                    finalist_ids.add(fitted_id)
            if generation["finalist_fingerprints"] != generation_finalists:
                _lifecycle_fail("Build finalist evidence drifted from its attempts")
            _require_sha256(
                generation["feedback_fingerprint"], "Build feedback fingerprint"
            )
        raw_shortlist = payload["shortlist"]
        if type(raw_shortlist) is not list:
            _lifecycle_fail("Build evidence shortlist is malformed")
        try:
            evidence_shortlist = tuple(
                _parse_fitted_policy(item)
                for item in cast(list[object], raw_shortlist)
            )
        except Exception as error:
            raise ChampionLifecycleError("Build evidence shortlist is malformed") from error
        evidence_ids = tuple(champion_fingerprint(item) for item in evidence_shortlist)
        if any(item not in finalist_ids for item in evidence_ids):
            _lifecycle_fail("Build evidence shortlist was not derived from finalists")
        if canonical_json_bytes([item.to_payload() for item in evidence_shortlist]) != (
            canonical_json_bytes(
                [item.to_payload() for item in checkpoint.proposal_archive]
            )
        ):
            _lifecycle_fail("checkpoint shortlist drifted from immutable Build evidence")
        raw_feedback = payload["combined_feedback"]
        if type(raw_feedback) is not dict:
            _lifecycle_fail("immutable Build combined feedback is malformed")
        evidence_feedback = _parse_sanitized_feedback(
            cast(dict[str, object], raw_feedback)
        )
        feedback_fingerprint = _require_sha256(
            payload["combined_feedback_fingerprint"],
            "Build combined feedback fingerprint",
        )
        if feedback_fingerprint != champion_fingerprint(
            evidence_feedback.to_payload()
        ):
            _lifecycle_fail("immutable Build combined feedback drifted")
        if canonical_json_bytes(evidence_feedback.to_payload()) != canonical_json_bytes(
            checkpoint.sanitized_feedback.to_payload()
        ):
            _lifecycle_fail("checkpoint Build feedback drifted from immutable evidence")

    def load_checkpoint(self) -> ChampionCheckpoint | None:
        path = self._path("checkpoint.json")
        if not path.exists():
            return None
        payload = self._read_payload(path)
        if set(payload) != {
            "schema_version",
            "input_fingerprint",
            "completed_stage",
            "active_parent",
            "proposal_archive",
            "sanitized_feedback",
            "build_evidence_fingerprint",
        }:
            _lifecycle_fail("checkpoint JSON has a malformed schema")
        try:
            archive_payload = payload["proposal_archive"]
            if type(archive_payload) is not list:
                _lifecycle_fail("checkpoint proposal archive must be a JSON array")
            feedback = payload["sanitized_feedback"]
            if type(feedback) is not dict:
                _lifecycle_fail("checkpoint feedback must be a JSON object")
            return ChampionCheckpoint(
                schema_version=payload["schema_version"],  # type: ignore[arg-type]
                input_fingerprint=payload["input_fingerprint"],  # type: ignore[arg-type]
                completed_stage=payload["completed_stage"],  # type: ignore[arg-type]
                active_parent=parse_champion_release(payload["active_parent"]),
                proposal_archive=tuple(
                    _parse_fitted_policy(item)
                    for item in cast(list[object], archive_payload)
                ),
                sanitized_feedback=_parse_sanitized_feedback(
                    cast(dict[str, object], feedback)
                ),
                build_evidence_fingerprint=payload[
                    "build_evidence_fingerprint"
                ],  # type: ignore[arg-type]
            )
        except ChampionLifecycleError:
            raise
        except Exception as error:
            raise ChampionLifecycleError("checkpoint JSON is malformed") from error

    def report_exists(self, split: Literal["calibration", "dev"]) -> bool:
        if type(split) is not str or split not in {"calibration", "dev"}:
            _lifecycle_fail("artifact store has no authority for this split")
        return self._path(f"{split}_report.json").exists()

    def write_report(self, report: ChampionEvaluationReport) -> Path:
        if type(report) is not ChampionEvaluationReport:
            _lifecycle_fail("report must be an exact ChampionEvaluationReport")
        ChampionEvaluationReport.__post_init__(report)
        path = self._path(f"{report.split}_report.json")
        self._atomic_create(path, report.to_payload())
        return path

    def bind_report(self, report: ChampionEvaluationReport) -> Path:
        """Create a report once, or verify the exact committed bytes on resume."""
        if type(report) is not ChampionEvaluationReport:
            _lifecycle_fail("report must be an exact ChampionEvaluationReport")
        ChampionEvaluationReport.__post_init__(report)
        path = self._path(f"{report.split}_report.json")
        if not path.exists():
            self._atomic_create(path, report.to_payload())
        elif canonical_json_bytes(self._read_payload(path)) != canonical_json_bytes(
            report.to_payload()
        ):
            _lifecycle_fail("immutable lifecycle report drifted or was replayed")
        return path

    def ensure_release(self, release: ChampionRelease) -> Path:
        expected = canonical_release_bytes(release)
        self._verify_root()
        try:
            has_current = self._name_exists("champion_release.json")
        except ChampionLifecycleError as error:
            self._restore_release_from_archive(release)
            raise ChampionLifecycleError(
                "stored Champion release alias drifted from the exact Parent"
            ) from error
        path = self.root / "champion_release.json"
        if not has_current:
            return self.publish_release(release, _accepted=False)
        try:
            payload = self._read_payload(path)
            stored = parse_champion_release(payload)
        except Exception as error:
            self._restore_release_from_archive(release)
            raise ChampionLifecycleError(
                "stored Champion release drifted from the exact Parent"
            ) from error
        if canonical_release_bytes(stored) != expected:
            self._restore_release_from_archive(release)
            _lifecycle_fail("active release drifted from the exact Parent")
        return path

    def _restore_release_from_archive(self, release: ChampionRelease) -> None:
        """Transactionally restore the exact verified Parent before failing closed."""
        expected = canonical_release_bytes(release)
        release_id = champion_fingerprint(release)
        archive_path = self.root / "releases" / f"{release_id}.json"
        try:
            archive_payload = self._read_payload(archive_path)
            archived = parse_champion_release(archive_payload)
        except Exception as error:
            raise ChampionLifecycleError(
                "immutable Parent archive is unavailable for restoration"
            ) from error
        if canonical_release_bytes(archived) != expected:
            _lifecycle_fail("immutable Parent archive drifted before restoration")
        self._force_replace_name_bytes("champion_release.json", expected)

    def publish_release(
        self, release: ChampionRelease, *, _accepted: bool = True
    ) -> Path:
        self._verify_root()
        self._recover_publication()
        content = canonical_release_bytes(release)
        release_id = champion_fingerprint(release)
        archive_path = self._release_archive.root / f"{release_id}.json"
        if self._release_archive._name_exists(archive_path.name):
            if canonical_json_bytes(
                self._release_archive._read_name_payload(archive_path.name)
            ) != canonical_json_bytes(release.to_payload()):
                _lifecycle_fail("immutable release archive drifted")
        else:
            self._atomic_create(archive_path, release.to_payload())
        current = self._path("champion_release.json")
        prior_release_id: str | None = None
        if self._name_exists(current.name):
            try:
                prior = parse_champion_release(self._read_name_payload(current.name))
            except Exception as error:
                raise ChampionLifecycleError(
                    "current release is malformed before publication"
                ) from error
            prior_id = champion_fingerprint(prior)
            prior_release_id = prior_id
            if self._archive_release_bytes(prior_id) != canonical_release_bytes(
                prior
            ):
                _lifecycle_fail("prior release archive drifted before publication")
        transaction_id = champion_fingerprint(
            {
                "prior_release_id": prior_release_id,
                "next_release_id": release_id,
                "accepted": _accepted,
            }
        )
        record: dict[str, object] = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "prior_release_id": prior_release_id,
            "next_release_id": release_id,
            "accepted": _accepted,
        }
        committed = False
        try:
            self._publication_hook("before_prepare")
            self._atomic_create_name("release_prepare.json", record)
            self._publication_hook("after_prepare")
            self._atomic_replace(current, content)
            self._publication_hook("after_current_replace")
            self._atomic_create_name("release_commit.json", record)
            self._publication_hook("after_commit")
            if _accepted:
                self._bind_accepted_release(transaction_id, release_id)
            committed = True
        except Exception:
            if not committed:
                self._unlink_root_name("release_commit.json")
                if self._name_exists("last_accepted_release.json"):
                    accepted_payload = self._read_name_payload(
                        "last_accepted_release.json"
                    )
                    if accepted_payload.get("transaction_id") == transaction_id:
                        self._unlink_root_name("last_accepted_release.json")
                self._restore_transaction_prior(prior_release_id)
                self._unlink_root_name("release_prepare.json")
                self._sync_root()
            raise
        self._unlink_root_name("release_commit.json")
        self._unlink_root_name("release_prepare.json")
        self._sync_root()
        return current


@dataclass(frozen=True)
class ChampionEvolutionConfig:
    """Pre-registered Build schedule and candidate gates."""

    build_size: int = 64
    calibration_size: int = 16
    screen_sizes: tuple[int, ...] = (8, 32, 64)
    generations: int = 1
    minimum_recipes: int = 5
    maximum_recipes: int = 10
    maximum_full_build_children: int = 3
    maximum_finalists: int = 3
    candidate_minimum_gain: float = 0.005
    research_target_gain: float = 0.05
    gate_config: ChampionGateConfig = field(
        default_factory=lambda: ChampionGateConfig(minimum_improved_folds=4)
    )
    build_task_ids: tuple[str, ...] = ()
    screen_task_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if type(self.build_size) is not int or type(self.calibration_size) is not int:
            _fail("Build and Calibration sizes must be exact integers")
        if type(self.screen_sizes) is not tuple or any(
            type(size) is not int for size in self.screen_sizes
        ):
            _fail("screen sizes must be an exact integer tuple")
        sizing = (self.build_size, self.calibration_size, self.screen_sizes)
        if sizing not in {_FORMAL_SIZES, _SMOKE_SIZES}:
            _fail("only the formal 64/16 or deterministic 8/2 smoke schedule is allowed")
        for name in (
            "generations",
            "minimum_recipes",
            "maximum_recipes",
            "maximum_full_build_children",
            "maximum_finalists",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"{name} must be an exact positive integer")
        if not 5 <= self.minimum_recipes <= self.maximum_recipes <= 10:
            _fail("each generation must propose five through ten recipes")
        if self.maximum_full_build_children > 3 or self.maximum_finalists > 3:
            _fail("full Build and finalist bounds cannot exceed three")
        for name in ("candidate_minimum_gain", "research_target_gain"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                _fail(f"{name} must be a finite nonnegative float")
        if self.research_target_gain < self.candidate_minimum_gain:
            _fail("research target cannot be below the candidate threshold")
        if type(self.gate_config) is not ChampionGateConfig:
            _fail("gate_config must be an exact ChampionGateConfig")
        ChampionGateConfig.__post_init__(self.gate_config)
        if type(self.build_task_ids) is not tuple or any(
            type(task_id) is not str or not task_id for task_id in self.build_task_ids
        ):
            _fail("Build task membership must be an exact string tuple")
        if self.build_task_ids and (
            len(self.build_task_ids) != self.build_size
            or len(self.build_task_ids) != len(set(self.build_task_ids))
        ):
            _fail("Build task membership must contain the fixed unique universe")
        if type(self.screen_task_ids) is not tuple or any(
            type(stage) is not tuple for stage in self.screen_task_ids
        ):
            _fail("screen membership must be an exact tuple of exact tuples")
        if self.screen_task_ids:
            if len(self.screen_task_ids) != len(self.screen_sizes):
                _fail("screen membership must cover every fixed stage")
            for size, stage in zip(self.screen_sizes, self.screen_task_ids, strict=True):
                if (
                    len(stage) != size
                    or any(type(task_id) is not str or not task_id for task_id in stage)
                    or len(stage) != len(set(stage))
                ):
                    _fail("each screen stage must have its exact unique membership")
            for earlier, later in zip(
                self.screen_task_ids, self.screen_task_ids[1:], strict=False
            ):
                if not set(earlier).issubset(later):
                    _fail("screen membership must be successively nested")

    @property
    def fingerprint(self) -> str:
        """Fingerprint schedule, gates, thresholds, and fixed task membership."""
        return champion_fingerprint(self)

    @property
    def screen_membership_sha256(self) -> str:
        return champion_fingerprint({"screen_task_ids": self.screen_task_ids})


@dataclass(frozen=True)
class BuildAttempt:
    """The latest trusted Build evidence for one expanded fitted Child."""

    fitted_id: str
    policy: FittedChampionPolicy
    stage_task_counts: tuple[int, ...]
    comparison: ChampionComparison | None
    status: Literal[
        "invalid", "pruned", "finalist", "below_candidate_threshold"
    ]
    invalid_reason: Literal["unscorable_child"] | None = None
    score_label: Literal["adaptive_train_build_diagnostic"] = (
        "adaptive_train_build_diagnostic"
    )
    independent_generalization_claim: Literal[False] = False


@dataclass(frozen=True)
class BuildGeneration:
    """One structural proposal generation evaluated against the same Parent."""

    number: int
    mutation_parent_sha256: str
    config_fingerprint: str
    stage_counts: tuple[int, ...]
    full_build_children: int
    attempts: tuple[BuildAttempt, ...]
    finalists: tuple[FittedChampionPolicy, ...]
    feedback: ProposerEvidence


@dataclass(frozen=True)
class BuildEvolutionResult:
    """Build diagnostics and shortlist; active_parent is never a Build Child."""

    active_parent: object
    generations: tuple[BuildGeneration, ...]
    shortlist: tuple[FittedChampionPolicy, ...]
    config: ChampionEvolutionConfig
    config_fingerprint: str
    score_label: Literal["adaptive_train_build_diagnostic"] = (
        "adaptive_train_build_diagnostic"
    )
    independent_generalization_claim: Literal[False] = False


@dataclass
class _AttemptState:
    fitted_id: str
    score_name: str
    policy: FittedChampionPolicy
    stage_task_counts: tuple[int, ...] = ()
    comparison: ChampionComparison | None = None
    evidence_rows: tuple[ChampionTaskRow, ...] = ()
    invalid_reason: Literal["unscorable_child"] | None = None


def _capture_object_graph(
    value: object,
) -> tuple[tuple[object, tuple[tuple[str, object], ...]], ...]:
    """Capture exact dataclass fields so hostile callbacks can be rolled back."""
    captured: list[tuple[object, tuple[tuple[str, object], ...]]] = []
    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        if is_dataclass(item) and not isinstance(item, type):
            seen.add(identity)
            values = tuple((entry.name, getattr(item, entry.name)) for entry in fields(item))
            captured.append((item, values))
            for _, nested in values:
                visit(nested)
            return
        if type(item) in {tuple, list}:
            seen.add(identity)
            for nested in cast(tuple[object, ...] | list[object], item):
                visit(nested)
        elif type(item) is dict:
            seen.add(identity)
            for key, nested in cast(dict[object, object], item).items():
                visit(key)
                visit(nested)

    visit(value)
    return tuple(captured)


def _restore_object_graph(
    captured: tuple[tuple[object, tuple[tuple[str, object], ...]], ...],
) -> None:
    for item, values in captured:
        field_names = {name for name, _ in values}
        try:
            extra_names = tuple(set(vars(item)) - field_names)
        except TypeError:
            extra_names = ()
        for name in extra_names:
            object.__delattr__(item, name)
        for name, value in values:
            object.__setattr__(item, name, value)


def _fingerprint_changed(value: object, expected: str) -> bool:
    try:
        return champion_fingerprint(value) != expected
    except Exception:
        return True


def _validated_parent(
    parent: object,
) -> tuple[ChampionRecipe, FittedChampionPolicy | None]:
    def validate_recipe(recipe: object) -> ChampionRecipe:
        if type(recipe) is not ChampionRecipe:
            _fail("active Parent contains an invalid recipe")
        canonical = cast(ChampionRecipe, recipe)
        for assumption in canonical.assumptions:
            if type(assumption) is not EvolutionAssumption:
                _fail("active Parent contains an invalid assumption")
            EvolutionAssumption.__post_init__(assumption)
        ChampionRecipe.__post_init__(canonical)
        return canonical

    try:
        if type(parent) is ChampionRelease:
            ChampionRelease.__post_init__(parent)
            normalized_lineage = tuple(
                _canonical_identity(item) for item in parent.lineage
            )
            if len(normalized_lineage) != len(set(normalized_lineage)):
                _lifecycle_fail(
                    "active Parent lineage contains duplicate normalized identities"
                )
            FittedChampionPolicy.__post_init__(parent.policy)
            return validate_recipe(parent.policy.recipe), parent.policy
        if type(parent) is FittedChampionPolicy:
            FittedChampionPolicy.__post_init__(parent)
            return validate_recipe(parent.recipe), parent
        if type(parent) is ChampionRecipe:
            return validate_recipe(parent), None
    except ChampionControllerError:
        raise
    except Exception as error:
        raise ChampionControllerError("active Parent violates its frozen contract") from error
    _fail("active Parent must be an exact ChampionRelease, policy, or recipe")
    raise AssertionError("unreachable")


def _validated_rows(
    rows: object,
    *,
    build_size: int,
) -> tuple[tuple[ChampionTaskRow, ...], tuple[str, ...], tuple[str, ...]]:
    if type(rows) not in {tuple, list} or not rows:
        _fail("Build rows must be a nonempty exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], rows))
    seen_keys: set[tuple[str, str]] = set()
    task_ids: list[str] = []
    identities: dict[str, tuple[object, object, int, str, object]] = {}
    candidate_tasks: dict[str, set[str]] = {}
    normalized_candidates: dict[str, str] = {}
    for raw_row in snapshot:
        if type(raw_row) is not ChampionTaskRow:
            _fail("Build rows must contain exact ChampionTaskRow values")
        row = cast(ChampionTaskRow, raw_row)
        try:
            ChampionTaskRow.__post_init__(row)
        except Exception as error:
            raise ChampionControllerError("Build row violates its trusted contract") from error
        if row.split != "build":
            _fail("Build evolution accepts Build rows only")
        normalized_name = _canonical_identity(row.candidate_name)
        prior_name = normalized_candidates.setdefault(
            normalized_name, row.candidate_name
        )
        if prior_name != row.candidate_name:
            _fail("normalized row inventory contains a candidate collision")
        key = (row.candidate_name, row.task_id)
        if key in seen_keys:
            _fail("Build rows contain a duplicate candidate/task key")
        seen_keys.add(key)
        if row.task_id not in identities:
            task_ids.append(row.task_id)
        identity = (row.profile, row.truth, row.fold, row.split, row.history)
        previous = identities.setdefault(row.task_id, identity)
        if identity != previous:
            _fail(
                "Build row universe drifted across profile, truth, fold, split, "
                "or history"
            )
        candidate_tasks.setdefault(row.candidate_name, set()).add(row.task_id)
    if len(task_ids) != build_size:
        _fail("Build rows do not contain the configured task universe")
    universe = set(task_ids)
    if any(task_set != universe for task_set in candidate_tasks.values()):
        _fail("every materialized candidate must cover the fixed Build universe")
    return (
        cast(tuple[ChampionTaskRow, ...], snapshot),
        tuple(task_ids),
        tuple(candidate_tasks),
    )


def _bound_config(
    config: ChampionEvolutionConfig,
    task_ids: tuple[str, ...],
) -> ChampionEvolutionConfig:
    if config.build_task_ids and config.build_task_ids != task_ids:
        _fail("Build task membership drifted from the pre-registered universe")
    screens = config.screen_task_ids or tuple(
        task_ids[:size] for size in config.screen_sizes
    )
    bound = _clone_config(
        config,
        build_task_ids=task_ids,
        screen_task_ids=screens,
    )
    if set(bound.screen_task_ids[-1]) != set(task_ids):
        _fail("the final screen must equal the complete fixed Build universe")
    if any(not set(stage).issubset(task_ids) for stage in bound.screen_task_ids):
        _fail("screen membership contains a task outside fixed Build")
    return bound


def _clone_gate(config: ChampionGateConfig) -> ChampionGateConfig:
    return ChampionGateConfig(
        **{entry.name: getattr(config, entry.name) for entry in fields(config)}
    )


def _clone_config(
    config: ChampionEvolutionConfig,
    *,
    build_task_ids: tuple[str, ...] | None = None,
    screen_task_ids: tuple[tuple[str, ...], ...] | None = None,
) -> ChampionEvolutionConfig:
    """Detach every nested registered value from caller-owned aliases."""
    return ChampionEvolutionConfig(
        build_size=config.build_size,
        calibration_size=config.calibration_size,
        screen_sizes=tuple(config.screen_sizes),
        generations=config.generations,
        minimum_recipes=config.minimum_recipes,
        maximum_recipes=config.maximum_recipes,
        maximum_full_build_children=config.maximum_full_build_children,
        maximum_finalists=config.maximum_finalists,
        candidate_minimum_gain=config.candidate_minimum_gain,
        research_target_gain=config.research_target_gain,
        gate_config=_clone_gate(config.gate_config),
        build_task_ids=(
            tuple(config.build_task_ids)
            if build_task_ids is None
            else tuple(build_task_ids)
        ),
        screen_task_ids=(
            tuple(tuple(stage) for stage in config.screen_task_ids)
            if screen_task_ids is None
            else tuple(tuple(stage) for stage in screen_task_ids)
        ),
    )


def _require_config_fingerprint(
    config: ChampionEvolutionConfig,
    expected: str,
) -> None:
    if _fingerprint_changed(config, expected):
        _fail("registered Champion evolution config mutated")


def _feature_value(policy: FittedChampionPolicy, task_row: ChampionTaskRow, index: int) -> float:
    assumption = policy.recipe.assumptions[index]
    feature = assumption.feature
    profile = task_row.profile
    try:
        if feature == "horizon_ratio":
            value: object = profile.horizon / profile.history_length
        elif feature in _PROFILE_FEATURES:
            value = getattr(profile, feature)
        else:
            _fail("fitted Child references an unsupported Build feature")
    except (ArithmeticError, AttributeError, TypeError, ValueError) as error:
        raise ChampionControllerError("fitted Child feature is invalid") from error
    if type(value) not in {int, float}:
        _fail("fitted Child feature must be exactly numeric")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail("fitted Child feature must be finite")
    return number


def _policy_forecast(
    policy: FittedChampionPolicy,
    task_rows: dict[str, ChampionTaskRow],
    *,
    executor: Callable[..., object] | None = None,
) -> tuple[tuple[float, ...] | None, str | None]:
    active_executor = execute_champion if executor is None else executor
    fallback = task_rows.get(policy.recipe.fallback_parent)
    if fallback is None or fallback.forecast is None:
        return None, "fallback_unavailable"
    parent_rows = tuple(task_rows.get(name) for name in policy.recipe.parents)
    enriched = fallback.history is not None or fallback.diagnostic is not None
    if enriched or any(
        row is not None and (row.history is not None or row.diagnostic is not None)
        for row in parent_rows
    ):
        if fallback.history is None:
            return None, "history_unavailable"
        forecasts: dict[str, tuple[float, ...]] = {}
        diagnostics: dict[str, CandidateDiagnostics] = {}
        for name, row in zip(policy.recipe.parents, parent_rows, strict=True):
            if row is None or row.forecast is None:
                return None, "parent_forecast_unavailable"
            if row.history != fallback.history:
                return None, "history_mismatch"
            if row.diagnostic is None or not _valid_history_diagnostic(row.diagnostic):
                return None, "history_diagnostic_unavailable"
            forecasts[name] = row.forecast
            diagnostics[name] = _runtime_diagnostic(row.diagnostic)
        execution = cast(
            ChampionExecution,
            active_executor(
                policy,
                forecasts,
                diagnostics,
                fallback.profile,
                fallback.history,
                fallback.profile.horizon,
            ),
        )
        if execution.fallback_reason not in {None, "assumption_not_satisfied"}:
            return None, f"runtime_{execution.fallback_reason}"
        return execution.forecast, None

    parents: dict[str, tuple[float, ...]] = {}
    for name in policy.recipe.parents:
        row = task_rows.get(name)
        if row is None or row.forecast is None:
            return fallback.forecast, None
        parents[name] = row.forecast
    for index, (assumption, (_, threshold)) in enumerate(
        zip(policy.recipe.assumptions, policy.thresholds, strict=True)
    ):
        value = _feature_value(policy, fallback, index)
        matches = value >= threshold if assumption.direction == "above" else value < threshold
        if not matches:
            return fallback.forecast, None

    kind = policy.recipe.kind
    if kind == "select":
        return parents[policy.recipe.parents[0]], None
    if kind == "route":
        routed = policy.recipe.assumptions[0].candidate_name
        if any(item.candidate_name != routed for item in policy.recipe.assumptions):
            return None, "invalid_route"
        return parents[routed], None
    left_name, right_name = policy.recipe.parents
    left, right = parents[left_name], parents[right_name]
    if kind == "horizon_route":
        split = int(len(left) * policy.horizon_split)
        return left[:split] + right[split:], None
    if kind == "weighted":
        return tuple(
            left[index] * policy.weights[0] + right[index] * policy.weights[1]
            for index in range(len(left))
        ), None
    if kind == "median":
        return tuple(
            left[index] / 2.0 + right[index] / 2.0 for index in range(len(left))
        ), None
    if kind == "bounded_overlay":
        return None, "history_scale_unavailable"
    return None, "unsupported_operator"


def _valid_history_diagnostic(diagnostic: object) -> bool:
    if type(diagnostic) is not ChampionHistoryDiagnostic:
        return False
    try:
        ChampionHistoryDiagnostic.__post_init__(diagnostic)
    except (ChampionEvidenceError, TypeError, ValueError):
        return False
    return True


def _runtime_diagnostic(
    diagnostic: ChampionHistoryDiagnostic,
) -> CandidateDiagnostics:
    """Construct Task 3 input with no fold objects, forecasts, truths, or cache."""
    return CandidateDiagnostics(
        name=diagnostic.name,
        family=diagnostic.family,
        folds=(),
        successful_folds=diagnostic.successful_folds,
        eligible=diagnostic.eligible,
        reason_code=diagnostic.reason_code,
        median_mase=diagnostic.median_mase,
        recent_mase=diagnostic.recent_mase,
        worst_mase=diagnostic.worst_mase,
        mase_mad=diagnostic.mase_mad,
        median_mae=diagnostic.median_mae,
        median_smape=diagnostic.median_smape,
        median_rmsse=diagnostic.median_rmsse,
        normalized_bias=diagnostic.normalized_bias,
        slope_error=diagnostic.slope_error,
        phase_error=diagnostic.phase_error,
        amplitude_ratio=diagnostic.amplitude_ratio,
        explosion=diagnostic.explosion,
        fold_forecasts=(),
        fold_truths=(),
        cache_key="",
        long_horizon_fold=None,
        long_horizon_coverage=diagnostic.long_horizon_coverage,
        median_joint_scaled_error=diagnostic.median_joint_scaled_error,
        recent_joint_scaled_error=diagnostic.recent_joint_scaled_error,
        worst_joint_scaled_error=diagnostic.worst_joint_scaled_error,
        median_smae=diagnostic.median_smae,
        recent_smae=diagnostic.recent_smae,
        worst_smae=diagnostic.worst_smae,
        smae_mad=diagnostic.smae_mad,
        median_srmse=diagnostic.median_srmse,
        recent_srmse=diagnostic.recent_srmse,
        worst_srmse=diagnostic.worst_srmse,
        srmse_mad=diagnostic.srmse_mad,
        worst_smae_raw=diagnostic.worst_smae_raw,
        worst_srmse_raw=diagnostic.worst_srmse_raw,
    )


def _materialize_policy(
    policy: FittedChampionPolicy,
    score_name: str,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    *,
    split: Literal["build", "calibration", "dev"] = "build",
    executor: Callable[..., object] | None = None,
) -> tuple[ChampionTaskRow, ...]:
    by_task: dict[str, dict[str, ChampionTaskRow]] = {task_id: {} for task_id in task_ids}
    identity: dict[str, ChampionTaskRow] = {}
    selected = set(task_ids)
    for row in rows:
        if row.task_id not in selected:
            continue
        by_task[row.task_id][row.candidate_name] = row
        identity.setdefault(row.task_id, row)
    materialized: list[ChampionTaskRow] = []
    for task_id in task_ids:
        base = identity[task_id]
        try:
            forecast, failure = _policy_forecast(
                policy, by_task[task_id], executor=executor
            )
        except (ArithmeticError, ChampionControllerError, OverflowError, ValueError):
            forecast, failure = None, "invalid_policy_materialization"
        if forecast is not None and any(not math.isfinite(value) for value in forecast):
            forecast, failure = None, "invalid_policy_materialization"
        materialized.append(
            ChampionTaskRow(
                task_id=task_id,
                candidate_name=score_name,
                profile=base.profile,
                truth=base.truth,
                forecast=forecast,
                failure_reason=failure if forecast is None else None,
                fold=base.fold,
                split=split,
            )
        )
    return tuple(materialized)


def _parent_rows(
    parent_recipe: ChampionRecipe,
    parent_policy: FittedChampionPolicy | None,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    *,
    executor: Callable[..., object] | None = None,
) -> tuple[str, tuple[ChampionTaskRow, ...]]:
    supplied = tuple(
        row
        for row in rows
        if row.task_id in set(task_ids) and row.candidate_name == parent_recipe.name
    )
    if supplied:
        if len(supplied) != len(task_ids):
            _fail("supplied Parent rows do not cover fixed screen membership")
        return parent_recipe.name, supplied
    if parent_policy is None:
        _fail("an unfitted Parent recipe requires trusted materialized Parent rows")
    split = rows[0].split
    if split not in {"build", "calibration", "dev"}:
        _fail("Parent materialization received an unauthorized split")
    return parent_recipe.name, _materialize_policy(
        parent_policy,
        parent_recipe.name,
        rows,
        task_ids,
        split=cast(Literal["build", "calibration", "dev"], split),
        executor=executor,
    )


def _stage_gate(config: ChampionEvolutionConfig, *, final: bool) -> ChampionGateConfig:
    del final
    return config.gate_config


def _evaluate_stage(
    state: _AttemptState,
    *,
    parent_recipe: ChampionRecipe,
    parent_policy: FittedChampionPolicy | None,
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    gate: ChampionGateConfig,
    executor: Callable[..., object] | None = None,
    scorer: Callable[..., object] | None = None,
    comparator: Callable[..., object] | None = None,
) -> _AttemptState:
    active_scorer = score_policy if scorer is None else scorer
    active_comparator = compare_champion if comparator is None else comparator
    parent_name, parent_rows = _parent_rows(
        parent_recipe, parent_policy, rows, task_ids, executor=executor
    )
    child_rows = _materialize_policy(
        state.policy,
        state.score_name,
        rows,
        task_ids,
        executor=executor,
    )
    evidence_rows = parent_rows + child_rows
    try:
        parent_score = cast(ChampionScore, active_scorer(evidence_rows, parent_name))
    except (ChampionEvidenceError, TypeError, ValueError) as error:
        raise ChampionControllerError("trusted Build scoring rejected the Parent") from error
    try:
        child_score = cast(
            ChampionScore, active_scorer(evidence_rows, state.score_name)
        )
        comparison = cast(
            ChampionComparison, active_comparator(parent_score, child_score, gate)
        )
    except (ChampionEvidenceError, TypeError, ValueError):
        return replace(
            state,
            stage_task_counts=state.stage_task_counts + (len(task_ids),),
            comparison=None,
            evidence_rows=evidence_rows,
            invalid_reason="unscorable_child",
        )
    return replace(
        state,
        stage_task_counts=state.stage_task_counts + (len(task_ids),),
        comparison=comparison,
        evidence_rows=evidence_rows,
        invalid_reason=None,
    )


def _rank(state: _AttemptState) -> tuple[float, float, float, str]:
    if state.comparison is None:
        _fail("attempt has no trusted comparison")
    comparison = cast(ChampionComparison, state.comparison)
    return (
        -comparison.joint_improvement,
        comparison.mean_delta_smae,
        comparison.mean_delta_srmse,
        state.fitted_id,
    )


def _diverse(
    states: tuple[_AttemptState, ...],
    *,
    limit: int,
) -> tuple[_AttemptState, ...]:
    ranked = tuple(sorted(states, key=_rank))
    selected: list[_AttemptState] = []
    seen_kinds: set[str] = set()
    seen_structures: set[tuple[str, tuple[str, ...]]] = set()
    for state in ranked:
        kind = state.policy.recipe.kind
        if kind in seen_kinds:
            continue
        selected.append(state)
        seen_kinds.add(kind)
        seen_structures.add((kind, state.policy.recipe.parents))
        if len(selected) == limit:
            return tuple(selected)
    for state in ranked:
        structure = (state.policy.recipe.kind, state.policy.recipe.parents)
        if state in selected or structure in seen_structures:
            continue
        selected.append(state)
        seen_structures.add(structure)
        if len(selected) == limit:
            break
    return tuple(selected)


def _feedback(states: tuple[_AttemptState, ...]) -> ProposerEvidence:
    scored_states = tuple(
        state
        for state in states
        if state.comparison is not None and state.evidence_rows
    )
    invalid_states = tuple(
        state
        for state in states
        if state.comparison is None and state.invalid_reason is not None
    )
    if len(scored_states) + len(invalid_states) != len(states):
        _fail("every Build attempt requires genuine or typed invalid feedback")
    invalid_attempts: list[_InvalidAttemptAggregate] = []
    for state in invalid_states:
        if (
            champion_fingerprint(state.policy) != state.fitted_id
            or not state.stage_task_counts
        ):
            _fail("invalid Build attempt feedback lost its host binding")
        invalid_attempts.append(
            _InvalidAttemptAggregate(
                structure_sha256=state.fitted_id,
                kind=state.policy.recipe.kind,
                stage_support=state.stage_task_counts[-1],
                reason_code=state.invalid_reason,
            )
        )
    grouped: dict[tuple[str, ...], list[_AttemptState]] = {}
    for state in scored_states:
        comparison = cast(ChampionComparison, state.comparison)
        task_ids = tuple(
            row.task_id
            for row in state.evidence_rows
            if row.candidate_name == comparison.parent_name
        )
        grouped.setdefault(task_ids, []).append(state)
    evidence_parts: list[ProposerEvidence] = []
    for group in grouped.values():
        parent_name = cast(ChampionComparison, group[0].comparison).parent_name
        parent_rows = tuple(
            row for row in group[0].evidence_rows if row.candidate_name == parent_name
        )
        child_rows = tuple(
            row
            for state in group
            for row in state.evidence_rows
            if row.candidate_name == state.score_name
        )
        comparisons = tuple(cast(ChampionComparison, state.comparison) for state in group)
        try:
            evidence_parts.append(
                sanitize_build_evidence(parent_rows + child_rows, comparisons)
            )
        except (ChampionEvidenceError, TypeError, ValueError) as error:
            raise ChampionControllerError("Build feedback could not be sanitized") from error
    return ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=tuple(
            sorted(
                (item for part in evidence_parts for item in part.morphology),
                key=lambda item: (item.group_id, item.candidate_name),
            )
        ),
        comparisons=tuple(
            sorted(
                (item for part in evidence_parts for item in part.comparisons),
                key=lambda item: item.candidate_name,
            )
        ),
        invalid_attempts=tuple(
            sorted(
                invalid_attempts,
                key=lambda item: (item.structure_sha256, item.reason_code),
            )
        ),
    )


def _call_proposer(
    proposer: Callable[[object, ProposerEvidence], object],
    parent: object,
    feedback: ProposerEvidence,
    *,
    minimum: int,
    maximum: int,
) -> tuple[ChampionRecipe, ...]:
    if not callable(proposer):
        _fail("proposer must be callable")
    try:
        proposed = proposer(parent, feedback)
    except Exception as error:
        raise ChampionControllerError("structural proposer callback failed") from error
    if type(proposed) not in {tuple, list}:
        _fail("proposer must return an exact tuple or list")
    recipes = tuple(cast(tuple[object, ...] | list[object], proposed))
    if not minimum <= len(recipes) <= maximum:
        _fail("proposer must return the configured five through ten recipes")
    validated: list[ChampionRecipe] = []
    for raw_recipe in recipes:
        if type(raw_recipe) is not ChampionRecipe:
            _fail("proposer output must contain exact ChampionRecipe values")
        recipe = cast(ChampionRecipe, raw_recipe)
        try:
            ChampionRecipe.__post_init__(recipe)
            detached = parse_champion_recipe(recipe.to_payload())
        except Exception as error:
            raise ChampionControllerError(
                "proposer returned an invalid Champion recipe"
            ) from error
        validated.append(detached)
    recipe_ids = tuple(champion_fingerprint(recipe) for recipe in validated)
    recipe_names = tuple(recipe.name for recipe in validated)
    canonical_names = tuple(_canonical_identity(name) for name in recipe_names)
    if len(recipe_ids) != len(set(recipe_ids)) or len(canonical_names) != len(
        set(canonical_names)
    ):
        _fail("proposer returned duplicate policy identities")
    for recipe in validated:
        canonical_parents = tuple(
            _canonical_identity(name) for name in recipe.parents
        )
        canonical_assumptions = tuple(
            _canonical_identity(item.assumption_id) for item in recipe.assumptions
        )
        if len(canonical_parents) != len(set(canonical_parents)) or len(
            canonical_assumptions
        ) != len(set(canonical_assumptions)):
            _fail("proposer returned duplicate policy identities")
    return tuple(validated)


def _stored_policies(
    generations: list[BuildGeneration],
    archive: list[_AttemptState],
) -> tuple[FittedChampionPolicy, ...]:
    values = [
        attempt.policy
        for generation in generations
        for attempt in generation.attempts
    ]
    values.extend(state.policy for state in archive)
    unique: list[FittedChampionPolicy] = []
    seen: set[int] = set()
    for policy in values:
        if id(policy) in seen:
            continue
        seen.add(id(policy))
        unique.append(policy)
    return tuple(unique)


def _stored_policy_fingerprints(
    generations: list[BuildGeneration],
    archive: list[_AttemptState],
) -> tuple[tuple[FittedChampionPolicy, str], ...]:
    return tuple(
        (policy, champion_fingerprint(policy))
        for policy in _stored_policies(generations, archive)
    )


def _stored_policy_changed(
    fingerprints: tuple[tuple[FittedChampionPolicy, str], ...],
) -> bool:
    return any(
        _fingerprint_changed(policy, expected)
        for policy, expected in fingerprints
    )


def _validate_stored_policy_ids(generations: list[BuildGeneration]) -> None:
    for generation in generations:
        for attempt in generation.attempts:
            if champion_fingerprint(attempt.policy) != attempt.fitted_id:
                _fail("stored fitted policy fingerprint drifted")


def run_build_evolution(
    parent: object,
    rows: tuple[ChampionTaskRow, ...] | list[ChampionTaskRow],
    proposer: Callable[[object, ProposerEvidence], object],
    config: ChampionEvolutionConfig,
    *,
    boundary_validator: Callable[[], None] | None = None,
    runtime_bindings: ChampionRuntimeBindings | None = None,
) -> BuildEvolutionResult:
    """Run Build-only structural evolution without replacing ``parent``."""
    if type(config) is not ChampionEvolutionConfig:
        _fail("config must be an exact ChampionEvolutionConfig")
    ChampionEvolutionConfig.__post_init__(config)
    if boundary_validator is not None and not callable(boundary_validator):
        _fail("Build boundary validator must be callable")
    if runtime_bindings is not None:
        if type(runtime_bindings) is not ChampionRuntimeBindings:
            _fail("Build runtime bindings must be exact")
        runtime_bindings.verify_live_globals()
        expander: Callable[..., object] = runtime_bindings.expander
        executor: Callable[..., object] = runtime_bindings.executor
        scorer: Callable[..., object] = runtime_bindings.scorer
        comparator: Callable[..., object] = runtime_bindings.comparator
    else:
        expander = expand_recipe
        executor = execute_champion
        scorer = score_policy
        comparator = compare_champion

    def require_boundary() -> None:
        if boundary_validator is not None:
            boundary_validator()
        if runtime_bindings is not None:
            runtime_bindings.verify_live_globals()

    require_boundary()
    input_config_sha256 = config.fingerprint
    parent_recipe, parent_policy = _validated_parent(parent)
    snapshot, task_ids, row_candidate_names = _validated_rows(
        rows, build_size=config.build_size
    )
    bound_config = _bound_config(config, task_ids)
    bound_config_sha256 = bound_config.fingerprint
    parent_sha256 = champion_fingerprint(parent)
    rows_sha256 = champion_fingerprint(snapshot)
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=(),
        comparisons=(),
    )
    seen_recipe_ids: set[str] = {champion_fingerprint(parent_recipe)}
    seen_policy_names: set[str] = {_canonical_identity(parent_recipe.name)}
    reserved_row_names = {
        _canonical_identity(candidate_name)
        for candidate_name in row_candidate_names
    }
    seen_fitted_ids: set[str] = set()
    generations: list[BuildGeneration] = []
    archive: list[_AttemptState] = []

    for generation_number in range(1, bound_config.generations + 1):
        _require_config_fingerprint(config, input_config_sha256)
        _require_config_fingerprint(bound_config, bound_config_sha256)
        _validate_stored_policy_ids(generations)
        parent_state = _capture_object_graph(parent)
        row_state = _capture_object_graph(snapshot)
        feedback_state = _capture_object_graph(feedback)
        config_state = _capture_object_graph(config)
        bound_config_state = _capture_object_graph(bound_config)
        stored_fingerprints = _stored_policy_fingerprints(generations, archive)
        stored_state = _capture_object_graph(
            tuple(policy for policy, _ in stored_fingerprints)
        )
        feedback_sha256 = champion_fingerprint(feedback)
        parent_changed = row_changed = feedback_changed = False
        config_changed = bound_config_changed = stored_changed = False
        callback_error: Exception | None = None
        recipes: tuple[ChampionRecipe, ...] | None = None
        try:
            try:
                require_boundary()
                recipes = _call_proposer(
                    proposer,
                    parent,
                    feedback,
                    minimum=bound_config.minimum_recipes,
                    maximum=bound_config.maximum_recipes,
                )
            except Exception as error:
                callback_error = error
            finally:
                parent_changed = _fingerprint_changed(parent, parent_sha256)
                row_changed = _fingerprint_changed(snapshot, rows_sha256)
                feedback_changed = _fingerprint_changed(feedback, feedback_sha256)
                config_changed = _fingerprint_changed(config, input_config_sha256)
                bound_config_changed = _fingerprint_changed(
                    bound_config, bound_config_sha256
                )
                stored_changed = _stored_policy_changed(stored_fingerprints)
        finally:
            _restore_object_graph(parent_state)
            _restore_object_graph(row_state)
            _restore_object_graph(feedback_state)
            _restore_object_graph(config_state)
            _restore_object_graph(bound_config_state)
            _restore_object_graph(stored_state)
        require_boundary()
        if parent_changed:
            _fail("proposer attempted to mutate the active Parent")
        if row_changed:
            _fail("proposer attempted to mutate the fixed Build rows")
        if feedback_changed:
            _fail("proposer attempted to mutate sanitized Build feedback")
        if config_changed or bound_config_changed:
            _fail("proposer attempted to mutate the registered config")
        if stored_changed:
            _fail("proposer attempted to mutate an archived fitted policy")
        _require_config_fingerprint(config, input_config_sha256)
        _require_config_fingerprint(bound_config, bound_config_sha256)
        _validate_stored_policy_ids(generations)
        if callback_error is not None:
            raise callback_error
        if recipes is None:
            _fail("structural proposer returned no recipes")
        validated_recipes = cast(tuple[ChampionRecipe, ...], recipes)

        states: list[_AttemptState] = []
        for recipe in validated_recipes:
            recipe_id = champion_fingerprint(recipe)
            policy_name = _canonical_identity(recipe.name)
            if policy_name in reserved_row_names:
                _fail("proposal policy ID collides with materialized row inventory")
            if recipe_id in seen_recipe_ids or policy_name in seen_policy_names:
                _fail("duplicate policy ID across Build generations")
            seen_recipe_ids.add(recipe_id)
            seen_policy_names.add(policy_name)
            try:
                policies = cast(tuple[FittedChampionPolicy, ...], expander(recipe, snapshot))
            except (ChampionProposalError, TypeError, ValueError) as error:
                raise ChampionControllerError(
                    "host numeric expansion rejected a recipe"
                ) from error
            for policy in policies:
                fitted_id = champion_fingerprint(policy)
                if fitted_id in seen_fitted_ids:
                    _fail("duplicate fitted policy ID across Build generations")
                seen_fitted_ids.add(fitted_id)
                states.append(
                    _AttemptState(
                        fitted_id=fitted_id,
                        score_name=f"build_child_{fitted_id}",
                        policy=policy,
                    )
                )
        if not states:
            _fail("host expansion produced an empty Build stage")

        active = tuple(states)
        for stage_index, stage_task_ids in enumerate(bound_config.screen_task_ids):
            if not active:
                break
            _require_config_fingerprint(config, input_config_sha256)
            _require_config_fingerprint(bound_config, bound_config_sha256)
            _validate_stored_policy_ids(generations)
            require_boundary()
            evaluated = tuple(
                _evaluate_stage(
                    state,
                    parent_recipe=parent_recipe,
                    parent_policy=parent_policy,
                    rows=snapshot,
                    task_ids=stage_task_ids,
                    gate=_stage_gate(
                        bound_config,
                        final=stage_index == len(bound_config.screen_task_ids) - 1,
                    ),
                    executor=executor,
                    scorer=scorer,
                    comparator=comparator,
                )
                for state in active
            )
            _require_config_fingerprint(config, input_config_sha256)
            _require_config_fingerprint(bound_config, bound_config_sha256)
            require_boundary()
            updates = {state.fitted_id: state for state in evaluated}
            states = [updates.get(state.fitted_id, state) for state in states]
            safe = tuple(
                state
                for state in evaluated
                if state.comparison is not None and state.comparison.accepted
            )
            if stage_index == len(bound_config.screen_task_ids) - 1:
                active = safe
            elif stage_index == len(bound_config.screen_task_ids) - 2:
                active = _diverse(
                    safe, limit=bound_config.maximum_full_build_children
                )
            else:
                survivor_count = max(
                    bound_config.maximum_full_build_children,
                    (len(safe) + 1) // 2,
                )
                active = _diverse(safe, limit=survivor_count)

        full_build_children = sum(
            bool(state.stage_task_counts)
            and state.stage_task_counts[-1] == bound_config.build_size
            for state in states
        )
        eligible = tuple(
            state
            for state in active
            if state.comparison is not None
            and state.comparison.joint_improvement
            >= bound_config.candidate_minimum_gain
        )
        finalists = _diverse(eligible, limit=bound_config.maximum_finalists)
        finalist_ids = {state.fitted_id for state in finalists}
        attempts = tuple(
            BuildAttempt(
                fitted_id=state.fitted_id,
                policy=state.policy,
                stage_task_counts=state.stage_task_counts,
                comparison=state.comparison,
                status=(
                    "invalid"
                    if state.invalid_reason is not None
                    else "finalist"
                    if state.fitted_id in finalist_ids
                    else "below_candidate_threshold"
                    if state in active
                    else "pruned"
                ),
                invalid_reason=state.invalid_reason,
            )
            for state in states
        )
        generation_feedback = _feedback(tuple(states))
        feedback = ProposerEvidence(
            label="adaptive_train_build_diagnostic",
            independent_generalization_claim=False,
            morphology=feedback.morphology + generation_feedback.morphology,
            comparisons=feedback.comparisons + generation_feedback.comparisons,
            invalid_attempts=tuple(
                sorted(
                    feedback.invalid_attempts + generation_feedback.invalid_attempts,
                    key=lambda item: (item.structure_sha256, item.reason_code),
                )
            ),
        )
        generations.append(
            BuildGeneration(
                number=generation_number,
                mutation_parent_sha256=parent_sha256,
                config_fingerprint=bound_config.fingerprint,
                stage_counts=bound_config.screen_sizes,
                full_build_children=full_build_children,
                attempts=attempts,
                finalists=tuple(state.policy for state in finalists),
                feedback=generation_feedback,
            )
        )
        archive.extend(finalists)

    shortlist = _diverse(tuple(archive), limit=bound_config.maximum_finalists)
    _require_config_fingerprint(config, input_config_sha256)
    _require_config_fingerprint(bound_config, bound_config_sha256)
    _validate_stored_policy_ids(generations)
    require_boundary()
    if champion_fingerprint(parent) != parent_sha256:
        _fail("Build evolution attempted to mutate the active Parent")
    if champion_fingerprint(snapshot) != rows_sha256:
        _fail("Build evolution attempted to mutate the fixed Build rows")
    return BuildEvolutionResult(
        active_parent=parent,
        generations=tuple(generations),
        shortlist=tuple(state.policy for state in shortlist),
        config=bound_config,
        config_fingerprint=bound_config.fingerprint,
    )


def _validated_lifecycle_tasks(
    tasks: object,
    *,
    label: str,
    expected_membership: tuple[tuple[str, str], ...],
    expected_content_hashes: tuple[tuple[str, str], ...],
) -> tuple[Task, ...]:
    if type(tasks) not in {tuple, list}:
        _lifecycle_fail(f"{label} tasks must be an exact tuple or list")
    snapshot = tuple(cast(tuple[object, ...] | list[object], tasks))
    if len(snapshot) != len(expected_membership):
        _lifecycle_fail(f"{label} task count drifted from the run manifest")
    validated: list[Task] = []
    for raw_task in snapshot:
        if type(raw_task) is not Task:
            _lifecycle_fail(f"{label} tasks must contain exact Task values")
        task = cast(Task, raw_task)
        if type(task.task_id) is not str or not task.task_id:
            _lifecycle_fail(f"{label} contains an invalid task ID")
        if (
            type(task.entity_name) is not str
            or not task.entity_name.strip()
            or _canonical_identity(task.entity_name) == "unknown"
        ):
            _lifecycle_fail(f"{label} contains an unknown entity")
        if type(task.history_values) is not tuple or not task.history_values:
            _lifecycle_fail(f"{label} task history must be a nonempty exact tuple")
        if type(task.future_values) is not tuple or not task.future_values:
            _lifecycle_fail(f"{label} task truth must be a nonempty exact tuple")
        for series_name, values in (
            ("history", task.history_values),
            ("truth", task.future_values),
        ):
            if any(
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                for value in values
            ):
                _lifecycle_fail(f"{label} task {series_name} must be exactly finite")
        if (
            type(task.prediction_length) is not int
            or task.prediction_length <= 0
            or task.prediction_length != len(task.future_values)
        ):
            _lifecycle_fail(f"{label} task horizon is malformed")
        if type(task.frequency) is not str or not task.frequency.strip():
            _lifecycle_fail(f"{label} task frequency is malformed")
        if task.seasonal_period is not None and type(task.seasonal_period) is not str:
            _lifecycle_fail(f"{label} task seasonal period is malformed")
        validated.append(task)
    result = tuple(validated)
    membership = tuple((task.task_id, task.entity_name) for task in result)
    if membership != expected_membership:
        _lifecycle_fail(f"{label} task identity or entity drifted from the manifest")
    _validated_membership(membership, f"{label} membership", expected_size=len(result))
    actual_hashes = tuple(
        (task.task_id, task_content_fingerprint(task)) for task in result
    )
    if actual_hashes != expected_content_hashes:
        _lifecycle_fail(f"{label} task content drifted from the run manifest")
    return result


def _validated_stage_rows(
    rows: object,
    *,
    tasks: tuple[Task, ...],
    split: Literal["build", "calibration", "dev"],
) -> tuple[ChampionTaskRow, ...]:
    if type(rows) not in {tuple, list} or not rows:
        _lifecycle_fail(f"{split} row provider returned no exact rows")
    snapshot = tuple(cast(tuple[object, ...] | list[object], rows))
    by_task = {task.task_id: task for task in tasks}
    if len(by_task) != len(tasks):
        _lifecycle_fail(f"{split} tasks contain duplicate IDs")
    seen_keys: set[tuple[str, str]] = set()
    normalized_candidates: dict[str, str] = {}
    candidate_tasks: dict[str, set[str]] = {}
    for raw_row in snapshot:
        if type(raw_row) is not ChampionTaskRow:
            _lifecycle_fail(f"{split} rows require exact ChampionTaskRow values")
        row = cast(ChampionTaskRow, raw_row)
        try:
            ChampionTaskRow.__post_init__(row)
        except Exception as error:
            raise ChampionLifecycleError(f"{split} row is malformed") from error
        if row.split != split:
            _lifecycle_fail(f"{split} row provider crossed a split boundary")
        task = by_task.get(row.task_id)
        if task is None:
            _lifecycle_fail(f"{split} rows contain a foreign task ID")
        if (
            row.truth != task.future_values
            or row.history != task.history_values
            or row.profile.history_length != len(task.history_values)
            or row.profile.horizon != task.prediction_length
            or row.profile.frequency != task.frequency
        ):
            _lifecycle_fail(f"{split} row identity or label projection drifted")
        key = (row.candidate_name, row.task_id)
        if key in seen_keys:
            _lifecycle_fail(f"{split} rows contain duplicate candidate/task IDs")
        seen_keys.add(key)
        normalized = _canonical_identity(row.candidate_name)
        prior_name = normalized_candidates.setdefault(normalized, row.candidate_name)
        if prior_name != row.candidate_name:
            _lifecycle_fail(f"{split} rows contain duplicate candidate identities")
        candidate_tasks.setdefault(row.candidate_name, set()).add(row.task_id)
    universe = set(by_task)
    if any(candidate_ids != universe for candidate_ids in candidate_tasks.values()):
        _lifecycle_fail(f"every {split} candidate must cover the exact task universe")
    return cast(tuple[ChampionTaskRow, ...], snapshot)


def _call_row_provider(
    provider: Callable[[tuple[Task, ...], str], object],
    tasks: tuple[Task, ...],
    split: Literal["build", "calibration", "dev"],
    *,
    protected: tuple[object, ...],
) -> tuple[ChampionTaskRow, ...]:
    if not callable(provider):
        _lifecycle_fail("row provider must be callable")

    def fingerprint_state() -> str:
        def safe_value(value: object) -> object:
            if type(value) is ChampionCheckpoint:
                return cast(ChampionCheckpoint, value).to_payload()
            if type(value) is ChampionRunManifest:
                return cast(ChampionRunManifest, value).to_payload()
            if type(value) is ProposerEvidence:
                return cast(ProposerEvidence, value).to_payload()
            if type(value) in {tuple, list}:
                return tuple(
                    safe_value(item)
                    for item in cast(tuple[object, ...] | list[object], value)
                )
            if type(value) is dict:
                return {
                    cast(str, key): safe_value(item)
                    for key, item in cast(dict[object, object], value).items()
                }
            return value

        return champion_fingerprint(safe_value((tasks, protected)))

    protected_state = _capture_object_graph((tasks, protected))
    protected_sha256 = fingerprint_state()
    callback_error: Exception | None = None
    supplied: object = None
    changed = False
    try:
        try:
            supplied = provider(tasks, split)
        except Exception as error:
            callback_error = error
        finally:
            try:
                changed = fingerprint_state() != protected_sha256
            except Exception:
                changed = True
    finally:
        _restore_object_graph(protected_state)
    if changed:
        _lifecycle_fail(f"{split} row provider mutated bound lifecycle inputs")
    if callback_error is not None:
        raise ChampionLifecycleError(f"{split} row provider failed") from callback_error
    return _validated_stage_rows(supplied, tasks=tasks, split=split)


@dataclass(frozen=True)
class _StageEvaluationState:
    policy: FittedChampionPolicy
    parent_score: ChampionScore
    child_score: ChampionScore | None
    comparison: ChampionComparison | None
    evidence_fingerprint: str


def _evaluate_lifecycle_stage(
    parent: ChampionRelease,
    policies: tuple[FittedChampionPolicy, ...],
    rows: tuple[ChampionTaskRow, ...],
    task_ids: tuple[str, ...],
    *,
    split: Literal["calibration", "dev"],
    gate: ChampionGateConfig,
    runtime_bindings: ChampionRuntimeBindings,
) -> tuple[_StageEvaluationState, ...]:
    if type(runtime_bindings) is not ChampionRuntimeBindings:
        _lifecycle_fail("lifecycle scoring requires exact runtime bindings")
    runtime_bindings.verify_live_globals()
    parent_recipe, parent_policy = _validated_parent(parent)
    parent_name, materialized_parent = _parent_rows(
        parent_recipe,
        parent_policy,
        rows,
        task_ids,
        executor=runtime_bindings.executor,
    )
    try:
        parent_score = cast(
            ChampionScore,
            runtime_bindings.scorer(materialized_parent, parent_name),
        )
    except (ChampionEvidenceError, TypeError, ValueError) as error:
        raise ChampionLifecycleError(
            f"trusted {split} scoring rejected the exact Parent"
        ) from error
    states: list[_StageEvaluationState] = []
    for policy in policies:
        policy_sha256 = champion_fingerprint(policy)
        child_name = f"{split}_child_{policy_sha256}"
        child_rows = _materialize_policy(
            policy,
            child_name,
            rows,
            task_ids,
            split=split,
            executor=runtime_bindings.executor,
        )
        evidence_fingerprint = champion_fingerprint(
            {
                "input_rows": rows,
                "parent_rows": materialized_parent,
                "child_rows": child_rows,
                "gate": gate,
                "policy": policy,
                "split": split,
            }
        )
        try:
            child_score = cast(
                ChampionScore, runtime_bindings.scorer(child_rows, child_name)
            )
            comparison = cast(
                ChampionComparison,
                runtime_bindings.comparator(parent_score, child_score, gate),
            )
        except (ChampionEvidenceError, TypeError, ValueError):
            child_score = None
            comparison = None
        states.append(
            _StageEvaluationState(
                policy=policy,
                parent_score=parent_score,
                child_score=child_score,
                comparison=comparison,
                evidence_fingerprint=evidence_fingerprint,
            )
        )
    return tuple(states)


def _stage_report(
    manifest: ChampionRunManifest,
    parent: ChampionRelease,
    split: Literal["calibration", "dev"],
    states: tuple[_StageEvaluationState, ...],
) -> ChampionEvaluationReport:
    return ChampionEvaluationReport(
        schema_version=1,
        input_fingerprint=manifest.input_fingerprint,
        split=split,
        parent_sha256=champion_fingerprint(parent),
        candidates=tuple(
            ChampionStageCandidateReport(
                policy_sha256=champion_fingerprint(state.policy),
                evidence_fingerprint=state.evidence_fingerprint,
                comparison=state.comparison,
                invalid_reason=(
                    "unscorable_child" if state.comparison is None else None
                ),
            )
            for state in states
        ),
    )


def _fresh_comparisons(
    states: tuple[_StageEvaluationState, ...],
    gate: ChampionGateConfig,
    comparator: Callable[..., object] | None = None,
) -> tuple[tuple[FittedChampionPolicy, ChampionComparison], ...]:
    fresh: list[tuple[FittedChampionPolicy, ChampionComparison]] = []
    active_comparator = compare_champion if comparator is None else comparator
    for state in states:
        if state.child_score is None:
            continue
        try:
            comparison = cast(
                ChampionComparison,
                active_comparator(
                    state.parent_score,
                    state.child_score,
                    gate,
                ),
            )
        except (ChampionEvidenceError, TypeError, ValueError) as error:
            raise ChampionLifecycleError(
                "bound lifecycle comparison could not be recomputed"
            ) from error
        if comparison.accepted:
            fresh.append((state.policy, comparison))
    return tuple(fresh)


def _lifecycle_rank(
    value: tuple[FittedChampionPolicy, ChampionComparison],
) -> tuple[float, float, float, str]:
    policy, comparison = value
    return (
        -comparison.joint_improvement,
        comparison.mean_delta_smae,
        comparison.mean_delta_srmse,
        champion_fingerprint(policy),
    )


def _combined_feedback(result: BuildEvolutionResult) -> ProposerEvidence:
    feedback = ProposerEvidence(
        label="adaptive_train_build_diagnostic",
        independent_generalization_claim=False,
        morphology=tuple(
            item
            for generation in result.generations
            for item in generation.feedback.morphology
        ),
        comparisons=tuple(
            item
            for generation in result.generations
            for item in generation.feedback.comparisons
        ),
        invalid_attempts=tuple(
            sorted(
                (
                    item
                    for generation in result.generations
                    for item in generation.feedback.invalid_attempts
                ),
                key=lambda item: (item.structure_sha256, item.reason_code),
            )
        ),
    )
    return feedback


class ChampionEvolutionController:
    """Run the formal one-shot Build, Calibration, and Dev lifecycle."""

    def __init__(
        self,
        *,
        manifest: ChampionRunManifest,
        config: ChampionEvolutionConfig,
        attestations: ChampionRunAttestations,
        proposer: ChampionProposerAdapter,
        row_provider: ChampionRowProviderAdapter,
        runtime_bindings: ChampionRuntimeBindings,
        artifact_store: ChampionArtifactStore,
        authority_store: ChampionAuthorityStore,
    ) -> None:
        if type(manifest) is not ChampionRunManifest:
            _lifecycle_fail("manifest must be an exact ChampionRunManifest")
        ChampionRunManifest.__post_init__(manifest)
        if type(config) is not ChampionEvolutionConfig:
            _lifecycle_fail("config must be an exact ChampionEvolutionConfig")
        ChampionEvolutionConfig.__post_init__(config)
        if type(attestations) is not ChampionRunAttestations:
            _lifecycle_fail("actual inputs must be exact ChampionRunAttestations")
        ChampionRunAttestations.__post_init__(attestations)
        if type(artifact_store) is not ChampionArtifactStore:
            _lifecycle_fail("artifact store must be an exact ChampionArtifactStore")
        if type(authority_store) is not ChampionAuthorityStore:
            _lifecycle_fail("authority store must be an exact ChampionAuthorityStore")
        if type(proposer) is not ChampionProposerAdapter:
            _lifecycle_fail("formal proposer requires an exact typed callable binding")
        if type(row_provider) is not ChampionRowProviderAdapter:
            _lifecycle_fail("formal row provider requires an exact typed callable binding")
        if type(runtime_bindings) is not ChampionRuntimeBindings:
            _lifecycle_fail("formal runtime requires exact closed bindings")
        proposer.verify()
        row_provider.verify()
        runtime_bindings.verify_live_globals()
        if (
            proposer.fingerprint != attestations.proposer_binding.fingerprint
            or row_provider.fingerprint
            != attestations.row_provider_binding.fingerprint
            or runtime_bindings.fingerprint != attestations.runtime_bindings.fingerprint
        ):
            _lifecycle_fail("executing callable bindings drifted from actual attestations")
        self.manifest = manifest
        self.config = config
        self.attestations = attestations
        self.proposer = proposer
        self.row_provider = row_provider
        self.runtime_bindings = runtime_bindings
        self.artifact_store = artifact_store
        self.authority_store = authority_store
        self._run_instance_fingerprint: str | None = None
        self._validate_store_separation()

    def _authority_run_fingerprint(self) -> str:
        if self._run_instance_fingerprint is None:
            _lifecycle_fail("run instance authority was not bootstrapped")
        return champion_fingerprint(
            {
                "input_fingerprint": self.manifest.input_fingerprint,
                "run_instance_fingerprint": self._run_instance_fingerprint,
            }
        )

    def _validate_store_separation(self) -> None:
        authority = self.authority_store.root
        disallowed = [self.artifact_store.root]
        disallowed.extend(path for _, path in self.attestations.source_files)
        disallowed.extend(path for _, path in self.attestations.dictionary_files)
        disallowed.extend(path for _, path in self.attestations.runtime_files)
        disallowed.extend(
            (
                self.attestations.split_manifest_file,
                self.attestations.forecast_store,
            )
        )
        for path in disallowed:
            absolute = path.absolute()
            if (
                authority == absolute
                or authority in absolute.parents
                or absolute in authority.parents
            ):
                _lifecycle_fail(
                    "authority root must be outside and non-aliasing with run inputs"
                )

    def _require_pre_callback_state(self, parent: ChampionRelease) -> None:
        self.attestations.verify(self.manifest)
        self.artifact_store.bind_manifest(self.manifest)
        if (
            self.artifact_store.bind_run_instance(self.manifest)
            != self._run_instance_fingerprint
        ):
            _lifecycle_fail("run instance authority drifted")
        self.artifact_store.ensure_release(parent)
        self.authority_store._verify_root()

    def _validate_bindings(
        self,
        parent: ChampionRelease,
        train_tasks: object,
    ) -> tuple[tuple[Task, ...], TrainPartitions]:
        if type(parent) is not ChampionRelease:
            _lifecycle_fail("formal evolution Parent must be an exact ChampionRelease")
        _validated_parent(parent)
        ChampionRunManifest.__post_init__(self.manifest)
        ChampionEvolutionConfig.__post_init__(self.config)
        self.attestations.verify(self.manifest)
        if (
            self.config.build_size != 64
            or self.config.calibration_size != 16
            or self.config.screen_sizes != (8, 32, 64)
        ):
            _lifecycle_fail("formal lifecycle requires the exact 64/16 schedule")
        if self.config.fingerprint != self.manifest.schedule_fingerprint:
            _lifecycle_fail("schedule fingerprint drifted from the run manifest")
        if (
            self.config.candidate_minimum_gain
            != self.manifest.candidate_minimum_gain
            or self.config.research_target_gain
            != self.manifest.research_target_gain
        ):
            _lifecycle_fail("threshold inputs drifted from the run manifest")
        if (
            parent.metric_policy_fingerprint
            != self.manifest.metric_policy_fingerprint
        ):
            _lifecycle_fail("metric policy drifted from the exact Parent")
        dictionary = dict(self.manifest.dictionary_hashes)
        if any(dictionary.get(name) != digest for name, digest in parent.source_hashes):
            _lifecycle_fail("Dictionary sources drifted from the exact Parent")
        train = _validated_lifecycle_tasks(
            train_tasks,
            label="Train",
            expected_membership=self.manifest.train_tasks,
            expected_content_hashes=self.manifest.train_task_hashes,
        )
        parts = partition_train_tasks(
            train,
            build_size=64,
            calibration_size=16,
            seed=self.manifest.partition_seed,
        )
        build_membership = tuple(
            (task.task_id, task.entity_name) for task in parts.build
        )
        calibration_membership = tuple(
            (task.task_id, task.entity_name) for task in parts.calibration
        )
        if (
            build_membership != self.manifest.build_tasks
            or calibration_membership != self.manifest.calibration_tasks
        ):
            _lifecycle_fail("internal split membership drifted from the run manifest")
        build_ids = tuple(task.task_id for task in parts.build)
        if self.config.build_task_ids != build_ids:
            _lifecycle_fail("Build task membership drifted from the schedule")
        if (
            not self.config.screen_task_ids
            or self.config.screen_task_ids[-1] != build_ids
        ):
            _lifecycle_fail("screen membership drifted from the Build partition")
        return train, parts

    def _require_checkpoint(
        self,
        checkpoint: ChampionCheckpoint,
        parent: ChampionRelease,
    ) -> None:
        if checkpoint.input_fingerprint != self.manifest.input_fingerprint:
            _lifecycle_fail("checkpoint belongs to a stale or foreign run")
        expected_stage = f"build_generation_{self.config.generations}"
        if checkpoint.completed_stage != expected_stage:
            _lifecycle_fail("checkpoint completed stage is stale or foreign")
        if champion_fingerprint(checkpoint.active_parent) != champion_fingerprint(parent):
            _lifecycle_fail("checkpoint active Parent is stale or foreign")
        self.artifact_store.require_build_evidence(
            checkpoint,
            manifest=self.manifest,
            parent=parent,
            config=self.config,
        )
        self.authority_store.require_build_evidence(
            run_authority_fingerprint=self._authority_run_fingerprint(),
            input_fingerprint=self.manifest.input_fingerprint,
            build_evidence_fingerprint=checkpoint.build_evidence_fingerprint,
        )

    def _require_durable_state(
        self,
        checkpoint: ChampionCheckpoint,
        parent: ChampionRelease,
    ) -> None:
        self.artifact_store.bind_manifest(self.manifest)
        self.artifact_store.ensure_release(parent)
        self.attestations.verify(self.manifest)
        self.authority_store._verify_root()
        if (
            self.artifact_store.bind_run_instance(self.manifest)
            != self._run_instance_fingerprint
        ):
            _lifecycle_fail("run instance authority drifted")
        stored = self.artifact_store.load_checkpoint()
        if stored is None:
            _lifecycle_fail("durable checkpoint disappeared before a boundary")
        self._require_checkpoint(stored, parent)
        if canonical_json_bytes(stored.to_payload()) != canonical_json_bytes(
            checkpoint.to_payload()
        ):
            _lifecycle_fail("durable checkpoint drifted before a lifecycle boundary")

    def evolve(
        self,
        parent: ChampionRelease,
        train_tasks: tuple[Task, ...] | list[Task],
        dev_tasks: object,
    ) -> ChampionEvolutionOutcome:
        """Run one formal lifecycle; Dev stays unopened until fresh Calibration acceptance."""
        parent_sha256 = champion_fingerprint(parent)
        config_sha256 = self.config.fingerprint
        manifest_sha256 = self.manifest.input_fingerprint
        _, parts = self._validate_bindings(parent, train_tasks)
        self.artifact_store.bind_manifest(self.manifest)
        self._run_instance_fingerprint = self.artifact_store.bind_run_instance(
            self.manifest
        )
        if self.artifact_store.has_accepted_release():
            _lifecycle_fail("one-shot lifecycle report cannot be overwritten or replayed")
        self.artifact_store.require_open_lifecycle(
            self._authority_run_fingerprint()
        )
        self.artifact_store.ensure_release(parent)
        checkpoint = self.artifact_store.load_checkpoint()
        build_result: BuildEvolutionResult | None = None
        if checkpoint is None:
            self._require_pre_callback_state(parent)
            build_rows = _call_row_provider(
                self.row_provider,
                parts.build,
                "build",
                protected=(parent, self.config, self.manifest),
            )
            self._require_pre_callback_state(parent)
            try:
                self._require_pre_callback_state(parent)
                build_result = run_build_evolution(
                    parent,
                    build_rows,
                    self.proposer,
                    self.config,
                    boundary_validator=lambda: self._require_pre_callback_state(
                        parent
                    ),
                    runtime_bindings=self.runtime_bindings,
                )
            except ChampionControllerError:
                raise
            except Exception as error:
                raise ChampionLifecycleError("Build evolution failed") from error
            self._require_pre_callback_state(parent)
            build_evidence_fingerprint = self.artifact_store.write_build_evidence(
                manifest=self.manifest,
                parent=parent,
                result=build_result,
                rows=build_rows,
            )
            self.authority_store.bind_build_evidence(
                run_authority_fingerprint=self._authority_run_fingerprint(),
                input_fingerprint=self.manifest.input_fingerprint,
                build_evidence_fingerprint=build_evidence_fingerprint,
            )
            checkpoint = ChampionCheckpoint(
                schema_version=1,
                input_fingerprint=self.manifest.input_fingerprint,
                completed_stage=f"build_generation_{self.config.generations}",
                active_parent=parent,
                proposal_archive=build_result.shortlist,
                sanitized_feedback=_combined_feedback(build_result),
                build_evidence_fingerprint=build_evidence_fingerprint,
            )
            self.artifact_store.write_checkpoint(checkpoint)
        else:
            self._require_checkpoint(checkpoint, parent)

        self._require_checkpoint(checkpoint, parent)
        self._require_durable_state(checkpoint, parent)
        if not checkpoint.proposal_archive:
            self.artifact_store.complete_lifecycle(
                run_authority_fingerprint=self._authority_run_fingerprint(),
                status="no_finalist",
                release=parent,
            )
            return ChampionEvolutionOutcome(
                manifest=self.manifest,
                checkpoint=checkpoint,
                release=parent,
                build_result=build_result,
                calibration_report=None,
                dev_report=None,
            )
        if (
            champion_fingerprint(parent) != parent_sha256
            or self.config.fingerprint != config_sha256
            or self.manifest.input_fingerprint != manifest_sha256
        ):
            _lifecycle_fail("bound lifecycle state drifted before Calibration")
        calibration_task_hashes = tuple(
            (task.task_id, task_content_fingerprint(task))
            for task in parts.calibration
        )
        calibration_candidate_fingerprint = champion_fingerprint(
            checkpoint.proposal_archive
        )
        self._require_durable_state(checkpoint, parent)
        calibration_lease, calibration_bundle = (
            self.authority_store.claim_or_resume_split(
                role="calibration",
                split_manifest_fingerprint=self.manifest.split_manifest_fingerprint,
                task_content_fingerprint=champion_fingerprint(
                    {"tasks": calibration_task_hashes}
                ),
                run_input_fingerprint=self._authority_run_fingerprint(),
                candidate_fingerprint=calibration_candidate_fingerprint,
                runtime_fingerprint=self.manifest.runtime_fingerprint,
            )
        )
        self._require_durable_state(checkpoint, parent)
        if calibration_bundle is None:
            calibration_rows = _call_row_provider(
                self.row_provider,
                parts.calibration,
                "calibration",
                protected=(parent, self.config, self.manifest, checkpoint),
            )
            self._require_durable_state(checkpoint, parent)
            self.authority_store.commit_provider_bundle(
                calibration_lease,
                candidate_fingerprint=calibration_candidate_fingerprint,
                runtime_fingerprint=self.manifest.runtime_fingerprint,
                rows=calibration_rows,
            )
        else:
            calibration_rows = _validated_stage_rows(
                calibration_bundle.rows,
                tasks=parts.calibration,
                split="calibration",
            )
        self._require_durable_state(checkpoint, parent)
        calibration_states = _evaluate_lifecycle_stage(
            parent,
            checkpoint.proposal_archive,
            calibration_rows,
            tuple(task.task_id for task in parts.calibration),
            split="calibration",
            gate=self.config.gate_config,
            runtime_bindings=self.runtime_bindings,
        )
        calibration_report = _stage_report(
            self.manifest,
            parent,
            "calibration",
            calibration_states,
        )
        if not self.authority_store.verify_evaluation(
            calibration_lease, calibration_report
        ):
            self.authority_store.commit_evaluation(
                calibration_lease, calibration_report
            )
        self.artifact_store.bind_report(calibration_report)
        self._require_durable_state(checkpoint, parent)
        if (
            champion_fingerprint(parent) != parent_sha256
            or self.config.fingerprint != config_sha256
            or self.manifest.input_fingerprint != manifest_sha256
        ):
            _lifecycle_fail("bound lifecycle state drifted at Calibration boundary")
        calibration_accepted = _fresh_comparisons(
            calibration_states,
            self.config.gate_config,
            self.runtime_bindings.comparator,
        )
        if not calibration_accepted:
            self.artifact_store.complete_lifecycle(
                run_authority_fingerprint=self._authority_run_fingerprint(),
                status="calibration_rejected",
                release=parent,
            )
            return ChampionEvolutionOutcome(
                manifest=self.manifest,
                checkpoint=checkpoint,
                release=parent,
                build_result=build_result,
                calibration_report=calibration_report,
                dev_report=None,
            )
        train_winner, _ = min(calibration_accepted, key=_lifecycle_rank)

        self._require_durable_state(checkpoint, parent)
        dev = _validated_lifecycle_tasks(
            dev_tasks,
            label="Dev",
            expected_membership=self.manifest.dev_tasks,
            expected_content_hashes=self.manifest.dev_task_hashes,
        )
        dev_task_hashes = tuple(
            (task.task_id, task_content_fingerprint(task)) for task in dev
        )
        dev_candidate_fingerprint = champion_fingerprint((train_winner,))
        self._require_durable_state(checkpoint, parent)
        dev_lease, dev_bundle = self.authority_store.claim_or_resume_split(
            role="dev",
            split_manifest_fingerprint=self.manifest.split_manifest_fingerprint,
            task_content_fingerprint=champion_fingerprint(
                {"tasks": dev_task_hashes}
            ),
            run_input_fingerprint=self._authority_run_fingerprint(),
            candidate_fingerprint=dev_candidate_fingerprint,
            runtime_fingerprint=self.manifest.runtime_fingerprint,
        )
        self._require_durable_state(checkpoint, parent)
        if dev_bundle is None:
            dev_rows = _call_row_provider(
                self.row_provider,
                dev,
                "dev",
                protected=(
                    parent,
                    train_winner,
                    self.config,
                    self.manifest,
                    checkpoint,
                ),
            )
            self._require_durable_state(checkpoint, parent)
            self.authority_store.commit_provider_bundle(
                dev_lease,
                candidate_fingerprint=dev_candidate_fingerprint,
                runtime_fingerprint=self.manifest.runtime_fingerprint,
                rows=dev_rows,
            )
        else:
            dev_rows = _validated_stage_rows(
                dev_bundle.rows, tasks=dev, split="dev"
            )
        self._require_durable_state(checkpoint, parent)
        dev_states = _evaluate_lifecycle_stage(
            parent,
            (train_winner,),
            dev_rows,
            tuple(task.task_id for task in dev),
            split="dev",
            gate=self.config.gate_config,
            runtime_bindings=self.runtime_bindings,
        )
        dev_report = _stage_report(self.manifest, parent, "dev", dev_states)
        if not self.authority_store.verify_evaluation(dev_lease, dev_report):
            self.authority_store.commit_evaluation(dev_lease, dev_report)
        self.artifact_store.bind_report(dev_report)
        self._require_durable_state(checkpoint, parent)
        if (
            champion_fingerprint(parent) != parent_sha256
            or champion_fingerprint(train_winner)
            != dev_report.candidates[0].policy_sha256
            or self.config.fingerprint != config_sha256
            or self.manifest.input_fingerprint != manifest_sha256
        ):
            _lifecycle_fail("bound lifecycle state drifted at Dev boundary")
        dev_accepted = _fresh_comparisons(
            dev_states,
            self.config.gate_config,
            self.runtime_bindings.comparator,
        )
        if not dev_accepted:
            self.artifact_store.complete_lifecycle(
                run_authority_fingerprint=self._authority_run_fingerprint(),
                status="dev_rejected",
                release=parent,
            )
            return ChampionEvolutionOutcome(
                manifest=self.manifest,
                checkpoint=checkpoint,
                release=parent,
                build_result=build_result,
                calibration_report=calibration_report,
                dev_report=dev_report,
            )
        if _canonical_identity(train_winner.recipe.name) in {
            _canonical_identity(item) for item in parent.lineage
        }:
            _lifecycle_fail("accepted release lineage contains a replayed policy ID")
        release = ChampionRelease(
            policy=train_winner,
            source_hashes=self.manifest.dictionary_hashes,
            metric_policy_fingerprint=self.manifest.metric_policy_fingerprint,
            lineage=parent.lineage + (train_winner.recipe.name,),
        )
        self.artifact_store.publish_release(release)
        self.artifact_store.complete_lifecycle(
            run_authority_fingerprint=self._authority_run_fingerprint(),
            status="accepted",
            release=release,
        )
        return ChampionEvolutionOutcome(
            manifest=self.manifest,
            checkpoint=checkpoint,
            release=release,
            build_result=build_result,
            calibration_report=calibration_report,
            dev_report=dev_report,
        )


__all__ = [
    "BuildAttempt",
    "BuildEvolutionResult",
    "BuildGeneration",
    "ChampionArtifactStore",
    "ChampionAuthorityStore",
    "ChampionCheckpoint",
    "ChampionCallableBinding",
    "ChampionControllerError",
    "ChampionEvaluationReport",
    "ChampionEvolutionController",
    "ChampionEvolutionConfig",
    "ChampionEvolutionOutcome",
    "ChampionLifecycleError",
    "ChampionProposerAdapter",
    "ChampionRowProviderAdapter",
    "ChampionRunManifest",
    "ChampionRunAttestations",
    "ChampionRuntimeBindings",
    "ChampionStageCandidateReport",
    "TrainPartitions",
    "canonical_release_bytes",
    "partition_train_tasks",
    "run_build_evolution",
    "task_content_fingerprint",
]
