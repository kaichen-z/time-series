"""JSON payload reading, canonical writing, and field validation shared across packages."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize a payload deterministically so content hashes stay reproducible."""
    return (
        json.dumps(
            standards_json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def standards_json_value(value: object) -> object:
    """Replace raw infinities with explicit typed sentinels; reject NaN."""
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN is not valid standards JSON")
        if math.isinf(value):
            return {
                "status": "positive_infinity" if value > 0 else "negative_infinity",
                "value": None,
            }
        return value
    if isinstance(value, Mapping):
        return {str(key): standards_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [standards_json_value(item) for item in value]
    return value


def decode_infinity_sentinel(value: object, field_name: str) -> float:
    """Decode one explicit raw-tail infinity sentinel under the active schema."""
    if value == {"status": "positive_infinity", "value": None}:
        return math.inf
    if value == {"status": "negative_infinity", "value": None}:
        return -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric or an infinity sentinel") from error
    if math.isnan(number):
        raise ValueError(f"{field_name} must not be NaN")
    return number


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write a payload as canonical JSON, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(payload))
    return destination


def read_json_object(path: str | Path) -> dict[str, object]:
    """Read a JSON file that must contain an object at the top level."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def require_object(value: object, field_name: str) -> dict[str, object]:
    """Return value as a string-keyed dict, or raise naming the offending field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def require_strings(
    value: object, field_name: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    """Return value as a tuple of stripped non-empty strings, treating None as empty."""
    if value is None:
        result: tuple[str, ...] = ()
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{field_name} must be a list of strings")
        result = tuple(str(item).strip() for item in value)
        if any(not item for item in result):
            raise ValueError(f"{field_name} must not contain empty values")
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def require_non_empty(value: object, field_name: str) -> str:
    """Return value as a stripped string, or raise naming the offending field."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def is_simple_filename(name: object) -> bool:
    """Report whether name is a bare filename that cannot escape its directory."""
    text = str(name)
    return bool(text) and Path(text).name == text
