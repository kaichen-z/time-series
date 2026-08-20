"""Load untrusted collection manifests and build deterministic releases."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

from common.payload import canonical_json_bytes

from .contracts import DatasetRelease, MethodCard, SourceRecord


RecordT = TypeVar("RecordT")


def _load_jsonl(
    path: str | Path,
    parser: Callable[[Mapping[str, object]], RecordT],
    identifier: Callable[[RecordT], str],
    identifier_name: str,
) -> tuple[RecordT, ...]:
    source = Path(path)
    records: list[RecordT] = []
    seen: set[str] = set()
    try:
        handle = source.open("r", encoding="utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{source}: invalid UTF-8") from exc
    try:
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                    if not isinstance(payload, Mapping):
                        raise ValueError("JSONL entry must be an object")
                    record = parser(payload)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
                record_id = identifier(record)
                if record_id in seen:
                    raise ValueError(
                        f"{source}:{line_number}: duplicate {identifier_name} {record_id!r}"
                    )
                seen.add(record_id)
                records.append(record)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: invalid UTF-8") from exc
    return tuple(records)


def load_source_records(path: str | Path) -> tuple[SourceRecord, ...]:
    return _load_jsonl(
        path, SourceRecord.from_payload, lambda record: record.source_id, "source_id"
    )


def load_method_cards(path: str | Path) -> tuple[MethodCard, ...]:
    return _load_jsonl(
        path, MethodCard.from_payload, lambda record: record.method_uid, "method_uid"
    )


def build_release(
    sources: Sequence[SourceRecord],
    methods: Sequence[MethodCard],
    *,
    dataset_id: str,
    release_date: str,
    collection_cutoff: str,
    taxonomy: Mapping[str, Sequence[str]],
    collection_batches: Sequence[Mapping[str, object]],
) -> DatasetRelease:
    return DatasetRelease(
        schema_version=1,
        dataset_id=dataset_id,
        release_date=release_date,
        collection_cutoff=collection_cutoff,
        sources=tuple(sorted(sources, key=lambda source: source.source_id)),
        methods=tuple(sorted(methods, key=lambda method: method.method_uid)),
        taxonomy={
            str(key): tuple(sorted(str(item) for item in values))
            for key, values in sorted(taxonomy.items())
        },
        collection_batches=tuple(dict(batch) for batch in collection_batches),
        content_hash="",
    )


def write_release(
    release: DatasetRelease,
    destination: str | Path,
    sha256_destination: str | Path | None = None,
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    unhashed = replace(release, content_hash="")
    content_digest = hashlib.sha256(canonical_json_bytes(unhashed.to_payload())).hexdigest()
    finalized = replace(release, content_hash=f"sha256:{content_digest}")
    release_bytes = canonical_json_bytes(finalized.to_payload())
    output.write_bytes(release_bytes)
    if sha256_destination is not None:
        sidecar = Path(sha256_destination)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        file_digest = hashlib.sha256(release_bytes).hexdigest()
        sidecar.write_text(f"{file_digest}  {output.name}\n", encoding="utf-8")
    return output
