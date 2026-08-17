from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from numerical_agent.collection.registry import (
    build_release,
    load_method_cards,
    load_source_records,
    write_release,
)


FIXTURES = Path(__file__).parent / "fixtures" / "method_collection"


def release_from_fixtures():
    return build_release(
        load_source_records(FIXTURES / "valid_sources.jsonl"),
        load_method_cards(FIXTURES / "valid_methods.jsonl"),
        dataset_id="forecast_method_dataset_v001",
        release_date="2026-08-17",
        collection_cutoff="2026-08-17",
        taxonomy={"statistical": ("baseline", "seasonal")},
        collection_batches=(),
    )


def test_registry_reports_path_and_line_for_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"sources\.jsonl:1"):
        load_source_records(path)

    path.write_text(
        json.dumps(
            {
                "source_id": "source_000001",
                "title": "Source",
                "authors": ["Author"],
                "year": 2024,
                "source_type": "paper",
                "url": "https://example.org/source",
                "retrieved_at": "2026-08-17",
            }
        )
        + "\nnot-json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"sources\.jsonl:2"):
        load_source_records(path)


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    source_line = (FIXTURES / "valid_sources.jsonl").read_text(encoding="utf-8").splitlines()[0]
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text(f"{source_line}\n{source_line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source_id"):
        load_source_records(source_path)

    method_line = (FIXTURES / "valid_methods.jsonl").read_text(encoding="utf-8").splitlines()[0]
    method_path = tmp_path / "methods.jsonl"
    method_path.write_text(f"{method_line}\n{method_line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate method_uid"):
        load_method_cards(method_path)


def test_release_builder_sorts_sources_and_methods() -> None:
    release = release_from_fixtures()

    assert [source.source_id for source in release.sources] == [
        "source_000001",
        "source_000002",
    ]
    assert [method.method_uid for method in release.methods] == [
        "method_000001",
        "method_000002",
    ]


def test_release_writer_is_byte_deterministic_and_hashes_canonical_payload(
    tmp_path: Path,
) -> None:
    release = release_from_fixtures()
    first = tmp_path / "first" / "forecast_method_dataset_v001.json"
    second = tmp_path / "second" / "forecast_method_dataset_v001.json"

    write_release(release, first, first.with_suffix(".sha256"))
    write_release(release, second, second.with_suffix(".sha256"))

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["content_hash"].startswith("sha256:")
    unhashed = dict(payload)
    unhashed["content_hash"] = ""
    canonical = (
        json.dumps(unhashed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert payload["content_hash"] == f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    sidecar = first.with_suffix(".sha256").read_text(encoding="utf-8")
    assert sidecar == f"{hashlib.sha256(first.read_bytes()).hexdigest()}  {first.name}\n"
