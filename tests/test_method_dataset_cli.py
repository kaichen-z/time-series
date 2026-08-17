from __future__ import annotations

import json
from pathlib import Path

from numerical_agent.main import build_parser, main


FIXTURES = Path(__file__).parent / "fixtures" / "method_collection"


def write_query_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_cutoff": "2026-08-17",
                "query_templates": ["time series forecasting {term} original paper"],
                "source_tiers": ["paper", "textbook"],
                "taxonomy": {
                    "statistical": {
                        "baseline": ["naive forecast baseline"],
                        "seasonal": ["seasonal naive"],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_collection_journal(
    path: Path,
    counts: list[int],
    duplicate_resolutions: list[dict[str, str]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "saturation_base_count": 2,
                "collection_batches": [
                    {
                        "batch_id": f"batch_{index + 1}",
                        "reviewed_source_count": 1,
                        "candidate_count": count,
                        "new_canonical_methods": count,
                        "duplicate_count": 0,
                        "rejected_count": 0,
                    }
                    for index, count in enumerate(counts)
                ],
                "duplicate_resolutions": duplicate_resolutions or [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dataset_subcommands_parse_required_inputs() -> None:
    parser = build_parser()

    collect = parser.parse_args(
        [
            "collect-methods",
            "--sources",
            "sources.jsonl",
            "--methods",
            "methods.jsonl",
            "--output-dir",
            "raw",
        ]
    )
    verify = parser.parse_args(
        [
            "verify-methods",
            "--sources",
            "sources.jsonl",
            "--methods",
            "methods.jsonl",
            "--queries",
            "queries.json",
            "--output",
            "audit.json",
        ]
    )
    build = parser.parse_args(
        [
            "build-dataset",
            "--sources",
            "sources.jsonl",
            "--methods",
            "methods.jsonl",
            "--queries",
            "queries.json",
            "--collection-journal",
            "journal.json",
            "--output",
            "release.json",
            "--audit-output",
            "audit.json",
            "--sha256-output",
            "release.sha256",
        ]
    )

    assert collect.command == "collect-methods"
    assert verify.command == "verify-methods"
    assert build.command == "build-dataset"


def test_collect_methods_writes_raw_registry_and_duplicate_report(
    tmp_path: Path, capsys
) -> None:
    output_dir = tmp_path / "raw"

    code = main(
        [
            "collect-methods",
            "--sources",
            str(FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(FIXTURES / "valid_methods.jsonl"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert (output_dir / "raw_method_registry.json").exists()
    duplicate_payload = json.loads(
        (output_dir / "duplicate_candidates.json").read_text(encoding="utf-8")
    )
    assert duplicate_payload == {"duplicate_candidates": []}
    summary = json.loads(capsys.readouterr().out)
    assert summary["method_count"] == 2


def test_verify_methods_returns_two_for_unpublishable_registry(
    tmp_path: Path, capsys
) -> None:
    queries = tmp_path / "queries.json"
    write_query_manifest(queries)
    methods = tmp_path / "methods.jsonl"
    payload = json.loads(
        (FIXTURES / "valid_methods.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    payload["verification_status"] = "unverified"
    payload["definition_source_ids"] = []
    methods.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    audit = tmp_path / "audit.json"

    code = main(
        [
            "verify-methods",
            "--sources",
            str(FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(methods),
            "--queries",
            str(queries),
            "--output",
            str(audit),
        ]
    )

    assert code == 2
    report = json.loads(audit.read_text(encoding="utf-8"))
    assert "method_not_verified" in report["verification"]["issue_codes"]
    assert json.loads(capsys.readouterr().out)["publishable"] is False


def test_build_dataset_requires_saturation_and_writes_verified_release(
    tmp_path: Path, capsys
) -> None:
    queries = tmp_path / "queries.json"
    journal = tmp_path / "journal.json"
    write_query_manifest(queries)
    write_collection_journal(journal, [0, 0, 0])
    release = tmp_path / "forecast_method_dataset_v001.json"
    audit = tmp_path / "audit.json"
    sidecar = tmp_path / "forecast_method_dataset_v001.sha256"

    code = main(
        [
            "build-dataset",
            "--sources",
            str(FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(FIXTURES / "valid_methods.jsonl"),
            "--queries",
            str(queries),
            "--collection-journal",
            str(journal),
            "--output",
            str(release),
            "--audit-output",
            str(audit),
            "--sha256-output",
            str(sidecar),
        ]
    )

    assert code == 0
    payload = json.loads(release.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "forecast_method_dataset_v001"
    assert len(payload["methods"]) == 2
    assert sidecar.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["saturated"] is True
    assert summary["method_count"] == 2


def test_build_dataset_rejects_unsaturated_collection(tmp_path: Path, capsys) -> None:
    queries = tmp_path / "queries.json"
    journal = tmp_path / "journal.json"
    write_query_manifest(queries)
    write_collection_journal(journal, [1, 0, 0])

    code = main(
        [
            "build-dataset",
            "--sources",
            str(FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(FIXTURES / "valid_methods.jsonl"),
            "--queries",
            str(queries),
            "--collection-journal",
            str(journal),
            "--output",
            str(tmp_path / "release.json"),
            "--audit-output",
            str(tmp_path / "audit.json"),
            "--sha256-output",
            str(tmp_path / "release.sha256"),
        ]
    )

    assert code == 2
    assert not (tmp_path / "release.json").exists()
    assert json.loads(capsys.readouterr().out)["saturated"] is False


def test_build_dataset_accepts_a_manually_resolved_distinct_wrapper_pair(
    tmp_path: Path, capsys
) -> None:
    duplicate_lines = (
        FIXTURES / "duplicate_methods.jsonl"
    ).read_text(encoding="utf-8").splitlines()[:2]
    methods = tmp_path / "methods.jsonl"
    methods.write_text("\n".join(duplicate_lines) + "\n", encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_cutoff": "2026-08-17",
                "query_templates": ["time series forecasting {term} original paper"],
                "source_tiers": ["paper", "textbook"],
                "taxonomy": {
                    "statistical": {
                        "automatic_selection": ["automatic ARIMA"],
                        "autoregressive": ["ARIMA"],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    journal = tmp_path / "journal.json"
    write_collection_journal(
        journal,
        [0, 0, 0],
        [
            {
                "left_method_uid": "method_arima",
                "right_method_uid": "method_auto_arima",
                "decision": "distinct_wrapper",
            }
        ],
    )

    code = main(
        [
            "build-dataset",
            "--sources",
            str(FIXTURES / "valid_sources.jsonl"),
            "--methods",
            str(methods),
            "--queries",
            str(queries),
            "--collection-journal",
            str(journal),
            "--output",
            str(tmp_path / "release.json"),
            "--audit-output",
            str(tmp_path / "audit.json"),
            "--sha256-output",
            str(tmp_path / "release.sha256"),
        ]
    )

    assert code == 0
    assert (tmp_path / "release.json").exists()
    assert json.loads(capsys.readouterr().out)["unresolved_duplicate_count"] == 0
