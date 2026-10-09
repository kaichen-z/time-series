"""Executable official benchmark loaders for the package evolution boundary.

The loaders deliberately materialize labels only for Train/Dev.  Test records are
represented by opaque identities so a caller cannot accidentally tune on them.
Anchor forecasts are supplied as a separately frozen cache and must cover every
Train/Dev identity exactly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from common.payload import canonical_json_bytes, canonical_json_line_bytes
from evolving_loop.benchmark_numeric_adapter import BenchmarkNumericWindow
from evolving_loop.data import Document


class OfficialBenchmarkLoaderError(ValueError):
    """Official data, split metadata, or anchor cache violated the contract."""


@dataclass(frozen=True)
class OfficialBenchmarkPartitions:
    train: tuple[BenchmarkNumericWindow, ...]
    dev: tuple[BenchmarkNumericWindow, ...]
    sealed_test_task_ids: tuple[str, ...]
    source_fingerprint: str

    def __post_init__(self) -> None:
        train_ids = {window.task_id for window in self.train}
        dev_ids = {window.task_id for window in self.dev}
        test_ids = set(self.sealed_test_task_ids)
        if not train_ids or not dev_ids or not test_ids:
            raise OfficialBenchmarkLoaderError("official partitions must be nonempty")
        if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
            raise OfficialBenchmarkLoaderError("official partitions must be disjoint")
        if len(test_ids) != len(self.sealed_test_task_ids):
            raise OfficialBenchmarkLoaderError("sealed test identities must be unique")


@dataclass(frozen=True)
class OfficialEngineInputFiles:
    tasks_file: Path
    split_manifest: Path
    anchor_bindings: Path
    receipt: Path
    fingerprint: str


def _opaque_id(namespace: str, external_key: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{external_key}".encode()).hexdigest()
    return f"official_{digest[:24]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchors_for(
    cache: Mapping[str, Mapping[str, Sequence[float]]] | None,
    task_id: str,
    horizon: int,
    *,
    allow_unbound: bool,
) -> Mapping[str, tuple[float, ...]]:
    if cache is None:
        if not allow_unbound:
            raise OfficialBenchmarkLoaderError("anchor cache is required")
        return MappingProxyType({"__unbound__": (0.0,) * horizon})
    raw = cache.get(task_id)
    if not isinstance(raw, Mapping) or not raw:
        raise OfficialBenchmarkLoaderError(f"missing anchor cache entry: {task_id}")
    anchors: dict[str, tuple[float, ...]] = {}
    for name, values in sorted(raw.items()):
        if type(name) is not str or not name or isinstance(values, (str, bytes)):
            raise OfficialBenchmarkLoaderError("invalid anchor cache entry")
        forecast = tuple(float(value) for value in values)
        if len(forecast) != horizon or not all(math.isfinite(value) for value in forecast):
            raise OfficialBenchmarkLoaderError(
                f"anchor horizon/value mismatch: {task_id}:{name}"
            )
        anchors[name] = forecast
    return MappingProxyType(anchors)


def load_anchor_cache(path: str | Path) -> Mapping[str, Mapping[str, tuple[float, ...]]]:
    """Load the only accepted anchor-cache schema and freeze its contents."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "official-anchor-cache-v1":
        raise OfficialBenchmarkLoaderError("unsupported anchor cache schema")
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise OfficialBenchmarkLoaderError("anchor cache entries must be nonempty")
    frozen: dict[str, Mapping[str, tuple[float, ...]]] = {}
    for key, forecasts in sorted(entries.items()):
        if type(key) is not str or not key or not isinstance(forecasts, dict):
            raise OfficialBenchmarkLoaderError("invalid anchor cache identity")
        frozen[key] = MappingProxyType(
            {name: tuple(float(value) for value in values) for name, values in sorted(forecasts.items())}
        )
    return MappingProxyType(frozen)


def load_official_timesx(
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    anchor_cache: Mapping[str, Mapping[str, Sequence[float]]] | None,
    allow_unbound_anchors: bool = False,
) -> OfficialBenchmarkPartitions:
    """Load the public PostTime scope with the frozen 2173/550 Train/Dev split."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "official-ts-alignment-v1":
        raise OfficialBenchmarkLoaderError("unsupported alignment manifest")
    section = manifest.get("timesx")
    if not isinstance(section, dict):
        raise OfficialBenchmarkLoaderError("missing TimesX alignment section")
    root = Path(repository_root)
    train: list[BenchmarkNumericWindow] = []
    dev: list[BenchmarkNumericWindow] = []
    sealed: list[str] = []
    source_rows = []
    for record in section.get("files", []):
        relative = Path(record["path"])
        # Manifest paths are rooted at repos/TimesX-project.
        parts = relative.parts
        relative_repo = Path(*parts[2:]) if parts[:2] == ("repos", "TimesX-project") else relative
        path = root / relative_repo
        if _sha256(path) != record["sha256"]:
            raise OfficialBenchmarkLoaderError(f"TimesX source hash mismatch: {relative}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        variable = payload["dataset_info"]["dataset_name"]
        expected = record["variable"]
        if variable != expected:
            raise OfficialBenchmarkLoaderError(f"TimesX variable mismatch: {relative}")
        for sample in payload["samples"]:
            future_ts = sample["future_time"]["timestamp"]
            start, end = future_ts[0][:10], future_ts[-1][:10]
            external_key = f"timesx:{variable}:{sample['idx']}"
            task_id = _opaque_id("timesx", external_key)
            declared_counts = record.get("split_counts", {})
            if start > "2025-01-30":
                if declared_counts.get("id_test", 0) or declared_counts.get("ood_test", 0):
                    sealed.append(task_id)
                continue
            # Only records declared post-training by the frozen manifest enter
            # Train/Dev. Extra repository and OOD variables have count zero.
            if not (start > "2023-01-01" and end < "2025-02-01"):
                continue
            if declared_counts.get("post_training", 0) == 0:
                continue
            part = dev if start >= "2024-09-01" else train
            truth = tuple(float(value) for value in sample["future_time"]["value"])
            documents = tuple(
                Document(
                    document_id=f"{task_id}_{field}",
                    content=str(sample[field]),
                )
                for field in (
                    "background",
                    "scenario",
                    "holiday_info",
                    "covariates_info",
                )
                if str(sample.get(field, "")).strip()
            )
            part.append(
                BenchmarkNumericWindow(
                    task_id=task_id,
                    group_id=f"{relative_repo.parts[1]}:{sample['freq']}",
                    history=tuple(float(value) for value in sample["past_time"]["value"]),
                    truth=truth,
                    frequency=str(sample["freq"]),
                    history_timestamps=tuple(
                        str(value) for value in sample["past_time"]["timestamp"]
                    ),
                    future_timestamps=tuple(str(value) for value in future_ts),
                    documents=documents,
                    target_description=str(
                        sample.get("background") or "numeric forecasting target"
                    ),
                    anchor_forecasts=_anchors_for(
                        anchor_cache,
                        task_id,
                        len(truth),
                        allow_unbound=allow_unbound_anchors,
                    ),
                )
            )
        source_rows.append((str(relative), record["sha256"]))
    counts = section.get("split_counts", {})
    if (len(train), len(dev), len(sealed)) != (2173, 550, counts.get("id_test", 0) + counts.get("ood_test", 0)):
        raise OfficialBenchmarkLoaderError(
            f"TimesX split count mismatch: {(len(train), len(dev), len(sealed))}"
        )
    fingerprint = hashlib.sha256(
        canonical_json_bytes({"manifest": _sha256(manifest_file), "sources": source_rows})
    ).hexdigest()
    return OfficialBenchmarkPartitions(tuple(train), tuple(dev), tuple(sealed), fingerprint)


def load_official_time_mmd(
    *,
    manifest_path: str | Path,
    repository_root: str | Path,
    anchor_cache: Mapping[str, Mapping[str, Sequence[float]]] | None,
    allow_unbound_anchors: bool = False,
) -> OfficialBenchmarkPartitions:
    """Materialize official MM-TSFlib Train/Dev windows; keep Test labels sealed."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    section = manifest.get("time_mmd")
    if manifest.get("schema") != "official-ts-alignment-v1" or not isinstance(section, dict):
        raise OfficialBenchmarkLoaderError("missing Time-MMD alignment section")
    protocol = section["protocol"]
    root = Path(repository_root)
    train: list[BenchmarkNumericWindow] = []
    dev: list[BenchmarkNumericWindow] = []
    sealed: list[str] = []
    source_rows = []
    for record in section["data_files"]:
        relative = Path(record["path"])
        parts = relative.parts
        relative_repo = Path(*parts[2:]) if parts[:2] == ("repos", "MM-TSFlib") else relative
        path = root / relative_repo
        if _sha256(path) != record["sha256"]:
            raise OfficialBenchmarkLoaderError(f"Time-MMD source hash mismatch: {relative}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        values = [
            float(row["OT"]) if row["OT"].strip() else math.nan
            for row in rows
        ]
        n = len(values)
        n_train, n_test = int(n * 0.7), int(n * 0.2)
        n_dev = n - n_train - n_test
        if not all(math.isfinite(value) for value in values[:n_train]):
            raise OfficialBenchmarkLoaderError(
                f"Time-MMD train contains missing/nonfinite target: {relative}"
            )
        mean = sum(values[:n_train]) / n_train
        variance = sum((value - mean) ** 2 for value in values[:n_train]) / n_train
        scale = math.sqrt(variance)
        if not math.isfinite(scale) or scale <= 0:
            raise OfficialBenchmarkLoaderError(f"invalid Time-MMD train scaler: {relative}")
        standardized = [(value - mean) / scale for value in values]
        lower_name = path.name.lower()
        cadence = "weekly" if "week" in lower_name else "daily" if "day" in lower_name else "monthly"
        lookback = int(protocol[cadence]["lookback"])
        domain = relative_repo.parts[-2]
        for horizon in protocol[cadence]["horizons"]:
            for part_name, first, last, output in (
                ("train", lookback, n_train - horizon, train),
                ("dev", n_train, n_train + n_dev - horizon, dev),
            ):
                for origin in range(first, last + 1):
                    external_key = f"time_mmd:{domain}:h{horizon}:{part_name}:t{origin}"
                    task_id = _opaque_id("time_mmd", external_key)
                    truth = tuple(standardized[origin : origin + horizon])
                    documents = tuple(
                        Document(
                            document_id=_opaque_id(
                                "time_mmd_document",
                                f"{relative_repo}:{origin}:{field}",
                            ),
                            content=rows[origin][field].strip(),
                        )
                        for field in ("Final_Search_2", "Final_Search_4", "Final_Search_6")
                        if field in rows[origin]
                        and rows[origin][field].strip()
                        and rows[origin][field].strip().upper() != "NA"
                    )
                    output.append(
                        BenchmarkNumericWindow(
                            task_id=task_id,
                            group_id=f"{domain}:h{horizon}",
                            history=tuple(standardized[origin - lookback : origin]),
                            truth=truth,
                            frequency=cadence,
                            documents=documents,
                            target_description=f"Time-MMD {domain} target series",
                            anchor_forecasts=_anchors_for(
                                anchor_cache,
                                task_id,
                                horizon,
                                allow_unbound=allow_unbound_anchors,
                            ),
                        )
                    )
            test_start = n - n_test
            test_origins = range(test_start, n - horizon + 1)
            # Match the paper loader's batch-size 32/drop_last execution set.
            executed = (len(test_origins) // int(protocol["paper_test_batch_size"])) * int(protocol["paper_test_batch_size"])
            for origin in tuple(test_origins)[:executed]:
                key = f"time_mmd:{domain}:h{horizon}:test:t{origin}"
                sealed.append(_opaque_id("time_mmd", key))
        source_rows.append((str(relative), record["sha256"]))
    if len(dev) != 5780 or len(sealed) != 11936:
        raise OfficialBenchmarkLoaderError(
            f"Time-MMD split count mismatch: dev={len(dev)} test={len(sealed)}"
        )
    fingerprint = hashlib.sha256(
        canonical_json_bytes({"manifest": _sha256(manifest_file), "sources": source_rows})
    ).hexdigest()
    return OfficialBenchmarkPartitions(tuple(train), tuple(dev), tuple(sealed), fingerprint)


def write_official_engine_inputs(
    partitions: OfficialBenchmarkPartitions,
    *,
    dataset: str,
    destination: str | Path,
) -> OfficialEngineInputFiles:
    """Write the exact Train/Dev inputs consumed by the package runner.

    Test labels never enter these files: the split manifest carries opaque test
    identities only.  Anchor forecasts are bound by the same opaque task IDs.
    """
    if type(partitions) is not OfficialBenchmarkPartitions:
        raise OfficialBenchmarkLoaderError("expected exact official partitions")
    if type(dataset) is not str or not dataset.strip():
        raise OfficialBenchmarkLoaderError("dataset identity must be nonempty")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    tasks_path = target / "tasks.jsonl"
    split_path = target / "split_manifest.json"
    anchors_path = target / "anchor_bindings.json"
    receipt_path = target / "input_receipt.json"

    task_records = []
    anchors: dict[str, dict[str, list[float]]] = {}
    for window in (*partitions.train, *partitions.dev):
        if "__unbound__" in window.anchor_forecasts:
            raise OfficialBenchmarkLoaderError(
                "unbound anchor jobs cannot be written as engine inputs"
            )
        task_records.append(
            {
                "benchmark_id": window.task_id,
                "labels_public": True,
                "series": {
                    "history_values": list(window.history),
                    "future_values": list(window.truth),
                    "history_timestamps": list(window.history_timestamps)
                    if window.history_timestamps
                    else [f"history_{index}" for index in range(len(window.history))],
                    "future_timestamps": list(window.future_timestamps)
                    if window.future_timestamps
                    else [f"future_{index}" for index in range(len(window.truth))],
                },
                "task_metadata": {
                    "prediction_length": len(window.truth),
                    "frequency": window.frequency,
                    "seasonal_period": None,
                    "target_description": window.target_description,
                },
                "entity_name": "official_benchmark_series",
                "target_name": "target",
                "documents": [
                    {"document_id": document.document_id, "content": document.content}
                    for document in window.documents
                ],
                "gt_evidence": [],
            }
        )
        anchors[window.task_id] = {
            name: list(values) for name, values in window.anchor_forecasts.items()
        }
    tasks_bytes = b"".join(canonical_json_line_bytes(record) for record in task_records)
    tasks_path.write_bytes(tasks_bytes)

    train_ids = [window.task_id for window in partitions.train]
    dev_ids = [window.task_id for window in partitions.dev]
    public_ids = list(partitions.sealed_test_task_ids)
    sizes = {
        "train": len(train_ids), "dev": len(dev_ids), "public_test": len(public_ids)
    }
    split = {
        "schema_version": 1,
        "dataset": dataset,
        "source_split": "official_train_dev_sealed_test",
        "seed": 20260903,
        "grouping": "official_benchmark_group",
        "stratification_features": [],
        "selection_uses_future_values": False,
        "selection_uses_gt_evidence": False,
        "selection_uses_document_labels": False,
        "target_sizes": sizes,
        "actual_sizes": sizes,
        "partitions": {
            "train": {"task_ids": train_ids, "entities": [], "distribution": {}},
            "dev": {"task_ids": dev_ids, "entities": [], "distribution": {}},
            "public_test": {"task_ids": public_ids, "entities": [], "distribution": {}},
        },
        "official_source_fingerprint": partitions.source_fingerprint,
    }
    split["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            split,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    split_path.write_bytes(canonical_json_bytes(split) + b"\n")
    anchor_payload = {"schema": "official-engine-anchor-bindings-v1", "entries": anchors}
    anchors_path.write_bytes(canonical_json_bytes(anchor_payload) + b"\n")

    receipt_payload = {
        "schema": "official-engine-input-receipt-v1",
        "dataset": dataset,
        "source_fingerprint": partitions.source_fingerprint,
        "counts": sizes,
        "tasks_sha256": _sha256(tasks_path),
        "split_manifest_sha256": _sha256(split_path),
        "anchor_bindings_sha256": _sha256(anchors_path),
        "test_labels_materialized": False,
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(receipt_payload)).hexdigest()
    receipt_payload["fingerprint"] = fingerprint
    receipt_path.write_bytes(canonical_json_bytes(receipt_payload) + b"\n")
    for path in (tasks_path, split_path, anchors_path, receipt_path):
        os.chmod(path, 0o444)
    return OfficialEngineInputFiles(
        tasks_file=tasks_path,
        split_manifest=split_path,
        anchor_bindings=anchors_path,
        receipt=receipt_path,
        fingerprint=fingerprint,
    )


__all__ = [
    "OfficialBenchmarkLoaderError",
    "OfficialBenchmarkPartitions",
    "OfficialEngineInputFiles",
    "load_anchor_cache",
    "load_official_time_mmd",
    "load_official_timesx",
    "write_official_engine_inputs",
]
