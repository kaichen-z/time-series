"""Fit a task-local Numerical ensemble on Train OOF and gate it once on Dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, get_args, get_origin, get_type_hints

from common.data import Task as DataTask, load_tasks_by_id
from common.evolution_core.contracts import METRIC_POLICY_FINGERPRINT
from common.payload import canonical_json_bytes

from .evolution.champion import champion_fingerprint
from .evolution.execution import Task as RuntimeTask
from .evolution.forecast_store import ForecastStore
from .evolution.module import read_module
from .evolution.numerical_selector import CandidateDiagnostics, HindcastFold, HindcastConfig, diagnose_candidate
from .evolution.portfolio import read_policy_file
from .evolution.screening import (
    ScreeningPolicy,
    materialize_active_dictionary,
    profile_task,
)
from .evolution.filtering import parse_filter_source
from .evolution.task_local_ensemble import (
    TaskLocalEnsembleReleaseV3,
    TaskLocalTournamentPolicy,
    canonical_task_local_release_bytes,
    task_local_fingerprint,
)
from .evolution.task_shortlist import (
    CandidatePriorV1, TaskCandidateShortlistV1, TaskShortlistPolicyV1,
    build_task_candidate_shortlist,
)
from .evolution.task_local_confidence import ConfidencePolicy
from .evolution.task_local_evolution import (
    ConditionalUpliftReport,
    TaskLocalTaskRow,
    build_group_fold_manifest,
    evaluate_task_local_release,
    fit_oof_release,
    task_morphology_key,
)
from .main import _add_tsfm_runtime_options, _runtime_registry
from .run_champion_evolution import (
    _clean_git_source,
    _load_parent,
    _load_screening_policy,
    _partition_ids,
    _source_files,
)
from .run_selector_evolution import _forecast_runtime_identity


_CONFIDENCE_HINDCAST_CONFIG = HindcastConfig(folds=5, min_successful_folds=3)


def _adaptive_hindcast_config(
    task: RuntimeTask,
    maximum: HindcastConfig,
) -> HindcastConfig:
    """Use every feasible origin up to five, retaining at least three when possible."""
    history_length = len(task.history)
    fold_horizon = min(task.horizon, max(1, history_length // 4))
    feasible = max(1, history_length // fold_horizon - 1)
    folds = min(maximum.folds, max(maximum.min_successful_folds, feasible))
    return replace(maximum, folds=folds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-dev-regression", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--split-file")
    parser.add_argument("--tasks-file")
    parser.add_argument("--anchor-release-dir")
    parser.add_argument("--forecast-store")
    parser.add_argument("--candidate-priors-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition-seed", type=int, default=20260902)
    _add_tsfm_runtime_options(parser)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, object] | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else canonical_json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()


def task_input_sha256(task: RuntimeTask) -> str:
    """Canonical history-only identity used by the persisted task shortlist."""
    return hashlib.sha256(canonical_json_bytes({
        "task_id": task.task_id, "history": list(task.history), "horizon": task.horizon,
        "frequency": task.frequency,
    })).hexdigest()


_task_input_sha = task_input_sha256  # Explicit compatibility for Task 4 callers.


def _evidence_sha(value):
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("task-local evidence requires a lowercase SHA-256")


def parse_shortlist_index(payload):
    if type(payload) is not dict or set(payload) != {"schema_version", "policy_sha256", "entries", "public_test_accessed"}:
        raise ValueError("shortlist index fields are malformed")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1 or payload["public_test_accessed"] is not False or type(payload["entries"]) is not list:
        raise ValueError("shortlist index is malformed")
    _evidence_sha(payload["policy_sha256"])
    ids = []
    for row in payload["entries"]:
        if type(row) is not dict or set(row) != {"task_id", "task_input_sha256", "shortlist_sha256", "diagnostics_sha256"} or type(row["task_id"]) is not str or not row["task_id"].strip():
            raise ValueError("shortlist index entry is malformed")
        ids.append(row["task_id"])
        for name in ("task_input_sha256", "shortlist_sha256", "diagnostics_sha256"):
            _evidence_sha(row[name])
    if ids != sorted(set(ids)):
        raise ValueError("shortlist index task IDs must be sorted and unique")
    return payload


@dataclass(frozen=True)
class TaskLocalEvidenceBundleV1:
    """Read-only Task 4 evidence passed to P2; never a forecast capability."""

    policy: TaskShortlistPolicyV1
    dictionary_sha256: str
    index: Mapping[str, object]
    by_task: Mapping[str, tuple[TaskCandidateShortlistV1, Mapping[str, object], str, str]]

    def __post_init__(self) -> None:
        if type(self.policy) is not TaskShortlistPolicyV1 or not isinstance(self.dictionary_sha256, str):
            raise ValueError("task-local evidence requires canonical policy and Dictionary SHA")
        if len(self.dictionary_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.dictionary_sha256):
            raise ValueError("task-local evidence Dictionary SHA is invalid")
        index = parse_shortlist_index(json.loads(canonical_json_bytes(dict(self.index))))
        if set(index) != {"schema_version", "policy_sha256", "entries", "public_test_accessed"} or index["schema_version"] != 1 or index["policy_sha256"] != self.policy.fingerprint() or index["public_test_accessed"] is not False:
            raise ValueError("task-local evidence index is not canonical")
        entries = index["entries"]
        if type(entries) is not list:
            raise ValueError("task-local evidence index entries must be a list")
        ids = []
        for entry in entries:
            if type(entry) is not dict or set(entry) != {"task_id", "task_input_sha256", "shortlist_sha256", "diagnostics_sha256"}:
                raise ValueError("task-local evidence index entry is malformed")
            ids.append(entry["task_id"])
        if ids != sorted(ids) or len(ids) != len(set(ids)) or set(ids) != set(self.by_task):
            raise ValueError("task-local evidence index task IDs are not canonical")
        for entry in entries:
            shortlist, diagnostics, shortlist_sha, diagnostics_sha = self.by_task[entry["task_id"]]
            diagnostics = _parse_diagnostics_payload(diagnostics, shortlist, entry["task_id"])
            if (type(shortlist) is not TaskCandidateShortlistV1 or shortlist.policy_sha256 != self.policy.fingerprint()
                    or shortlist.dictionary_sha256 != self.dictionary_sha256
                    or shortlist.fingerprint() != shortlist_sha
                    or hashlib.sha256(canonical_json_bytes(dict(diagnostics))).hexdigest() != diagnostics_sha
                    or (entry["task_input_sha256"], entry["shortlist_sha256"], entry["diagnostics_sha256"])
                    != (shortlist.task_input_sha256, shortlist_sha, diagnostics_sha)):
                raise ValueError("task-local evidence task binding mismatch")
        object.__setattr__(self, "index", _freeze_json(index))
        object.__setattr__(self, "by_task", MappingProxyType({task_id: (shortlist, _freeze_json(_parse_diagnostics_payload(diagnostics, shortlist, task_id)), shortlist_sha, diagnostics_sha)
            for task_id, (shortlist, diagnostics, shortlist_sha, diagnostics_sha) in self.by_task.items()}))


def _freeze_json(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_diagnostic_value(value, annotation):
    """Validate the exact dataclass JSON schema, including typed infinities."""
    if get_origin(annotation) is types.UnionType:
        for option in get_args(annotation):
            try:
                _validate_diagnostic_value(value, option)
                return
            except ValueError:
                pass
        raise ValueError("invalid optional diagnostic field")
    if annotation in (CandidateDiagnostics, HindcastFold):
        if type(value) is not dict or set(value) != {field.name for field in fields(annotation)}:
            raise ValueError("diagnostic fields are malformed")
        for name, field_type in get_type_hints(annotation).items():
            _validate_diagnostic_value(value[name], field_type)
    elif get_origin(annotation) is tuple:
        if type(value) is not list:
            raise ValueError("diagnostic vector must be a list")
        for item in value:
            _validate_diagnostic_value(item, get_args(annotation)[0])
    elif annotation is float:
        if value in ({"status": "positive_infinity", "value": None}, {"status": "negative_infinity", "value": None}):
            return
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("diagnostic numeric field is malformed")
    elif type(value) is not annotation:
        raise ValueError("diagnostic primitive type is malformed")


def parse_diagnostics_payload(payload):
    value = payload
    if type(value) is not dict or set(value) != {"schema_version", "task_id", "task_input_sha256", "rows", "public_test_accessed"} or type(value["schema_version"]) is not int or value["schema_version"] != 1 or type(value["task_id"]) is not str or not value["task_id"].strip() or value["public_test_accessed"] is not False or type(value["rows"]) is not list:
        raise ValueError("task-local diagnostics payload is malformed")
    _evidence_sha(value["task_input_sha256"])
    names = []
    for row in value["rows"]:
        if type(row) is not dict or set(row) != {"candidate_name", "failure_reason", "diagnostic"} or type(row["candidate_name"]) is not str or not row["candidate_name"].strip() or row["failure_reason"] is not None and type(row["failure_reason"]) is not str:
            raise ValueError("task-local diagnostics row is malformed")
        if row["diagnostic"] is not None:
            _validate_diagnostic_value(row["diagnostic"], CandidateDiagnostics)
            if row["diagnostic"]["name"] != row["candidate_name"]:
                raise ValueError("diagnostic candidate identity mismatch")
        names.append(row["candidate_name"])
    if names != sorted(set(names)):
        raise ValueError("task-local diagnostics are not an ordered shortlist subset")
    return value


def _parse_diagnostics_payload(payload: Mapping[str, object], shortlist: TaskCandidateShortlistV1, expected_task_id: str) -> dict[str, object]:
    value = parse_diagnostics_payload(json.loads(canonical_json_bytes(dict(payload))))
    if value["task_id"] != expected_task_id or value["task_input_sha256"] != shortlist.task_input_sha256 or not {row["candidate_name"] for row in value["rows"]} <= set(shortlist.candidate_names):
        raise ValueError("task-local diagnostics shortlist binding mismatch")
    return value


def load_task_local_evidence_bundle(output: Path, *, dictionary_sha256: str) -> TaskLocalEvidenceBundleV1:
    """Load the immutable Task 4 artifacts without materializing any forecast."""
    try:
        def canonical(path: Path):
            raw = path.read_bytes(); payload = json.loads(raw)
            if canonical_json_bytes(payload) != raw:
                raise ValueError("task-local evidence bytes are noncanonical")
            return payload
        index = canonical(output / "task_shortlist_index.json")
        policy = TaskShortlistPolicyV1.from_payload(canonical(output / "task_shortlist_policy.json"))
        by_task = {}
        for entry in index["entries"]:
            shortlist = TaskCandidateShortlistV1.from_payload(canonical(output / "task_shortlists" / f"{entry['task_input_sha256']}.json"))
            diagnostics = canonical(output / "task_diagnostics" / f"{entry['task_input_sha256']}.json")
            by_task[entry["task_id"]] = (shortlist, diagnostics, entry["shortlist_sha256"], entry["diagnostics_sha256"])
        return TaskLocalEvidenceBundleV1(policy, dictionary_sha256, index, by_task)
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError("task-local evidence bundle is malformed") from error


def _load_candidate_priors_bundle(
    path: Path, *, grouping_fingerprint: str, dictionary_sha256: str,
    families: dict[str, str], fold_count: int,
) -> tuple[dict[int, tuple[CandidatePriorV1, ...]], tuple[CandidatePriorV1, ...]]:
    """Read the closed Task 3 prior artifact without accepting permissive variants."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("candidate priors artifact is malformed") from error
    expected = {"schema_version", "grouping_fingerprint", "dictionary_sha256", "fold_priors", "final_priors", "payload_fingerprint"}
    if type(payload) is not dict or set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("candidate priors artifact is malformed")
    unsigned = {key: value for key, value in payload.items() if key != "payload_fingerprint"}
    fingerprint = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if payload["payload_fingerprint"] != fingerprint or payload["grouping_fingerprint"] != grouping_fingerprint or payload["dictionary_sha256"] != dictionary_sha256:
        raise ValueError("candidate priors artifact authority mismatch")
    raw_folds = payload["fold_priors"]
    if type(raw_folds) is not dict or set(raw_folds) != {str(index) for index in range(fold_count)} or type(payload["final_priors"]) is not list:
        raise ValueError("candidate priors artifact fold schema is malformed")
    try:
        folds = {int(key): tuple(CandidatePriorV1.from_payload(item) for item in raw_folds[key]) for key in sorted(raw_folds, key=int)}
        final = tuple(CandidatePriorV1.from_payload(item) for item in payload["final_priors"])
    except (TypeError, ValueError) as error:
        raise ValueError("candidate priors artifact values are malformed") from error
    expected_names = tuple(sorted(families))
    for values in (*folds.values(), final):
        if tuple(item.candidate_name for item in values) != expected_names or any(families[item.candidate_name] != item.family for item in values):
            raise ValueError("candidate priors artifact namespace mismatch")
    return folds, final


def _report_payload(report: ConditionalUpliftReport) -> dict[str, object]:
    return asdict(report)


def _smoke_profile(task_id: str, *, periodic: bool):
    from .evolution.screening import TaskProfile

    return TaskProfile(
        task_id=task_id,
        frequency="D",
        history_length=4,
        horizon=2,
        zero_fraction=0.0,
        signed=False,
        integer_valued=True,
        trend_direction="flat",
        trend_strength=0.1,
        periodicity_periods=(2,) if periodic else (),
        periodicity_strength=0.8 if periodic else 0.1,
        periodicity_confidence=0.9 if periodic else 0.1,
        outlier_fraction=0.0,
        noise_relative_scale=0.1,
        likely_stationary=True,
        stationarity_score=0.9,
        recent_regime_start=None,
        recent_regime_confidence=0.1,
        intermittency_adi=1.0,
        intermittency_cv2=0.1,
    )


def _smoke_rows(
    task_ids: Iterable[str],
    *,
    split: str,
    specialist_regression: bool = False,
) -> tuple[TaskLocalTaskRow, ...]:
    from .evolution.numerical_selector import CandidateDiagnostics

    rows: list[TaskLocalTaskRow] = []
    for index, task_id in enumerate(task_ids):
        profile = _smoke_profile(task_id, periodic=bool(index % 2))
        truth = (10.0, 10.0)
        history = (7.0, 8.0, 9.0, 10.0)
        for name, family, full, folds in (
            ("toto_2_0", "tsfm", (8.0, 8.0), ((8.0, 8.0),) * 5),
            (
                "seasonal_naive",
                "statistical",
                (0.0, 0.0) if specialist_regression else (10.0, 10.0),
                ((10.0, 10.0),) * 5,
            ),
        ):
            diagnostic = CandidateDiagnostics.synthetic(
                name=name,
                family=family,
                median_mase=1.0,
                fold_forecasts=folds,
                fold_truths=(truth,) * 5,
                median_smae=1.0,
                median_srmse=1.0,
            )
            rows.append(
                TaskLocalTaskRow(
                    task_id=task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=history,
                    truth=truth,
                    forecast=full,
                    diagnostic=diagnostic,
                    split=split,
                )
            )
    return tuple(rows)


def _run_smoke(output: Path, *, dev_regression: bool) -> int:
    train_ids = tuple(f"train_{index}" for index in range(8))
    dev_ids = ("dev_periodic", "dev_dense")
    tasks = tuple(
        DataTask(
            task_id=task_id,
            history_values=(float(index + 1), 2.0, 3.0, 4.0),
            future_values=(10.0, 10.0),
            prediction_length=2,
            frequency="D",
            seasonal_period="2",
            entity_name=f"Entity {index}",
        )
        for index, task_id in enumerate(train_ids)
    )
    policy = TaskLocalTournamentPolicy(
        minimum_activation_support=2,
        minimum_activation_groups=2,
    )
    manifest = build_group_fold_manifest(tasks, seed=17)
    release, oof = fit_oof_release(
        _smoke_rows(train_ids, split="train"),
        manifest,
        anchor_release_sha256="a" * 64,
        anchor_name="toto_2_0",
        source_hashes=(("dictionary", "b" * 64),),
        policy=policy,
        confidence_policy=ConfidencePolicy(
            exact_minimum_support=2,
            coarse_minimum_support=2,
            global_minimum_support=2,
        ),
    )
    _write_once(output / "group_folds.json", manifest.to_payload())
    _write_once(output / "oof_report.json", _report_payload(oof))
    if not oof.accepted:
        _write_once(
            output / "evaluation_complete.json",
            {
                "schema_version": 1,
                "status": "oof_rejected",
                "train_oof_tasks": 8,
                "dev_tasks": 0,
            },
        )
        return 0
    dev = evaluate_task_local_release(
        release,
        _smoke_rows(
            dev_ids,
            split="dev",
            specialist_regression=dev_regression,
        ),
        task_ids=dev_ids,
        split="dev",
    )
    _write_once(output / "dev_report.json", _report_payload(dev))
    status = "accepted" if dev.accepted else "dev_rejected"
    if dev.accepted:
        _write_once(output / "task_local_release.json", canonical_task_local_release_bytes(release))
    _write_once(
        output / "evaluation_complete.json",
        {
            "schema_version": 1,
            "status": status,
            "train_oof_tasks": 8,
            "dev_tasks": 2,
            "release_fingerprint": task_local_fingerprint(release),
        },
    )
    return 0


def _reviewed_candidates(
    module: object,
    portfolio: object,
    screening: ScreeningPolicy,
) -> tuple[tuple[str, str], ...]:
    method_names = tuple(item.name for item in module.methods)  # type: ignore[attr-defined]
    runtime = [(name, "statistical") for name in method_names]
    runtime.extend(
        (
            item.name,
            "tsfm" if item in portfolio.tsfm else "combined",  # type: ignore[attr-defined]
        )
        for item in portfolio.all_policies  # type: ignore[attr-defined]
    )
    entries = {entry.name: entry for entry in screening.entries}
    if set(entries) != {name for name, _family in runtime}:
        raise ValueError("task-local runtime and screening namespaces differ")
    return tuple(
        (name, family)
        for name, family in runtime
        if entries[name].status in {"keep", "specialized"}
    )


def _materialize_rows(
    store: ForecastStore,
    tasks: Iterable[DataTask],
    candidates: tuple[tuple[str, str], ...],
    screening: ScreeningPolicy,
    *,
    split: str,
    hindcast_config: HindcastConfig,
) -> tuple[TaskLocalTaskRow, ...]:
    rows: list[TaskLocalTaskRow] = []
    for source in tasks:
        task = RuntimeTask(
            source.task_id,
            tuple(source.history_values),
            source.prediction_length,
            source.frequency,
            tuple(source.future_values),
        )
        profile = profile_task(task)
        active = {
            item.name for item in materialize_active_dictionary(screening, profile).active
        }
        for name, family in candidates:
            forecast = None
            failure: str | None = None
            diagnostic = None
            if name not in active and name != "toto_2_0":
                failure = "NotApplicable: screening_policy"
            else:
                try:
                    forecast = store.forecast(name, task.history, task.horizon, task.frequency)
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"[:10000]
                try:
                    task_hindcast_config = _adaptive_hindcast_config(
                        task, hindcast_config
                    )
                    diagnostic = diagnose_candidate(
                        task,
                        name,
                        family,
                        store.forecast,
                        task_hindcast_config,
                        runtime_settings={"forecast_store": store.identity_hash},
                    )
                except Exception as error:
                    diagnostic = None
            rows.append(
                TaskLocalTaskRow(
                    task_id=task.task_id,
                    candidate_name=name,
                    family=family,
                    profile=profile,
                    history=tuple(task.history),
                    truth=tuple(task.future),
                    forecast=forecast,
                    diagnostic=diagnostic,
                    split=split,
                    failure_reason=failure,
                )
            )
    return tuple(rows)


def materialize_task_shortlist_rows(
    store: ForecastStore,
    task: RuntimeTask,
    shortlist: TaskCandidateShortlistV1,
    families: dict[str, str],
    *,
    split: str,
    hindcast_config: HindcastConfig,
) -> tuple[TaskLocalTaskRow, ...]:
    """Phase B only: materialize the closed, history-only shortlist in order."""
    if type(shortlist) is not TaskCandidateShortlistV1:
        raise TypeError("shortlist materialization requires an exact shortlist")
    profile = profile_task(task)
    rows: list[TaskLocalTaskRow] = []
    for name in shortlist.candidate_names:
        family = families.get(name)
        if family is None:
            raise ValueError(f"shortlisted candidate has no family: {name}")
        forecast = None
        diagnostic = None
        failure: str | None = None
        try:
            forecast = store.forecast(name, task.history, task.horizon, task.frequency)
        except Exception as error:
            failure = f"shortlisted_runtime_failure: {type(error).__name__}: {error}"[:10000]
        try:
            diagnostic = diagnose_candidate(
                task, name, family, store.forecast,
                _adaptive_hindcast_config(task, hindcast_config),
                runtime_settings={"forecast_store": store.identity_hash},
            )
        except Exception as error:
            diagnostic = None
        rows.append(TaskLocalTaskRow(
            task_id=task.task_id, candidate_name=name, family=family, profile=profile,
            history=tuple(task.history), truth=tuple(task.future), forecast=forecast,
            diagnostic=diagnostic, split=split, failure_reason=failure,
        ))
    return tuple(rows)


def _shortlist_rows_for_tasks(
    store: ForecastStore,
    tasks: Iterable[DataTask],
    *,
    dictionary: object,
    screening: ScreeningPolicy,
    families: dict[str, str],
    priors: tuple[object, ...],
    anchor_name: str,
    policy: TaskShortlistPolicyV1,
    split: str,
    hindcast_config: HindcastConfig,
    output: Path | None = None,
) -> tuple[tuple[TaskLocalTaskRow, ...], tuple[TaskCandidateShortlistV1, ...]]:
    """Close each history-only shortlist before touching its forecast runtime."""
    from .evolution.filtering import FilterDictionary
    from .evolution.task_shortlist import CandidatePriorV1
    if type(dictionary) is not FilterDictionary or any(type(item) is not CandidatePriorV1 for item in priors):
        raise TypeError("shortlist phase requires exact dictionary and candidate priors")
    rows: list[TaskLocalTaskRow] = []
    shortlists: list[TaskCandidateShortlistV1] = []
    for source in tasks:
        task = RuntimeTask(source.task_id, tuple(source.history_values), source.prediction_length,
                           source.frequency, tuple(source.future_values))
        shortlist = build_task_candidate_shortlist(
            dictionary=dictionary, profile=profile_task(task), task_input_sha256=_task_input_sha(task),
            anchor_name=anchor_name, available_names=tuple(families), priors=priors,
            policy=policy, screening=screening,
        )
        if len(shortlist.candidate_names) > TaskLocalTournamentPolicy(anchor_name=anchor_name).maximum_candidates:
            raise ValueError("shortlist exceeds the current tournament candidate limit")
        if output is not None:
            _write_once(output / "task_shortlists" / f"{shortlist.task_input_sha256}.json", shortlist.canonical_bytes())
        shortlists.append(shortlist)
        rows.extend(materialize_task_shortlist_rows(store, task, shortlist, families, split=split, hindcast_config=hindcast_config))
    return tuple(rows), tuple(shortlists)


def _v3_oof_rows(
    store: ForecastStore, tasks: tuple[DataTask, ...], *, manifest: object,
    dictionary: object, screening: ScreeningPolicy, families: dict[str, str],
    anchor_name: str, policy: TaskShortlistPolicyV1, hindcast_config: HindcastConfig,
    output: Path, fold_priors: dict[int, tuple[CandidatePriorV1, ...]],
) -> tuple[tuple[TaskLocalTaskRow, ...], dict[str, TaskCandidateShortlistV1]]:
    """Use the exact Task 3 complement-prior tuple for each held-out fold."""
    from .evolution.task_local_evolution import GroupFoldManifest
    if type(manifest) is not GroupFoldManifest:
        raise TypeError("V3 OOF requires an exact group-fold manifest")
    by_id = {task.task_id: task for task in tasks}
    rows: list[TaskLocalTaskRow] = []
    shortlists: dict[str, TaskCandidateShortlistV1] = {}
    folds = dict(manifest.task_fold_map)
    for fold in range(manifest.fold_count):
        held_out = tuple(by_id[task_id] for task_id in sorted(folds) if folds[task_id] == fold)
        fold_rows, fold_shortlists = _shortlist_rows_for_tasks(
            store, held_out, dictionary=dictionary, screening=screening, families=families,
            priors=fold_priors[fold], anchor_name=anchor_name, policy=policy,
            split="train", hindcast_config=hindcast_config, output=output,
        )
        rows.extend(fold_rows)
        shortlists.update({task.task_id: shortlist for task, shortlist in zip(held_out, fold_shortlists, strict=True)})
    return tuple(rows), shortlists


def _shortlist_index(tasks: Iterable[DataTask], shortlists: Mapping[str, TaskCandidateShortlistV1], rows: Iterable[TaskLocalTaskRow], *, policy: TaskShortlistPolicyV1 | None = None) -> dict[str, object]:
    rows_by_task: dict[str, list[TaskLocalTaskRow]] = {}
    for row in rows:
        rows_by_task.setdefault(row.task_id, []).append(row)
    entries = []
    for task in tasks:
        shortlist = shortlists[task.task_id]
        payload = _diagnostics_payload(task.task_id, shortlist.task_input_sha256, rows_by_task.get(task.task_id, []))
        entries.append({"task_id": task.task_id, "task_input_sha256": shortlist.task_input_sha256,
            "shortlist_sha256": shortlist.fingerprint(), "diagnostics_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest()})
    entries.sort(key=lambda item: (item["task_id"], item["task_input_sha256"]))
    policy = policy or TaskShortlistPolicyV1()
    return {"schema_version": 1, "policy_sha256": policy.fingerprint(), "entries": entries,
            "public_test_accessed": False}


def _diagnostics_payload(task_id: str, task_input_sha256: str, rows: Iterable[TaskLocalTaskRow]) -> dict[str, object]:
    values = [{"candidate_name": row.candidate_name, "failure_reason": row.failure_reason,
               "diagnostic": asdict(row.diagnostic) if row.diagnostic is not None else None}
              for row in rows]
    values.sort(key=lambda row: row["candidate_name"])
    return {"schema_version": 1, "task_id": task_id, "task_input_sha256": task_input_sha256,
            "rows": values, "public_test_accessed": False}


def _bind_shortlist_index(output: Path, run_manifest: dict[str, object], shortlist_index: dict[str, object], *, rows: Iterable[TaskLocalTaskRow] = ()) -> None:
    policy = TaskShortlistPolicyV1()
    if shortlist_index.get("policy_sha256") != policy.fingerprint():
        raise ValueError("shortlist index policy mismatch")
    _write_once(output / "task_shortlist_policy.json", policy.canonical_bytes())
    by_task: dict[str, list[TaskLocalTaskRow]] = {}
    for row in rows:
        by_task.setdefault(row.task_id, []).append(row)
    for entry in shortlist_index["entries"]:
        payload = _diagnostics_payload(entry["task_id"], entry["task_input_sha256"], by_task.get(entry["task_id"], []))
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != entry["diagnostics_sha256"]:
            raise ValueError("shortlist diagnostics index mismatch")
        _write_once(output / "task_diagnostics" / f"{entry['task_input_sha256']}.json", payload)
    _write_once(output / "task_shortlist_index.json", shortlist_index)
    run_manifest["shortlist_index_fingerprint"] = hashlib.sha256(canonical_json_bytes(shortlist_index)).hexdigest()
    _write_once(output / "run_manifest.json", run_manifest)


def _formal_main(args: argparse.Namespace, output: Path) -> int:
    required = {
        "repo": args.repo,
        "split_file": args.split_file,
        "tasks_file": args.tasks_file,
        "anchor_release_dir": args.anchor_release_dir,
        "forecast_store": args.forecast_store,
        "candidate_priors_file": args.candidate_priors_file,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise ValueError(f"formal task-local evolution is missing {missing}")
    repo = Path(args.repo).resolve()
    _clean_git_source(repo)
    train_ids, dev_ids = _partition_ids(args.split_file, include_dev=True)
    if len(train_ids) != 80 or len(dev_ids) != 20:
        raise ValueError("formal task-local evolution requires exactly 80 Train and 20 Dev")
    train_by_id = {task.task_id: task for task in load_tasks_by_id(args.tasks_file, train_ids)}
    if set(train_by_id) != set(train_ids):
        raise ValueError("formal task-local evolution is missing Train tasks")
    train = tuple(train_by_id[task_id] for task_id in train_ids)
    source_files = _source_files(repo)
    source_hashes = tuple((name, _sha256(path)) for name, path in source_files)
    anchor_path = Path(args.anchor_release_dir) / "champion_release.json"
    anchor = _load_parent(anchor_path)
    module = read_module(repo / "methods.py")
    portfolio = read_policy_file(repo / "policies.py")
    portfolio.validate_namespace(module.names())
    screening = _load_screening_policy(repo / "dictionary.py")
    dictionary = parse_filter_source((repo / "dictionary.py").read_text(encoding="utf-8"))
    candidates = _reviewed_candidates(module, portfolio, screening)
    dictionary_sha256 = hashlib.sha256(canonical_json_bytes({"entries": [
        {"name": item.name, "family": item.family, "status": item.status,
         "applicability": list(item.applicability), "reason": item.reason}
        for item in dictionary.entries
    ]})).hexdigest()
    runtimes = _runtime_registry(args)
    store: ForecastStore | None = None
    try:
        store = ForecastStore(
            Path(args.forecast_store),
            repo / "methods.py",
            repo / "skills.py" if (repo / "skills.py").is_file() else None,
            portfolio,
            runtimes,
            screening_hash=screening.fingerprint(),
            runtime_identity=_forecast_runtime_identity(args),
        )
        manifest = build_group_fold_manifest(
            train, seed=int(args.partition_seed)
        )
        families = dict(candidates)
        shortlist_policy = TaskShortlistPolicyV1()
        tournament_policy = TaskLocalTournamentPolicy(anchor_name=anchor.policy.recipe.fallback_parent)
        fold_priors, priors = _load_candidate_priors_bundle(
            Path(args.candidate_priors_file), grouping_fingerprint=manifest.grouping_fingerprint,
            dictionary_sha256=dictionary_sha256, families=families, fold_count=manifest.fold_count,
        )
        train_shortlist_rows, train_shortlists = _v3_oof_rows(
            store, train, manifest=manifest, dictionary=dictionary, screening=screening,
            families=families, anchor_name=anchor.policy.recipe.fallback_parent,
            policy=shortlist_policy, hindcast_config=_CONFIDENCE_HINDCAST_CONFIG, output=output,
            fold_priors=fold_priors,
        )
        provisional = TaskLocalEnsembleReleaseV3(
            schema_version=3,
            anchor_release_sha256=champion_fingerprint(anchor),
            anchor_name=anchor.policy.recipe.fallback_parent,
            tournament_policy=tournament_policy,
            shortlist_policy=shortlist_policy,
            candidate_priors=priors,
            dictionary_sha256=dictionary_sha256,
            grouping_fingerprint=manifest.grouping_fingerprint,
            oof_report_sha256="0" * 64,
            source_hashes=source_hashes,
            metric_policy_fingerprint=METRIC_POLICY_FINGERPRINT,
            lineage=("task_local_shortlist_v3",),
            confidence_evidence=None,
        )
        oof = evaluate_task_local_release(provisional, train_shortlist_rows, task_ids=train_ids, split="oof")
        release = TaskLocalEnsembleReleaseV3(
            **{**provisional.__dict__, "oof_report_sha256": oof.report_fingerprint}
        )
        run_manifest = {
            "schema_version": 2,
            "split_sha256": _sha256(Path(args.split_file)),
            "source_hashes": dict(source_hashes),
            "anchor_release_sha256": champion_fingerprint(anchor),
            "forecast_store_fingerprint": store.identity_hash,
            "grouping_fingerprint": manifest.grouping_fingerprint,
            "tournament_policy_fingerprint": task_local_fingerprint(release.tournament_policy),
            "shortlist_policy_fingerprint": release.shortlist_policy.fingerprint(),
            "candidate_priors_fingerprint": hashlib.sha256(canonical_json_bytes([item.to_payload() for item in priors])).hexdigest(),
            "dictionary_sha256": release.dictionary_sha256,
            "hindcast_config_fingerprint": task_local_fingerprint(
                {
                    "mode": "adaptive_three_to_five_origins",
                    "maximum": asdict(_CONFIDENCE_HINDCAST_CONFIG),
                }
            ),
            "train_task_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
            "dev_task_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode()).hexdigest(),
        }
        _write_once(output / "group_folds.json", manifest.to_payload())
        _write_once(output / "oof_report.json", _report_payload(oof))
        if not oof.accepted:
            shortlist_index = _shortlist_index(train, train_shortlists, train_shortlist_rows)
            _bind_shortlist_index(output, run_manifest, shortlist_index, rows=train_shortlist_rows)
            _write_once(
                output / "evaluation_complete.json",
                {
                    "schema_version": 1,
                    "status": "oof_rejected",
                    "train_oof_tasks": 80,
                    "dev_tasks": 0,
                },
            )
            return 0

        # Dev task bodies are deliberately loaded only after frozen OOF acceptance.
        dev_by_id = {task.task_id: task for task in load_tasks_by_id(args.tasks_file, dev_ids)}
        if set(dev_by_id) != set(dev_ids):
            raise ValueError("formal task-local evolution is missing Dev tasks")
        dev = tuple(dev_by_id[task_id] for task_id in dev_ids)
        dev_rows, shortlists = _shortlist_rows_for_tasks(
            store, dev, dictionary=dictionary, screening=screening, families=families,
            priors=priors, anchor_name=release.anchor_name, policy=release.shortlist_policy,
            split="dev", hindcast_config=_CONFIDENCE_HINDCAST_CONFIG, output=output,
        )
        all_shortlists = dict(train_shortlists)
        all_shortlists.update({task.task_id: shortlist for task, shortlist in zip(dev, shortlists, strict=True)})
        shortlist_index = _shortlist_index(train + dev, all_shortlists, train_shortlist_rows + dev_rows)
        _bind_shortlist_index(output, run_manifest, shortlist_index, rows=train_shortlist_rows + dev_rows)
        dev_report = evaluate_task_local_release(
            release, dev_rows, task_ids=dev_ids, split="dev"
        )
        _write_once(output / "dev_report.json", _report_payload(dev_report))
        status = "accepted" if dev_report.accepted else "dev_rejected"
        if dev_report.accepted:
            _write_once(
                output / "task_local_release.json",
                canonical_task_local_release_bytes(release),
            )
        _write_once(
            output / "evaluation_complete.json",
            {
                "schema_version": 1,
                "status": status,
                "train_oof_tasks": 80,
                "dev_tasks": 20,
                "release_fingerprint": task_local_fingerprint(release),
            },
        )
        return 0
    finally:
        if store is not None:
            store.close()
        runtimes.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if (output / "evaluation_complete.json").exists():
        raise ValueError("task-local evolution has already completed")
    if args.smoke:
        return _run_smoke(output, dev_regression=bool(args.smoke_dev_regression))
    if args.smoke_dev_regression:
        raise ValueError("--smoke-dev-regression requires --smoke")
    return _formal_main(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
