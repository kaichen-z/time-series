"""Executable lifecycle boundaries for task-local Numerical evolution."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest

from common.payload import read_json_object
from numerical_agent.evaluate_frozen_task_local_ensemble import main as public_main
from numerical_agent.evolution.task_local_ensemble import parse_task_local_release
from numerical_agent.run_task_local_ensemble_evolution import main
from numerical_agent.run_task_local_ensemble_evolution import (
    _CONFIDENCE_HINDCAST_CONFIG,
    _adaptive_hindcast_config,
    materialize_task_shortlist_rows,
    _load_candidate_priors_bundle,
)
from numerical_agent.evolution.execution import Task as RuntimeTask
from numerical_agent.evolution.numerical_selector import CandidateDiagnostics
from numerical_agent.evolution.task_shortlist import TaskCandidateShortlistV1
from numerical_agent.evolution.task_shortlist import CandidatePriorV1, TaskShortlistPolicyV1
from numerical_agent.evolution.filtering import FilterDictionary, FilterEntry
from numerical_agent.evolution.screening import ApplicabilityPolicy, ScreeningEntry, ScreeningPolicy
from numerical_agent.evolution.task_local_evolution import GroupFoldManifest
from numerical_agent.evolution.task_local_ensemble import (
    TaskLocalTournamentPolicy,
    execute_task_local_ensemble,
)
from common.data import Task as DataTask


def test_smoke_runs_eight_train_two_dev_and_freezes_after_acceptance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "accepted"

    assert main(["--smoke", "--output-dir", str(output)]) == 0

    complete = read_json_object(output / "evaluation_complete.json")
    assert complete["train_oof_tasks"] == 8
    assert complete["dev_tasks"] == 2
    assert complete["status"] == "accepted"
    assert (output / "task_local_release.json").is_file()
    assert (output / "oof_report.json").is_file()
    assert (output / "dev_report.json").is_file()
    release = parse_task_local_release(
        read_json_object(output / "task_local_release.json")
    )
    assert release.schema_version == 2
    assert release.confidence_evidence is not None


def test_confidence_hindcast_uses_three_origins_for_long_horizon_history() -> None:
    task = RuntimeTask(
        "long_horizon", tuple(float(i) for i in range(112)), 168, "D", ()
    )

    config = _adaptive_hindcast_config(task, _CONFIDENCE_HINDCAST_CONFIG)

    assert config.folds == 3
    assert config.min_successful_folds == 3


def test_shortlist_materializer_never_executes_excluded_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    class Store:
        identity_hash = "store"
        def forecast(self, name: str, history: tuple[float, ...], horizon: int, frequency: str) -> tuple[float, ...]:
            calls.append(name)
            if name == "bad":
                raise RuntimeError("boom")
            return (1.0,) * horizon
    def diagnose(task: RuntimeTask, name: str, family: str, forecast, config, **_kwargs):
        calls.append(f"diagnose:{name}")
        return CandidateDiagnostics.synthetic(name=name, family=family, median_mase=1.0,
            fold_forecasts=((1.0,),) * 3, fold_truths=((1.0,),) * 3,
            median_smae=1.0, median_srmse=1.0)
    monkeypatch.setattr("numerical_agent.run_task_local_ensemble_evolution.diagnose_candidate", diagnose)
    selected = ("toto_2_0", *(f"good{index}" for index in range(7)))
    excluded = tuple((name, "ranked_out") for name in sorted(f"method{index}" for index in range(8, 20)))
    shortlist = TaskCandidateShortlistV1(1, "a" * 64, "b" * 64, "c" * 64,
        selected, excluded, False, False)
    rows = materialize_task_shortlist_rows(Store(), RuntimeTask("task", (1.0, 2.0, 3.0, 4.0), 1, "D", (5.0,)), shortlist,
        {"toto_2_0": "tsfm", **{name: "statistical" for name in selected[1:]}}, split="dev",
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG)
    assert calls == [item for name in selected for item in (name, f"diagnose:{name}")]
    assert {row.candidate_name for row in rows} == set(selected)
    assert not {name for name in calls if name.startswith("method")}


def test_shortlist_materializer_checks_deadline_between_candidates() -> None:
    calls: list[str] = []

    class Store:
        identity_hash = "store"

        def forecast(self, name, _history, horizon, _frequency):
            calls.append(name)
            return (1.0,) * horizon

    shortlist = TaskCandidateShortlistV1(
        1, "a" * 64, "b" * 64, "c" * 64,
        ("toto_2_0", "naive_last"), (), True, False,
    )
    checks = 0

    def check_deadline():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RuntimeError("deadline")

    with pytest.raises(RuntimeError, match="deadline"):
        materialize_task_shortlist_rows(
            Store(),
            RuntimeTask("task", (1.0, 2.0, 3.0, 4.0), 1, "D", (5.0,)),
            shortlist,
            {"toto_2_0": "tsfm", "naive_last": "statistical"},
            split="dev",
            hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
            check_deadline=check_deadline,
        )

    assert calls and set(calls) == {"toto_2_0"}


def test_forecast_exception_materializes_failure_rows_in_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numerical_agent.run_task_local_ensemble_evolution import _materialize_rows

    class Store:
        identity_hash = "store"

        def forecast(self, name: str, history, horizon: int, frequency: str):
            raise RuntimeError("forecast failed")

    def diagnose(task, name, family, forecast, config, **_kwargs):
        return CandidateDiagnostics.synthetic(
            name=name, family=family, median_mase=1.0,
            fold_forecasts=((1.0,),) * 3, fold_truths=((1.0,),) * 3,
            median_smae=1.0, median_srmse=1.0,
        )

    monkeypatch.setattr("numerical_agent.run_task_local_ensemble_evolution.diagnose_candidate", diagnose)
    shortlist = TaskCandidateShortlistV1(
        1, "a" * 64, "b" * 64, "c" * 64, ("toto_2_0",), (), True, False,
    )
    task = RuntimeTask("task", (1.0, 2.0, 3.0, 4.0), 1, "D", (5.0,))
    shortlisted = materialize_task_shortlist_rows(
        Store(), task, shortlist, {"toto_2_0": "tsfm"}, split="dev",
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
    )
    full = _materialize_rows(
        Store(), (DataTask("task", task.history, task.future, task.horizon, task.frequency, None, "entity"),),
        (("toto_2_0", "tsfm"),),
        ScreeningPolicy((ScreeningEntry("toto_2_0", "tsfm", "keep", ApplicabilityPolicy(), "safe"),), ("toto_2_0",)),
        split="train", hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
    )
    assert shortlisted[0].forecast is None
    assert shortlisted[0].failure_reason == "shortlisted_runtime_failure: RuntimeError: forecast failed"
    assert full[0].forecast is None
    assert full[0].failure_reason == "RuntimeError: forecast failed"


def test_anchor_diagnostic_failure_preserves_its_full_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store:
        identity_hash = "store"

        def forecast(self, name: str, history, horizon: int, frequency: str):
            return (1.0,) * horizon

    def diagnose(*_args, **_kwargs):
        raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr("numerical_agent.run_task_local_ensemble_evolution.diagnose_candidate", diagnose)
    shortlist = TaskCandidateShortlistV1(
        1, "a" * 64, "b" * 64, "c" * 64, ("toto_2_0",), (), True, False,
    )
    row = materialize_task_shortlist_rows(
        Store(), RuntimeTask("task", (1.0, 2.0, 3.0, 4.0), 1, "D", (5.0,)),
        shortlist, {"toto_2_0": "tsfm"}, split="dev",
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
    )[0]
    assert row.forecast == (1.0,)
    assert row.diagnostic is None
    assert row.failure_reason is None


def test_diagnostic_failure_preserves_forecast_and_tournament_records_anchor_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store:
        identity_hash = "store"

        def forecast(self, name: str, history, horizon: int, frequency: str):
            return (1.0,) * horizon if name == "toto_2_0" else (9.0,) * horizon

    def diagnose(task, name, family, forecast, config, **_kwargs):
        if name == "specialist":
            raise RuntimeError("diagnostics unavailable")
        return CandidateDiagnostics.synthetic(
            name=name, family=family, median_mase=1.0,
            fold_forecasts=((1.0,),) * 3, fold_truths=((1.0,),) * 3,
            median_smae=1.0, median_srmse=1.0,
        )

    monkeypatch.setattr("numerical_agent.run_task_local_ensemble_evolution.diagnose_candidate", diagnose)
    shortlist = TaskCandidateShortlistV1(
        1, "a" * 64, "b" * 64, "c" * 64,
        ("toto_2_0", "specialist"), (), True, False,
    )
    rows = materialize_task_shortlist_rows(
        Store(), RuntimeTask("task", (1.0, 2.0, 3.0, 4.0), 1, "D", (5.0,)),
        shortlist, {"toto_2_0": "tsfm", "specialist": "statistical"}, split="dev",
        hindcast_config=_CONFIDENCE_HINDCAST_CONFIG,
    )
    by_name = {row.candidate_name: row for row in rows}
    assert by_name["specialist"].forecast == (9.0,)
    assert by_name["specialist"].diagnostic is None
    assert by_name["specialist"].failure_reason is None
    result = execute_task_local_ensemble(
        TaskLocalTournamentPolicy(), candidate_names=shortlist.candidate_names,
        forecasts={name: row.forecast for name, row in by_name.items() if row.forecast is not None},
        diagnostics={name: row.diagnostic for name, row in by_name.items() if row.diagnostic is not None},
        horizon=1,
    )
    assert result.forecast == (1.0,)
    assert result.selected_names == ("toto_2_0",)
    assert result.activated is False
    assert result.fallback_reason == "no_eligible_specialist"


def test_prior_bundle_requires_exact_complement_and_final_prior_authority(tmp_path: Path) -> None:
    prior = {"candidate_name": "toto_2_0", "family": "tsfm", "success_rate": 1.0,
             "mean_joint": 0.1, "p90_joint": 0.1, "morphology_scores": []}
    unsigned = {"schema_version": 1, "grouping_fingerprint": "a" * 64,
                "dictionary_sha256": "b" * 64, "fold_priors": {"0": [prior], "1": [prior]},
                "final_priors": [prior]}
    unsigned["payload_fingerprint"] = __import__("hashlib").sha256(
        __import__("common.payload", fromlist=["canonical_json_bytes"]).canonical_json_bytes(unsigned)
    ).hexdigest()
    path = tmp_path / "priors.json"
    path.write_text(__import__("json").dumps(unsigned), encoding="utf-8")
    folds, final = _load_candidate_priors_bundle(path, grouping_fingerprint="a" * 64,
        dictionary_sha256="b" * 64, families={"toto_2_0": "tsfm"}, fold_count=2)
    assert folds[0] == final
    for mutation, kwargs in (
        ({"payload_fingerprint": "0" * 64}, {}),
        ({}, {"grouping_fingerprint": "c" * 64}),
        ({}, {"dictionary_sha256": "d" * 64}),
        ({"fold_priors": {"0": [prior]}}, {}),
        ({"final_priors": [{**prior, "candidate_name": "other"}]}, {}),
    ):
        broken = {**unsigned, **mutation}
        broken["payload_fingerprint"] = mutation.get("payload_fingerprint", __import__("hashlib").sha256(
            __import__("common.payload", fromlist=["canonical_json_bytes"]).canonical_json_bytes(broken)
        ).hexdigest())
        path.write_text(__import__("json").dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_candidate_priors_bundle(path, grouping_fingerprint=kwargs.get("grouping_fingerprint", "a" * 64),
                dictionary_sha256=kwargs.get("dictionary_sha256", "b" * 64), families={"toto_2_0": "tsfm"}, fold_count=2)


def test_v3_oof_materializes_only_each_fold_prior_shortlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from numerical_agent.run_task_local_ensemble_evolution import _v3_oof_rows, _shortlist_index
    names = ("toto_2_0", *(f"m{index:02d}" for index in range(20)))
    families = {name: "tsfm" if name == "toto_2_0" else "statistical" for name in names}
    dictionary = FilterDictionary(tuple(FilterEntry(name, families[name], "keep", (), "safe") for name in names))
    screening = ScreeningPolicy(tuple(ScreeningEntry(name, families[name], "keep", ApplicabilityPolicy(), "safe") for name in names), ("toto_2_0",))
    implementation = __import__("numerical_agent.evolution.task_local_evolution", fromlist=["_GROUPING_IMPLEMENTATION"])._GROUPING_IMPLEMENTATION
    manifest = GroupFoldManifest(1, 1, 2, (("0" * 64, ("late",), 1), ("1" * 64, ("early",), 0)), implementation)
    tasks = (DataTask("late", tuple(float(i) for i in range(100, 120)), (1.0,), 1, "D", None, "late"),
             DataTask("early", tuple(float(i) for i in range(20)), (1.0,), 1, "D", None, "early"))
    def priors(chosen: tuple[str, ...]) -> tuple[CandidatePriorV1, ...]:
        return tuple(CandidatePriorV1(name, families[name], 1.0, 0.0 if name in chosen else 9.0, 0.0 if name in chosen else 9.0, ()) for name in sorted(names))
    expected = {0: ("toto_2_0", *(f"m{index:02d}" for index in range(7))), 1: ("toto_2_0", *(f"m{index:02d}" for index in range(7, 14)))}
    calls: list[tuple[str, str]] = []
    class Store:
        identity_hash = "store"
        def forecast(self, name, history, horizon, frequency):
            calls.append((history[0], name)); return (1.0,)
    def diagnose(task, name, family, forecast, config, **kwargs):
        calls.append((task.task_id, f"diagnose:{name}"))
        return CandidateDiagnostics.synthetic(name=name, family=family, median_mase=1.0, fold_forecasts=((1.0,),) * 3, fold_truths=((1.0,),) * 3, median_smae=1.0, median_srmse=1.0)
    monkeypatch.setattr("numerical_agent.run_task_local_ensemble_evolution.diagnose_candidate", diagnose)
    _rows, shortlists = _v3_oof_rows(Store(), tasks, manifest=manifest, dictionary=dictionary, screening=screening, families=families, anchor_name="toto_2_0", policy=TaskShortlistPolicyV1(), hindcast_config=_CONFIDENCE_HINDCAST_CONFIG, output=tmp_path, fold_priors={0: priors(expected[0]), 1: priors(expected[1])})
    assert shortlists["early"].candidate_names == expected[0]
    assert shortlists["late"].candidate_names == expected[1]
    assert {name.removeprefix("diagnose:") for task, name in calls if task == "early"} == set(expected[0])
    assert {name.removeprefix("diagnose:") for task, name in calls if task == "late"} == set(expected[1])
    assert {name for marker, name in calls if marker == 0.0} == set(expected[0])
    assert {name for marker, name in calls if marker == 100.0} == set(expected[1])
    index = _shortlist_index(tasks, shortlists, _rows)
    assert [entry["task_id"] for entry in index["entries"]] == ["early", "late"]
    assert {entry["task_id"]: entry["task_input_sha256"] for entry in index["entries"]} == {task_id: shortlists[task_id].task_input_sha256 for task_id in shortlists}


@pytest.mark.parametrize("terminal", ["accepted", "oof_rejected"])
def test_terminal_index_binding_writes_matching_manifest(tmp_path: Path, terminal: str) -> None:
    from numerical_agent.run_task_local_ensemble_evolution import _bind_shortlist_index, _diagnostics_payload
    from numerical_agent.evolution.task_shortlist import TaskShortlistPolicyV1
    payload = _diagnostics_payload("a", "1" * 64, ())
    index = {"schema_version": 1, "policy_sha256": TaskShortlistPolicyV1().fingerprint(), "public_test_accessed": False,
             "entries": [{"task_id": "a", "task_input_sha256": "1" * 64, "shortlist_sha256": "2" * 64,
                          "diagnostics_sha256": __import__("hashlib").sha256(__import__("common.payload", fromlist=["canonical_json_bytes"]).canonical_json_bytes(payload)).hexdigest()}]}
    manifest: dict[str, object] = {"terminal": terminal}
    output = tmp_path / terminal
    _bind_shortlist_index(output, manifest, index)
    assert read_json_object(output / "run_manifest.json")["shortlist_index_fingerprint"] == __import__("hashlib").sha256(__import__("common.payload", fromlist=["canonical_json_bytes"]).canonical_json_bytes(index)).hexdigest()
    assert read_json_object(output / "task_shortlist_index.json") == index


def test_failed_dev_publishes_no_task_local_release(tmp_path: Path) -> None:
    output = tmp_path / "rejected"

    assert main(
        ["--smoke", "--smoke-dev-regression", "--output-dir", str(output)]
    ) == 0

    complete = read_json_object(output / "evaluation_complete.json")
    assert complete["status"] == "dev_rejected"
    assert not (output / "task_local_release.json").exists()


def test_completed_smoke_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "once"
    assert main(["--smoke", "--output-dir", str(output)]) == 0

    with pytest.raises(ValueError, match="already completed"):
        main(["--smoke", "--output-dir", str(output)])


def test_shell_runner_dry_run_names_the_formal_train_dev_command() -> None:
    env = {
        **os.environ,
        "TASK_LOCAL_REPO": "runs/method_evolution/v001",
        "TASK_LOCAL_SPLIT_FILE": "splits/drcik_public_80_20_99_v1.json",
        "TASK_LOCAL_TASKS_FILE": "/tmp/tasks.jsonl",
        "TASK_LOCAL_ANCHOR_RELEASE_DIR": "runs/champion_evolution/latest",
        "TASK_LOCAL_FORECAST_STORE": "runs/champion_forecasts/latest",
        "TASK_LOCAL_CANDIDATE_PRIORS_FILE": "runs/task_local_priors/v001.json",
        "TASK_LOCAL_OUTPUT_DIR": "runs/task_local_ensemble/v001",
    }
    result = subprocess.run(
        ["bash", "scripts/run_task_local_ensemble_evolution.sh", "--dry-run"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "run_task_local_ensemble_evolution" in result.stdout
    assert "--anchor-release-dir runs/champion_evolution/latest" in result.stdout
    assert "--candidate-priors-file runs/task_local_priors/v001.json" in result.stdout


def test_public99_runs_once_after_acceptance_and_compares_toto(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    output = tmp_path / "public"
    assert main(["--smoke", "--output-dir", str(release_dir)]) == 0

    assert public_main(
        ["--smoke", "--release-dir", str(release_dir), "--output-dir", str(output)]
    ) == 0

    report = read_json_object(output / "public_regression_report.json")
    comparison = report["comparison"]
    assert report["public_task_count"] == 99
    assert comparison["baseline"] == "toto_2_0"
    assert {
        "child_mean_smae",
        "child_mean_srmse",
        "child_p90_smae_raw",
        "child_p95_srmse_raw",
        "wins",
        "ties",
        "losses",
    } <= set(comparison)
    assert (output / "evaluation_complete.json").is_file()

    with pytest.raises(ValueError, match="already completed"):
        public_main(
            ["--smoke", "--release-dir", str(release_dir), "--output-dir", str(output)]
        )


def test_public99_rejects_a_dev_rejected_release_before_scoring(tmp_path: Path) -> None:
    release_dir = tmp_path / "rejected"
    assert main(
        ["--smoke", "--smoke-dev-regression", "--output-dir", str(release_dir)]
    ) == 0

    with pytest.raises(ValueError, match="accepted"):
        public_main(
            [
                "--smoke",
                "--release-dir",
                str(release_dir),
                "--output-dir",
                str(tmp_path / "public"),
            ]
        )
