from __future__ import annotations

import copy

import pytest

from evolving_loop.accuracy_profile import (
    BASELINE_PANEL,
    build_accuracy_profile,
    parse_baseline_log,
    task_difficulty_features,
    validate_accuracy_profile,
)


def _scores() -> dict[str, dict[str, float]]:
    values = {
        "task_a": 0.0,
        "task_b": 1.0,
        "task_c": 2.0,
        "task_d": 3.0,
    }
    return {model: dict(values) for model in BASELINE_PANEL}


def _sources() -> dict[str, dict[str, str]]:
    return {
        model: {
            "path": f"runs/baselines/{model}_dev.log",
            "sha256": str(index + 1) * 64,
        }
        for index, model in enumerate(BASELINE_PANEL)
    }


def _profile() -> dict:
    return build_accuracy_profile(
        _scores(),
        source_commit="1" * 40,
        source_files=_sources(),
    )


def test_parse_baseline_log_rejects_duplicate_task_scores() -> None:
    """Catches a restarted log silently overwriting an earlier score for one task."""
    text = "\n".join(
        (
            "[1/2] task_1 H=24 paths=25/25 sMAE=0.125 mean=0.125",
            "[2/2] task_1 H=24 paths=25/25 sMAE=0.250 mean=0.188",
        )
    )

    with pytest.raises(ValueError, match="duplicate task score"):
        parse_baseline_log(text)


def test_profile_is_canonical_under_model_and_task_reordering() -> None:
    """Catches source dictionary order leaking into the frozen profile or its digest."""
    scores = _scores()
    reversed_scores = {
        model: dict(reversed(tuple(scores[model].items())))
        for model in reversed(BASELINE_PANEL)
    }
    reversed_sources = dict(reversed(tuple(_sources().items())))

    first = _profile()
    second = build_accuracy_profile(
        reversed_scores,
        source_commit="1" * 40,
        source_files=reversed_sources,
    )

    assert first == second
    assert first["task_count"] == 4
    assert first["panel_models"] == list(BASELINE_PANEL)


def test_profile_validation_rejects_incomplete_or_invalid_scores() -> None:
    """Catches missing task coverage and invalid capped sMAE entering split selection."""
    incomplete = _scores()
    incomplete[BASELINE_PANEL[0]].pop("task_d")
    with pytest.raises(ValueError, match="task coverage"):
        build_accuracy_profile(
            incomplete,
            source_commit="1" * 40,
            source_files=_sources(),
        )

    invalid = _scores()
    invalid[BASELINE_PANEL[0]]["task_a"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_accuracy_profile(
            invalid,
            source_commit="1" * 40,
            source_files=_sources(),
        )

    boolean = _scores()
    boolean[BASELINE_PANEL[0]]["task_a"] = True
    with pytest.raises(ValueError, match="number"):
        build_accuracy_profile(
            boolean,
            source_commit="1" * 40,
            source_files=_sources(),
        )


def test_profile_digest_detects_score_tampering() -> None:
    """Catches an edited task score retaining the provenance identity of old evidence."""
    changed = copy.deepcopy(_profile())
    changed["tasks"]["task_a"]["smae"][BASELINE_PANEL[0]] = 0.5

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_accuracy_profile(changed)


def test_task_difficulty_features_use_hand_checked_percentile_buckets() -> None:
    """Catches rank direction or boundary errors that would invert difficulty strata."""
    features = task_difficulty_features(_profile())

    assert features["task_a"]["difficulty_score"] == 0.0
    assert features["task_a"]["difficulty_decile"] == "0"
    assert features["task_b"]["difficulty_score"] == pytest.approx(1.0 / 3.0)
    assert features["task_b"]["difficulty_decile"] == "3"
    assert features["task_c"]["difficulty_score"] == pytest.approx(2.0 / 3.0)
    assert features["task_c"]["difficulty_decile"] == "6"
    assert features["task_c"]["model_quintiles"][BASELINE_PANEL[0]] == "3"
    assert features["task_d"]["difficulty_score"] == 1.0
    assert features["task_d"]["difficulty_decile"] == "9"
    assert features["task_d"]["model_quintiles"][BASELINE_PANEL[0]] == "4"
