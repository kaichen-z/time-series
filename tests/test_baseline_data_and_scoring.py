from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.data import DEV, HIDDEN, load_benchmark
from baseline.forecasters import SeasonalNaive, quantile_paths, seasonal_period
from baseline.scoring import WINSOR_CAP, sample_mean, score_task, summarize


def record(benchmark_id: str, labels_public: bool = True, horizon: int = 2) -> dict:
    future = [10.0, 11.0] if labels_public else [None, None]
    return {
        "benchmark_id": benchmark_id,
        "labels_public": labels_public,
        "series": {
            "history_values": [1.0, 2.0, 3.0, 4.0],
            "history_timestamps": ["t1", "t2", "t3", "t4"],
            "future_values": future,
            "future_timestamps": ["t5", "t6"],
        },
        "task_metadata": {
            "prediction_length": horizon,
            "frequency": "1 day",
            "seasonal_period": "7",
        },
    }


def write(tmp_path: Path, records: list[dict]) -> Path:
    destination = tmp_path / "tasks.jsonl"
    destination.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return destination


def test_hidden_tasks_are_loaded_not_dropped(tmp_path: Path) -> None:
    """common.data.load_tasks filters these out; the submission is built from exactly them."""
    path = write(tmp_path, [record("task_1"), record("task_2", labels_public=False)])

    tasks = load_benchmark(path)

    assert [task.benchmark_id for task in tasks] == ["task_1", "task_2"]
    assert [task.split for task in tasks] == [DEV, HIDDEN]


def test_a_withheld_label_reads_as_absent_not_as_a_number(tmp_path: Path) -> None:
    path = write(tmp_path, [record("task_2", labels_public=False)])

    hidden = load_benchmark(path, split=HIDDEN)[0]

    assert hidden.future_values == ()
    assert hidden.prediction_length == 2  # the horizon is still known, only the answer is not
    assert hidden.future_timestamps == ("t5", "t6")


def test_the_task_carries_no_text_that_could_leak_into_a_prompt(tmp_path: Path) -> None:
    """No Context means the forecaster sees numbers; an absent field cannot be prompted with."""
    path = write(tmp_path, [record("task_1")])

    task = load_benchmark(path)[0]

    assert not hasattr(task, "target_description")
    assert not hasattr(task, "documents")


def test_a_nan_gap_in_history_is_interpolated_not_propagated(tmp_path: Path) -> None:
    """A handful of hidden tasks carry NaN gaps; every forecaster would else return garbage."""
    entry = record("task_1")
    entry["series"]["history_values"] = [1.0, None, None, 4.0]
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(entry, allow_nan=True).replace("null", "NaN"), encoding="utf-8")

    task = load_benchmark(path)[0]

    assert task.history_values == (1.0, 2.0, 3.0, 4.0)


def test_labels_that_disagree_with_the_horizon_fail_loudly(tmp_path: Path) -> None:
    path = write(tmp_path, [record("task_1", horizon=5)])

    with pytest.raises(ValueError, match="expected 5"):
        load_benchmark(path)


def test_the_point_forecast_is_the_mean_across_trajectories() -> None:
    assert sample_mean([[1.0, 2.0], [3.0, 6.0]]) == (2.0, 4.0)


def test_a_ragged_ensemble_is_a_failure_not_a_crash() -> None:
    score = score_task("task_1", [1.0, 2.0], [[1.0, 2.0], [1.0]])

    assert score.smae is None
    assert "differ in length" in score.failure


def test_one_wild_task_cannot_dominate_the_average() -> None:
    """A near-zero-magnitude series produces a huge scaled error; the paper winsorizes at 5."""
    score = score_task("task_1", [0.01, 0.01], [[100.0, 100.0]])

    assert score.smae == WINSOR_CAP
    assert score.srmse == WINSOR_CAP


def test_coverage_reports_the_tasks_a_baseline_failed() -> None:
    scores = [
        score_task("task_1", [1.0, 2.0], [[1.0, 2.0]]),
        score_task("task_2", [1.0, 2.0], [[1.0]]),
    ]

    report = summarize("dummy", scores)

    assert report.scored == 1 and report.tasks == 2
    assert report.coverage == 0.5
    assert report.mean_smae == 0.0


def test_a_trajectory_follows_one_quantile_level_across_every_step() -> None:
    """Independent per-step draws would zig-zag across the band and lose the predicted shape."""
    quantiles = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]  # levels 0.1 / 0.5 / 0.9

    paths = quantile_paths(quantiles, [0.1, 0.5, 0.9], samples=20, seed=1)

    assert len(paths) == 20
    for path in paths:
        assert path[0] == path[1]  # same level read at both steps
        assert 0.0 <= path[0] <= 2.0


def test_seasonal_period_prefers_the_task_field_then_the_frequency() -> None:
    assert seasonal_period("1 day", "24") == 24
    assert seasonal_period("1 hour", None) == 24
    assert seasonal_period("1 day", None) == 7
    assert seasonal_period("something odd", None) == 1
    assert seasonal_period("1 day", "1") == 7  # a period of 1 is no seasonality; fall through


def test_seasonal_naive_repeats_the_last_cycle() -> None:
    history = [1.0, 2.0, 3.0] * 4
    forecaster = SeasonalNaive(frequency="1 day", period_field="3")

    paths = forecaster.forecast_samples(history, horizon=3, samples=50)

    assert len(paths) == 50 and all(len(path) == 3 for path in paths)
    # History is exactly periodic, so the residual spread is zero and every path is the cycle.
    assert paths[0] == (1.0, 2.0, 3.0)


def test_naive_forecast_is_a_random_walk_from_the_last_value() -> None:
    from baseline.classical import NaiveForecaster

    forecaster = NaiveForecaster(seed=0)
    paths = forecaster.forecast_samples([1.0, 2.0, 3.0], horizon=4, samples=30)

    assert len(paths) == 30 and all(len(path) == 4 for path in paths)


def test_ses_and_ets_and_arima_return_the_requested_shape() -> None:
    from baseline.classical import ARIMAForecaster, ETSForecaster, SESForecaster

    history = [10.0 + (i % 7) + 0.01 * i for i in range(60)]
    for forecaster in (SESForecaster(seed=0), ETSForecaster(seed=0), ARIMAForecaster(seed=0)):
        paths = forecaster.forecast_samples(history, horizon=5, samples=20)
        assert len(paths) == 20 and all(len(path) == 5 for path in paths)


def test_a_series_that_cannot_be_fit_still_gets_a_naive_fallback() -> None:
    """A constant series makes ARIMA's differencing degenerate; coverage must not drop to zero."""
    from baseline.classical import ARIMAForecaster

    paths = ARIMAForecaster(seed=0).forecast_samples([5.0] * 10, horizon=3, samples=10)

    assert len(paths) == 10 and all(len(path) == 3 for path in paths)


def test_moirai_turns_its_quantile_grid_into_trajectories() -> None:
    """The checkpoint is not loaded here; only the grid-to-paths wiring is under test."""
    from baseline.moirai import MoiraiSampleForecaster

    forecaster = MoiraiSampleForecaster()
    forecaster._levels = (0.1, 0.5, 0.9)
    forecaster.quantiles = lambda history, horizon: [[0.0] * horizon,
                                                     [5.0] * horizon,
                                                     [9.0] * horizon]

    paths = forecaster.forecast_samples([1.0, 2.0], horizon=3, samples=40)

    assert len(paths) == 40 and all(len(path) == 3 for path in paths)
    assert all(min(path) >= 0.0 and max(path) <= 9.0 for path in paths)
    # Every step of a path reads the same quantile level, so a flat grid gives a flat path.
    assert all(len(set(path)) == 1 for path in paths)


def test_chronos_turns_its_quantile_grid_into_trajectories() -> None:
    """The checkpoint is not loaded here; only the [batch,step,level] to [level][step] wiring is."""
    import types

    from baseline.chronos_baseline import ChronosSampleForecaster

    forecaster = ChronosSampleForecaster()

    class Pipeline:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            import torch

            grid = torch.zeros(1, prediction_length, len(quantile_levels))
            for level_index, level in enumerate(quantile_levels):
                grid[0, :, level_index] = level * 10
            return grid, None

    forecaster._pipeline = Pipeline()
    forecaster._ensure_pipeline = lambda: (forecaster._pipeline, __import__("torch"))

    paths = forecaster.forecast_samples([1.0, 2.0], horizon=3, samples=15)

    assert len(paths) == 15 and all(len(path) == 3 for path in paths)
    assert all(len(set(path)) == 1 for path in paths)  # flat grid gives a flat path


def test_a_quantile_grid_that_does_not_match_its_levels_is_rejected() -> None:
    """A checkpoint with a different grid must fail loudly rather than be misread as levels."""
    with pytest.raises(ValueError, match="quantile rows"):
        quantile_paths([[1.0], [2.0]], levels=[0.1, 0.5, 0.9], samples=4)


def test_aurora_token_length_tracks_the_task_period() -> None:
    from baseline.aurora import DEFAULT_TOKEN_LEN, MAX_TOKEN_LEN, token_length_for

    assert token_length_for("1 hour", None) == 24
    assert token_length_for("1 day", None) == 7
    assert token_length_for("1 week", None) == 52  # under the cap, so used as-is
    assert token_length_for("1 day", str(4 * MAX_TOKEN_LEN)) == MAX_TOKEN_LEN  # capped
    assert token_length_for("unknown", None) == DEFAULT_TOKEN_LEN


def test_aurora_is_put_in_eval_mode_before_it_forecasts() -> None:
    """Left in training mode it rejects a single series and silently shortens the horizon."""
    import sys
    import types

    from baseline.aurora import AuroraConfig, AuroraForecaster

    class Model:
        training = True

        def eval(self):
            self.training = False

    stub = types.ModuleType("aurora")
    stub.load_model = lambda repo_id, device: Model()
    original = sys.modules.get("aurora")
    sys.modules["aurora"] = stub
    try:
        forecaster = AuroraForecaster(AuroraConfig(device="cpu"))
        model, _torch = forecaster._ensure_model()
        assert model.training is False
    finally:
        if original is None:
            del sys.modules["aurora"]
        else:
            sys.modules["aurora"] = original


def test_the_forecast_prompt_carries_the_series_and_nothing_else() -> None:
    """No Context: the prompt may show numbers and cadence, never a description of the entity."""
    from baseline.gemini import build_prompt

    prompt = build_prompt([1.0, 2.5], ["2020-01-01", "2020-01-02"], horizon=3, frequency="1 day")

    assert "(2020-01-01, 1)" in prompt and "(2020-01-02, 2.5)" in prompt
    assert "1 day" in prompt and "3" in prompt


def test_an_unusable_answer_is_dropped_rather_than_scored() -> None:
    """A short, non-numeric, or non-finite path must never reach the ensemble."""
    from baseline.gemini import _parse_forecast

    assert _parse_forecast('{"forecast": [1.0, 2.0]}', horizon=2) == (1.0, 2.0)
    assert _parse_forecast('{"forecast": [1.0]}', horizon=2) is None       # wrong length
    assert _parse_forecast('{"forecast": ["a", "b"]}', horizon=2) is None  # not numeric
    assert _parse_forecast('{"forecast": [1.0, Infinity]}', horizon=2) is None
    assert _parse_forecast("sorry, I cannot", horizon=2) is None


def test_a_forecast_is_on_disk_before_the_next_task_starts(tmp_path: Path) -> None:
    """A sweep that dies hours in must keep the tasks it already paid for."""
    from baseline.run import _forecast_writer, _write_record

    destination = tmp_path / "nested" / "out.jsonl"
    with _forecast_writer(destination) as handle:
        _write_record(handle, "task_1", [(1.0, 2.0)])
        mid_run = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
        _write_record(handle, "task_2", [(3.0, 4.0)])

    assert mid_run == [{"benchmark_id": "task_1", "samples": [[1.0, 2.0]]}]
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["benchmark_id"] for line in lines] == ["task_1", "task_2"]


def test_a_scoring_only_sweep_writes_nothing() -> None:
    from baseline.run import _forecast_writer

    with _forecast_writer(None) as handle:
        assert handle is None


def test_a_rate_limit_refusal_is_obeyed_not_guessed_at() -> None:
    from baseline.gemini import _retry_delay

    assert _retry_delay(Exception("429 ... Please retry in 11.598459829s.")) == pytest.approx(12.098, abs=0.01)
    assert _retry_delay(Exception("429 RESOURCE_EXHAUSTED, no delay given")) == 15.0
    assert _retry_delay(Exception("500 internal error")) is None  # not a rate limit; normal backoff
    assert _retry_delay(Exception("429 retry in 9999s")) == 90.0  # capped


def test_the_rate_limiter_holds_calls_to_its_budget() -> None:
    import time

    from baseline.gemini import RateLimiter

    limiter = RateLimiter(per_minute=3)
    started = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - started < 0.5  # first three pass straight through
    assert len(limiter._started) == 3

    RateLimiter(per_minute=0).acquire()  # unlimited is a no-op, not a division by zero
