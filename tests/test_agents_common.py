"""Trend/seasonality detection in agents/common.py: the fix for the naive endpoint-diff bug."""

from __future__ import annotations

from dr_cik.agents.common import _declared_seasonal_period, render_task_brief, trend_phrase, trend_word
from dr_cik.models import AgentDocument, TaskView

from .conftest import requires_sample

# Real task_42 history (sales volume, Nuance Cosmetic Lab): a genuine ~22-step repeating
# cycle. The last-20-point window happens to run from a local peak down to a mid-recovery
# point, which is exactly what fooled the old window[-1]-window[0] heuristic into "falling".
_TASK_42_HISTORY = (
    45.104, 79.022, 108.38, 130.293, 349.321, 558.734, 758.739, 950.305, 1134.104, 1311.993,
    1487.25, 1661.686, 1632.464, 1607.898, 1590.064, 1344.266, 1112.177, 899.515, 709.815,
    546.336, 410.91, 304.957, 228.124, 180.906, 161.755, 186.75, 204.405, 211.847, 210.322,
    199.754, 181.145, 155.469, 125.881, 94.922, 64.674, 37.938, 17.032, 3.75, 0.0, 5.167,
    19.71, 42.721, 72.445, 107.101, 143.529, 179.847, 213.078, 240.783, 261.322, 272.925,
    275.457, 268.052, 252.399, 229.542, 201.278, 170.548, 139.593, 111.322, 87.949, 71.523,
    63.278, 64.584, 75.658, 94.827, 121.851, 154.264, 190.013, 226.625, 260.826, 290.835,
    314.003, 329.26, 335.22, 427.572, 148.605, 387.783, 219.602, 548.669, 123.435, 460.28,
    241.474, 464.064, 679.559, 500.189, 710.574, 869.328, 985.116, 1010.289, 1556.433,
    1466.911, 2040.045, 2242.954, 2268.545, 2129.057, 1981.546, 1827.236, 1666.784, 1502.243,
    1336.196, 1171.494, 1012.456, 861.111, 722.21, 597.396, 490.346, 402.934, 335.914,
    289.969, 265.354, 260.163, 273.003, 308.037, 343.29, 375.824, 404.092, 425.24, 438.063,
    441.988, 436.512, 422.706, 401.084, 374.322, 344.268, 313.215, 284.356, 259.517, 241.076,
    231.206, 229.303, 237.256, 253.609, 277.697, 307.739, 341.404, 376.012, 409.731, 439.519,
    463.372, 479.169, 486.462, 485.026, 473.855, 455.588, 430.521, 401.333, 370.526, 340.438,
    313.735, 292.573, 278.284, 272.846, 276.658, 289.07, 309.891, 336.781, 368.826,
)


def _view(history_values: tuple[float, ...], seasonal_period: object = None) -> TaskView:
    horizon = 3
    return TaskView(
        benchmark_id="test",
        entity_name="Nuance Cosmetic Lab",
        target_name="sales volume",
        target_description="sales volume of a store",
        frequency="1 day",
        prediction_length=horizon,
        seasonal_period=seasonal_period,
        history_timestamps=tuple(str(i) for i in range(len(history_values))),
        history_values=history_values,
        future_timestamps=tuple(str(i) for i in range(len(history_values), len(history_values) + horizon)),
        documents=(AgentDocument(document_id="doc_1", text="irrelevant"),),
    )


def test_trend_word_is_not_fooled_by_a_peak_to_mid_cycle_window() -> None:
    """The old window[-1]-window[0] heuristic said 'falling' here; the series is actually stable/cyclical."""
    assert trend_word(_TASK_42_HISTORY) == "stable"
    assert trend_word(_TASK_42_HISTORY) != "falling"


def test_trend_phrase_detects_the_real_cycle() -> None:
    phrase = trend_phrase(_view(_TASK_42_HISTORY, seasonal_period="D"))
    assert "22-step cycle" in phrase
    assert "detected" in phrase
    strength = float(phrase.rsplit(" ", 1)[-1].rstrip(")"))
    assert strength >= 0.9


def test_render_task_brief_has_a_trend_line_and_no_wrong_word() -> None:
    brief = render_task_brief(_view(_TASK_42_HISTORY, seasonal_period="D"))
    assert "Trend:" in brief
    assert "falling" not in brief


def test_trend_word_volatile_path_is_unaffected() -> None:
    noisy = tuple(100.0 + (50.0 if i % 2 == 0 else -50.0) for i in range(24))
    assert trend_word(noisy) == "volatile"


def test_declared_seasonal_period_accepts_only_positive_ints() -> None:
    assert _declared_seasonal_period("D") is None
    assert _declared_seasonal_period("5T") is None
    assert _declared_seasonal_period(24) == 24
    assert _declared_seasonal_period(None) is None
    assert _declared_seasonal_period(True) is None
    assert _declared_seasonal_period(0) is None
    assert _declared_seasonal_period(-3) is None


@requires_sample
def test_task_42_end_to_end_through_real_data_loading(sample_tasks) -> None:
    task = next((t for t in sample_tasks if t.benchmark_id == "task_42"), None)
    if task is None:
        return  # task_42 isn't in the 3-task official sample bundle on every machine
    view = task.agent_view()
    assert trend_word(view.history_values) != "falling"
    assert "cycle" in trend_phrase(view) or "no strong cycle" in trend_phrase(view)
