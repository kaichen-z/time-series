"""Reasoning extraction and bundle-to-prompt splicing."""

from __future__ import annotations

import dataclasses
import json

import pytest
from dr_cik.llm import parse_json_object

from evolving_agents.agents.common import (
    extract_reasoning,
    get_code_template,
    render_code_template_block,
    render_fewshot_block,
    render_numeric_brief,
)
from evolving_agents.bundles import load_seed
from evolving_agents.models import NumericTaskView

VIEW = NumericTaskView(
    benchmark_id="task_x",
    history_values=tuple(float(10 + (index % 12)) for index in range(120)),
    prediction_length=6,
    frequency="H",
    seasonal_period=12,
)


def test_extract_reasoning_splits_thinking_from_the_answer() -> None:
    payload = json.dumps({"assumption": "flat", "code": "def forecast(): pass"})
    reasoning, answer = extract_reasoning(f"<think>Weekly dips dominate.</think>{payload}")
    assert reasoning == "Weekly dips dominate."
    assert parse_json_object(answer)["assumption"] == "flat"


def test_extract_reasoning_returns_none_without_a_think_block() -> None:
    reasoning, answer = extract_reasoning('{"assumption": "flat"}')
    assert reasoning is None
    assert answer == '{"assumption": "flat"}'


def test_extract_reasoning_keeps_an_unterminated_block() -> None:
    reasoning, answer = extract_reasoning("<think>I was cut off mid")
    assert reasoning == "I was cut off mid"
    assert answer == ""


def test_extract_reasoning_survives_a_multiline_block() -> None:
    reasoning, answer = extract_reasoning("<think>line one\nline two</think>  {}")
    assert reasoning == "line one\nline two"
    assert answer == "{}"


def test_numeric_brief_describes_the_series_without_any_text_fields() -> None:
    brief = render_numeric_brief(VIEW)
    assert "Forecast horizon: 6 steps" in brief
    assert "History length: 120 points" in brief
    assert "Trend:" in brief
    for leaked in ("document", "doc_", "corpus", "task_x"):
        assert leaked not in brief.lower()


def test_fewshot_block_renders_seed_examples() -> None:
    block = render_fewshot_block(load_seed("coding"))
    assert block.startswith("Worked examples:")
    assert "Example 1:" in block


def test_fewshot_block_is_empty_without_examples() -> None:
    assert render_fewshot_block(dataclasses.replace(load_seed("coding"), fewshot_examples=())) == ""


def test_code_template_block_lists_every_template() -> None:
    bundle = load_seed("coding")
    block = render_code_template_block(bundle)
    for name in bundle.code_templates:
        assert f"# template: {name}" in block


def test_get_code_template_names_the_alternatives_when_missing() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_code_template(load_seed("coding"), "no_such_template")
    assert "seasonal_naive" in str(excinfo.value)


def test_get_code_template_returns_the_source() -> None:
    assert "def forecast(" in get_code_template(load_seed("coding"), "linear_trend")
