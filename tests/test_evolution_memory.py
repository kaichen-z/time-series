"""Tests for the per-generation memory file: banners, content, and the fallback when it fails."""
from __future__ import annotations

from pathlib import Path

from numerical_agent.evolution.memory import (
    MEMORY_FILENAME,
    append_memory,
    banner,
    build_summary_request,
    record_generation,
)


class StubLLM:
    """A summarizer that records what it was asked and answers with fixed prose."""

    def __init__(self, text: str = "The module lost two flat methods and kept auto_tbats.") -> None:
        self.text = text
        self.requests: list[str] = []

    def complete(self, *, system, messages, temperature=0.0):
        from common.llm import LLMResponse

        self.requests.append(messages[-1]["content"])
        return LLMResponse(text=self.text)


class FailingLLM:
    def complete(self, *, system, messages, temperature=0.0):
        raise RuntimeError("model is not loaded")


def test_each_generation_is_written_under_its_own_banner(tmp_path: Path) -> None:
    record_generation(tmp_path, StubLLM(), generation=1, applied=("delete flat: no shape",),
                      method_count=69)
    record_generation(tmp_path, StubLLM("Second generation."), generation=2,
                      applied=("delete other: beaten",), method_count=68)

    text = (tmp_path / MEMORY_FILENAME).read_text(encoding="utf-8")

    assert text.count(banner(1)) == 1
    assert text.count(banner(2)) == 1
    assert text.index(banner(1)) < text.index(banner(2))
    assert "Second generation." in text


def test_the_summarizer_is_told_the_operations_and_the_counts() -> None:
    request = build_summary_request(
        generation=3, applied=("delete stl_naive: flat",), method_count=41, val_best_smae=0.34,
        reasoning="stl_naive never tracks the series.",
    )

    assert "Generation 3" in request
    assert "41 methods" in request
    assert "0.34" in request
    assert "delete stl_naive: flat" in request
    assert "stl_naive never tracks the series." in request


def test_a_failed_summarizer_still_records_the_operations(tmp_path: Path) -> None:
    """A generation with no entry reads later as a generation that did nothing."""
    record_generation(tmp_path, FailingLLM(), generation=4,
                      applied=("delete ses: duplicate of naive_last",), method_count=12)

    text = (tmp_path / MEMORY_FILENAME).read_text(encoding="utf-8")

    assert banner(4) in text
    assert "delete ses: duplicate of naive_last" in text
    assert "Summarizer failed" in text


def test_no_summarizer_at_all_still_records_the_generation(tmp_path: Path) -> None:
    record_generation(tmp_path, None, generation=5, applied=(), method_count=8)

    text = (tmp_path / MEMORY_FILENAME).read_text(encoding="utf-8")

    assert banner(5) in text
    assert "No operations were applied" in text


def test_an_empty_summary_falls_back_rather_than_writing_a_blank_entry(tmp_path: Path) -> None:
    record_generation(tmp_path, StubLLM("   "), generation=6,
                      applied=("rewrite bats: repaired the period guard",), method_count=20)

    text = (tmp_path / MEMORY_FILENAME).read_text(encoding="utf-8")

    assert "rewrite bats: repaired the period guard" in text


def test_appending_never_rewrites_an_earlier_entry(tmp_path: Path) -> None:
    append_memory(tmp_path, 1, "First.")
    append_memory(tmp_path, 2, "Second.")

    text = (tmp_path / MEMORY_FILENAME).read_text(encoding="utf-8")

    assert "First." in text and "Second." in text
