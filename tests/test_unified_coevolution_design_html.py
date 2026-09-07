from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "docs" / "unified-coevolution-design-2026-09-07.html"


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.scripts: list[str | None] = []
        self.stylesheets: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))
        if tag == "script":
            self.scripts.append(values.get("src"))
        if tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(str(values.get("href", "")))


def _page() -> tuple[str, _PageParser]:
    text = HTML_PATH.read_text(encoding="utf-8")
    parser = _PageParser()
    parser.feed(text)
    return text, parser


def test_design_html_is_self_contained_and_accessible() -> None:
    text, parser = _page()

    assert '<html lang="zh-CN">' in text
    assert 'name="viewport"' in text
    assert "<style>" in text
    assert parser.stylesheets == []
    assert all(src is None for src in parser.scripts)
    assert not any(link.startswith(("http://", "https://")) for link in parser.links)


def test_design_html_covers_the_complete_unified_flow() -> None:
    text, parser = _page()

    expected_ids = {
        "overview",
        "architecture",
        "supply",
        "numerical",
        "retrieval",
        "decision",
        "feedback",
        "protocol",
        "artifacts",
        "implementation",
    }
    assert expected_ids <= parser.ids
    for phrase in (
        "Dictionary Supply Epoch",
        "Numerical → Retrieval → Decision",
        "8 → 32 → Train-80 → Dev-20",
        "Public-99",
        "DictionarySupplyRelease",
        "UnifiedCoEvolutionBundle",
        "sMAE",
        "sRMSE",
        "select / route / horizon_route / weighted / median / bounded_overlay",
        "多个可证伪 assumptions",
    ):
        assert phrase in text


def test_design_html_distinguishes_existing_and_new_work() -> None:
    text, _parser = _page()

    assert "现有 · 直接复用" in text
    assert "新增 · 本次桥接" in text
    assert "不参与 evolve" in text
    assert "设计已批准，统一 runner 尚未实现" in text
    assert "不是实验结果" in text
    assert "64 Build" not in text
    assert "16 Calibration" not in text


def test_design_html_has_working_internal_navigation_targets() -> None:
    _text, parser = _page()

    internal_targets = {
        link.removeprefix("#") for link in parser.links if link.startswith("#")
    }
    assert internal_targets
    assert internal_targets <= parser.ids
