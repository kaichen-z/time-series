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
        self.contract_stages: dict[str, set[str]] = {}
        self.handoffs: set[str] = set()

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
        if values.get("data-contract") and values.get("data-stage"):
            contract = str(values["data-contract"])
            self.contract_stages.setdefault(contract, set()).add(
                str(values["data-stage"])
            )
        if values.get("data-handoff"):
            self.handoffs.add(str(values["data-handoff"]))


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
        "global-coevolve",
        "protocol",
        "artifacts",
        "implementation",
    }
    assert expected_ids <= parser.ids
    for phrase in (
        "Base Self-Evolve Loop",
        "Parent → Children → Train/Dev Gate → Accept or Rollback",
        "每个坐标都是完整 self-evolve loop",
        "Global Bundle Co-Evolve",
        "完整系统 Bundle Child",
        "Warm-up Sweep",
        "Weakest-coordinate Scheduler",
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


def test_design_html_places_global_coevolution_above_coordinate_loops() -> None:
    text, _parser = _page()

    base = text.index("Base Self-Evolve Loop")
    coordinate = text.index("Coordinate Self-Evolve Loops")
    global_loop = text.index("Global Bundle Co-Evolve")
    assert base < coordinate < global_loop
    for coordinate in ("Dictionary Loop", "Numerical Loop", "Retrieval Loop", "Decision Loop"):
        assert coordinate in text
    assert "Co-evolve 不替代 self-evolve" in text


def test_meta_harness_v2_documents_controls_joint_child_and_train_only_memory() -> None:
    text, _parser = _page()

    for phrase in (
        "三个单角色 ablation 和一个 joint Child",
        "requested_child_kind",
        "Train-only memory",
        "coding、retrieval、decision、joint",
        "至少改变两个角色",
        "Dictionary/Champion package adapter 尚未接入",
        "Dev 与 Public 永不回流",
    ):
        assert phrase in text


def test_retrieval_section_separates_inference_from_self_evolution() -> None:
    text, parser = _page()

    assert {"retrieval-inference-diagram", "retrieval-evolution-diagram"} <= parser.ids
    inference = text.index("单题执行：Two-stage Retrieval")
    evolution = text.index("跨代进化：Retrieval Self-Evolve")
    assert inference < evolution
    for phrase in (
        "这不是 self-evolve",
        "这才是 Retrieval self-evolve",
        "Retrieval Parent = Genome + Prompt + Budgets + Skills",
        "Child A · Round 1",
        "Child B · Evidence Verification",
        "Child C · Round 2",
        "每评估一个 Retrieval Child，都会在每道任务上重新执行上面的",
        "R1 → D0 → R2 → Verify",
        "Accepted Retrieval Release",
    ):
        assert phrase in text


def test_each_coordinate_exposes_the_same_prompt_output_host_handoff_contract() -> None:
    _text, parser = _page()

    required_stages = {
        "parent-input",
        "llm-prompt",
        "llm-output",
        "host-actions",
        "release-output",
        "next-input",
    }
    for contract in ("dictionary", "numerical", "retrieval", "decision", "bundle"):
        assert parser.contract_stages.get(contract) == required_stages


def test_page_connects_every_release_to_the_next_coordinate_input() -> None:
    text, parser = _page()

    assert parser.handoffs == {
        "dictionary-to-numerical",
        "numerical-to-retrieval",
        "retrieval-to-decision",
        "decision-to-bundle",
        "bundle-to-next-generation",
    }
    for phrase in (
        "LLM 可以 evolve",
        "Host 固定，不能 evolve",
        "DictionarySupplyRelease → Numerical Parent Input",
        "NumericalForecastPackage → Retrieval Parent Input",
        "VerifiedRetrievalCard → Decision Parent Input",
        "DecisionTrace → Bundle Evaluation Input",
        "Accepted Bundle → 下一代 Bundle Parent",
        '"operations"',
        '"recipes"',
        '"evidence_chains"',
        '"action": "select"',
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
