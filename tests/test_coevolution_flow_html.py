from html.parser import HTMLParser
from pathlib import Path


FLOW_HTML = Path(__file__).parents[1] / "docs" / "three-agent-coevolution-flow.html"


class _FlowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.agent_names: set[str] = set()
        self.buttons: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if values.get("data-agent"):
            self.agent_names.add(str(values["data-agent"]))
        if tag == "button" and values.get("data-view"):
            self.buttons.add(str(values["data-view"]))


def test_coevolution_flow_html_covers_the_live_pipeline() -> None:
    source = FLOW_HTML.read_text(encoding="utf-8")
    parser = _FlowParser()
    parser.feed(source)

    assert parser.agent_names == {"numerical", "retrieval", "decision"}
    assert parser.buttons == {"pipeline", "evolution", "demo", "artifacts"}
    assert {
        "pipeline",
        "evolution",
        "demo",
        "artifacts",
        "agent-detail",
    } <= parser.ids
    assert "0.640444097" in source
    assert "task_114" in source
    assert "Public-99" in source
    assert "fallback" in source


def test_coevolution_flow_html_is_self_contained_and_small() -> None:
    source = FLOW_HTML.read_text(encoding="utf-8")

    assert "<!doctype html>" in source.lower()
    assert "<script" in source
    assert "<style" in source
    assert "fetch(" not in source
    assert FLOW_HTML.stat().st_size < 1_000_000
