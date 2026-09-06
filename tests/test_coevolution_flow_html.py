from html.parser import HTMLParser
from pathlib import Path
import json
import re
import subprocess

from evolving_loop.retrieval_agent.schemas import (
    RetrievalAssumption,
    RetrievalGap,
    RetrievalRoundResult,
)


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
        "example-input",
        "example-output",
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


def test_agent_click_updates_illustrative_input_and_output_payloads() -> None:
    source = FLOW_HTML.read_text(encoding="utf-8")
    script_match = re.search(r"<script>([\s\S]*?)</script>", source)
    assert script_match is not None
    harness = r"""
const fs = require('fs');
const source = fs.readFileSync(0, 'utf8');
class Element {
  constructor(id = '', dataset = {}) {
    this.id = id;
    this.dataset = dataset;
    this.textContent = '';
    this.listeners = {};
    this.classList = { toggle() {} };
    this.style = { setProperty() {} };
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute() {}
  replaceChildren(...children) { this.children = children; }
}
const viewButtons = ['pipeline', 'evolution', 'demo', 'artifacts'].map(
  (view) => new Element('', {view})
);
const views = viewButtons.map((button) => new Element(button.dataset.view));
const cards = ['numerical', 'retrieval', 'decision'].map(
  (agent) => new Element('', {agent})
);
const ids = {};
[
  'agent-detail', 'detail-input-title', 'detail-output-title',
  'detail-limit-title', 'detail-input', 'detail-output', 'detail-limit',
  'agent-example', 'example-title', 'example-input', 'example-output',
  'example-handoff-label', 'example-handoff-text'
].forEach((id) => { ids[id] = new Element(id); });
global.document = {
  querySelectorAll(selector) {
    if (selector === '[data-view]') return viewButtons;
    if (selector === '.view') return views;
    if (selector === '.agent-card') return cards;
    return [];
  },
  getElementById(id) { return ids[id]; },
  createElement() { return new Element(); }
};
new Function(source)();
cards[1].listeners.click();
console.log(JSON.stringify({
  input: ids['example-input'].textContent,
  output: ids['example-output'].textContent
}));
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        input=script_match.group(1),
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert '"documents"' in result["input"]
    assert '"assumptions"' in result["input"]
    assert '"evidence_chains"' in result["output"]
    assert '"exact_quote"' in result["output"]

    retrieval_input = json.loads(result["input"])
    retrieval_output = json.loads(result["output"])
    assert [
        RetrievalAssumption.from_payload(item)
        for item in retrieval_input["assumptions"]
    ]
    assert [RetrievalGap.from_payload(item) for item in retrieval_input["gaps"]]
    assert RetrievalRoundResult.from_payload(retrieval_output).sufficient is True
