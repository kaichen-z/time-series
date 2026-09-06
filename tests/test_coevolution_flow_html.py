from html.parser import HTMLParser
from pathlib import Path
import json
import re
import subprocess

FLOW_HTML = Path(__file__).parents[1] / "docs" / "three-agent-coevolution-flow.html"


class _FlowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.agent_names: set[str] = set()
        self.buttons: set[str] = set()
        self.orchestration_stages: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if values.get("id"):
            element_id = str(values["id"])
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)
        if values.get("data-agent"):
            self.agent_names.add(str(values["data-agent"]))
        if tag == "button" and values.get("data-view"):
            self.buttons.add(str(values["data-view"]))
        if tag == "button" and values.get("data-orchestration-stage"):
            self.orchestration_stages.add(
                str(values["data-orchestration-stage"])
            )


def test_coevolution_flow_html_covers_the_live_pipeline() -> None:
    source = FLOW_HTML.read_text(encoding="utf-8")
    parser = _FlowParser()
    parser.feed(source)

    assert not parser.duplicate_ids
    assert parser.agent_names == {"numerical", "retrieval", "decision"}
    assert parser.buttons == {
        "pipeline",
        "dictionary",
        "champion",
        "evolution",
        "demo",
        "artifacts",
    }
    assert {
        "pipeline",
        "dictionary",
        "champion",
        "evolution",
        "demo",
        "artifacts",
        "agent-detail",
        "example-input",
        "example-output",
        "real-coordinate-example",
        "numerical-self-evolve",
        "dictionary-evolve-example",
        "champion-evolve-example",
        "full-method-evolution-prompt",
        "full-filter-prompt",
        "full-screening-prompt",
        "story-prompt",
        "orchestration-detail",
        "orchestration-stage",
        "orchestration-title",
        "orchestration-status",
        "orchestration-provenance",
        "orchestration-input",
        "orchestration-action",
        "orchestration-evolve",
        "orchestration-output",
        "orchestration-failure",
        "orchestration-authority",
        "orchestration-handoff",
        "orchestration-prompt-label",
        "orchestration-prompt",
        "orchestration-carry",
        "orchestration-prev",
        "orchestration-next",
        "orchestration-contract-mode",
        "orchestration-recorded-mode",
    } <= parser.ids
    assert "0.640444097" in source
    assert "task_114" in source
    assert "Public-99" in source
    assert "balanced-v3 实测 Toto baseline" in source
    assert "0.388409" in source
    assert "0.392069" in source
    assert "0.381407" in source
    assert "0.601756" in source
    assert "0.587150" in source
    assert "0.590347" in source
    assert "sMAE 最大相对均值差 2.77%" in source
    assert "三 Agent 正式运行增益仍为 0" in source
    assert "fallback" in source
    assert "80 Train cross-fit" in source
    assert "20 Dev 唯一验收门" in source
    assert "64 Build" not in source
    assert "16 Calibration" not in source
    assert "8 → 32 → 64" not in source
    assert "详细 task feedback 只来自 Train" in source
    assert "历史旧协议记录：不是新版 8/N 的结果" in source
    assert "Stage 1 · Dictionary Evolution" in source
    assert "Stage 2 · Champion Evolution" in source
    assert "Stage 3 · Three-Agent Co-Evolve" in source
    assert "原始 Catalog 不是 available_methods" in source
    assert parser.orchestration_stages == {
        "method_evolution",
        "dictionary_filter",
        "task_screening",
        "champion_proposal",
        "numerical_execution",
        "retrieval_round1",
        "retrieval_round2",
        "decision",
        "gate",
    }
    assert "当前代码事实" in source
    assert "设计规格 · 待代码实现" in source
    assert "numerical_champion_release" in source
    assert "methods.py + policies.py + dictionary.py" in source
    assert "context_pair" in source


def test_coevolution_flow_html_is_self_contained_and_small() -> None:
    source = FLOW_HTML.read_text(encoding="utf-8")

    assert "<!doctype html>" in source.lower()
    assert "<script" in source
    assert "<style" in source
    assert "fetch(" not in source
    assert FLOW_HTML.stat().st_size < 1_000_000


def test_orchestration_uses_only_provenanced_contracts_and_artifacts() -> None:
    source = FLOW_HTML.read_text(encoding="utf-8")

    assert "train_method_reports" not in source
    assert "catalog_candidates: ['toto', 'seasonal_naive', 'linear_trend', 'croston']" not in source
    assert "CURRENT CONTRACT" in source
    assert "RECORDED RUN" in source
    assert "NOT RECORDED" in source
    assert "runs/method_evolution/v001/bootstrap_summary.json" in source
    assert "93 generated / 93 succeeded" in source
    assert "legacy metrics schema" in source
    for label in (
        "SYSTEM PROMPT",
        "USER INPUT",
        "RAW MODEL OUTPUT",
        "HOST PARSE / VALIDATE",
        "EVOLVE / MUTATION",
        "RESULT",
        "ARTIFACTS",
        "DOWNSTREAM HANDOFF",
    ):
        assert label in source


def test_truth_modes_render_all_nine_live_contracts_and_recorded_evidence() -> None:
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
    this.hidden = false;
    this.disabled = false;
  }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  setAttribute() {}
  replaceChildren(...children) { this.children = children; }
}
const viewButtons = [
  'pipeline', 'dictionary', 'champion', 'evolution', 'demo', 'artifacts'
].map(
  (view) => new Element('', {view})
);
const views = viewButtons.map((button) => new Element(button.dataset.view));
const cards = ['numerical', 'retrieval', 'decision'].map(
  (agent) => new Element('', {agent})
);
const orchestrationStages = [
  'method_evolution', 'dictionary_filter', 'task_screening',
  'champion_proposal', 'numerical_execution', 'retrieval_round1',
  'retrieval_round2', 'decision', 'gate'
].map((orchestrationStage) => new Element('', {orchestrationStage}));
const ids = {};
[
  'agent-detail', 'detail-input-title', 'detail-output-title',
  'detail-limit-title', 'detail-input', 'detail-output', 'detail-limit',
  'agent-example', 'example-title', 'example-input', 'example-output',
  'example-handoff-label', 'example-handoff-text',
  'story-step', 'story-title', 'story-agent', 'story-copy',
  'story-input', 'story-output', 'story-feedback',
  'story-prompt', 'story-prompt-label', 'story-prompt-text',
  'story-prev', 'story-next', 'full-method-evolution-prompt',
  'full-filter-prompt', 'full-screening-prompt', 'full-numerical-prompt',
  'orchestration-detail', 'orchestration-stage', 'orchestration-title',
  'orchestration-status', 'orchestration-input', 'orchestration-action',
  'orchestration-output', 'orchestration-failure', 'orchestration-authority',
  'orchestration-provenance', 'orchestration-evolve', 'orchestration-handoff',
  'orchestration-prompt-label', 'orchestration-prompt',
  'orchestration-carry', 'orchestration-prev', 'orchestration-next',
  'orchestration-contract-mode', 'orchestration-recorded-mode'
].forEach((id) => { ids[id] = new Element(id); });
global.document = {
  querySelectorAll(selector) {
    if (selector === '[data-view]') return viewButtons;
    if (selector === '.view') return views;
    if (selector === '.agent-card') return cards;
    if (selector === '[data-orchestration-stage]') return orchestrationStages;
    return [];
  },
  getElementById(id) { return ids[id]; },
  createElement() { return new Element(); }
};
new Function(source)();
const payloads = {};
for (const card of cards) {
  card.listeners.click();
  payloads[card.dataset.agent] = {
    input: ids['example-input'].textContent,
    output: ids['example-output'].textContent,
    handoff: ids['example-handoff-text'].textContent
  };
}
payloads.orchestration = [];
for (let index = 0; index < orchestrationStages.length; index += 1) {
  payloads.orchestration.push({
    stage: ids['orchestration-stage'].textContent,
    title: ids['orchestration-title'].textContent,
    status: ids['orchestration-status'].textContent,
    provenance: ids['orchestration-provenance'].textContent,
    promptLabel: ids['orchestration-prompt-label'].textContent,
    prompt: ids['orchestration-prompt'].textContent,
    input: ids['orchestration-input'].textContent,
    action: ids['orchestration-action'].textContent,
    evolve: ids['orchestration-evolve'].textContent,
    output: ids['orchestration-output'].textContent,
    failure: ids['orchestration-failure'].textContent,
    authority: ids['orchestration-authority'].textContent,
    handoff: ids['orchestration-handoff'].textContent,
    carry: ids['orchestration-carry'].textContent,
    previousDisabled: ids['orchestration-prev'].disabled,
    nextLabel: ids['orchestration-next'].textContent
  });
  if (index < orchestrationStages.length - 1) {
    ids['orchestration-next'].listeners.click();
  }
}
ids['orchestration-recorded-mode'].listeners.click();
payloads.recorded = [];
for (const stage of orchestrationStages) {
  stage.listeners.click();
  payloads.recorded.push({
    status: ids['orchestration-status'].textContent,
    provenance: ids['orchestration-provenance'].textContent,
    input: ids['orchestration-input'].textContent,
    output: ids['orchestration-output'].textContent,
    result: ids['orchestration-failure'].textContent,
    handoff: ids['orchestration-handoff'].textContent
  });
}
payloads.story = [];
for (let index = 0; index < 6; index += 1) {
  payloads.story.push({
    step: ids['story-step'].textContent,
    title: ids['story-title'].textContent,
    agent: ids['story-agent'].textContent,
    input: ids['story-input'].textContent,
    output: ids['story-output'].textContent,
    feedback: ids['story-feedback'].textContent,
    promptLabel: ids['story-prompt-label'].textContent,
    promptHidden: ids['story-prompt'].hidden
  });
  if (index < 5) ids['story-next'].listeners.click();
}
console.log(JSON.stringify(payloads));
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        input=script_match.group(1),
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    orchestration = result["orchestration"]
    assert [item["stage"] for item in orchestration] == [
        "STEP 1 / 9",
        "STEP 2 / 9",
        "STEP 3 / 9",
        "STEP 4 / 9",
        "STEP 5 / 9",
        "STEP 6 / 9",
        "STEP 7 / 9",
        "STEP 8 / 9",
        "STEP 9 / 9",
    ]
    assert orchestration[0]["previousDisabled"] is True
    assert orchestration[-1]["nextLabel"] == "从头再看 →"
    assert orchestration[0]["status"] == "CURRENT CONTRACT"
    assert "evolve_once" in orchestration[0]["provenance"]
    assert "run_module" in orchestration[0]["action"]
    assert "batch" in orchestration[0]["evolve"]
    assert "FILTER_SYSTEM" in orchestration[1]["promptLabel"]
    assert "SCREENING_SYSTEM" in orchestration[2]["promptLabel"]
    assert "CHAMPION_PROPOSAL_SYSTEM" in orchestration[3]["promptLabel"]
    assert "NOT APPLICABLE" in orchestration[4]["prompt"]
    assert "Retrieval Round 1" in orchestration[5]["prompt"]
    assert "Retrieval Round 2" in orchestration[6]["prompt"]
    assert "Decision Agent" in orchestration[7]["prompt"]
    assert "projection" in orchestration[4]["handoff"]
    assert "sanitized four-field assumptions" in orchestration[5]["handoff"]
    assert "Host validation" in orchestration[7]["handoff"]

    recorded = result["recorded"]
    assert len(recorded) == 9
    assert "LEGACY 2026-08-22" in recorded[0]["status"]
    assert "93 generated / 93 succeeded" in recorded[0]["input"]
    assert '"operations"' in recorded[0]["output"]
    assert recorded[1]["output"].startswith("NOT RECORDED")
    assert '"mean_active_candidates": 80.1625' in recorded[2]["result"]
    assert '"recipes"' in recorded[3]["output"]
    assert '"selected_candidate_id": "toto_2_0"' in recorded[7]["output"]
    assert '"accepted_steps":0' in recorded[8]["result"]
