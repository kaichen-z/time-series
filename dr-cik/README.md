# dr-cik

Reproductions of two of Dr-CiK's "Deep Research" agent baselines — **OpenDR** and
**DRBench** — paired with **Chronos** (Dr-CiK's own pretrained-TSFM baseline, used
zero-shot only, never fine-tuned on Dr-CiK data). Fully standalone: no imports from
`src/drcik_agent`.

## Install

```bash
cd dr-cik
pip install -e '.[chronos,gemini,dev]'
```

## Configure

```bash
export GEMINI_API_KEY="your_key"   # powers OpenDR/DRBench's LLM calls and the evidence-recall judge
```

## Quick start (no download needed)

The official 3-task sample already lives at
`/raid/home/air/khoutaibi/external/Dr-CiK/sample`:

```bash
dr-cik run --agent drbench \
  --sample-dir /raid/home/air/khoutaibi/external/Dr-CiK/sample \
  --output-dir outputs/drbench-sample

dr-cik run --agent opendr \
  --sample-dir /raid/home/air/khoutaibi/external/Dr-CiK/sample \
  --output-dir outputs/opendr-sample
```

## Full dataset and model checkpoint

```bash
dr-cik download-data    # -> /raid/home/air/khoutaibi/time_series_dataset/Dr-CiK
dr-cik download-models  # -> /raid/home/air/khoutaibi/models (Chronos checkpoint)

dr-cik run --agent drbench --data-dir /raid/home/air/khoutaibi/time_series_dataset/Dr-CiK \
  --split public-dev --output-dir outputs/drbench-dev
```

## Outputs

`--output-dir` gets `forecasts.jsonl` and `deep_research.jsonl` (exact `SUBMISSION.md`
shape), plus `run_report.jsonl` (full agent trace + metrics per task) and `summary.json`
(mean metrics). `smae`/`srmse`/`scrps` are local development-proxy scores (`S=25`, per
the paper's formula — separate from the `>=100` samples written to `forecasts.jsonl`);
`evidence_recall` is our own Gemini-judge approximation, not Dr-CiK's private official
scorer. Both are explicitly labeled as such in `summary.json`'s `note` field.

## Tests

```bash
pytest tests/          # offline, no API key or GPU needed
```

## Leakage safety

Agent code only ever sees `TaskView`/`AgentDocument` (id + text only) — never `role`,
`subtype`, `future_values`, or `gt_evidence`. This is structural, not conventional:
`tests/test_models_leakage.py` AST-scans every agent-facing module and fails the build
if it references the labeled `Document`/`ForecastTask` types.
