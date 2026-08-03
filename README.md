# time-series

## Minimal Dr-CiK Forecasting Agent

This repository contains a dependency-free vertical slice for the Dr-CiK task:

```text
historical time series + noisy document corpus
                    |
                    v
        Time-Series Diagnosis Agent
                    |
                    v
             Retrieval Agent
                    |
                    v
         Evidence Synthesis Agent
                    |
                    v
       Probabilistic Forecast Agent
                    |
                    v
 retrieval metrics + forecast metrics + official submission files
```

The baseline is intentionally small and deterministic. It establishes a runnable
evaluation harness before Nexus-, BLF-, or CORAL-style components are introduced.
Benchmark role/subtype labels are stripped before retrieval, so the agent cannot use
ground-truth document labels during inference.

### What it produces

- `forecasts.jsonl`: Dr-CiK forecasting-track format with 100 sample trajectories per task.
- `deep_research.jsonl`: cited document IDs and extracted evidence.
- `run_report.jsonl`: diagnosis, retrieval scores, forecast mean, and per-task metrics.
- `summary.json`: aggregate development metrics.

Local `sMAE`, `sRMSE`, and `sCRPS` are explicitly labeled as development proxies. The
official hidden-test scores are computed by the Dr-CiK maintainers.

### Quick start with the official sample

```bash
git clone https://github.com/ServiceNow/Dr-CiK.git external/Dr-CiK
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/official-sample
```

The same command can be run without installation:

```bash
PYTHONPATH=src python3 -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/official-sample
```

### Run the full public development split

```bash
pip install -e '.[huggingface]'
drcik-agent run-hf --public-dev --output-dir outputs/public-dev
```

Use `--limit 5` while developing. The hidden test split can be executed with
`--hidden-test`; because its labels are withheld, the system creates submission files
but no forecast scores.

### Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### Current baseline and next replacements

| Module | v0.1 implementation | Planned upgrade |
|---|---|---|
| Diagnosis | Trend, seasonality, residual scale, information needs | LLM time-series diagnosis |
| Retrieval | BM25 over the task corpus | Multi-hop forecast-aware retrieval |
| Evidence | Extractive structured claims | Evidence verifier with conflict checks |
| Forecast | Seasonal naive + exact contextual values + uncertainty samples | Nexus-style macro/micro synthesis |
| Memory/loop | One pass | BLF-style belief state and forecast-feedback loop |

The next scientifically meaningful experiment is not to add every component at once.
It is to compare `no context`, `oracle context`, this one-pass retrieval baseline, and a
forecast-aware iterative retrieval variant under the same forecaster.
