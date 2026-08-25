# Complete Usage Guide

**Repository:** <https://github.com/kaichen-z/time-series>

This guide documents the functionality that is currently implemented and executable. It covers:

- environment and data setup;
- the unified command-line interface;
- all 14 fixed baselines;
- the current three-agent harness;
- the four Coding Agent candidate-generation settings;
- persistent skill learning;
- prompt-, genome-, and source-level evolution;
- outputs, metrics, experimental boundaries, and troubleshooting.

## 1. Project in one minute

The repository supports two distinct experiment families:

```text
Fixed evaluation:
Dr-CiK task -> selected baseline -> forecast -> metrics

Self-evolution:
training tasks -> three-agent harness -> failure attribution -> child harnesses
               -> train ranking -> entity-disjoint dev gate -> accept or reject
```

The unified entrypoint is:

```bash
python -m evolving_loop --baseline <baseline-name> [options]
python -m evolving_loop --evolution <prompt|genome|source> [options]
```

A baseline evaluates one fixed method and does not alter the harness. An evolution run uses resolved
public training labels to diagnose failure trajectories and propose changes to the system.

## 2. Installation

### 2.1 Clone the repository

```bash
git clone https://github.com/kaichen-z/time-series.git
cd time-series
```

### 2.2 Create a Python environment

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2.3 Install dependencies

For tests and statistical runs:

```bash
pip install -e '.[dev,huggingface]'
```

For the default Chronos baseline and the `tsfm` or `combined` Coding settings:

```bash
pip install -e '.[dev,huggingface,chronos]'
```

For the optional TimesFM baseline:

```bash
pip install -e '.[timesfm]'
```

TimesFM can introduce PyTorch-version conflicts. A separate virtual environment is recommended for
the TimesFM ablation.

The optional local-Qwen backend also requires:

```bash
pip install torch transformers accelerate
```

Qwen is not the default backend and its default checkpoint is large. Use the Codex backend unless a
suitable local GPU environment is available.

### 2.4 Install and authenticate Codex CLI

`codex-triad`, `codex-direct`, `codex-contract`, `evolving-harness`, and all three evolution modes
can use the authenticated Codex CLI installed on the machine.

For macOS/Linux:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex
```

Follow the sign-in flow on the first launch, then verify the installation:

```bash
codex --version
```

See the official [Codex CLI documentation](https://learn.chatgpt.com/docs/codex/cli). This project
does not call an OpenAI model directly with an API key. It invokes local `codex exec`. Each agent
completion runs in an ephemeral read-only directory and must return schema-constrained JSON.

## 3. Dr-CiK data

### 3.1 Download the official three-task sample

```bash
mkdir -p external
git clone https://github.com/ServiceNow/Dr-CiK.git external/Dr-CiK
```

The sample layout is:

```text
external/Dr-CiK/sample/
├── tasks/
│   ├── task_42.json
│   ├── task_163.json
│   └── task_201.json
└── documents/
```

Use this sample for smoke tests and mechanism inspection, not aggregate performance claims.

### 3.2 Load Public Dev or Hidden Test directly from Hugging Face

Most fixed baselines support:

```bash
python -m evolving_loop --baseline statistical --public-dev --limit 10
python -m evolving_loop --baseline chronos --hidden-test
```

`--public-dev` selects the 199 labeled public tasks. `--hidden-test` selects the 80 tasks whose
future labels are withheld. Hidden Test can produce submission files, but local forecast scores are
unavailable.

### 3.3 Download file-per-task JSON for evolution

The evolution loader accepts one JSON file, one JSONL file, or a directory of task JSON files. The
official archive can be downloaded as follows:

```bash
mkdir -p external/Dr-CiK/full-download
hf download ServiceNow/Dr-CiK Dr-CiK_public.tar.gz \
  --repo-type dataset \
  --local-dir external/Dr-CiK/full-download

tar -xzf external/Dr-CiK/full-download/Dr-CiK_public.tar.gz \
  -C external/Dr-CiK/full-download
```

The task path is then:

```text
external/Dr-CiK/full-download/Dr-CiK_public/tasks
```

Only tasks containing resolved `future_values` are eligible for training and evolution. Hidden Test
labels are never an evolution signal.

### 3.4 Input argument summary

| Argument | Accepted input | Intended use |
|---|---|---|
| `--sample-dir` | Dr-CiK sample root containing `tasks/` | Fixed baselines and fast `evolving-harness` tests |
| `--public-dev` / `--hidden-test` | Automatic Hugging Face loading | Most fixed baselines |
| `--tasks-file` | One JSON, JSONL, or a directory of task JSON files | New harness, skill learning, and evolution |

Exception: the legacy `skill-fresh` and `skill-library` baselines accept labeled JSONL only. The
normalized Hugging Face `data/tasks/train.jsonl` is not the same schema expected by this legacy
loader. Build a compatible JSONL from the downloaded file-per-task directory:

```bash
mkdir -p external/Dr-CiK/derived
python -c 'import json; from pathlib import Path; source=Path("external/Dr-CiK/full-download/Dr-CiK_public/tasks"); destination=Path("external/Dr-CiK/derived/public_tasks.jsonl"); destination.write_text("\n".join(json.dumps(json.loads(item.read_text())) for item in sorted(source.glob("*.json"))) + "\n")'
```

## 4. Verify the installation

List every method:

```bash
python -m evolving_loop --list-methods
```

Run the tests:

```bash
pytest -q
```

Run a lightweight real-task smoke test without an LLM or TSFM:

```bash
python -m evolving_loop \
  --baseline statistical \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --samples 25 \
  --output-dir outputs/smoke/statistical-task42
```

## 5. Current three-agent architecture

```text
historical numbers
    -> Coding Agent
    -> executable assumptions, Python programs, and/or a TSFM candidate
    -> sandbox execution
    -> historical hindcast validation

candidate assumptions + document corpus
    -> Retrieval Agent
    -> evidence that distinguishes candidates
    -> exact-quote verification
    -> structured evidence-to-impact translation

executed candidates + verified evidence
    -> Decision Agent
    -> preserve the hindcast winner or make a citation-backed switch
    -> final forecast

future values arrive after inference
    -> Coding, Retrieval, Decision, and forecast metrics
    -> reusable skill learning and/or outer evolution feedback
```

### 5.1 Coding Agent

At inference time, the Coding Agent may see:

- historical values;
- forecast horizon;
- frequency and the supplied seasonal-period field;
- summaries of previously validated numerical skills;
- execution errors and historical hindcast errors for its own programs.

It may not see documents, retrieved evidence, `gt_evidence`, or real future values.

Its inner loop is:

1. generate candidates according to `--setting`;
2. require each generated candidate to include an assumption, failure condition, and executable
   Python `forecast` function;
3. execute each program in the sandbox;
4. construct rolling hindcast folds from the end of the observed history;
5. rank candidates by historical hindcast sMAPE;
6. send the best generated parent and its fold errors to the LLM for mutation;
7. promote a mutation to the next mutation round only when its hindcast improves;
8. choose the lowest-hindcast-sMAPE candidate from all initial and mutated candidates;
9. save an executable Coding Skill only if it also beats repeat-last.

A hindcast temporarily hides an already observed part of the history and treats it as simulated
future data. It never exposes the actual evaluation horizon.

### 5.2 Retrieval Agent

The Retrieval Agent receives task identity and time ranges, the document corpus, Coding assumptions
and failure conditions, hindcast scores, validated Retrieval Skills, and any unresolved gaps from a
prior retrieval round.

Its goal is not generic semantic similarity. It should find evidence that can falsify an assumption,
distinguish candidates, identify a resolved anomaly, identify a future event, or otherwise change
the forecast decision.

Every claim must contain a document ID and an exact quote. A Python verifier confirms that the quote
actually occurs in the named document. Accepted evidence is then translated into a structured
impact containing:

- mechanism layer: `observation`, `latent_process`, `future_driver`, `regime`, or `irrelevant`;
- temporal relation to the forecast horizon;
- direction and permanence;
- `preserve`, `multiply`, `add`, or no numerical action;
- an explicit event window and magnitude when applicable.

Ungrounded quotes, wrong document IDs, and quantitative adjustments without explicit magnitude and
timing are rejected or downgraded to no numerical change.

### 5.3 Decision Agent

The Decision Agent must choose among already executed candidates. It cannot invent a trajectory.

The host default is the candidate with the lowest historical hindcast sMAPE. An override must select
a known candidate, cite verified evidence, and specifically falsify the default assumption or support
the alternative. Evidence-adjusted candidates must cite every document used to construct them.

Invalid JSON, unknown candidates, unverified citations, or unsupported overrides fall back to the
historically validated host default.

### 5.4 Evidence-adjusted candidates

When verified evidence explicitly provides a future event window and numerical effect, the host may
derive a bounded candidate from the current hindcast winner:

```text
base candidate: forecast y for 20 steps
evidence: explicit +20% effect from steps 5 through 10
derived candidate: y x 1.20 only for steps 5 through 10
```

This is a traceable restricted programmatic edit, not free-form LLM number generation.

## 6. Fixed baselines

General form:

```bash
python -m evolving_loop \
  --baseline <name> \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/baselines/<name>-task42
```

### 6.1 `chronos`

Runs `amazon/chronos-bolt-small` on numbers only. It reads no documents and runs no Retrieval or
Decision Agent.

```bash
python -m evolving_loop \
  --baseline chronos \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --chronos-device-map cpu \
  --output-dir outputs/baselines/chronos-task42
```

An offline cached run can add `--chronos-cache-dir <path> --chronos-local-files-only`. Chronos load
failures raise an error unless `--allow-statistical-fallback` is explicitly supplied. Do not enable
that fallback in a formal Chronos comparison because it changes the method being evaluated.

### 6.2 `timesfm`

Runs TimesFM on numbers only. It is an alternative TSFM comparison and does not consume documents.

```bash
python -m evolving_loop \
  --baseline timesfm \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/baselines/timesfm-task42
```

### 6.3 `statistical`

Runs a deterministic, LLM-free statistical model. It uses seasonal naive or drifted seasonal naive
when a seasonal lag is detected, otherwise linear trend extrapolation. It uses no documents.

```bash
python -m evolving_loop \
  --baseline statistical \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/baselines/statistical-task42
```

This is not the same as `--setting statistics`. The baseline is one fixed implementation; the
setting asks an LLM to generate multiple programs with a statistical-method dictionary.

### 6.4 `one-pass`

Runs one numerical diagnosis, one retrieval pass, evidence synthesis, and probabilistic forecasting.
It has no iterative gap filling and no self-evolution.

```bash
python -m evolving_loop \
  --baseline one-pass \
  --sample-dir external/Dr-CiK/sample \
  --backbone chronos \
  --top-k 5 \
  --output-dir outputs/baselines/one-pass
```

### 6.5 `iterative`

Runs the earlier safe contextual loop:

```text
gap diagnosis -> query -> retrieve -> utility rerank -> verify evidence
              -> sufficiency check -> repeat if needed
              -> macro/micro reasoning -> restricted revise-or-preserve
```

The safe mode permits automatic revision only for explicit future values or a history-validated
normal-regime projection. Generic textual up/down claims do not become arbitrary multipliers.

```bash
python -m evolving_loop \
  --baseline iterative \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --backbone chronos \
  --max-steps 10 \
  --top-k 5 \
  --output-dir outputs/baselines/iterative-task42
```

### 6.6 `iterative-unsafe`

Runs the same loop while allowing generic text-derived add/multiply revisions without historical
validation. This is a negative safety ablation, not a recommended runtime.

```bash
python -m evolving_loop \
  --baseline iterative-unsafe \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/baselines/iterative-unsafe-task42
```

### 6.7 `oracle-context`

Bypasses real retrieval and supplies public `gt_evidence` to the safe iterative forecaster. It asks
how well the downstream evidence-to-number stage could perform under perfect retrieval. It is a
ceiling diagnostic, not a deployable baseline, and is forbidden on Hidden Test.

```bash
python -m evolving_loop \
  --baseline oracle-context \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/baselines/oracle-task42
```

### 6.8 `rules-triad`

Runs the earlier fixed three-agent implementation with deterministic reasoning:

- Coding creates fixed backbone, statistical, robust-history, and local-level families;
- Retrieval uses rule-based retrieval and evidence-to-impact translation;
- Decision applies validation and evidence-compatibility rules;
- multiple retrieval/decision rounds are possible, but prompts and source never evolve.

```bash
python -m evolving_loop \
  --baseline rules-triad \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --max-steps 3 \
  --output-dir outputs/baselines/rules-triad-task42
```

### 6.9 `codex-triad`

Uses the same earlier Triad host, but backs the Coding, Retrieval, and Decision reasoning roles with
schema-constrained Codex calls. Chronos/Python still execute and hindcast numerical candidates.

This is not the current `evolving_loop` harness and does not run prompt, genome, or source evolution.

```bash
python -m evolving_loop \
  --baseline codex-triad \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --max-steps 3 \
  --output-dir outputs/baselines/codex-triad-task42
```

### 6.10 `codex-direct`

Gives Codex the task, Chronos baseline, and complete local corpus. Codex researches the documents
and emits the complete future array directly. The host validates its length and types; invalid
output falls back to the numerical backbone.

This tests a strong direct-LLM baseline but lacks executable-candidate comparison and restricted
numerical edits.

```bash
python -m evolving_loop \
  --baseline codex-direct \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --output-dir outputs/baselines/codex-direct-task42
```

### 6.11 `codex-contract`

Codex emits a structured, falsifiable Forecast Contract rather than a trajectory. The contract
describes the expected regime, whether an event has ended, and whether baseline seasonality should
be preserved. Python then validates citations, opens compatible model gates, independently builds
and validates numerical candidates, and executes changes through the restricted workspace.

This separation of textual reasoning from numerical generation is the most stable agent baseline
among the aggregate runs currently stored in the repository.

```bash
python -m evolving_loop \
  --baseline codex-contract \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --output-dir outputs/baselines/codex-contract-task42
```

### 6.12 `skill-fresh`

This is the collaborator's original Voyager-style numbers-only skill baseline, not the three-agent
harness:

```text
numeric task -> local Qwen writes a Python skill -> sandbox
             -> one error-informed retry -> repeat-last if both attempts fail
```

Fresh mode neither stores nor reuses skills.

```bash
python -m evolving_loop \
  --baseline skill-fresh \
  --tasks-file external/Dr-CiK/derived/public_tasks.jsonl \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --device cuda:0 \
  --limit 10 \
  --output-dir outputs/baselines/skill-fresh
```

### 6.13 `skill-library`

Uses the same one-task workflow but can retrieve, execute, create, persist, and score reusable
Python skills across tasks.

```bash
python -m evolving_loop \
  --baseline skill-library \
  --tasks-file external/Dr-CiK/derived/public_tasks.jsonl \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --device cuda:0 \
  --library-path runs/skill-baseline/coding_skills.json \
  --limit 10 \
  --output-dir outputs/baselines/skill-library
```

The two legacy `skill-*` baselines currently use Qwen regardless of `--llm-backend`. Use
`evolving-harness` or an evolution mode for the Codex-backed implementation. Build the compatible
JSONL from the file-per-task archive as described in Section 3.4; do not pass the normalized Hugging
Face task table directly to this legacy loader.

### 6.14 `evolving-harness`

Evaluates a fixed snapshot of the current Coding/Retrieval/Decision harness:

- runs the current seed or supplied policy;
- uses the selected Coding `--setting`;
- defaults to Codex for all three reasoning roles;
- sandboxes and hindcasts Coding programs;
- verifies Retrieval citations in Python;
- restricts Decision to executed candidates;
- does not run outer prompt/genome/source evolution;
- does not persist skills learned from the evaluation tasks.

```bash
python -m evolving_loop \
  --baseline evolving-harness \
  --sample-dir external/Dr-CiK/sample \
  --setting statistics \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --limit 3 \
  --output-dir outputs/baselines/evolving-harness-sample
```

## 7. Four Coding Agent settings

These settings control how the Coding Agent produces numerical candidates. They are not outer
evolution modes.

| Setting | LLM Python programs | Statistical dictionary | Real Chronos candidate | Research question |
|---|---:|---:|---:|---|
| `llm_only` | Yes | No | No | Can the LLM invent useful forecasting methods from scratch? |
| `statistics` | Yes | Yes | No | Does statistical prior knowledge improve hypotheses and code? |
| `tsfm` | No | No | Yes | How strong is the TSFM candidate alone? |
| `combined` | Yes | Yes | Yes | Are transparent programs and a TSFM complementary? |

### 7.1 `llm_only`

The LLM receives numbers and the executable output contract, but no statistical dictionary and no
TSFM candidate. It must create the assumptions, failure conditions, and Python programs itself.

### 7.2 `statistics`

This is the default. The LLM also receives a non-exhaustive dictionary containing robust local
level, damped local trend, seasonal naive, seasonal trend, moving-average residual, and
Fourier/harmonic ideas. It may combine, modify, or go beyond them. The dictionary does not select a
fixed answer; historical hindcast still ranks the generated programs.

### 7.3 `tsfm`

Executes a real Chronos candidate and applies the same historical hindcast protocol. The current
implementation does not ask the LLM to create extra Python programs in this setting. Retrieval and
Decision can still use the selected LLM backend. For a completely LLM-free run, use the fixed
`chronos` baseline instead.

### 7.4 `combined`

Combines statistical-dictionary-guided Python programs, existing Coding Skills, a real Chronos
candidate, and mutation of generated programs. All candidates are compared by the same hindcast
protocol; the LLM does not imitate Chronos weights.

Example setting comparison:

```bash
python -m evolving_loop --baseline evolving-harness \
  --sample-dir external/Dr-CiK/sample --setting llm_only \
  --output-dir outputs/settings/llm-only

python -m evolving_loop --baseline evolving-harness \
  --sample-dir external/Dr-CiK/sample --setting statistics \
  --output-dir outputs/settings/statistics

python -m evolving_loop --baseline evolving-harness \
  --sample-dir external/Dr-CiK/sample --setting tsfm \
  --output-dir outputs/settings/tsfm

python -m evolving_loop --baseline evolving-harness \
  --sample-dir external/Dr-CiK/sample --setting combined \
  --output-dir outputs/settings/combined
```

## 8. Skill libraries

The current system has three separate stores:

```text
runs/evolving/skills.json             # executable Coding programs
runs/evolving/retrieval_skills.json   # retrieval and verification strategies
runs/evolving/decision_skills.json    # candidate-selection rules
```

Missing files start as empty libraries. Prompts expose applicable skill summaries to agents, but
the Python host—not the LLM—validates and writes the JSON files.

### 8.1 Coding Skill gate

A generated program is stored only if it executes in the sandbox, returns the correct horizon,
completes historical hindcasts, beats repeat-last, and comes from LLM generation or mutation rather
than an external TSFM.

### 8.2 Retrieval Skill gate

After public outcomes resolve, the mean of retrieval precision, supporting recall, and distractor
avoidance must reach the current 0.6 threshold, and verified evidence must exist.

### 8.3 Decision Skill gate

At least two candidates must exist and the selected candidate must have zero selection regret
against the best candidate available in that set.

### 8.4 Anti-memorization gate

The Skill Curator rejects task IDs, document IDs, entity names, exact timestamps, realized future
values, and task-specific answers. A saved skill must describe a cross-task strategy.

### 8.5 Explicit public-training skill run

The fixed `evolving-harness` baseline does not persist evaluation-task learning. To learn all three
skill libraries on an explicitly authorized public training stream, use the legacy-compatible `run`
subcommand:

```bash
python -m evolving_loop run \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --setting statistics \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --learn-from-public-outcomes \
  --limit 20 \
  --library-path runs/training/coding_skills.json \
  --retrieval-library-path runs/training/retrieval_skills.json \
  --decision-library-path runs/training/decision_skills.json \
  --results-path runs/training/results.jsonl
```

Never enable outcome learning on Hidden Test.

## 9. Outer evolution modes

General form:

```bash
python -m evolving_loop \
  --evolution <prompt|genome|source> \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --setting statistics \
  --llm-backend codex \
  --generations 3 \
  --children 2
```

### 9.1 Shared train/dev/holdout protocol

All three modes:

1. shuffle entities using `--seed`;
2. reserve complete entities for Dev and untouched Public Holdout according to
   `--dev-fraction` and `--holdout-fraction`;
3. run the parent on Train;
4. strip `future_values`, `gt_evidence`, and document role/subtype before every inference call;
5. score only after label-free inference completes;
6. use module rewards and the worst failure trajectories to generate children;
7. evaluate all children on Train and retain the Train-best child;
8. evaluate only that child on held-out Dev;
9. accept it only when its Dev reward is strictly greater than the parent's Dev reward;
10. otherwise preserve the parent;
11. after evolution is frozen, evaluate the accepted artifact once on Public Holdout.

An LLM-proposed change is therefore not an accepted evolution until it passes the Dev gate. Public
Holdout never generates a child, selects a child, or changes the accepted artifact. Every evolution
run writes `--split-manifest-path`, which records task IDs and entities for all three partitions.

For the full 199-task public set, do not pass `--limit`. Use the same seed and fractions for all
three compared modes:

```bash
--seed 7 --dev-fraction 0.25 --holdout-fraction 0.20 \
--split-manifest-path runs/<mode>/split_manifest.json
```

The repository provides a wrapper that enforces these shared settings and isolates all outputs:

```bash
scripts/run_llm_only_evolutions.sh \
  external/Dr-CiK/full-download/Dr-CiK_public/tasks all
```

Run only one condition by replacing `all` with `prompt`, `genome`, or `source`. Common overrides are
environment variables, for example:

```bash
EA_GENERATIONS=1 EA_CHILDREN=1 EA_LIMIT=5 EA_DRY_RUN=1 \
  scripts/run_llm_only_evolutions.sh /path/to/tasks all
```

The main controls are `EA_CODEX_MODEL`, `EA_REASONING_EFFORT`, `EA_GENERATIONS`, `EA_CHILDREN`,
`EA_SEED`, `EA_DEV_FRACTION`, `EA_HOLDOUT_FRACTION`, `EA_LIMIT`, and `EA_RUNS_DIR`. Source mode
requires a clean tracked worktree. Run artifacts remain untracked.

The wrapper's formal-pilot default is **2 generations × 3 children**. Every mode writes
`checkpoint.json` and append-only `progress.jsonl` beside its final trace. Re-running the same
command resumes from the last completed generation and reuses validated Codex call caches. Use
`--no-resume` only when intentionally starting a new lineage. Malformed model JSON is parsed
leniently and then sent through at most two syntax-only repair calls; an unrepaired response or a
failed child is rejected without terminating the other candidates.

### 9.2 `--evolution prompt`

The weakest training module is identified, and exactly one complete prompt belonging to that role
may be replaced:

- Coding: generation or revision prompt;
- Retrieval: retrieval prompt;
- Decision: decision prompt.

Candidate budgets, hindcast settings, topology, aggregation, source, scorer, and safety boundaries
remain fixed. This is the controlled prompt-only baseline.

```bash
python -m evolving_loop \
  --evolution prompt \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --setting statistics \
  --llm-backend codex \
  --limit 30 \
  --generations 3 \
  --children 2 \
  --policy-path runs/prompt/best_policy.json \
  --trace-path runs/prompt/evolution_trace.json
```

### 9.3 `--evolution genome`

The Meta-Harness may jointly change:

- all four role prompts;
- the number of initial Coding programs;
- mutation rounds and mutation children;
- hindcast fold count and validation horizon;
- the order and repetition of `retrieve` and `decide` stages;
- whether evidence-adjusted candidates are enabled;
- the maximum number of evidence adjustments;
- last-round versus majority aggregation across decisions.

The data split, label firewall, scorer, sandbox, citation checks, resource caps, and held-out
acceptance test remain immutable.

```bash
python -m evolving_loop \
  --evolution genome \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --setting combined \
  --llm-backend codex \
  --limit 30 \
  --generations 3 \
  --children 2 \
  --policy-path runs/genome/best_policy.json \
  --trace-path runs/genome/evolution_trace.json
```

Continue from an accepted genome:

```bash
python -m evolving_loop \
  --evolution genome \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --setting combined \
  --seed-policy-path runs/genome/best_policy.json \
  --policy-path runs/genome-next/best_policy.json \
  --trace-path runs/genome-next/evolution_trace.json \
  --generations 3 \
  --children 2
```

### 9.4 `--evolution source`

This is the most open and expensive mode. In an isolated Git worktree, the Codex Source Engineer
may rewrite the Coding, Retrieval, Decision, and Harness implementations; add modules under
`evolving_loop/generated/`; create new agent roles; and redesign ranking, validation, memory,
stopping, or communication algorithms.

It may not edit the CLI, data loaders, scorer, LLM transport, sandbox, evolution host, tests, or
label firewall.

Each child follows:

```text
Codex source edit -> isolated worktree -> path/AST/text safety audit -> full pytest
                  -> Train evaluation -> select Train-best child
                  -> held-out Dev evaluation -> accept only on Dev improvement
                  -> save cumulative patch
```

Source evolution requires a clean tracked worktree.

```bash
python -m evolving_loop \
  --evolution source \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --setting statistics \
  --llm-backend codex \
  --limit 20 \
  --generations 1 \
  --children 1 \
  --source-patch-path runs/source/best_source.patch \
  --trace-path runs/source/evolution_trace.json
```

The accepted patch is not automatically applied to the current branch. Inspect and apply it
explicitly:

```bash
git apply --check runs/source/best_source.patch
git apply runs/source/best_source.patch
pytest -q
```

When no child passes the Dev gate, the best patch remains the incumbent and may be empty on the
first run.

Continue from an accepted source patch:

```bash
python -m evolving_loop \
  --evolution source \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --seed-source-patch runs/source/best_source.patch \
  --source-patch-path runs/source-next/best_source.patch \
  --trace-path runs/source-next/evolution_trace.json \
  --generations 1 \
  --children 1
```

## 10. Frozen public/hidden inference

Evolution and inference are separate operations. `--evolution` may read resolved public outcomes;
`--inference` never learns skills, mutates an artifact, or invokes the scorer unless
`--score-public` is explicitly requested.

Evaluate a frozen prompt/genome artifact on the untouched Public Holdout:

```bash
python -m evolving_loop \
  --inference genome \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --split-manifest runs/genome/split_manifest.json \
  --split-name holdout \
  --policy-path runs/genome/best_policy.json \
  --setting llm_only \
  --score-public \
  --output-dir outputs/genome-public-holdout
```

Generate the official-format Hidden Test files without local scores:

```bash
python -m evolving_loop \
  --inference genome \
  --hidden-test \
  --policy-path runs/genome/best_policy.json \
  --setting llm_only \
  --samples 100 \
  --output-dir outputs/genome-hidden
```

For source evolution, run the accepted cumulative patch in an isolated worktree:

```bash
python -m evolving_loop \
  --inference source \
  --hidden-test \
  --source-patch-path runs/source/best_source.patch \
  --setting llm_only \
  --samples 100 \
  --output-dir outputs/source-hidden
```

Hidden rows are retained by the inference loader but their `future_values`, `gt_evidence`, document
`role`, and document `subtype` fields are cleared before mutable code runs. `--score-public` is
rejected for Hidden Test. The output contains `forecasts.jsonl`, `deep_research.jsonl`,
`run_report.jsonl`, and `summary.json`.

The current evolving harness emits a point forecast. For schema compatibility it exports that point
forecast 100 times, producing a degenerate empirical distribution. This supports submission plumbing
but is not a complete probabilistic model; calibrated trajectory generation remains future work.

## 11. Inner Coding evolution versus outer harness evolution

| Level | Runs per task? | Mutable object | Feedback | Persists across tasks? |
|---|---:|---|---|---:|
| Inner Coding search | Yes | Current task's Python programs | Historical hindcast sMAPE and execution errors | Through validated Coding Skills |
| Outcome Skill learning | Public training only | Generalized Coding/Retrieval/Decision Skills | Resolved future and public retrieval labels | Three JSON libraries |
| Prompt evolution | Per generation | One role prompt | Train failure trajectories plus Dev reward | Policy JSON |
| Genome evolution | Per generation | Prompts, budgets, topology, and aggregation | Train failure trajectories plus Dev reward | Genome JSON |
| Source evolution | Per generation | Agent and orchestration Python | Train failures, tests, and Dev reward | Cumulative patch |

Mutating a forecasting program inside one task does not mean the whole three-agent harness has
changed. Outer evolution changes the framework inherited by later tasks.

## 12. Output files

### 11.1 Most fixed Dr-CiK baselines

`chronos`, `timesfm`, `statistical`, `one-pass`, `iterative`, `iterative-unsafe`, `oracle-context`,
`rules-triad`, `codex-triad`, `codex-direct`, and `codex-contract` write:

```text
<output-dir>/
├── forecasts.jsonl       # Dr-CiK forecast submission format
├── deep_research.jsonl   # evidence submission format
├── run_report.jsonl      # per-task diagnosis, forecast, and metrics
├── loop_trace.jsonl      # only when a multi-step trace exists
└── summary.json          # aggregate development metrics
```

### 11.2 `skill-fresh` and `skill-library`

```text
<output-dir>/results.jsonl
<output-dir>/run.log
<library-path>             # library mode only
```

### 11.3 `evolving-harness`

`<output-dir>/results.jsonl` contains the resolved outcome, selected candidate, host default,
retrieved documents, rejection reasons, used skills, and every Coding assumption and hindcast score.

### 11.4 Prompt and genome evolution

```text
<policy-path>   # final accepted HarnessPolicy JSON
<trace-path>    # parent/child train and dev rewards, target role, acceptance
```

### 11.5 Source evolution

```text
<source-patch-path>   # final accepted cumulative Git patch
<trace-path>          # audits, tests, rewards, and rejection reasons
```

Use a separate output path for every experiment. Existing files may otherwise be overwritten.

## 12. Metrics

### 12.1 MAE

Mean absolute error averages the absolute difference between each predicted and true future value.
It retains the target's unit and is lower-is-better. MAE should not be compared directly across
targets with very different scales.

### 12.2 sMAPE

For each step, sMAPE divides absolute prediction error by the sum of the absolute prediction and
truth, then averages across the horizon. The implementation ranges from 0 to 200 and is
lower-is-better. It is more scale-comparable than MAE, but near-zero series still require care.

### 12.3 Retrieval metrics

- retrieval precision: fraction of retrieved documents that are supporting;
- supporting recall: fraction of all supporting documents retrieved;
- distractor avoidance: for non-empty citation sets, one minus the fraction of cited documents
  that are distractors; empty citation sets receive 1.0 in this local diagnostic.

All are higher-is-better. Document roles are used only after inference for scoring.

### 12.4 Coding and Decision diagnostics

- Coding oracle sMAPE: post-outcome error of the best candidate available in the generated set;
- Decision selection regret: selected-candidate sMAPE minus best-available-candidate sMAPE.

Lower is better. Zero Decision regret means the selector chose the best available candidate.

### 12.5 System reward

Outer evolution currently uses a 0-to-1 higher-is-better score:

```text
system reward = 80% transformed final-forecast score + 20% retrieval score
```

The retrieval score is the mean of precision, recall, and distractor avoidance. Coding and Decision
module rewards diagnose the weakest role but are not independently weighted into the current system
reward; they affect it indirectly through the final forecast.

The main runner's sMAE, sRMSE, and sCRPS are transparent local development proxies. Official Hidden
Test scores are calculated by Dr-CiK maintainers and must not be conflated with these proxies.

## 13. Recommended experiment order

1. Validate data and fixed `statistical`, `chronos`, and `codex-contract` baselines.
2. Compare the four Coding settings with identical data, LLM, budgets, seed, and hindcast protocol.
3. Run prompt evolution to test whether one controlled role prompt can improve held-out results.
4. Run genome evolution only after the fixed harness is stable.
5. Run source evolution only when evaluation is stable, task coverage is sufficient, and the much
   larger compute budget is acceptable.

## 14. Fair comparison rules

Keep the following identical across compared methods:

- task set, order, and entity-disjoint split;
- LLM, model version, and reasoning effort;
- call, candidate, retrieval, and wall-clock budgets;
- backbone checkpoint;
- seed and number of probabilistic samples;
- metric implementation;
- candidate output contract and hindcast protocol when comparing Coding settings.

Use separate output directories. Never use Hidden Test to generate or accept prompts, skills,
policies, or source patches.

The new harness shuffles before applying `--limit`, so record both `--limit` and `--seed`. A frozen
task manifest is preferable for final comparisons.

## 15. Interpreting the existing results

See [`BASELINE_METHODS_AND_RESULTS.md`](BASELINE_METHODS_AND_RESULTS.md) for the full audit. The key
stored observations are:

- unrestricted early context revision worsened 199-task MAE by 61.70%, demonstrating the need for
  an evidence gate;
- the later safe gate prevented broad harm but produced only a small 199-task improvement;
- Codex Contract improved mean MAE by 3.85% on a frozen 30-task subset, with 2 improved, 28
  unchanged, and 0 harmed tasks;
- the earlier free-form Codex Triad worsened MAE by 34.84% on the corresponding 30-task diagnostic,
  motivating executable-candidate constraints and Decision fallback;
- the old Codex Triad reduced `task_42` MAE from 72.73 to 31.65, but this is a one-task mechanism
  diagnostic, not an aggregate claim;
- one genome child passed its Dev gate, while one source child passed audit and tests but worsened
  Train and Dev rewards and was correctly rejected.

Results using different task sets, backbones, or metric families are not directly comparable.

## 16. Troubleshooting

### `Codex CLI was not found`

Run `codex --version`. Install Codex if the command is unavailable; if it exists but agent calls fail,
launch `codex` once and complete authentication.

### `Chronos is configured but is not installed`

```bash
pip install -e '.[chronos]'
```

### Model download failure

Confirm Hugging Face access, or point to an existing cache and use
`--chronos-local-files-only`. Do not silently enable statistical fallback in a formal backbone run.

### `source evolution requires a clean tracked worktree`

Source mode refuses to begin when tracked changes exist. Commit or safely preserve the work first.
Do not use destructive Git commands that could discard collaborator changes.

### Evolution produces an empty Train or Dev split

Evolution splits by entity. A single task cannot produce a valid entity-disjoint Train/Dev split.
Use more tasks or the complete Public Dev set.

### Evidence was found but the forecast did not change

This is expected when evidence does not overlap the future, lacks an explicit numerical effect,
fails exact-quote verification, does not match a candidate mechanism, or cannot justify a Decision
override. Correct retrieval does not imply that a strong numerical baseline must be changed.

### A skill file was not created

Possible causes include a read-only fixed evaluation, outcome learning not being enabled, failure to
pass a skill quality gate, a Coding program not beating repeat-last, or a different configured path.

## 17. Minimal copy-paste workflow

```bash
git clone https://github.com/kaichen-z/time-series.git
cd time-series
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,huggingface,chronos]'

mkdir -p external
git clone https://github.com/ServiceNow/Dr-CiK.git external/Dr-CiK

pytest -q
python -m evolving_loop --list-methods

python -m evolving_loop \
  --baseline statistical \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/quick/statistical

python -m evolving_loop \
  --baseline evolving-harness \
  --sample-dir external/Dr-CiK/sample \
  --setting statistics \
  --llm-backend codex \
  --codex-reasoning-effort high \
  --limit 3 \
  --output-dir outputs/quick/evolving-harness
```

Before a long experiment, download the file-per-task Public Dev archive and give every method its own
output, policy, trace, and skill-library paths.
