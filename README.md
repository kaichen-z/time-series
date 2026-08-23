# Foresight-Driven Retrieval for Time-Series Forecasting

Repository: <https://github.com/kaichen-z/time-series>

## New canonical evolving-agent implementation

The collaborator's top-level [`evolving_loop/`](evolving_loop/) package is now the base for the
self-evolving experiment. The original Fresh-vs-Skill-Library baseline is preserved. The integrated
version adds numbers-only program evolution with historical hindcasting, outcome-validated
Retrieval and Decision skill libraries, verified contextual retrieval, a citation-constrained
Decision Agent, a real Chronos ablation, and failure-attributed three-agent co-evolution with
held-out acceptance. The `evolve` entrypoint has three controlled levels:
`--evolution-mode prompt` (one prompt only), `genome` (prompts, budgets, and topology), and
`source` (audited Python source patches generated in isolated Git worktrees). See
[`docs/EVOLVING_AGENT.md`](docs/EVOLVING_AGENT.md) for the exact information boundaries, flow,
metrics, and commands. The initial English experiment report is available at
[`docs/EVOLUTION_METHODS_AND_RESULTS.md`](docs/EVOLUTION_METHODS_AND_RESULTS.md). A consolidated
audit of all earlier baseline families and saved results is available at
[`docs/BASELINE_METHODS_AND_RESULTS.md`](docs/BASELINE_METHODS_AND_RESULTS.md).
For a concise map of the generic Self-Harness, dictionary curation adapter, three-agent
Meta-Harness, evolution depths, and data protocol, see
[`docs/SELF_EVOLUTION_FRAMEWORK.md`](docs/SELF_EVOLUTION_FRAMEWORK.md).
The complete installation, baseline, Coding-setting, skill-learning, and evolution manual is
available at [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md).

## Parameterized Self-Evolution framework

The repository also contains a domain-independent Parent/Child evolution core plus a numerical
tool-dictionary curation adapter. The framework accepts externally supplied base methods,
implementers, runtimes, tasks, metrics, and evolution parameters. It now ships an auditable
**166-method / 115-source** definition dataset spanning Statistical, TSFM, and Combined methods;
these are method cards, not bundled executable forecasting runtimes. See
[`numerical_agent/README.md`](numerical_agent/README.md) for the component contracts, artifacts,
release build command, and offline fake-provider smoke command.

An opt-in Setting 2 extension adds diagnostic-selected, source-backed forecasting guidance to the
existing Coding evolution loop. See [`docs/setting2/README.md`](docs/setting2/README.md) for the
integration flow, knowledge provenance, commands, and historical results.

This repository contains an auditable, gap-guided agent loop for the
[Dr-CiK](https://github.com/ServiceNow/Dr-CiK) contextual time-series forecasting
benchmark. Its research hypothesis is that a passage should be retrieved because it is
expected to improve the downstream forecast—not merely because it is lexically similar
to a query. A numerical forecasting backbone creates an immutable baseline; a structured
retrieval controller fills explicit information gaps; and a reviser may change only the
future forecast through a small, evidence-backed action language.

The numerical backbone is now Amazon Chronos-Bolt by default. The paper's
forecast workspace and restricted actions are combined with this project's
structured gap judging, forecast-utility retrieval, evidence grounding, and
evidence-to-impact translation. The current runtime remains deterministic and exposes
interfaces for learned retrievers, judges, and PostTime-style revisers. It does not claim
that its label-free utility proxy is an already trained PRM.

## Unified experiment command

All executable baselines and evolution levels now share one Python entrypoint. List the names with:

```bash
python -m evolving_loop --list-methods
```

Run a baseline by name:

```bash
python -m evolving_loop \
  --baseline chronos \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --output-dir outputs/baselines/chronos-task42

python -m evolving_loop \
  --baseline codex-triad \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --output-dir outputs/baselines/codex-triad-task42
```

Run one evolution level by name:

```bash
python -m evolving_loop \
  --evolution genome \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --generations 3 \
  --children 2
```

Run all three controlled `llm_only` evolution conditions with one reproducible Bash script:

```bash
scripts/run_llm_only_evolutions.sh \
  external/Dr-CiK/full-download/Dr-CiK_public/tasks all
```

The script gives `prompt`, `genome`, and `source` the same model, seed, entity split, generations,
children, and timeouts while keeping every artifact in a separate subdirectory under
`runs/llm_only_evolution/`. Use `EA_DRY_RUN=1` to print the three commands without executing them.

Evolution now reserves entity-disjoint Train, Dev, and Public Holdout partitions and writes a split
manifest. After freezing the artifact, run label-free Hidden Test inference separately:

```bash
python -m evolving_loop \
  --inference genome \
  --hidden-test \
  --policy-path runs/evolving/best_policy.json \
  --setting llm_only \
  --output-dir outputs/evolving-hidden
```

Use `--inference prompt` for a prompt policy or `--inference source
--source-patch-path runs/evolving/best_source.patch` for an accepted source patch. Hidden inference
writes submission-format files and never scores, learns skills, or evolves the harness.

Baseline names are `skill-fresh`, `skill-library`, `chronos`, `timesfm`, `statistical`, `one-pass`, `iterative`,
`iterative-unsafe`, `oracle-context`, `rules-triad`, `codex-triad`, `codex-direct`,
`codex-contract`, and `evolving-harness`. Evolution names are `prompt`, `genome`, and `source`.
`chronos`, `timesfm`, and `statistical` are true numbers-only runs: they do not retrieve or consume
documents. `oracle-context` is restricted to public development diagnostics. The previous
`evolving-agent run/evolve` and `drcik-agent run-sample/run-hf` interfaces remain supported.
The two `skill-*` baselines use the original numbers-only JSONL interface through `--tasks-file`.

## Baseline methods and current results

### Method definitions

| Baseline | Numerical forecaster | Uses documents? | Agent behavior |
|---|---|:---:|---|
| **Chronos** | Chronos-Bolt Small | No | Numbers-only time-series foundation-model reference. |
| **TimesFM** | TimesFM 2.5 | No | Numbers-only TSFM alternative to Chronos. |
| **Statistical** | Trend, robust level, and seasonal-repeat programs | No | Deterministic non-TSFM reference. |
| **Skill Fresh** | LLM-generated Python forecasting skill | No | Generates a fresh executable skill for every task. |
| **Skill Library** | LLM-generated or previously saved Python skill | No | May create, save, retrieve, and reuse forecasting skills across tasks. |
| **One Pass** | Configured numerical backbone | Yes | Retrieves context once, then forecasts without iterative gap resolution. |
| **Iterative** | Configured numerical backbone | Yes | Repeatedly diagnoses gaps, retrieves and verifies evidence, and applies restricted revisions. |
| **Iterative Unsafe** | Configured numerical backbone | Yes | Allows weakly validated textual multiply/add revisions; retained as a negative ablation. |
| **Oracle Context** | Configured numerical backbone | GT evidence | Bypasses retrieval and supplies public ground-truth evidence; diagnostic only and not deployable. |
| **Rules Triad** | Chronos/Python candidates | Yes | Deterministic Coding, Retrieval, and Decision roles. |
| **Codex Direct** | Chronos prior plus a directly generated trajectory | Yes | Codex reads the complete task and directly emits the final forecast. |
| **Codex Contract** | Chronos plus Python-generated contract-compatible candidates | Yes | Codex emits a structured regime contract; Python hindcasts candidates and applies restricted revisions. |
| **Codex Triad** | Chronos/Python executable candidates | Yes | Separate Codex Coding, Retrieval, and Decision roles propose, investigate, and select candidates. |
| **Evolving Harness** | Coding-Agent-generated executable candidates | Yes | Current three-agent system before Prompt/Genome/Source evolution. |

### Full 199-task public-development results

These rows use the same public task set and the same Chronos backbone. Lower is better for all
forecasting metrics. The scaled metrics are local development proxies, not official hidden-test
scores.

| Method | Tasks | MAE | RMSE | sMAE proxy | sCRPS proxy | Improved | Harmed | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Chronos only** | 199 | 797.6268 | 1121.9784 | 2.6981 | 2.5370 | -- | -- | Numerical reference. |
| **Iterative Unsafe** | 199 | 1289.7688 | 1591.6977 | 2.8350 | 2.6919 | 12 | 29 | MAE worsened by 61.70%; relevant text was converted into harmful numerical edits. |
| **Safe grounded Iterative** | 199 | **797.1767** | **1121.4952** | **2.6687** | **2.5075** | 7 | 0 | Sparse conservative revisions; all seven accepted changes improved MAE. |
| **Oracle Context** | 199 | 797.1506 | 1121.4600 | 2.6641 | 2.5038 | -- | -- | Uses public GT evidence and is not deployable. |

### Frozen 30-task Codex results

| Method | Tasks | Baseline MAE | Final MAE | Baseline RMSE | Final RMSE | Improved / unchanged / harmed | Retrieval precision | Supporting recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Codex Contract** | 30 | 162.6270 | **156.3621** (-3.85%) | 301.0117 | **296.4659** (-1.51%) | 2 / 28 / 0 | 0.7791 | 0.4120 |
| **Codex Triad** | 30 | 162.9749 | **219.7548** (+34.84%) | 301.0281 | **335.9287** (+11.59%) | 11 / 8 / 11 | 0.4467 | 0.2805 |

The earlier Codex Triad generated useful candidates on some tasks, but its Decision role and runtime
were unreliable: the saved run records 683 Codex-stage calls and 252 stage failures. This negative
result motivated citation-constrained selection, fallback rules, and the current evolution work.

### Three-task development results

These runs use the three official public samples (`task_42`, `task_163`, and `task_201`) and are
mechanism checks rather than unbiased benchmark results.

| Method | Tasks | Baseline MAE | Final MAE | Final RMSE | Retrieval precision | Improved / unchanged / harmed |
|---|---:|---:|---:|---:|---:|---:|
| **One-Pass Statistical** | 3 | 9.4101 | 9.4101 | 17.9559 | 0.3333 | 0 / 3 / 0 |
| **Iterative Statistical** | 3 | 9.4101 | 9.4101 | 17.9559 | 0.4180 | 0 / 3 / 0 |
| **Chronos + Regime Retrieval** | 3 | 27.7416 | 14.0457 | 23.2439 | 0.4180 | 1 / 2 / 0 |
| **Chronos + Regime-Table Retrieval** | 3 | 27.7416 | **13.5600** | **22.3424** | 0.4180 | 2 / 1 / 0 |
| **Codex Contract** | 3 | 27.7416 | 14.0457 | 23.2439 | 0.9048 | 1 / 2 / 0 |
| **Contract + validated explicit points** | 3 | 27.7416 | 13.6800 | 22.5559 | 0.9048 | 2 / 1 / 0 |

### Additional saved diagnostics and missing comparisons

| Method/run | Scope | Recorded result | Status |
|---|---|---|---|
| **Codex Direct** | `task_42` | MAE 72.7346 -> 72.7346; retrieval precision 1.0000 | Found clean evidence but did not convert it into an effective numerical change. |
| **Rules Triad** | `task_42` | MAE 72.7346 -> 31.6472 | Single-task mechanism result. |
| **Codex Contract / improved Codex Triad** | `task_42` | MAE 72.7346 -> 31.6472 | Single-task mechanism result; strongly exposed during development. |
| **Numbers-only Coding self-evolution** | `task_42` | Initial best MAE 47.3090; post-mutation MAE 47.3090 | Mutation was rejected; validates the inner gate but shows no evolution gain. |
| **Initial Evolving Harness replay** | `task_42` | MAE 104.4267; sMAPE 23.6486 | Different candidate system; smoke test only. |
| **Skill Library** | 50 tasks | Mean sMAPE 53.6033; first half 53.6158; second half 53.5908 | 25 skill writes, 10 reuses, and 15 fallbacks; no matched Skill-Fresh comparison. |
| **Skill Fresh** | -- | Not yet recorded on a matched task set | Implemented, but no complete comparable result. |
| **TimesFM** | -- | No complete comparable aggregate recorded | Implemented numbers-only alternative. |
| **Prompt/Genome/Source evolution** | New matched 30-task pilot | Running | Not reported as a completed result. |

The main comparison for the current project is therefore:

```text
Chronos
vs Codex Direct
vs Codex Contract
vs Codex Triad
vs the evolved three-agent harness
```

These numbers come from local public-development experiments, not the official Dr-CiK hidden-test
leaderboard. Results from different task sets are not directly comparable. The new matched
Prompt/Genome/Source evolution pilot is still running and is intentionally not reported here as a
completed result. See [`docs/BASELINE_METHODS_AND_RESULTS.md`](docs/BASELINE_METHODS_AND_RESULTS.md)
for the complete audit and artifact-level details.

## System flow

### Experimental co-evolving three-agent loop

The image-inspired architecture is now executable as `--system triad`:

```text
numbers only ──> Coding Agent ──> multiple executable forecast candidates
                                      │
documents ────> Retrieval Agent ──> verified evidence + typed impacts
                                      │
                                      v
                               Decision Agent
                         select / combine / ask again
                                      │
                                      v
                         probabilistic final forecast

delayed ground truth ──> coding coverage + retrieval quality + selection regret
                     ──> separate feedback for the three agents
```

The Coding Agent initially sees only numbers and generates backbone, transparent
statistical, robust-history, and local-level hypotheses. The Retrieval Agent searches for
evidence that distinguishes them and converts accepted prose into typed impacts. The Coding
Agent can then generate evidence-conditioned candidates, while the Decision Agent selects or
ensembles candidates and can request another retrieval round. Ground truth is used only after
the future resolves, never during inference.

Candidate families do not receive hand-written model priors. Before forecasting the real
horizon, each executable program is evaluated on up to three rolling historical cutoffs. Its
base reliability is `1 / (1 + mean scaled validation MAE)`. Text evidence is a compatibility
constraint rather than an arbitrary score bonus: a grounded active event can require an
evidence-adjusted candidate, while a resolved event adds no numerical preference for any model
family. If the history is too short to validate, the system conservatively preserves the
configured backbone. The Decision Agent also avoids fixed score-margin ensembles; combining
programs is deferred until out-of-fold stacking weights can demonstrate a validation gain.

Run the loop on a public Dr-CiK sample and write the delayed-feedback records:

```bash
python3 -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --system triad \
  --backbone chronos \
  --max-steps 3 \
  --top-k 5 \
  --learn-from-public-outcomes \
  --feedback-file outputs/triad/agent-feedback.jsonl \
  --evolution-file outputs/triad/evolution-policy.json \
  --output-dir outputs/triad
```

The current self-evolution layer is an interpretable online policy rather than neural-weight
training. It updates learned candidate-family preferences, useful retrieval vocabulary, and
decision-tag preferences after each resolved public task; these delayed learned preferences
are logged separately from historical validation scores. The feedback log also exposes three distinct future
training targets: candidate-set coverage for Coding, evidence quality for Retrieval, and
selection regret for Decision.

### Numbers-only Coding Agent and co-evolution

The executable Coding Agent now has an explicit information boundary and a nested evolution
contract:

```text
Input: historical timestamps + historical values only
  -> analyze statistical properties
  -> generate multiple falsifiable assumptions
  -> translate every assumption into executable Python
  -> sandbox execution
  -> rolling historical hindcast
  -> revise one failed assumption/program once
Output: assumption + failure condition + code + hindcast score
```

It never receives documents, retrieved evidence, `gt_evidence`, future values, or Retrieval
Agent output. [`scripts/run_code_evolution.py`](scripts/run_code_evolution.py) executes this
inner task-level loop. The returned program is selected by history-only validation rather than
by an LLM-authored numerical answer.

The new outer loop in [`co_evolution.py`](drcik_agent/co_evolution.py) evolves the reusable
generation policy across resolved training tasks. It evaluates a population of versioned prompt
bundles, attributes the weakest module from candidate coverage, evidence quality, and decision
regret, then changes exactly one eligible prompt. The sequence is:

```text
Coding assumptions + executable candidates
        -> Retrieval searches for evidence that tests those assumptions
        -> Decision cross-checks candidates against verified evidence
        -> final forecast
        -> delayed ground truth and labels score all three modules
        -> failure attribution chooses Coding OR Retrieval OR Decision
        -> mutate one prompt -> train/dev evaluation -> retain only validated bundles
```

Run the minimal population loop with [`scripts/run_co_evolution.py`](scripts/run_co_evolution.py).
An evolved JSON bundle can be loaded into the Codex triad with `--agent-bundle`. This is prompt-
policy evolution, not neural-weight training and not graph-topology search.

All three reasoning roles can instead be backed by schema-constrained Codex calls:

```bash
.venv/bin/python -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --system triad \
  --reasoning-agent codex \
  --backbone chronos \
  --chronos-model-id outputs/model-cache/chronos-bolt-small \
  --chronos-local-files-only \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --top-k 6 \
  --max-steps 2 \
  --output-dir outputs/task42-codex-triad
```

In this mode Codex proposes the executable numerical hypothesis families, autonomously
searches the local corpus with exact-quote grounding, and selects among executed candidates.
Python/Chronos still execute, backtest, and validate the numerical programs; Codex cannot emit
or overwrite the final trajectory directly. Invalid citations, unknown candidate IDs, schema
failures, and CLI failures are rejected or fall back visibly to the deterministic implementation.
The exact prompts and Dr-CiK comparison are documented in
[`docs/CODEX_TRIAD.md`](docs/CODEX_TRIAD.md).

On the public `task_42` mechanism test, the improved prompt found `doc_1560`, the Coding Agent
materialized its “historical baseline and seasonality” claim as a backtested 22-step harmonic
regime candidate, and the Decision Agent selected that executable candidate:

| Method | MAE | Retrieval precision | Supporting recall | Harmful revision |
|---|---:|---:|---:|---:|
| Chronos baseline | 72.7346 | – | – | – |
| Codex triad | **31.6472** | 0.8333 | 0.3846 | 0 |

This is one public sample and demonstrates the mechanism only; it is not an aggregate claim.

```text
history ──> diagnosis ──> Chronos-Bolt ──> immutable y_baseline
   │
documents ──> gap judge ──> next query ──> BM25 candidate pool
                                │                    │
                                │                    v
                                │        forecast-utility reranker
                                │                    │
                                │                    v
                                └──── insufficient ─ evidence grounding
                                                     │
                                                     v
                                       sentence-level Evidence State
                                                     │
                                          sufficient / missing gaps
                                                     │
                                                     v
                       importance compression ──> macro + micro outlook
                                                     │
                                                     v
                                      PostTime-style revise/preserve
                                                     │
                                                     v
                                     Last-Mile restricted workspace
                                                     │
                                                     v
                                    y_final + trajectories + audit trace

resolved training tasks ──> forecast-utility labels ──> frozen learned scorer
realized outcomes ──> post-hoc memory ──> later chronological tasks
```

The complete English architecture diagram, including the online inference path and
offline learning/evolution path, is available in two standalone formats:
[`docs/architecture.html`](docs/architecture.html) and
[`docs/architecture.svg`](docs/architecture.svg).

The Chronos baseline is generated exactly once. Retrieval cannot rewrite historical values or
`y_baseline`; it can only propose changes to `y_final`. This makes it possible to ask
whether context improved the forecast instead of hiding the backbone and the contextual
revision inside one opaque prompt.

## Paper-derived design

The runtime is deliberately a synthesis rather than many complete frameworks stacked
together:

| Work | What is integrated now | What is not placed in online inference |
|---|---|---|
| [Last-Mile Forecasting](https://arxiv.org/pdf/2606.02497) | Immutable baseline, forecast workspace, evidence-backed restricted actions | None of its case-specific prompts are required |
| [PostTime](https://arxiv.org/pdf/2605.29401) | LLM-as-reviser role, explicit revise-or-preserve gate, improvement-over-baseline metrics, hard-case fallback behavior | SFT/RLVR weight training requires a separate training corpus and GPUs |
| [From Long News to Accurate Forecast](https://arxiv.org/abs/2606.03097) | Candidate-pool utility reranking and forecast-aware long-document compression | The current scorer is a frozen label-free proxy; learned RM/PRM training is an offline next step |
| [S2G-RAG](https://aclanthology.org/2026.acl-long.1185/) | Explicit evidence sufficiency, structured missing gaps, and gap-guided queries | Its QA-specific gap labels are replaced with forecast gaps |
| [ReflectiveRAG](https://aclanthology.org/2026.eacl-industry.27/) | Adaptive stopping and relevance-minus-redundancy filtering | Its QA answer controller is not reused as a forecast judge |
| [Agentic-R](https://aclanthology.org/2026.findings-acl.785/) | Trainable retriever interface combining local relevance with global task utility | Global answer correctness becomes downstream forecast improvement |
| [BLF](https://arxiv.org/pdf/2604.18576) | Compact linguistic belief state updated in log-odds space instead of accumulating raw text | Binary Platt calibration and logit aggregation do not directly apply to continuous trajectories |
| [NEXUS](https://arxiv.org/pdf/2605.14389) | Separate macro numerical outlook and micro event outlook before final synthesis | LLM prompts can replace the deterministic agents after controlled ablations |
| [CORAL](https://arxiv.org/pdf/2604.01658) | Shared persistent artifacts and evaluator separation inform the architecture | Long-running autonomous evolution belongs outside task inference to prevent leakage and uncontrolled benchmark search |

See [`docs/PAPER_INTEGRATION.md`](docs/PAPER_INTEGRATION.md) for the module mapping and
recommended ablations, and [`docs/CHRONOS_BACKBONE.md`](docs/CHRONOS_BACKBONE.md) for
the default backbone configuration. TimesFM remains available as an optional comparison;
see [`docs/TIMESFM_BACKBONE.md`](docs/TIMESFM_BACKBONE.md).

Each loop iteration begins with a structured sufficiency decision. The controller
selects the highest-value unresolved gap, constructs the next query, and can create new
follow-up gaps from grounded evidence. For example, finding an anomaly dynamically
creates a `resolution_permanence` gap; finding an unquantified future event creates an
`event_magnitude` gap. The loop stops when gaps are resolved, the corpus is exhausted,
progress stalls, or expected information gain falls below the configured cost threshold.

The verifier checks entity identity, temporal alignment, target relevance, whether the
document answers the current question, and whether textual claims conflict with the
observed numerical pattern. A document rejected only because it answers a different
question remains available to later iterations.

Accepted evidence is translated into structured forecast impacts containing the event
window, direction, permanence, forecast-horizon overlap, magnitude, confidence, and an
auditable adjustment rule. The safe default does not directly apply generic textual
effects such as `increase by 20%` or `2 times the usual demand`: these magnitudes often
refer to a different baseline or are already reflected in history. Only an explicit
future timestamp-value pair or a history-backtested normal-regime projection may revise
the numerical prior automatically. Other multiply/add candidates are preserved for an
explicit unsafe ablation. An event that ended before the horizon produces a
`return_to_baseline` instruction and is not extrapolated.

Each impact then becomes a proposal with an event type, affected range, action type,
value, source documents, confidence, rationale, and any retrieved memory IDs. The
workspace executor accepts only these actions:

| Action | Meaning |
|---|---|
| `preserve` | Keep the baseline when an event ended, is already reflected in history, or lacks a defensible magnitude |
| `multiply` | Candidate multiplier/percentage action; disabled by the safe default unless explicitly enabled for ablation |
| `add` | Candidate absolute/residual-scaled action; disabled by the safe default unless explicitly enabled for ablation |
| `clip` | Enforce an explicit lower or upper bound |
| `override` | Revise a specific point when verified context provides an explicit future value |

Unsupported edits, out-of-horizon ranges, low-confidence changes, unsafe multipliers,
and duplicate actions are rejected and recorded. Corroborating documents therefore do
not multiply the same event effect twice.

To reproduce the deliberately unsafe generic-event ablation, add
`--allow-unvalidated-event-revisions` to an iterative run.

### Optional Codex reasoning agents

For a clean Codex-powered comparison, `codex-direct` is an intentionally unguarded
full-corpus agent baseline. Chronos generates the same numerical prior used by our method;
Codex then searches the task's local document directory and directly returns cited evidence
and the complete final trajectory in one call. This baseline does **not** use our BM25
candidate stage, verifier, evidence-to-impact translator, memory, or restricted revision
workspace:

```bash
python3 -m drcik_agent run-hf \
  --public-dev \
  --task-id task_117 \
  --system codex-direct \
  --backbone chronos \
  --chronos-model-id outputs/model-cache/chronos-bolt-small \
  --chronos-local-files-only \
  --codex-reasoning-effort high \
  --output-dir outputs/codex-direct-task117
```

The default reasoning effort for `codex-direct` is `high`, matching the style of the
Codex configuration reported by Dr-CiK. The exact available Codex model depends on the
installed CLI and account; use `--codex-model` to freeze it in a formal experiment. The
output records Codex calls, cache hits, failures, latency, cited-document retrieval metrics,
and forecast metrics. Invalid or failed Codex forecasts fall back visibly to the immutable
Chronos baseline instead of silently using our hybrid agents.

A real `task_42` smoke run of this baseline used one uncached high-effort Codex call
(201.6 seconds). Codex cited four supporting documents and no distractors, but preserved
Chronos exactly because it found no defensible numerical adjustment:

| Method | MAE | Retrieval precision | Supporting-document recall | Harmful revision |
|---|---:|---:|---:|---:|
| Chronos | 72.7346 | – | – | – |
| Codex-Direct | 72.7346 | 1.0000 | 0.3077 | 0 |
| Codex-Contract | **31.6472** | 1.0000 | 0.3846 | 0 |
| Proposed safe hybrid | **31.6472** | 0.8333 | 0.3846 | 0 |

This is a single public smoke task, not an aggregate result. It illustrates the intended
ablation: autonomous Codex found clean evidence, while the proposed evidence-to-impact and
restricted revision path was still necessary to turn context into a numerical improvement.

`codex-contract` tests that bridge explicitly. Codex may search the full corpus and emit
evidence plus a structured regime hypothesis, but it is forbidden to emit future numbers.
For a grounded `normal_seasonal` contract, a numerical tool fits a trend-harmonic candidate,
validates it on historical holdouts, blends it with Chronos according to validation gain, and
applies the result through the restricted workspace:

```bash
python3 -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --system codex-contract \
  --backbone chronos \
  --chronos-model-id outputs/model-cache/chronos-bolt-small \
  --chronos-local-files-only \
  --codex-reasoning-effort high \
  --output-dir outputs/codex-contract-task42
```

On `task_42`, Codex produced a 0.98-confidence `normal_seasonal` contract, correctly marked
the software anomaly and promotion windows, and cited five supporting documents with no
distractors. The history-only candidate achieved validation MAE 12.7622 versus 43.9295 for
seasonal naive, resulting in a 0.709 blend weight. Final MAE fell from 72.7346 to 31.6472
(56.5% relative gain), matching the proposed safe hybrid on this task. The uncached Codex call
took 121.4 seconds. This remains a single-task mechanism test rather than aggregate evidence.

Across all three official repository sample tasks, the contract system improved two tasks,
preserved one, and harmed none. Explicit values are accepted only when a dated value in one
grounded source is independently corroborated by another grounded source at the same local time
and remains plausible under the observed numerical scale:

| Task | Contract | Chronos MAE | Contract MAE | Outcome | Retrieval precision |
|---|---|---:|---:|---|---:|
| `task_42` | `normal_seasonal` | 72.7346 | **31.6472** | improved | 1.0000 |
| `task_163` | `explicit_future_values` | 10.4337 | **9.3365** | improved | 1.0000 |
| `task_201` | `explicit_future_values` | 0.0563 | 0.0563 | unchanged | 0.7143 |

A frozen, label-free 30-task public-development evaluation is documented in
[`docs/EVAL_30_RESULTS.md`](docs/EVAL_30_RESULTS.md). It improves mean MAE from
162.6270 to 156.3621, with 2 improved tasks, 28 unchanged tasks, and no harmed tasks.

For `task_163`, eight of 24 future timestamps were present in a dated table and repeated in a
second independently formatted report. Blending those corroborated anchors at weight 0.75
reduced MAE by 10.5%. For `task_201`, Codex still cited two time-series distractors, but their
unanchored single-source numeric range supplied no corroborated timestamp-value pairs; all
explicit revisions were rejected and Chronos was preserved.

Mean MAE fell from 27.7416 to 13.6800 (50.7%), and mean RMSE fell from 39.4859
to 22.5559 (42.9%). The three-task result remains a mechanism test, not evidence that every
task or contract type improves.

The deterministic text modules remain the default for reproducibility. A logged-in local
Codex CLI can replace query planning, semantic evidence verification, and evidence-to-impact
translation while leaving Chronos and the restricted forecast workspace unchanged:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --task-id task_42 \
  --backbone chronos \
  --chronos-model-id outputs/model-cache/chronos-bolt-small \
  --chronos-local-files-only \
  --reasoning-agent codex \
  --codex-stages query,verify,impact \
  --codex-reasoning-effort low \
  --codex-cache-dir outputs/codex-cache
```

Every Codex call uses an ephemeral read-only sandbox, a strict JSON output schema, exact-quote
grounding, source-ID validation, and an on-disk content-addressed cache. If the CLI times out,
is unavailable, or returns invalid JSON, that stage falls back to the deterministic agent.
Codex never receives `future_values`, `gt_evidence`, document role labels, or document subtype
labels, and it cannot execute forecast revisions directly. Verifier-selected exact quotes are
pinned through context compression. Daily timestamp-value blocks are parsed only inside these
verified quote boundaries, preventing superficially similar distractor tables from changing the
forecast.

For lower-cost experiments, use only the semantic verifier:

```bash
--reasoning-agent codex --codex-stages verify
```

On the public `task_42` smoke test, the two-step full Codex loop matched the safe rule loop:
MAE decreased from the Chronos baseline of 72.7346 to 31.6472, with no harmful revision.
Codex increased retrieval precision from 0.8333 to 1.0000 on this single task but reduced the
supporting-document recall proxy from 0.3846 to 0.2308. Six uncached Codex calls took 188.9
seconds.

An exploratory oracle bottleneck scan then identified tasks where the safe rule loop preserved
Chronos but correct retrieved context could improve the downstream forecast. On two such hard
cases, verifier-only Codex recovered the exact future-value blocks and reached the oracle ceiling:

| Task | Chronos / rule MAE | Codex-verifier MAE | Oracle MAE | Gain | Uncached verifier cost |
|---|---:|---:|---:|---:|---:|
| `task_117` | 228.8206 | **222.7202** | 222.7202 | 6.1004 | 1 call / 46.3 s |
| `task_116` | 15.8733 | **12.0222** | 12.0222 | 3.8510 | 2 calls / 84.5 s |

The matched current rule loop remained unchanged on both tasks, while Codex produced no harmful
revision. These cases were selected using public oracle diagnostics, so they demonstrate mechanism
and implementation—not an unbiased aggregate accuracy estimate. The unchanged default rule system
was rerun on all 199 public tasks after adding the Codex-only parsing gate and retained its prior
7-improved / 0-harmed safety result.

In the normal inference path, Dr-CiK's `role`, `subtype`, `future_values`, and
`gt_evidence` fields are never exposed to the loop. Public labels are used only after a
run to calculate development metrics. The explicitly separate `--oracle-evidence`
diagnostic described below is the sole exception: it bypasses retrieval on public tasks
to measure the downstream evidence-to-forecast ceiling and is rejected for hidden-test
inference.

Post-hoc memory follows the same separation. A run never reads its own future values.
Only after an outcome is explicitly recorded can the system compare `y_baseline`,
`y_final`, and the actual series, store whether a revision helped, and use that lesson
as a shrinkage prior for later matching events.

## Quick start

Clone Dr-CiK and run its three official sample tasks:

```bash
git clone https://github.com/ServiceNow/Dr-CiK.git external/Dr-CiK
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[chronos]'

drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop
```

The project entrypoint can run directly from source after installing the Chronos runtime:

```bash
pip install 'chronos-forecasting>=2.2.0'
python3 -m drcik_agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop
```

The package and the `amazon/chronos-bolt-small` checkpoint are loaded lazily on the first
forecast. The checkpoint is downloaded from Hugging Face unless it is already cached.

Useful loop controls:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/agent-loop \
  --system iterative \
  --max-steps 10 \
  --top-k 5 \
  --max-no-progress 4 \
  --convergence-tolerance 0.002 \
  --candidate-multiplier 3 \
  --context-character-budget 12000 \
  --min-information-gain 0.05 \
  --revision-threshold 0.60
```

Chronos controls:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --backbone chronos \
  --chronos-model-id amazon/chronos-bolt-small \
  --chronos-device-map cpu \
  --chronos-max-context 2048 \
  --chronos-max-horizon 1024 \
  --chronos-cache-dir outputs/model-cache
```

Chronos is the default and failure is explicit. The system does not silently switch to
the old statistical model. For an intentional degraded-mode run, add
`--allow-statistical-fallback`; the recorded baseline method will begin with
`statistical_fallback:`. For a clean ablation, use `--backbone statistical`.

`--top-k 5 --candidate-multiplier 3` retrieves 15 candidates, scores all 15 for
forecasting utility, and sends only the best 5 to the verifier. The ranking and
compression modules never see `future_values`.

Optional outcome memory for sequential research runs:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/with-memory \
  --memory-file outputs/forecast-memory.jsonl \
  --learn-from-public-outcomes
```

`--learn-from-public-outcomes` is intentionally opt-in and is rejected for the hidden
test split. For a clean benchmark comparison, keep it off and record outcomes only in a
separate, chronologically valid backtest.

`--top-k` is the number of new documents inspected at each iteration. The original
one-pass baseline is preserved for ablations:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/one-pass \
  --system one-pass \
  --top-k 8
```

## Outputs

- `forecasts.jsonl`: Dr-CiK forecasting submission format with 100 trajectories per task.
- `deep_research.jsonl`: accepted document IDs and extracted evidence.
- `loop_trace.jsonl`: every query, candidate, verifier verdict, structured evidence
  utility score, compression decision, macro/micro outlook, revision decision,
  accepted/rejected action, belief update, forecast summary, and stop decision.
- `run_report.jsonl`: per-task diagnosis, belief state, development metrics, and the full
  forecast workspace containing historical observations, immutable `y_baseline`, editable
  `y_final`, proposals, action results, and memory references.
- `summary.json`: aggregate development metrics.

Local `sMAE`, `sRMSE`, and `sCRPS` values are explicitly development proxies. Official
hidden-test scores are calculated by the Dr-CiK maintainers. When a workspace is used,
the report also includes `baseline_mae`, `revision_value_mae`,
`relative_revision_gain`, and `harmful_revision` to measure whether the last-mile agent
actually improved the forecasting backbone. It also reports `revision_accept_rate`,
`revision_fallback_rate`, `mean_predicted_revision_utility`,
`context_retention_ratio`, `mean_belief_sufficiency`, `gap_coverage`,
`mean_expected_information_gain`, `retrieval_turns`, and `documents_inspected`.

## Public development split

```bash
pip install -e '.[chronos,huggingface]'
drcik-agent run-hf --public-dev --output-dir outputs/public-dev
```

Use `--limit 5` for a short development run. `--hidden-test` creates submission files
without local forecast scores because the hidden labels are unavailable.

### Oracle-evidence bottleneck diagnostic

Use this only on public labeled development tasks. It replaces retrieved documents with
the task's `gt_evidence` annotations while keeping the same backbone, impact translator,
revision gate, workspace actions, and uncertainty sampler:

```bash
drcik-agent run-sample \
  --sample-dir external/Dr-CiK/sample \
  --output-dir outputs/chronos-oracle-evidence \
  --oracle-evidence
```

This is not a valid test-time system and is not an official Dr-CiK result. The CLI and
runtime both reject it when labels are hidden. Its purpose is causal debugging: if
oracle evidence helps but normal retrieval does not, retrieval is a bottleneck; if even
oracle evidence does not help, evidence-to-impact translation or revision is also a
bottleneck.

With Chronos-Bolt Small, seed 7, and the three official repository samples, the current
controlled diagnostic is:

| Task | Constrained Chronos MAE | Earlier preserve-only loop | Improved retrieval loop | Oracle evidence |
|---|---:|---:|---:|---:|
| `task_163` | 10.433750 | 10.433750 | 9.336523 | 9.336523 |
| `task_201` | 0.056257 | 0.056257 | 0.056257 | 0.056257 |
| `task_42` | 72.734644 | 72.734644 | 31.647156 | 31.647156 |
| Mean | 27.741550 | 27.741550 | 13.679979 | 13.679979 |

Two changes close the oracle gap on these three samples. First, the context-point parser
now reads wide Markdown forecast tables, allowing the normal loop to recover eight
explicit future irradiance points for `task_163`. Second, a regime-normalization reviser
translates verified "return to normal" evidence into a numerical trajectory: it fits a
trend-plus-harmonic model to the most recent two observed cycles, validates it on the
last cycle, and blends it with Chronos only when this history-only backtest beats seasonal
naive. On `task_42`, its 22-step model has validation MAE 12.76 versus 43.93 for seasonal
naive and reduces future MAE by 56.5%. No future values or benchmark role labels enter
either decision.

### Full public-split safety evaluation

We then evaluated all 199 labeled public synthetic tasks with the same Chronos-Bolt
Small backbone. Hidden human tasks were not used. The `sMAE`, `sRMSE`, and `sCRPS`
figures below are local proxies produced by this repository, not official leaderboard
scores.

| System | MAE | CRPS | sMAE proxy | sCRPS proxy | Improved | Harmed |
|---|---:|---:|---:|---:|---:|---:|
| Chronos only | 797.6268 | 708.5187 | 2.6981 | 2.5370 | — | — |
| Unsafe generic event revisions | 1289.7688 | 1198.7457 | 2.8350 | 2.6919 | 12 | 29 |
| Safe grounded loop | **797.1767** | **708.2300** | **2.6687** | **2.5075** | **7** | **0** |

The safe loop changed only 7 of 199 forecasts (3.5%): all seven changes improved MAE
and none harmed it. Five were history-backtested normal-regime projections and two were
explicit future timestamp-value overrides. This result is why generic textual
multiply/add actions are now opt-in rather than the default.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Current scope

| Component | Current implementation | Natural next experiment |
|---|---|---|
| Diagnosis | Trend, robust residual scale, conservative seasonality inference | Specialized TSFM diagnostics |
| Control | Structured sufficiency decision, dynamic forecast gaps, gap-guided query, marginal-gain stopping | Distill a lightweight S2G-style judge from chronological traces |
| Retrieval | BM25 candidate pool plus injectable forecast-utility scorer; label-free proxy is the default ablation | Train the Agentic-R/long-news scorer on chronological forecast-gain labels |
| Grounding | Entity/time/target checks plus sentence-level claims with provenance, magnitude, and persistence | LLM entailment and cross-document corroboration |
| Context | Importance-aware sentence retention under a global character budget | Learned article reward model and pairwise fusion |
| Working memory | BLF-inspired linguistic belief state plus unified forecast workspace | Multi-trial continuous-trajectory aggregation |
| Evidence impact | Event window, direction, permanence, explicit magnitude, Markdown table values, and history-backtested normal-regime projection | LLM causal-impact estimator with calibrated uncertainty |
| Reasoning | NEXUS-style macro numerical and micro event outlooks | LLM outlook agents with schema-constrained outputs |
| Forecast backbone | Chronos-Bolt Small median forecast; TimesFM and statistical backbones retained for comparisons | Use Chronos quantiles directly for calibrated trajectory sampling |
| Last-mile revision | PostTime-style revise/preserve gate plus restricted workspace actions | Post-train a compact reviser with SFT and improvement-ratio RLVR |
| Outcome memory | Optional post-resolution calibration lessons in JSONL | Event embeddings and leakage-safe chronological retrieval |
| Offline evolution | Evaluator-separated memory and outcome label interfaces | CORAL-style policy evolution on isolated development runs |

The first controlled comparison should be `backbone only` vs. `oracle context` vs.
`one-pass retrieval` vs. `iterative retrieval + unrestricted revision` vs. `iterative
retrieval + restricted workspace revision`, all using the same backbone. Report both
forecast accuracy and revision value (`baseline error - final error`) so retrieval gains
are separated from backbone quality and harmful context edits.
