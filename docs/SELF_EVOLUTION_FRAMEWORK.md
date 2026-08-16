# Self-Evolution Framework

This document gives a concise map of the repository's self-evolution architecture. The system has
two related layers: a reusable domain-independent Self-Harness and a time-series-specific
three-agent Meta-Harness.

## 1. Generic Self-Harness

The reusable controller is implemented in
[`evolving_agent/evolution_core/controller.py`](../evolving_agent/evolution_core/controller.py).
It treats the object being evolved as an artifact and delegates domain behavior to injected
components.

```text
Parent artifact
  -> evaluate on Train and collect failure traces
  -> propose multiple child artifacts
  -> optionally screen children on small Train/Dev prefixes
  -> evaluate promoted children on complete Train
  -> select the best Train child
  -> compare that child with the parent on disjoint Dev
  -> accept only if the configured Dev metric improves
  -> persist the accepted artifact as the next parent
```

The controller receives six interchangeable components:

- an artifact adapter that validates, serializes, and restores artifacts;
- a mutator that proposes children from the parent and sanitized Train failures;
- an executor that runs an artifact on task items;
- an evaluator that converts execution results into metrics and failure diagnostics;
- an acceptance gate that compares Parent and Child on Dev;
- an artifact store that saves checkpoints, traces, and accepted generations.

The controller contains no forecasting assumptions. The same lifecycle can therefore evolve a
tool dictionary, an executable forecasting program, or another structured artifact.

## 2. Numerical Dictionary Curation Adapter

The dictionary-specific implementation is in
[`numerical_agent/adapters/dictionary_curation.py`](../numerical_agent/adapters/dictionary_curation.py).
It plugs a `ToolDictionary` into the generic Self-Harness.

```text
Externally supplied method definitions
  -> implement unimplemented methods
  -> run each implementation on Train series
  -> calculate method-level forecasting diagnostics
  -> classify each method
       accepted / specialized / quarantined / discarded / unavailable
  -> revise eligible quarantined methods from sanitized failure feedback
  -> produce a Child Dictionary
  -> accept it only if it improves the held-out Dev objective
```

The framework does not hard-code the final Statistical, TSFM, or Combined method collection. Those
base methods are supplied as experiment inputs. This separates the evolution mechanism from the
knowledge being curated.

## 3. Numbers-Only Program Self-Evolution

The Numerical/Coding Agent implementation is in
[`evolving_agent/coding_agent/evolution.py`](../evolving_agent/coding_agent/evolution.py).
It cannot see documents, retrieved evidence, ground-truth evidence, or future values.

```text
Historical numbers + horizon + frequency
  -> LLM generates multiple falsifiable forecasting assumptions
  -> each assumption becomes executable Python forecast code
  -> sandbox execution
  -> rolling historical hindcasting
  -> revise the strongest failed program from hindcast diagnostics
  -> retain accurate and diverse validated candidates
  -> save a reusable numerical skill only when it beats repeat-last hindcasting
```

Every generated program must implement the same `forecast(history, horizon, frequency)` contract.
This allows LLM-only, statistical-dictionary, TSFM, and combined conditions to share the same
evaluation interface.

## 4. Three-Agent Forecasting Harness

The end-to-end system combines three restricted roles:

| Role | Input | Output |
|---|---|---|
| Numerical Agent | Historical values, frequency, horizon, prior numerical skills, hindcast feedback | Executed candidate forecasts, assumptions, failure conditions, hindcast scores |
| Retrieval Agent | Candidate assumptions, task metadata, documents, validated retrieval skills | Exact citations, relevant documents, mechanism classes, typed evidence impacts |
| Decision Agent | Executed candidates, hindcast scores, verified evidence, validated decision skills | Selected candidate and supporting citations |

The inference path is:

```text
Numerical Agent
  -> executable candidates
  -> historical hindcasting
Retrieval Agent
  -> verified contextual evidence
Decision Agent
  -> evidence-constrained candidate selection
  -> final forecast
```

The Decision Agent cannot invent a new numerical trajectory. It must select an already executed
candidate, and a contextual override must cite verified evidence.

## 5. Whole-Harness Co-Evolution

The structured three-agent Meta-Harness is implemented in
[`evolving_agent/co_evolution.py`](../evolving_agent/co_evolution.py). During task inference, all
future labels are removed before the agents run. Labels become visible only to the trusted scorer
after a forecast has been frozen.

After resolution, the system reports:

- **Numerical reward:** whether the candidate set contained a strong forecast;
- **Retrieval reward:** supporting-document precision/recall and distractor avoidance;
- **Decision reward:** regret between the selected candidate and the best generated candidate;
- **System reward:** 80% final forecast reward plus 20% retrieval reward.

Module rewards diagnose what failed. The system reward remains the end-to-end inheritance
objective.

```text
Accepted Parent Harness
  -> run Train and learn isolated validated skills
  -> summarize module rewards and worst failure trajectories
  -> Meta-Harness proposes structurally diverse children
  -> screen weak children with successive halving
  -> rank survivors on complete Train
  -> compare the best Train child with Parent on read-only Dev
  -> accept only if Child Dev reward is strictly higher
  -> freeze prompts, budgets, topology, policies, and skill snapshots
```

Train tasks may update each child's isolated skill libraries. Dev tasks are read-only and cannot
write skills. Public Test is never used for mutation or acceptance.

## 6. Three Evolution Depths

| Mode | Mutable surface | Accepted artifact |
|---|---|---|
| `prompt` | Exactly one complete prompt owned by the diagnosed target role | `best_policy.json` |
| `genome` | Role prompts, Numerical search/hindcast budgets, Retrieval/Decision topology, evidence policy, aggregation, and validated skill snapshots | `best_policy.json` |
| `source` | Mutable Agent/Harness Python and new modules under `evolving_agent/generated/` | `best_source.patch` |

Source evolution runs Codex in an isolated Git worktree. A proposed patch must pass the static
label/safety audit, the full test suite, Train selection, and held-out Dev acceptance before it can
become the next repository generation. Scoring, task splitting, label removal, the sandbox, tests,
and the evolution host remain immutable.

This is harness and artifact evolution, not LLM weight training. The system evolves prompts,
skills, dictionaries, candidate budgets, validation settings, communication topology, and—only in
source mode—audited implementation code.

## 7. Frozen Dr-CiK Data Protocol

The recommended public split is stored in
[`splits/drcik_public_80_20_99_v1.json`](../splits/drcik_public_80_20_99_v1.json).

| Partition | Tasks | Use |
|---|---:|---|
| Train | 80 | Mutations, dictionary/skill learning, failure analysis |
| Dev | 20 | Parent/Child acceptance and early stopping only |
| Public Test | 99 | One frozen final evaluation |

The split is entity-disjoint. The official 80 hidden tasks are excluded and remain reserved for the
benchmark's hidden evaluation. Assignment uses metadata only and never inspects future values,
ground-truth evidence, or document relevance labels.

## 8. Main Implementation Map

| Responsibility | File |
|---|---|
| Generic Parent/Child lifecycle | `evolving_agent/evolution_core/controller.py` |
| Generic contracts and configuration | `evolving_agent/evolution_core/contracts.py` |
| Dev acceptance gate | `evolving_agent/evolution_core/acceptance.py` |
| Artifact/checkpoint persistence | `evolving_agent/evolution_core/persistence.py` |
| Dictionary filtering and curation | `numerical_agent/adapters/dictionary_curation.py` |
| Numbers-only executable program evolution | `evolving_agent/coding_agent/evolution.py` |
| Three-agent forecast runtime | `evolving_agent/harness.py` |
| Prompt/Genome co-evolution | `evolving_agent/co_evolution.py` |
| Source-level evolution | `evolving_agent/source_evolution.py` |
| Full operational guide | `docs/EVOLVING_AGENT.md` |
