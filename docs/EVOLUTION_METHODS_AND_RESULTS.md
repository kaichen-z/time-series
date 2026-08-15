# Three-Level Harness Evolution: Methods and Initial Results

**Repository:** <https://github.com/kaichen-z/time-series>

**Experiment date:** August 13, 2026

**Dataset:** Three public Dr-CiK sample tasks (`task_42`, `task_163`, and `task_201`)

**LLM:** `gpt-5.6-sol`, high reasoning effort

Machine-readable summary: [`results/initial_evolution_results.json`](results/initial_evolution_results.json)

For the complete audit of earlier Chronos, statistical, Codex Contract, Triad, Direct-Prompt, and
Dr-CiK reproduction baselines, see
[`BASELINE_METHODS_AND_RESULTS.md`](BASELINE_METHODS_AND_RESULTS.md).

## 1. Objective

The project separates contextual time-series forecasting into three inference roles:

1. **Coding Agent:** receives historical numbers only, proposes falsifiable forecasting
   assumptions, translates them into executable Python programs, and validates them by historical
   hindcasting.
2. **Retrieval Agent:** receives the candidate assumptions and the document corpus, retrieves exact
   evidence, rejects distractors, and converts accepted evidence into typed causal impacts.
3. **Decision Agent:** compares executed numerical candidates with verified evidence and selects an
   existing forecast. It cannot invent a new trajectory.

An outer evolution loop uses resolved training outcomes to improve this system. Future values,
ground-truth evidence annotations, and document role labels are removed before inference. They are
available only to the immutable evaluator after a forecast has been produced.

## 2. Shared evolution protocol

All three evolution modes use the same scientific selection structure:

```text
parent system
  -> generate child candidates from resolved training failures
  -> evaluate children on training tasks
  -> select the best training child
  -> evaluate that child on an entity-disjoint development split
  -> accept only if held-out development reward improves
  -> otherwise retain the parent
```

The system reward used in these experiments combines final forecasting quality and retrieval
quality:

```text
system reward = 0.8 * forecast reward + 0.2 * retrieval reward
```

Forecast reward is derived from sMAPE and retrieval reward averages supporting-document precision,
supporting recall, and distractor avoidance. Higher system reward is better. The scorer, task split,
label firewall, generated-code sandbox, and acceptance rule are not evolvable.

## 3. Evolution modes

### 3.1 Prompt-only evolution (`--evolution-mode prompt`)

This is the most constrained baseline. The evaluator identifies the weakest role, and the Prompt
Evolver may replace exactly one complete prompt owned by that role:

- Coding generation prompt;
- Coding revision prompt;
- Retrieval prompt; or
- Decision prompt.

It cannot change budgets, topology, source code, metrics, or safety boundaries. The accepted
artifact is a `best_policy.json` file.

**Current result:** the mode is implemented and covered by unit tests, but no forecasting evolution
run has yet been recorded. It must therefore remain an unbenchmarked baseline in the current
comparison.

### 3.2 Harness Genome evolution (`--evolution-mode genome`)

Genome mode is the current structured Meta-Harness. One inheritable genome contains:

- all Coding, Retrieval, and Decision prompts;
- the number of initial Coding programs and mutation children;
- historical validation folds and horizon;
- the ordered Retrieval/Decision workflow;
- evidence-adjustment controls; and
- the Decision aggregation rule.

The model may redesign these fields jointly. This permits topology changes such as:

```text
retrieve -> decide
```

becoming:

```text
retrieve -> decide -> retrieve -> decide
```

without allowing arbitrary source edits.

### 3.3 Source-level evolution (`--evolution-mode source`)

Source mode is the most expressive option. A Source Harness Engineer may modify:

- `evolving_agent/coding_agent/evolution.py`;
- `evolving_agent/retrieval_agent/agent.py`;
- `evolving_agent/decision_agent/agent.py`;
- `evolving_agent/harness.py`; and
- new Python modules under `evolving_agent/generated/`.

Each source candidate is created in a detached temporary Git worktree. It must pass:

1. mutable-path and forbidden-code audit;
2. label-access audit;
3. Python syntax and import safety checks;
4. the complete test suite;
5. training evaluation; and
6. entity-disjoint development acceptance.

Rejected candidates never modify the parent checkout. An accepted generation is stored as a
cumulative patch that can seed the next source generation.

## 4. Dataset split used in the smoke experiments

The deterministic entity-level split with seed 7 and development fraction 0.34 was:

| Split | Task | Entity |
|---|---|---|
| Train | `task_163` | Arid Heights Research Annex |
| Train | `task_201` | Blue Creek Technical Site |
| Development | `task_42` | Nuance Cosmetic Lab |

This is only a mechanism smoke test. Three tasks are not sufficient for a scientific performance
claim.

## 5. Harness Genome result

One previously recorded Genome generation produced one child (`v001`). The mutation targeted the
weak Retrieval module but jointly changed the workflow and role instructions. Its principal change
was a two-pass contrastive evidence process:

```text
retrieve -> decide -> retrieve -> decide
```

The first Decision round exposes a missing discriminator; the second Retrieval round searches for
counterevidence and temporal endpoints; the final Decision uses the merged evidence.

| Metric | Parent `v000` | Child `v001` | Change |
|---|---:|---:|---:|
| Training system reward | 0.659675 | 0.850589 | +0.190915 |
| Development system reward | 0.859456 | 0.870863 | +0.011406 |

**Decision:** accepted. The accepted child improved both training reward and held-out development
reward. The complete aggregate result is preserved in the machine-readable summary linked above;
generated caches and raw trajectories remain local run artifacts rather than versioned source.

## 6. Source-level result

The initial Source smoke run used a deliberately small budget:

- one generation;
- one source child;
- one initial Coding program;
- no inner Coding mutation;
- one historical validation fold; and
- statistics-only Coding mode.

The Source Engineer generated a cutoff-stability tournament with adaptive challengers,
recency-robust validation, stability-aware Decision logic, and stricter Retrieval coverage. It
modified all three agents and the Harness and created
`evolving_agent/generated/cutoff_tournament.py`.

The candidate passed the source audit and all **163 tests**, but its empirical reward decreased:

| Metric | Parent | Source child | Change |
|---|---:|---:|---:|
| Training system reward | 0.898000 | 0.874556 | -0.023444 |
| Development system reward | 0.859363 | 0.847298 | -0.012065 |

**Decision:** rejected. The accepted cumulative patch was therefore empty, so the repository
retained the parent implementation. This is a successful safety and selection result, not an
accuracy improvement: the system generated a substantial architecture mutation, verified that it
was executable, measured its degradation, and refused to inherit it.

## 7. Standalone `task_42` diagnostic

`task_42` was also replayed separately with a fresh, empty skill library and the same reduced
statistics-only Coding budget used by the Source smoke configuration. This standalone replay is a
diagnostic of the parent pipeline; it is **not** the exact development evaluation from Source
evolution. During Source evolution, the parent processes the two training tasks first and may carry
hindcast-validated numeric skills into development inference.

| Metric | Standalone parent result |
|---|---:|
| Final sMAPE | 23.648640 |
| Final MAE | 104.426708 |
| Coding hindcast sMAPE | 12.059662 |
| Retrieval precision | 0.750000 |
| Supporting-document recall | 0.230769 |
| Distractor avoidance | 0.960000 |
| Decision selection regret | 0.000000 |

The Retrieval Agent selected:

- `doc_1548`: historical software bug evidence;
- `doc_1552`: evidence that the bug was neutralized;
- `doc_1560`: future return to normal seasonal behavior; and
- `doc_1562`: a distractor that should have been rejected.

The Decision Agent selected `validated_seasonal_repeat__evidence_3`, an evidence-adjusted version of
the executed seasonal-repeat program. Decision regret was zero among the available candidates, but
low supporting recall and one accepted distractor show that Retrieval remains the primary weakness
in this reduced-budget run.

## 8. What these initial results establish

The experiments establish the following engineering facts:

1. All three evolution levels are available through one CLI.
2. Genome mutations can be inherited only after held-out improvement.
3. Source mutations can create new executable agent logic without touching the parent checkout.
4. Passing static checks and tests is not treated as evidence of forecasting improvement.
5. A source candidate that degrades both train and development reward is correctly rejected.

They do **not** establish that Source evolution is worse than Genome evolution. The runs used
different Coding budgets, only one Source child was sampled, and the dataset contained only three
tasks. A valid comparison requires identical budgets, multiple seeds, more children and
generations, and an entity-disjoint Dr-CiK development/test evaluation.

## 9. Reproduction commands

Prompt-only baseline:

```bash
evolving-agent evolve \
  --tasks-file /path/to/Dr-CiK/tasks \
  --evolution-mode prompt \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --setting statistics \
  --generations 1 \
  --children 1
```

Harness Genome:

```bash
evolving-agent evolve \
  --tasks-file /path/to/Dr-CiK/tasks \
  --evolution-mode genome \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --setting statistics \
  --generations 1 \
  --children 1
```

Reduced Source smoke run:

```bash
evolving-agent evolve \
  --tasks-file /path/to/Dr-CiK/sample/tasks \
  --evolution-mode source \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --setting statistics \
  --coding-initial-programs 1 \
  --coding-mutations 0 \
  --coding-validation-folds 1 \
  --generations 1 \
  --children 1 \
  --dev-fraction 0.34 \
  --seed 7 \
  --source-patch-path runs/evolving/source_smoke/best_source.patch \
  --trace-path runs/evolving/source_smoke/evolution_trace.json
```

## 10. Recommended next experiment

Run Prompt, Genome, and Source modes under an identical budget on a larger entity-disjoint subset.
For each mode, report multiple seeds, mean and standard deviation of sMAPE/system reward, retrieval
precision and recall, Decision regret, acceptance rate, token cost, and wall-clock time. Source mode
should sample multiple children because one rejected architecture is not informative about the
search space.

## 11. Fresh 30-task frozen-Genome evaluation

Two frozen LLM-only Harness Genomes and three fixed baselines were evaluated on the same separately
sampled 30-task public Dr-CiK manifest. This set was not used to generate either Genome. The raw
manifest, frozen v000/v003 policies, and machine-readable aggregate are stored under
`runs/fresh30_four_method_20260815/`.

| Method | Tasks | Mean MAE | Median MAE | Mean sMAPE | Median sMAPE |
|---|---:|---:|---:|---:|---:|
| retry2 v003 | 30 | **23.057969** | 9.098874 | **24.519263** | **15.857966** |
| v000 | 30 | 33.068265 | **7.246025** | 32.409864 | 24.614434 |
| Codex-Contract | 30 | 50.186415 | 8.383005 | not emitted | not emitted |
| Chronos | 30 | 51.566175 | 8.383005 | not emitted | not emitted |
| Codex-Direct | 30 | 64.255378 | 7.667382 | not emitted | not emitted |

Relative to v000, v003 reduced mean MAE by **30.27%** and mean sMAPE by **24.35%**. Its paired
per-task outcome was 14 wins, 6 ties, and 10 losses for both MAE and sMAPE. The result supports a
mean improvement from this generated Genome on the frozen manifest, but not uniform improvement:
v003's median MAE was worse and one third of tasks degraded. Moreover, v003 changed Coding search,
hindcasting, prompts, and the Retrieval/Decision topology together, so this comparison cannot
attribute the gain to any single component.

The v003 policy was a Meta-Harness-generated child, not the accepted incumbent of the original
retry2 selection run: it failed that run's screen threshold and v000 was retained. This fresh
evaluation is therefore an exploratory frozen-child comparison, not a retrospective change to the
original acceptance decision.
