# Evolving Agent: Integrated Experimental System

This directory is now the canonical implementation of the new experiment. It preserves the
collaborator's Voyager-style `CodingSkillAgent` baseline and adds a controlled three-agent
forecasting harness.

## What each agent sees

| Role | Allowed input | Forbidden input | Output |
|---|---|---|---|
| Coding Agent | Historical numbers, horizon, frequency, prior numeric skills, historical hindcast errors | Documents, retrieved evidence, GT evidence, future values | Falsifiable assumption, failure condition, executable Python program, hindcast score |
| Retrieval Agent | Candidate assumptions, task identity/time window, corpus documents, validated retrieval-skill summaries | Future values and GT evidence | Exact cited evidence and typed mechanism/impact |
| Decision Agent | Executed forecasts, hindcast scores, verified evidence, validated decision-skill summaries | Future values | Candidate ID and matching citations; it cannot invent values |
| Harness Evolver | Aggregate resolved-task failures on the training split | Labels during task inference | A complete child Harness Genome: prompts, Coding search, workflow/topology, evidence policy, and aggregation |

## Runtime flow

```text
historical numbers
  -> generate/reuse multiple executable skills
  -> sandbox
  -> rolling hindcast on historical cutoffs
  -> revise the best failed program once
  -> validated numeric candidates

candidate assumptions + documents
  -> hypothesis-guided retrieval
  -> exact-quote and document-ID verification
  -> observation / latent-process / future-driver / regime classification
  -> typed evidence impact

executed candidates + verified evidence
  -> conservative host default (lowest hindcast error)
  -> evidence-cited Decision override when justified
  -> final forecast
```

After the true future becomes available, the system computes separate diagnostic rewards:

- Coding: quality/coverage of the best candidate available to Decision.
- Retrieval: supporting-document precision/recall and distractor avoidance.
- Decision: regret between the selected candidate and the best generated candidate.
- Whole system: final forecast score plus retrieval quality.

Successful public training trajectories may also produce two persistent, generalized libraries:

- `retrieval_skills.json`: query and verification strategies learned only from retrieval runs that
  pass the configured precision/recall/avoidance threshold.
- `decision_skills.json`: candidate-selection rules learned only when Decision has negligible
  regret relative to the best available executed candidate.

These skills contain reusable applicability and failure conditions, not raw task answers. The host
rejects generated skills containing task IDs, document IDs, entity names, or exact timestamps.
Skills remain advice: Retrieval still needs exact quotes, and Decision still needs matching evidence
citations. Hidden labels can neither score nor write skills.

The weakest role is reported as a diagnosis, but it no longer restricts what may evolve. A child
may jointly rewrite all role prompts, Coding candidate/mutation budgets, hindcast folds and horizon,
the ordered multi-round Retrieval/Decision topology, evidence-adjustment policy, and Decision
aggregation. It may invent any numerical framework that compiles to the sandboxed `forecast()`
contract, including adaptive selectors, decompositions, ensembles, or new algorithms. Children are
ranked on training tasks and accepted only if the best child improves a disjoint, entity-level
development split. All three
skill libraries are cloned per child. Training tasks may grow those isolated libraries sequentially;
development tasks are strictly read-only and cannot generate skills. This is prompt/skill/harness
evolution, not neural weight training.

The scorer, split, label firewall, citation verifier, forecast-code sandbox, executable interface,
and resource ceilings are deliberately immutable. Allowing a model to rewrite those would reward
leakage or unsafe execution rather than forecasting progress.

## Coding Agent ablations

All settings use identical data splits, LLM, output schema, hindcasting budget, and metrics.

1. `llm_only`: the LLM invents reusable Python skills without a supplied method dictionary.
2. `statistics`: the LLM receives a statistical method dictionary and generates executable skills.
3. `tsfm`: Chronos is executed as the required numeric-only candidate.
4. `combined`: generated statistical programs and Chronos compete under the same hindcast metric.

Generated skills enter `SkillLibrary` only when they execute successfully and beat repeat-last on
historical hindcasts. Actual future labels update evaluation/evolution only; they never approve a
skill during inference.

## Commands

Install both the original `src/drcik_agent` and top-level `evolving_agent` packages:

```bash
pip install -e '.[dev]'
```

Run the complete harness:

```bash
evolving-agent run \
  --tasks-file /path/to/Dr-CiK/data/tasks/train.jsonl \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --setting statistics \
  --limit 30 \
  --learn-from-public-outcomes \
  --results-path runs/evolving/statistics_30.jsonl
```

`codex` is the default LLM backend. Each agent call uses the authenticated local Codex CLI in an
ephemeral, read-only temporary workspace and receives only its role-specific prompt payload. Qwen
is retained only as an explicit ablation via `--llm-backend qwen`; selecting Codex never imports or
loads Qwen weights. A single Dr-CiK JSON task can be passed directly as `--tasks-file` in addition
to the normal JSONL dataset.

The first real Codex smoke test used `gpt-5.6-sol` with high reasoning on public `task_42`, three
statistics-guided Coding candidates, no Coding mutation, and outcome skill learning. The full
Coding → Retrieval → Decision → skill-learning path completed. Final sMAPE was 23.6486 and
retrieval precision was 0.8333. The best available Coding trajectory would have achieved sMAPE
5.2417, but Decision applied a `+5.2%` candidate supported by a confounding document, producing
selection regret 18.4070. This is a useful failure trajectory: the Codex integration is operational,
while temporal/source verification and conservative Decision gating remain the immediate accuracy
bottleneck. It is one mechanism test, not an aggregate benchmark result.

By default, this writes:

```text
runs/evolving/skills.json
runs/evolving/retrieval_skills.json
runs/evolving/decision_skills.json
```

Use `--library-path`, `--retrieval-library-path`, and `--decision-library-path` to isolate
experimental runs.

`--learn-from-public-outcomes` is intentionally explicit. Without it, every task receives an
isolated in-memory clone and `run` is read-only with respect to all three libraries, so an
evaluation task cannot teach later evaluation tasks. Use the
flag only on a chronological public training stream. The `evolve` command enables learning inside
each isolated training evaluation. A child carries the skills learned on its training sequence into
development evaluation, but development remains read-only and cannot add or update skills.

Run the held-out co-evolution loop:

```bash
evolving-agent evolve \
  --tasks-file /path/to/Dr-CiK/data/tasks/train.jsonl \
  --setting combined \
  --limit 50 \
  --generations 3 \
  --children 2
```

The accepted genome is written to `runs/evolving/best_policy.json`. Continue evolution from that
exact inherited framework with:

```bash
evolving-agent evolve \
  --tasks-file /path/to/Dr-CiK/data/tasks/train.jsonl \
  --llm-backend codex \
  --seed-policy-path runs/evolving/best_policy.json \
  --policy-path runs/evolving/best_policy_next.json \
  --generations 3 \
  --children 2
```

The old one-shot comparison remains runnable with
`python -m evolving_agent.coding_agent.baseline --mode fresh|library ...`.

## Important experiment boundary

Dr-CiK has public labels that make stage-wise diagnosis possible. These labels are suitable for
training/evolution splits, but reported claims must use entity-disjoint dev/test tasks. The hidden
test set must never be used to generate prompts, skills, or acceptance decisions.
