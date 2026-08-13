# Evolving Agent: Integrated Experimental System

This directory is now the canonical implementation of the new experiment. It preserves the
collaborator's Voyager-style `CodingSkillAgent` baseline and adds a controlled three-agent
forecasting harness.

## What each agent sees

| Role | Allowed input | Forbidden input | Output |
|---|---|---|---|
| Coding Agent | Historical numbers, horizon, frequency, prior numeric skills, historical hindcast errors | Documents, retrieved evidence, GT evidence, future values | Falsifiable assumption, failure condition, executable Python program, hindcast score |
| Retrieval Agent | Candidate assumptions, task identity/time window, corpus documents | Future values and GT evidence | Exact cited evidence and typed mechanism/impact |
| Decision Agent | Executed forecasts, hindcast scores, verified evidence | Future values | Candidate ID and matching citations; it cannot invent values |
| Harness Evolver | Aggregate resolved-task failures on the training split | Labels during task inference | One full prompt replacement for the weakest role |

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

The weakest role is eligible for one prompt mutation. Children are ranked on training tasks and
accepted only if the best child improves a disjoint, entity-level development split. This is
prompt/skill/harness evolution, not neural weight training.

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
  --setting statistics \
  --limit 30 \
  --results-path runs/evolving/statistics_30.jsonl
```

Run the held-out co-evolution loop:

```bash
evolving-agent evolve \
  --tasks-file /path/to/Dr-CiK/data/tasks/train.jsonl \
  --setting combined \
  --limit 50 \
  --generations 3 \
  --children 2
```

The old one-shot comparison remains runnable with
`python -m evolving_agent.coding_agent.baseline --mode fresh|library ...`.

## Important experiment boundary

Dr-CiK has public labels that make stage-wise diagnosis possible. These labels are suitable for
training/evolution splits, but reported claims must use entity-disjoint dev/test tasks. The hidden
test set must never be used to generate prompts, skills, or acceptance decisions.
