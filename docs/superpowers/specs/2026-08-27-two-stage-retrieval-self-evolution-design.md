# Two-Stage Retrieval Self-Evolution Design

**Date:** 2026-08-27

**Status:** Approved design; implementation not started

**Scope:** `evolving_loop/retrieval_agent`, its harness integration, trusted
evaluation, and Retrieval-targeted evolution

## 1. Purpose

The repository already has a Retrieval Agent, exact-quote verification, a
persistent Retrieval Skill Library, delayed-outcome skill learning, and
Prompt/Genome/Source evolution. The current Retrieval Agent nevertheless sees
Coding hypotheses during its first pass and represents most retrieval behavior
inside one large prompt. That makes confirmation bias and failure attribution
hard to control.

This design turns Retrieval into an independently evolvable subsystem with two
separated inference stages:

1. an assumption-blind evidence pass;
2. an assumption-guided gap and counterevidence pass.

The subsystem evolves a typed Retrieval Genome and typed Retrieval Skills. It
does not evolve the trusted verifier, labels, scorer, split, or leakage
boundary.

## 2. Goals

- Retrieve evidence independently before numerical assumptions can bias it.
- Allow a second pass to investigate explicit gaps in numerical assumptions.
- Represent evidence as verified causal chains rather than isolated quotes.
- Attribute retrieval quality separately from Decision selection quality.
- Give complete evidence chains Train-only marginal sMAE/sRMSE credit.
- Evolve Retrieval prompts, strategies, budgets, topology, and skills through
  typed mutations and Train/Dev elitism.
- Preserve label-free, write-free frozen inference on Dev, Test, and hidden
  tasks.
- Reuse current host-side quote, provenance, temporal, candidate, and metric
  gates instead of replacing them.

## 3. Non-Goals

- Retrieval does not write forecasting code or forecast values.
- Retrieval does not modify TSFM weights or checkpoints.
- Retrieval does not receive document role/subtype labels, GT evidence, or
  resolved future values during inference.
- Round 2 does not receive candidate forecast arrays, source code, candidate
  names, or hindcast scores.
- This phase does not implement an open-ended CORAL-style agent society.
- This phase does not permit source-level mutation of the trusted verifier,
  evaluator, data loader, split, CLI security boundary, or tests.

## 4. Recommended Approach

Three implementation depths were considered:

1. **Prompt-only:** cheap but conflates query planning, evidence composition,
   and verification behavior inside one prompt.
2. **Typed Retrieval Genome:** selected. It exposes prompts, strategies,
   budgets, stage triggers, and skills while retaining immutable host gates.
3. **Source-level multi-agent evolution:** deferred until the typed Genome has
   demonstrated held-out value.

The typed Genome is the smallest approach that supports causal attribution,
versioning, and safe evolution without turning every change into an arbitrary
source patch.

## 5. End-to-End Architecture

```text
Historical values ──> Morphology Reasoner ──> Morphology Card ─────┐
                                                                  │
Target + timestamps + documents ──> Round 1 Independent Retrieval │
                                    └─> Verified Evidence Ledger ──┤
                                                                  v
                                                    Provisional Decision
                                                       / Gap Critic
                                                                  │
                                                                  v
                                             Sanitized named gap request
                                                                  │
                                                                  v
                                        Round 2 Assumption-Guided Retrieval
                                                                  │
                                                                  v
                                           Host merge + deterministic verifier
                                                                  │
                                                                  v
                                                 Final Retrieval Card
                                                                  │
                                                                  v
                                                    Final Decision Agent
```

The Morphology Reasoner and Round 1 may execute independently. The host, not an
agent, converts a Morphology Card and provisional Decision result into the
sanitized Round 2 input.

## 6. Inference Schemas

### 6.1 Round 1 Input

Round 1 receives only:

```json
{
  "target": {
    "entity_name": "entity",
    "target_name": "target",
    "description": "description",
    "frequency": "D",
    "forecast_window": ["first timestamp", "last timestamp"]
  },
  "documents": [
    {"document_id": "doc_1", "content": "document text"}
  ],
  "retrieval_skills": []
}
```

Round 1 does not receive numerical candidates, assumptions, hindcast scores,
future values, GT evidence, document role/subtype labels, prior task outcomes,
or method-selection results.

### 6.2 Round 1 Output

```json
{
  "evidence_chains": [
    {
      "chain_id": "chain_1",
      "claim": "A scheduled intervention overlaps the horizon.",
      "entity_match": true,
      "target_match": true,
      "temporal_relation": "overlaps_future",
      "mechanism": "latent_process",
      "direction": "up",
      "magnitude_kind": "unknown",
      "magnitude_value": null,
      "start_timestamp": "timestamp or null",
      "end_timestamp": "timestamp or null",
      "citations": [
        {"document_id": "doc_1", "exact_quote": "verbatim span"}
      ],
      "missing_links": ["explicit_magnitude"],
      "used_skill_ids": ["explicit_window_search"]
    }
  ],
  "counterevidence": [],
  "missing_information": ["explicit_magnitude"],
  "sufficient": false
}
```

All quoted evidence, cited document IDs, time bounds, magnitude claims, and
chain fields remain untrusted until the host verifier accepts them.

### 6.3 Provisional Decision / Gap Critic

The existing Decision layer may inspect verified Round 1 evidence and executed
numeric candidates. Its externally visible Round 2 output is restricted to:

```json
{
  "gaps": [
    {
      "assumption_id": "a_trend",
      "gap_type": "continuation_or_reversal",
      "missing_information": "Evidence of continuation or reversal",
      "priority": "high"
    }
  ]
}
```

The host rejects free-form candidate values, scores, source code, document
labels, or future values in this interface.

### 6.4 Round 2 Input

Round 2 receives:

- the verified Round 1 ledger;
- the sanitized gap request;
- anonymized assumption IDs, kinds, claims, and failure conditions from the
  Morphology Card;
- validated Retrieval Skills applicable to Round 2.

It does not receive candidate names, forecast arrays, source code, hindcast
scores, future values, GT evidence, or document labels.

### 6.5 Round 2 Output

Round 2 uses the same Evidence Chain schema as Round 1 and adds:

- the addressed `assumption_id` values;
- whether a chain supports, challenges, or leaves an assumption unresolved;
- explicit unresolved contradictions;
- a delta-only list of newly found evidence and gaps.

The host verifies Round 2 independently, then merges it with Round 1 by stable
chain and citation identity. Round 2 cannot overwrite or erase verified Round
1 evidence.

### 6.6 Final Retrieval Card

The merged card contains:

- verified evidence and counterevidence chains;
- verified typed impacts;
- assumption support/challenge status;
- missing links and unresolved contradictions;
- selected document IDs;
- exact skill IDs used by each chain;
- per-stage rejection reasons and completeness status.

Only this card may enter the final Decision Agent.

## 7. Retrieval Genome

Each generation owns one immutable `RetrievalGenome`:

```json
{
  "schema_version": 1,
  "version": "v003",
  "parent": "v002",
  "round1_prompt": "complete prompt",
  "round2_prompt": "complete prompt",
  "round1_strategy": "timeline_first",
  "round2_strategy": "counterevidence_first",
  "second_round_trigger": "on_named_gap",
  "max_selected_documents": 8,
  "max_evidence_chains": 4,
  "max_citations_per_chain": 4,
  "require_counterevidence_search": true,
  "require_target_match": true,
  "require_temporal_overlap": true,
  "active_skill_ids": ["explicit_window_search"]
}
```

Allowed strategy values and numeric ranges are enumerated in the schema. The
LLM proposes a complete child Genome; Python parses and validates it before any
task execution.

The immutable host always enforces exact quotes, provenance, time parsing,
magnitude eligibility, label isolation, resource limits, and fallback behavior.
A Genome may strengthen these boundaries but cannot disable them.

## 8. Retrieval Skill Schema and Lifecycle

Retrieval Skills are declarative, Git-tracked data rather than executable
Python. Execution and verification remain host-owned.

```json
{
  "skill_id": "explicit_window_search",
  "version": 2,
  "parent_version": 1,
  "stage": "round1",
  "status": "specialized",
  "name": "explicit_window_search",
  "description": "Find exact event start and end boundaries.",
  "applicability": {
    "assumption_kinds": ["future_event", "regime_change"],
    "gap_types": ["missing_start", "missing_end"],
    "temporal_relations": ["overlaps_future"]
  },
  "query_steps": [
    "Find the named event.",
    "Find its first affected timestamp.",
    "Find its last affected timestamp."
  ],
  "required_chain_fields": [
    "entity", "target", "mechanism", "start_timestamp", "end_timestamp"
  ],
  "counterevidence_rule": "Search for cancellation, postponement, containment, or recovery.",
  "failure_conditions": [
    "The document concerns another entity.",
    "The event ended before the forecast window."
  ]
}
```

The Evolver may submit only:

- `add`: create a new candidate skill;
- `repair`: replace one skill version while preserving identity and lineage;
- `specialize`: narrow applicability after subgroup evidence;
- `merge`: replace redundant skills with one lineage-aware successor;
- `quarantine`: disable a dangerous or invalid skill without erasing history.

There is no destructive delete. Rejected and quarantined skills remain
auditable. Reactivation creates a new version and must pass normal validation.

A skill becomes `accepted` only after use on at least three Train tasks across
at least two entities, 100% exact-quote validity, non-worse sMAE and sRMSE than
the no-skill counterfactual, at least one strict metric improvement, no increase
in catastrophic candidates, and one read-only Dev acceptance. A skill that
passes only within a typed subgroup becomes `specialized`.

## 9. Outcome Attribution and Rewards

Inference first runs on a sanitized task with no labels. Only after all
Retrieval and Decision artifacts are frozen may the trusted evaluator read
public Train labels, GT evidence, or document relevance labels.

Retrieval uses a metric vector rather than one blended scalar:

- supporting-document and GT-evidence recall;
- distractor avoidance;
- exact-quote validity;
- complete-chain rate;
- contextual-oracle sMAE gain;
- contextual-oracle sRMSE gain;
- invalid and catastrophic candidate counts.

The contextual gains are:

```text
coding-oracle error - contextual-oracle error
```

They measure whether Retrieval created a better available candidate. Final
Decision regret is reported separately and is not assigned to Retrieval.

### 9.1 Evidence-Chain Credit

One causal chain becomes forecast-eligible only when its required entity,
target, mechanism, time-window, and magnitude fields are complete. A qualitative
chain may be verified without becoming eligible for a numeric adjustment.

For each newly completed chain, the trusted evaluator compares the candidate
pool before and after adding the chain and records the marginal change in capped
Dr-CiK sMAE and sRMSE. An incomplete quote may receive evidence-recall credit but
never forecast-utility credit.

If a chain names multiple skills, its credit is initially joint. Before skill
promotion, leave-one-skill-out replay on matching Train cases must demonstrate
which skill is necessary. The implementation must not copy the full chain
reward independently to every named skill.

## 10. Self-Evolution Procedure

The working Retrieval Agent performs inference. A separately contracted
Retrieval Evolver runs only after Train outcomes resolve. Both roles may use the
same base LLM, but they have different inputs and permissions.

Each generation creates three scoped children:

| Child | Mutable scope |
|---|---|
| A | Round 1 query, relevance, and document-selection strategy |
| B | Evidence-chain composition, time semantics, contradiction search |
| C | Round 2 assumption-gap strategy and trigger |

Skills may be mutated only when the child scope owns the corresponding stage.
One child cannot edit all three scopes.

The Train-only search is:

1. evaluate Parent and three Children on the same eight screen Train tasks;
2. prune invalid or clearly dominated children;
3. promote at most two children to the complete 80-task Train split;
4. compare entity-disjoint Train folds;
5. retain the Pareto-safe Train winner as the provisional parent;
6. repeat for the configured number of Train generations;
7. compare the original Parent and final Train winner exactly once on the
   20-task Dev split.

No Dev result is passed back to the Evolver. The 99-task public regression set
is not used for mutation or acceptance. Hidden tasks are inference-only.

## 11. Acceptance Gates

Every candidate must first satisfy:

- zero leakage and zero forbidden evaluator access;
- 100% exact-quote validity for accepted evidence;
- full task coverage;
- no increase in Crash or Invalid counts;
- no increase in capped/catastrophic sMAE or sRMSE task counts;
- supporting recall and distractor avoidance each no more than 0.02 below the
  Parent;
- all resource and deterministic-verifier tests.

The final Dev acceptance additionally requires:

- contextual-oracle mean sMAE no worse than Parent;
- contextual-oracle mean sRMSE no worse than Parent;
- at least one of those two metrics strictly better;
- final-system mean sMAE and sRMSE no worse than Parent;
- P90 and P95 sMAE no worse than Parent;
- no new task-level catastrophic failure.

Floating-point comparisons use the repository's fixed tolerance. The tolerance,
metric cap, split hash, and evaluator hash are frozen before evolution.

## 12. Failure and Fallback Behavior

- Round 1 parse/runtime failure yields an empty verified ledger and the pure
  Numerical fallback.
- Round 2 failure preserves verified Round 1 evidence.
- Invalid quotes are removed individually and recorded; they do not become
  free-form Decision context.
- Incomplete chains may remain qualitative but cannot create numeric edits.
- Unknown skills are rejected and recorded.
- Transient network/model failures are retried and checkpointed; they are not
  scored as forecasting failures.
- Budget exhaustion produces a deterministic partial result and stop reason.
- Dev, Test, hidden, and frozen inference prohibit skill writes, Genome writes,
  and Evolver calls.

## 13. Release and Audit Artifacts

The existing repository remains the only Git repository. No nested `.git`
directory is created.

```text
evolving_loop/retrieval_agent/releases/
└── v003/
    ├── genome.json
    ├── round1_prompt.md
    ├── round2_prompt.md
    ├── skills.json
    └── manifest.json
```

The manifest binds:

- version and Parent version;
- Train/Dev split hash;
- prompts and Skill Library hashes;
- verifier, evaluator, and metric hashes;
- resource budgets;
- Train/Dev summaries and acceptance reason.

Rejected children and detailed traces remain under `runs/`; they are never
presented as releases.

## 14. Co-Evolution Sequence

Initial development uses coordinate isolation:

1. freeze Numerical/Morphology and Decision; evolve Retrieval;
2. freeze Numerical/Morphology and Retrieval; evolve Decision;
3. after each module has a held-out-safe release, enable alternating
   co-evolution.

Alternating co-evolution mutates only one principal module per generation:

```text
Numerical/Morphology -> Retrieval -> Decision -> diagnose weakest module -> repeat
```

This preserves attribution. Open-ended simultaneous source mutation is deferred.

## 15. Implementation Boundaries

New modules are expected under:

```text
evolving_loop/retrieval_agent/
├── schemas.py
├── two_stage_agent.py
├── policy.py
├── credit.py
├── evolution.py
├── skill_library.py
└── releases/
```

The implementation will extend, not replace:

- `evolving_loop/harness.py` for the two-stage topology and safe fallback;
- `evolving_loop/co_evolution.py` for Retrieval-scoped typed children;
- `evolving_loop/evaluation.py` for trusted Retrieval diagnostics;
- the CLI for explicit evolution and frozen-inference modes;
- existing tests for leakage, quote verification, skills, and co-evolution.

The Retrieval package consumes a sanitized Morphology schema. It must not import
Numerical Agent implementation internals or create a cyclic package dependency.

## 16. Required Tests

### Unit contracts

- exact Round 1, gap, Round 2, Genome, Skill, and final-card schemas;
- rejection of unknown fields, tools, skills, IDs, timestamps, and operations;
- immutable verifier behavior across every Genome;
- deterministic merge and stable chain identity;
- skill lineage and non-destructive quarantine.

### Leakage and adversarial tests

- future values, GT evidence, role/subtype labels, forecast arrays, hindcast
  scores, candidate names, and source code never enter Retrieval prompts;
- prompt injection cannot disable exact-quote or provenance checks;
- document order and IDs are not treated as relevance labels;
- fabricated, partial, cross-entity, cross-target, and temporally invalid chains
  cannot produce numeric adjustments.

### Attribution tests

- Retrieval receives contextual-oracle gain while Decision regret remains
  separate;
- incomplete chains receive no forecast-utility credit;
- joint skill credit is not duplicated;
- leave-one-skill-out replay identifies necessary skills;
- network failures do not become forecast failures.

### Evolution tests

- each Child changes only its owned scope;
- screening uses Train only;
- Dev is read-only and evaluated once;
- rejected children cannot modify the release or Skill Library;
- resumed runs preserve hashes and task completion state;
- hidden/frozen mode performs zero writes and zero mutation calls.

### End-to-end gates

- current single-pass behavior remains available as an explicit baseline;
- two-stage Parent and children run on identical cases and budgets;
- release creation occurs only after every Train/Dev and safety gate passes.

## 17. Completion Criteria

The subsystem is complete when:

1. both retrieval rounds and the gap sanitizer obey the schemas above;
2. the trusted verifier and fallback remain immutable;
3. chain- and skill-level Train credit is reproducible;
4. typed Retrieval evolution supports all three child scopes;
5. a complete 80 Train / 20 Dev run can freeze or reject a release without
   accessing the 99-task regression or hidden sets;
6. frozen inference is write-free and Evolver-free;
7. all focused and full repository tests pass;
8. reports clearly separate Retrieval quality, contextual-oracle utility,
   Decision regret, and final-system performance.
