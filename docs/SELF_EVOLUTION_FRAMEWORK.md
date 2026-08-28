# Self-Evolution Framework

This document gives a concise map of the repository's self-evolution architecture. The system has
two related layers: a reusable domain-independent Self-Harness and a time-series-specific
three-agent Meta-Harness.

## 1. Generic Self-Harness

The reusable controller is implemented in
[`common/evolution_core/controller.py`](../common/evolution_core/controller.py).
It treats the object being evolved as an artifact and delegates domain behavior to injected
components.

```text
Parent artifact
  -> evaluate on Train and collect failure traces
  -> propose multiple child artifacts
  -> evaluate each child on complete Train
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
[`numerical_agent/curation/__init__.py`](../numerical_agent/curation/__init__.py).
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
[`evolving_loop/coding_agent/evolution.py`](../evolving_loop/coding_agent/evolution.py).
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

The end-to-end system combines three restricted roles. The legacy candidate-aware Retrieval path
remains available as `single-pass` at the CLI (`single_pass` in `HarnessRuntimeConfig`) and is the
backward-compatible runtime default. The fixed two-stage path is explicit and uses a typed
Retrieval release.

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

## 5. Two-Stage Retrieval Boundary

The implementation is in
[`evolving_loop/retrieval_agent/two_stage_agent.py`](../evolving_loop/retrieval_agent/two_stage_agent.py),
with the public assembly in [`evolving_loop/harness.py`](../evolving_loop/harness.py).

```text
safe task target + document text
  -> assumption-blind Round 1
  -> deterministic host verification
  -> provisional Decision over executed candidates
  -> named gaps + four-field sanitized Morphology assumptions
  -> optional gap-directed Round 2
  -> deterministic merge into FinalRetrievalCard
  -> final Decision over executed host candidates
```

Round 1 has exactly the top-level keys `target`, `documents`, and `retrieval_skills`. Round 2 has
exactly `target`, `documents`, `round1`, `gaps`, `assumptions`, and `retrieval_skills`. A Round 2
assumption contains only `assumption_id`, `kind`, `claim`, and `failure_condition`; candidate IDs,
forecasts, hindcast metrics, source code, future values, ground-truth evidence, and evaluator-only
document `role`/`subtype` fields do not cross this boundary.

Both stages return this strict wire shape (the last three keys are optional):

```text
{
  "evidence_chains": [EvidenceChain],
  "counterevidence": [EvidenceChain],
  "missing_information": [str],
  "sufficient": bool,
  "gaps"?: [{"assumption_id", "gap_type", "missing_information", "priority"}],
  "rejected"?: [str],
  "unresolved_contradictions"?: [str]
}
```

An `EvidenceChain` has exactly `chain_id`, `claim`, `entity_match`, `target_match`,
`temporal_relation`, `mechanism`, `direction`, `magnitude_kind`, `magnitude_value`,
`start_timestamp`, `end_timestamp`, `citations`, `missing_links`, `used_skill_ids`,
`addressed_assumption_ids`, `stance`, and `numeric_eligible`. A citation has only `document_id` and
`exact_quote`. The host rechecks the exact quote, identity, target, time window, mechanism,
magnitude, assumption IDs, and configured document/chain/citation budgets; model-authored
`numeric_eligible` is not authority.

The merged host output has exactly `round1`, nullable `round2`, `chains`,
`selected_document_ids`, `rejected`, `unresolved_contradictions`, `complete`, and `gaps`. Invalid
Round 1 JSON fails to the pure numerical route and skips Round 2. Invalid Round 2 JSON keeps
verified Round 1. Frozen Public/hidden inference disables scoring as appropriate, Skill writes,
learning, and evolution.

The current unified CLI deliberately supplies a conservative empty Morphology provider. It can
exercise verified Round 1 safely, but it skips Round 2 because no sanitized Numerical assumptions
exist. The complete two-stage public assembly is proven with a deterministic fake Morphology
provider in `tests/test_retrieval_e2e.py`; a real accepted Numerical/Morphology provider remains a
prerequisite for a real Round 2 experiment.

Post-resolution learning writes Retrieval Skills only as candidates. An internal mutation Child
may name one of those candidates in its desired eventual `active_skill_ids`; only exact IDs named
by that Child are projected through the real two-stage agent and verifier, and only during exact
internal Train shadow stages. The trusted evaluator runs a second pre-label harness replay with a
used candidate withheld and retains that omitted execution's actual final candidate pool,
including any alternative contextual candidate. Inherited accepted Skills remain available as
context; leave-one-out and promotion evidence apply only to named candidate IDs actually used by
the Child. Once the complete screen/fold batch is fully scored, it applies the
cross-task gate to the candidate-specific library: at least three tasks from two entities,
exact-quote validity `1.0`, necessary leave-one-Skill-out replays, non-worse sMAE and sRMSE with one
strict gain, and no added catastrophe. A named candidate that remains unpromoted cannot become the
Train winner, reach Dev, or be published. The transition is one provenance-bound append; the
shared seed/Parent library is not aliased, while Dev, Public, unknown stages, and frozen inference
remain active-only and unchanged.

## 6. Retrieval Genome Evolution

Each generation creates exactly three complete typed Child Genomes, one per immutable mutation
scope. Any out-of-scope change is rejected:

| Child scope | Owned fields | Owned Skill stage |
|---|---|---|
| A · Round 1 | `round1_prompt`, `round1_strategy`, `max_selected_documents` | `round1` |
| B · evidence-chain policy | `max_evidence_chains`, `max_citations_per_chain`, counterevidence-search, target-match, and temporal-overlap requirements | `both` |
| C · Round 2 | `round2_prompt`, `round2_strategy`, `second_round_trigger` | `round2` |

The fixed protocol requires exactly 80 Train and 20 Dev tasks. It chooses exactly eight Train
screening cases from one or more complete entities by default. Those screening entities are
disjoint from the entities in every remaining Train fold; the eight cases are not required to be
internally entity-unique. It promotes at most two children, evaluates them on the remaining
entity-disjoint Train folds, and fixes one Train winner. Only then does it open Dev
once for Parent and Child with persistence, writers, and evolvers disabled. The Child must make a
strict contextual-oracle gain while preserving mean final sMAE/sRMSE, P90/P95, retrieval-quality
tolerances, exact-quote validity, and invalid/catastrophic counts. Otherwise the original Parent is
selected byte-for-byte. Public Regression is never loaded into the mutation/acceptance path.

Checkpoint resume is authenticated and monotonic. The authority record, head, and append-only
anchor ledger must be outside the run root; the wrapper requires at least 32 shell characters and
the CLI requires at least 32 UTF-8 bytes through `RETRIEVAL_CHECKPOINT_AUTHORITY_KEY`. A fresh run atomically publishes
`bootstrap.json` before the first checkpoint transaction. Its `external_anchor` value must be
retained independently, then supplied on every restart through
`RETRIEVAL_CHECKPOINT_AUTHORITY_EXPECTED`; it must not be rediscovered from mutable run artifacts
at resume time. The CLI removes both values from its environment before any LLM subprocess.
Checkpoint schema v2 additionally commits a canonical, deduplicated Skill snapshot table, the
pre/post snapshot hashes of every completed evaluation, and the current snapshot for every Genome
fingerprint in the same atomic publication. Before an evaluation-cache hit can be consumed, resume
validates the checkpoint digest/epoch, exact Skill history and active origins, canonical promotion
evidence recomputed from retained with-Skill and actual omitted candidate pools. Every contextual
forecast must reproduce exactly from its source chain after that chain is reverified against the
immutable task documents and the independently parsed Child Genome's named Skill IDs. All replays
for one task must retain the same primary execution: the baseline must reproduce its trusted
task-trace pool digest and coding-oracle metrics, and the final primary pool must reproduce its
contextual-pool digest and contextual-oracle metrics. Evaluator-computed gates/metrics and an exact candidate-to-accepted copy whose policy
fields are unchanged may then be checked. The authenticated ordered completed-batch sequence binds every pre-library cache key
and pre/post snapshot. After its transition chain and final candidate snapshot validate, each
matching record is consumed exactly once—including earlier unchanged batches and earlier
promotions—without re-running it. Within an authenticated host record, missing, mismatched,
semantically inconsistent, Dev-derived, unbound, or duplicate state is rejected.
Schema-v1 checkpoints fail closed; there is no silent migration.

The operator HMAC key and trusted host evaluator are the checkpoint trust root. Resume does not
independently attest nondeterministic harness execution beyond the authenticated host record. A
trusted operator holding the key and current external anchor may intentionally reissue or migrate
coherent state; that administrative operation is outside the untrusted-model and unauthenticated-
tamper boundary. Without current operator authority, even a coherent rewrite is rejected. The key
and expected-anchor value are scrubbed before construction of any LLM subprocess.

The checked-in `evolving_loop/retrieval_agent/releases/v000` is an unevaluated seed, not an
accepted Child. As of 2026-08-28, only deterministic fake-LLM tests have run: no real Retrieval
80/20 LLM experiment, post-`v000` accepted Retrieval release, Retrieval Public Regression
evaluation, or hidden score exists. Existing Numerical experiment tables elsewhere in the
repository are Numerical results only.

```bash
python -m pytest -q tests/test_retrieval_e2e.py
# Future authorized real run only; this command has not produced a reported result:
scripts/run_retrieval_evolution.sh
```

Before that future command, freeze a run manifest containing the implementation commit, seed
release hash, model and reasoning effort, task/token budgets, split hash, verifier hash, metric
hash/cap, and output directory. Do not start it merely as part of documentation or verification.

## 7. Whole-Harness Co-Evolution

The structured three-agent Meta-Harness is implemented in
[`evolving_loop/co_evolution.py`](../evolving_loop/co_evolution.py). During task inference, all
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

## 8. Three Evolution Depths

| Mode | Mutable surface | Accepted artifact |
|---|---|---|
| `prompt` | Exactly one complete prompt owned by the diagnosed target role | `best_policy.json` |
| `genome` | Role prompts, Numerical search/hindcast budgets, Retrieval/Decision topology, evidence policy, aggregation, and validated skill snapshots | `best_policy.json` |
| `source` | Mutable Agent/Harness Python and new modules under `evolving_loop/generated/` | `best_source.patch` |

Source evolution runs Codex in an isolated Git worktree. A proposed patch must pass the static
label/safety audit, the full test suite, Train selection, and held-out Dev acceptance before it can
become the next repository generation. Scoring, task splitting, label removal, the sandbox, tests,
and the evolution host remain immutable.

This is harness and artifact evolution, not LLM weight training. The system evolves prompts,
skills, dictionaries, candidate budgets, validation settings, communication topology, and—only in
source mode—audited implementation code.

## 9. Frozen Dr-CiK Data Protocol

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

For dictionary curation, run `scripts/run_dictionary_frozen_test.sh` only after the accepted
`working_dictionary.json` is frozen. The command evaluates the 99 Public Test tasks without an LLM,
without artifact write-back, and without exposing its result to mutation or Dev acceptance.

## 10. Main Implementation Map

| Responsibility | File |
|---|---|
| Generic Parent/Child lifecycle | `common/evolution_core/controller.py` |
| Generic contracts and configuration | `common/evolution_core/contracts.py` |
| Dev acceptance gate | `common/evolution_core/acceptance.py` |
| Artifact/checkpoint persistence | `common/evolution_core/persistence.py` |
| Dictionary filtering and curation | `numerical_agent/curation/__init__.py` |
| Numbers-only executable program evolution | `evolving_loop/coding_agent/evolution.py` |
| Three-agent forecast runtime | `evolving_loop/harness.py` |
| Two-stage Retrieval runtime and safe schemas | `evolving_loop/retrieval_agent/two_stage_agent.py`, `evolving_loop/retrieval_agent/schemas.py` |
| Deterministic Retrieval verification | `evolving_loop/retrieval_agent/verifier.py` |
| Retrieval Genome, release, and evolution protocol | `evolving_loop/retrieval_agent/policy.py`, `evolving_loop/retrieval_agent/evolution.py` |
| Frozen Public/hidden inference | `evolving_loop/frozen_inference.py` |
| Prompt/Genome co-evolution | `evolving_loop/co_evolution.py` |
| Source-level evolution | `evolving_loop/source_evolution/__init__.py` |
| Full operational guide | `docs/EVOLVING_AGENT.md` |
