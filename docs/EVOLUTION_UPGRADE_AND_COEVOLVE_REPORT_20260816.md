# Single-Agent Evolution and Three-Agent Co-Evolution Upgrade

**Date:** 2026-08-16

**Repository:** `time-series-setting2`

**Branch:** `codex/setting2-domain-knowledge`
**Base HEAD:** `7fdc98b26f8abef4902cd4f16948fc61d4083ace` (the evaluated implementation is an uncommitted working-tree extension of this commit)

## Executive summary

This work completed the requested two phases in order.

1. The inner, single Coding Agent evolution loop was redesigned around deployment-horizon diagnostics, multiple parents, one-mechanism mutations, and conservative acceptance. On four fresh Dr-CiK public-development tasks, the selected Coding trajectory reduced mean future MAE from **1.62718 to 1.54090** (**5.30%**), with **2 wins, 2 ties, and 0 losses** relative to the initial best program.
2. A role-attributed `coevolve` mode was then added for Coding, Retrieval, and Decision. It mutates all three roles separately, identifies task-level specialists, and can merge complementary role prompts only when they share a common parent. It does not add another runtime agent or evolve the evaluator.
3. A real one-generation search selected Retrieval specialist `v001`. On the search split, `v001` improved composite train reward from **0.86994 to 0.87464** and entity-disjoint dev reward from **0.83499 to 0.86652**, improving both dev tasks. A Retrieval+Coding merge `v004` also passed the deployment gate but ranked second on dev.
4. The frozen `v001` did **not** generalize to a separate four-task confirmation set. Mean MAE changed from **68.4171 to 71.4602** (**4.45% worse**), with **1 win, 2 ties, and 1 loss**. Therefore `v001` is not promoted; the seed policy remains the deployable policy.

The principal scientific result is not that co-evolution already improves forecasting. It is that the repository now contains a functioning, auditable role-level co-evolution mechanism, and its first frozen experiment exposed a concrete failure: optimizing document coverage can improve supporting recall while increasing verifier rejections and contextual route regret.

## 1. Scope and evaluation firewall

All experiments used `ServiceNow/Dr-CiK` public-development tasks. Public labels were used only by the host evaluator after forecasts were produced. No online outcome learning was enabled in the four-task confirmation comparison.

The splits were fixed before the three-agent search:

| Purpose | Task IDs | Use |
|---|---|---|
| Single-agent held-out check | 66, 87, 179, 185 | Evaluate the final inner Coding evolution design |
| Three-agent search | 44, 46, 48 | Entity-disjoint split: task 48 train; tasks 44 and 46 dev |
| Frozen confirmation | 49, 50, 51, 52 | Never used to generate, choose, merge, or accept a policy |

The three-agent task manifests explicitly excluded the frozen-30 suite, the earlier outer-evolution pilot tasks, and the single-agent holdout tasks. Confirmation was run only after `best_policy.json` had been written.

All model calls used the authenticated local Codex CLI:

- `codex-cli 0.146.0`;
- model `gpt-5.6-sol`;
- reasoning effort `high`;
- ephemeral read-only temporary workspace;
- 900-second per-call timeout;
- 12 capacity retries with 30-second delay;
- no MaaS or QS API.

No timeout, capacity failure, parser failure, or fallback occurred in the reported three-agent search and confirmation runs.

## 2. Related work and what was adopted

The design was based on mechanisms from primary papers and official implementations, not on adding a general multi-agent framework.

### GEPA

[GEPA](https://arxiv.org/abs/2507.19457) and its [official implementation](https://github.com/gepa-ai/gepa) provide the closest pattern for reflective evolution: rich execution feedback, one-component mutation, Pareto-aware parent selection, and common-ancestor merging of complementary component improvements.

Adopted here:

- actionable structured diagnostics rather than a scalar score alone;
- one owned component per mutation;
- task-level specialist retention;
- common-ancestor prompt merging followed by full re-evaluation.

Not adopted:

- unconstrained mutation of arbitrary system components;
- test-set adaptation;
- evaluator mutation.

### ADAS

[ADAS](https://arxiv.org/abs/2408.08435) and its [official repository](https://github.com/ShengranHu/ADAS) motivate archive-based agent search. Its open-ended source generation is too permissive for this forecasting setting because the scorer, label firewall, forecast sandbox, and evidence verifier must remain immutable. We retained the archive idea but kept the mutation grammar narrow.

### Structural and temporal credit assignment

[Unifying Temporal and Structural Credit Assignment in LLM-Based Multi-Agent Prompt Optimization](https://arxiv.org/abs/2605.30227) motivates assigning failure to a role and using block-coordinate prompt updates instead of rewriting the whole system after every failure. [MAPRO](https://arxiv.org/abs/2510.07475) similarly emphasizes topology-aware downstream blame.

Adopted here:

- task-level losses for Coding, Retrieval, and Decision;
- role-specific failure traces;
- one prompt mutation for each role in every co-evolution generation;
- downstream forecast utility remains part of the immutable system reward.

### Red Queen Gödel Machine

[The Red Queen Gödel Machine](https://arxiv.org/abs/2606.26294) studies co-evolution of agents and evaluators. This repository deliberately does not do that. The evaluator and acceptance thresholds remain fixed within and across the experiment; otherwise the system could improve its apparent score by weakening the judge.

## 3. Phase 1 — redesigning single-agent Coding evolution

### 3.1 Failure patterns in the previous loop

The old inner loop had several mismatches with this project.

1. **Short-fold overfitting.** A mutation could look good on several eight-step folds and still extrapolate poorly over a long deployment horizon.
2. **One-parent tunnel vision.** Only the aggregate hindcast winner was revised, even when another program was much better on the longest fold.
3. **Unstructured mutation.** The model could make a broad rewrite without identifying a single falsifiable failure mechanism.
4. **Cosmetic improvements.** A child could change validation logic yet return essentially the same deployment forecast.
5. **Mean-only acceptance.** Lower average hindcast error could hide severe short-fold regression or larger long-horizon directional bias.

The historical trajectories demonstrate the problem. On task 45, an early revision selected `left_censored_event_cycle` because its aggregate hindcast improved, even though its deployment-scale normalized bias was `0.4883` and its future MAE was worse than its parent. On task 67, a narrow repeat-last route produced MAE `4247.74`, while a structurally different analogue method was far better. These cases motivated both deployment-scale diagnostics and multiple structural parents.

### 3.2 Implemented inner-loop design

The final Coding evolution loop now has the following behavior.

#### Multi-scale causal validation

Each program is evaluated on the normal short rolling folds plus one additional deployment-scale fold when history permits. Fold horizons and normalized signed biases are recorded. The aggregate score is weighted by the number of held-out samples, so the long fold cannot be treated as one small vote equivalent to an eight-step fold.

#### Rich actionable side information

The mutation prompt receives:

- the parent assumption and failure condition;
- every fold's horizon, sMAPE, normalized bias, and execution error;
- the parent deployment forecast;
- the worst fold;
- peer program summaries and their fold scores.

The child must diagnose one failure mechanism and change exactly that mechanism. It is explicitly required to protect folds where the parent already works and to make a material change to the deployment forecast.

#### Two parents per branch

For each open or knowledge-conditioned branch, the system keeps:

1. the lowest aggregate hindcast-error parent; and
2. when distinct, a specialist that is best on the longest available fold.

This retains structural alternatives without a free-form population manager.

#### Host-side mutation gates

A child can replace its parent only when all applicable checks pass:

- the deployment trajectory changes by more than numerical noise;
- absolute hindcast improvement is at least `0.25` sMAPE;
- relative hindcast improvement is at least `1%`;
- short-horizon regression stays below the larger of `0.25` sMAPE or `5%` of the parent score;
- absolute deployment-scale bias regression stays below the larger of `0.02` or `10%` of the parent bias.

The final task-level selector starts from the best initial program and accepts only a mutation that passes the same gate. The parent always remains available. Model decoding is deterministic (`temperature=0`) so parent/child comparisons are not confounded by sampling temperature.

### 3.3 Fresh four-task result

The table compares the initial-best Coding program with the final safely selected Coding program. It does not compare whole-system contextual forecasts.

| Task | Initial-best MAE | Evolved-selected MAE | Result |
|---|---:|---:|---|
| 66 | 0.000901 | 0.000901 | tie |
| 87 | 3.732352 | 3.396047 | win |
| 179 | 2.608333 | 2.608333 | tie |
| 185 | 0.167149 | 0.158333 | win |
| **Mean** | **1.627184** | **1.540904** | **5.30% reduction** |

Across the 14 mutation children with an explicit parent lineage, 9 improved future MAE, 3 worsened it, and 2 tied their parent. The important safety result is that none of the three harmful children was selected: final selection was 2 wins, 2 ties, and 0 losses.

This is a small mechanism check, not a benchmark claim. It establishes that the safer evolution loop can produce useful changes while preserving strong parents on these four unseen tasks.

## 4. Phase 2 — role-attributed three-agent co-evolution

### 4.1 What was missing before

The existing outer loop had train/dev evaluation, a policy archive, and a prompt mode, but the earlier pilot only changed Retrieval. It did not guarantee that Coding, Retrieval, and Decision were each tested; did not attribute task failures structurally; and did not implement a validated merge of complementary role specialists.

### 4.2 Implemented `coevolve` algorithm

One generation now proceeds as follows.

1. Evaluate the incumbent on train and entity-disjoint dev tasks.
2. Convert each resolved training outcome into role losses:
   - Coding: capped oracle forecast sMAPE;
   - Retrieval: one minus the mean of precision, supporting recall, and distractor avoidance;
   - Decision: capped positive selection regret.
3. Rank roles by failure, but generate at least one child for **every** role regardless of rank.
4. Constrain each child to replace exactly its owned prompt: `coding_generation_prompt`, `retrieval_prompt`, or `decision_prompt`.
5. Evaluate each direct child on train. Children that regress beyond the immutable train allowance never see dev.
6. Evaluate train-safe children on dev and identify task specialists: a specialist must improve at least one shared dev task relative to the common parent.
7. If at least two different roles have specialists, combine their disjoint prompt changes only when they have the same parent. Re-evaluate the merge from scratch on train and dev.
8. A deployable candidate must satisfy all of:
   - no forbidden train regression;
   - aggregate dev improvement over the incumbent;
   - no individual dev-task regression beyond the configured tolerance.
9. Select the deployable candidate with the highest dev reward. Other task specialists may remain in the bounded Pareto archive, but do not become the deployed policy.

The evaluator, task split, metric, exact-quote verifier, label boundary, and forecast-code sandbox are not mutable. The trace records role failure scores, role ownership, train/dev task rewards, rejection reasons, merge lineage, archive coverage, and the complete set of policies that passed the deployment gate.

### 4.3 Actual generated mutations

The first real run diagnosed Retrieval as the dominant training failure:

| Role | Failure score |
|---|---:|
| Retrieval | 0.552222 |
| Coding | 0.024198 |
| Decision | 0.000338 |

Nevertheless, all three roles were mutated once.

- **Retrieval `v001`:** introduced a mandatory entity-event-target-window coverage ledger before pruning. It attempts to preserve every same-chain event, mechanism, target, magnitude, window, resolution, and contradiction link.
- **Coding `v002`:** required a robust geometrically damped recent-trend candidate that becomes level-like when slope estimates disagree, while retaining plateau/change-point alternatives.
- **Decision `v003`:** introduced an assumption-entailment gate: an override must contradict a necessary assumption of the host default or uniquely support another candidate.

Retrieval `v001` and Coding `v002` showed complementary task specialties, so the system constructed common-parent merge `v004`. Decision `v003` failed the train guard and did not enter dev.

### 4.4 Search result

The reward is the repository's fixed composite system reward, not MAE alone.

| Policy | Mutated role(s) | Train reward | Dev reward | Dev task 44 | Dev task 46 | Status |
|---|---|---:|---:|---:|---:|---|
| `v000` | seed | 0.869941 | 0.834986 | 0.823530 | 0.846443 | incumbent |
| `v001` | Retrieval | 0.874640 | **0.866520** | **0.854319** | **0.878720** | selected |
| `v002` | Coding | 0.873853 | 0.834017 | 0.800876 | 0.867159 | dev regression; not deployable |
| `v003` | Decision | 0.856389 | not evaluated | — | — | train regression |
| `v004` | Retrieval + Coding merge | **0.892927** | 0.852758 | 0.829677 | 0.875840 | deployable, ranked second |

`v001` improved both dev tasks and raised mean dev reward by `0.031534`. `v004` also improved both tasks relative to `v000`, proving that the common-ancestor merge path ran successfully; however, its dev reward was lower than `v001`, so it was archived rather than selected. The regenerated trace explicitly records `deployment_candidate_versions = ["v001", "v004"]`.

## 5. Frozen confirmation result

After search, `v001` was frozen and compared against `v000` on tasks 49–52. Both runs used identical Coding prompts, caches, libraries, model, and runtime configuration. Outcome learning was disabled.

| Task | Seed MAE | `v001` MAE | Delta (`v001` − seed) | Result |
|---|---:|---:|---:|---|
| 49 | 160.227921 | 160.227921 | 0.000000 | tie |
| 50 | 0.618644 | 0.618644 | 0.000000 | tie |
| 51 | 25.720544 | 24.148230 | −1.572314 | win |
| 52 | 87.101276 | 100.845919 | +13.744643 | loss |
| **Mean** | **68.417096** | **71.460178** | **+3.043082** | **4.45% worse** |
| **Median** | **56.410910** | **62.497074** | **+6.086164** | worse |

Mean sMAPE changed from `67.130265` to `67.181845`. The exact bootstrap over all `4^4 = 256` task resamples assigns only **25.39%** probability to a positive mean-MAE improvement. With only four tasks this is descriptive, but it clearly does not support promotion.

### 5.1 Failure mechanism on task 52

The Retrieval mutation behaved as designed at the coverage level:

- supporting recall increased from `0.5833` to `0.9167`;
- retrieval precision changed from `0.6364` to `0.6471`;
- distractor avoidance decreased from `0.84` to `0.76`.

However, the expanded ledger also emitted quotes that the deterministic verifier rejected. Because the host uses a conservative all-rejections gate, contextual route weight changed from `0.2` to `0.0`. The final forecast therefore moved from the selected `validated_recurrent_pulse` toward the unconditioned top-three median. That increased MAE from `87.10` to `100.85`.

Task 51 shows the opposite side of the same mechanism: route weight also changed from `0.2` to `0.0`, but the numeric median happened to improve MAE by `1.57`. The policy therefore changed forecasting through verifier/gating interactions rather than through a uniformly better evidence interpretation.

### 5.2 Decision

`v001` is rejected for deployment despite passing the small search split. The seed `v000` remains the default. Confirmation tasks 49–52 are now considered observed and must not be used for the next policy-selection cycle.

## 6. What the experiment teaches us

1. **Role-level co-evolution is feasible in the existing architecture.** No additional agent framework, shared-memory service, vector database, or fourth runtime agent was required.
2. **A common-ancestor merge can be generated and fully evaluated.** The system produced a valid Retrieval+Coding merge and correctly left it in the archive when a single-role specialist ranked higher.
3. **Composite reward on three search tasks is too weak a selector.** The chosen policy improved retrieval-oriented reward on the search split but did not improve frozen forecast MAE.
4. **More recall is not automatically useful.** Coverage expansion can increase both supporting recall and invalid or distracting evidence, which interacts discontinuously with a conservative gate.
5. **The single-agent inner loop is currently the more reliable improvement.** Its safe selector delivered a modest positive result without selected-task regression in the four-task holdout.

## 7. Recommended next experiment

The next co-evolution run should not modify `v001` using tasks 49–52. Instead:

1. allocate a larger, entity-disjoint search split, for example at least 12–20 train tasks and 8–12 dev tasks;
2. keep forecast utility as the primary acceptance objective and treat retrieval recall as a constrained secondary objective;
3. add a host-side negative-control constraint for verifier rejection rate and route-weight flips, rather than merely asking the Retrieval prompt to be cautious;
4. keep one-role mutation and common-parent merge unchanged;
5. preregister another untouched confirmation group before search;
6. promote a policy only if it improves aggregate forecast error without a harmful-task tail and preserves retrieval precision/avoidance.

No further confirmation-driven prompt tuning was performed in this work.

## 8. Verification and artifacts

The final test suite result is:

```text
184 passed in 6.92s
git diff --check: clean
```

Key artifacts:

- `runs/single_agent_evolution_20260816/holdout4_results.jsonl`
- `runs/three_agent_coevolution_20260816/search_manifest.json`
- `runs/three_agent_coevolution_20260816/confirmation_manifest.json`
- `runs/three_agent_coevolution_20260816/best_policy.json`
- `runs/three_agent_coevolution_20260816/policy_archive.json`
- `runs/three_agent_coevolution_20260816/evolution_trace.json`
- `runs/three_agent_coevolution_20260816/confirmation_seed.jsonl`
- `runs/three_agent_coevolution_20260816/confirmation_v001.jsonl`

SHA-256:

```text
60c650ac648a393d9142c4294d75b677ef02e700dbad282145cc14c0578519e6  holdout4_results.jsonl
10c05140e6fd3a84c1afe6980ac1148c0e3bc376951140c466ed8b2d0a0da9d4  search_manifest.json
3f02fafd97c12bd0f85ad00de084ac579c08e51033916e686aaf1593f2d73a91  confirmation_manifest.json
a7366c5b2055614beab8b427176a763178b48181f86c0cbf52d8d1896c6dfb0d  best_policy.json
6d99c0f35c4f89e0b2605bcac93bd7e96386f6e4432140202b3088ddcf91655a  policy_archive.json
6804cabd0ddbd9ad6d160b476136d572fe18612bf810aeed691cacef2b8d209f  evolution_trace.json
98887b0dc643cd91d041eb25be41c8ebc12cc0693d19b31a6f047960409cc342  confirmation_seed.jsonl
af6491b26aacdf691fc6fb58fdeab6b5103c75636e2bbc0377b4f837aba76b93  confirmation_v001.jsonl
```
