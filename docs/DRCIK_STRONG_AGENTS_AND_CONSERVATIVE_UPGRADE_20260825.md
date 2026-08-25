# Strong Dr-CiK Agents and the Conservative v004 Upgrade

**Date:** 2026-08-25
**Official Dr-CiK revision inspected:** `4acbafe11f2e7caec792277caed606001abaf12c`
**MoiraiAgent revision inspected:** `cfd46d4510ed8896f263116f32928eede05b0a75`
**Scope:** analyze the strongest systems evaluated by Dr-CiK, extract mechanisms that transfer to the existing three-role harness, implement a bounded upgrade, and evaluate it without adding another agent framework.

## 1. Main conclusion

The strongest evidence-producing system in the Dr-CiK paper is Codex with GPT-5.5 High. Under Codex context, the downstream winners differ by metric: MoiraiAgent has the best sRMSE and average ranks, Gemini medium has the best zero-failure sMAE and sCRPS, and Qwen3.5-9B has the lowest sCRPS with 11 failures. The transferable pattern is not a large multi-agent graph:

1. keep numerical extrapolation in an executable forecasting model;
2. make context acquisition iterative and task-specific;
3. preserve exact evidence provenance and quantitative details;
4. permit only a small set of typed, verifiable context operations;
5. retain a numerical-only fallback when evidence is incomplete or contradictory.

The existing Setting 2 path already has the right high-level roles—Coding, Retrieval, and Decision—and a frozen `retrieve -> decide -> retrieve -> decide` policy. The conservative upgrade therefore fixes missing information flow and verification inside that topology rather than introducing a new RAG service, vector database, critic team, or orchestration dependency.

## 2. What the strongest evaluated agents do well

### 2.1 Codex: iterative file-level research rather than one-shot retrieval

The Dr-CiK authors ran Codex CLI 0.128.0 with GPT-5.5 and high reasoning effort. Codex autonomously:

- searched and opened task documents;
- wrote partial summaries while research was still incomplete;
- revised its reasoning after reading additional documents;
- synthesized a final structured evidence report with citations.

This differs from a one-shot embedding top-k pipeline. The intermediate state gives Codex opportunities to notice entity or target mismatches and to pursue missing parts of a multi-document chain. In the paper's controlled evaluation, Codex reached 38.5% Evidence Recall, 48.9% Supporting Document Recall, and 41.0% Distractor Avoidance; every other evaluated deep-research agent remained below 5% Evidence Recall.

The comparison is partially confounded: Codex used GPT-5.5 High, whereas the other four deep-research scaffolds used Gemini 3 Flash. The result therefore supports the complete Codex system, not a clean causal claim that its loop alone creates the full gain.

### 2.2 MoiraiAgent: language coordinates numerical tools

The open MoiraiAgent implementation pairs an LLM coordinator with a time-series foundation-model service and a general Python sandbox. Its documented workflow can:

- select a useful lookback region;
- identify historical anomalies that should not be extrapolated;
- repair or reinterpret corrupted observations;
- anticipate the effects of documented future context;
- choose among forecasting experts.

The public implementation does not itself enforce a typed or bounded adjustment language: both the Python tool and final generation are general-purpose. The typed, host-verified operations introduced below are therefore our conservative adaptation of the numerical-tool principle, not a reproduction of a MoiraiAgent guarantee.

This produces the strongest original-context reference in the paper: `sMAE 0.242`, `sRMSE 0.343`, and `sCRPS 0.206`. With imperfect Codex evidence, MoiraiAgent improves no-context `sCRPS` from `0.338` to `0.310` and `sRMSE` from `0.545` to `0.516`, but worsens `sMAE` from `0.356` to `0.375`. This mixed result is direct evidence that context requires a quality gate rather than unconditional application.

### 2.3 What not to infer from related agents

TimeClaw is relevant as an architectural reference but reports on CiK, not Dr-CiK. Its reusable ideas include executable numeric tools, held-out admission of learned skills, and two-stage retrieval using text plus a numerical series fingerprint. Its published scores cannot be included in a Dr-CiK result table.

Likewise, the new numerical-agent work on current `main` demonstrates useful task profiles, multi-fold diagnostics, candidate screening, and guarded ensembles. On its 20-task development split, its 103-to-56 candidate screen substantially reduced crash exposure while preserving the task oracle. However, its frozen selector did not generalize on the 99-task public test: mean MASE worsened from 2.883 to 3.185 and mean MAE from 555.7 to 1340.2. We therefore borrow diagnostic and safety principles, not that learned selector.

## 3. Failure modes identified by Dr-CiK

### 3.1 Time-series distractors

Time-series distractors are the dominant distractor category even though each distractor subtype has the same corpus count. Rejecting them requires comparing a document's interpretation with the actual observed trajectory and forecast horizon. Topic or entity matching alone is insufficient.

Before v004, the Retrieval Agent received history timestamps but **not history values**. It could not perform this comparison even if its prompt requested it.

### 3.2 Specificity collapse

The paper's Codex case study shows that an agent may find relevant source documents yet lose the forecast-critical details during synthesis. Examples include:

- dropping an exact hour such as `01:00`;
- dropping a count or numerical magnitude;
- losing the start or end of an effect window;
- rewriting a permanent level shift as a continuing trend;
- omitting a modal qualifier or condition.

This explains why Supporting Document Recall exceeds Evidence Recall: locating a source is easier than compiling its precise content into a forecast-safe representation.

### 3.3 Incomplete multi-hop bridges

An event document may provide timing while another document connects the event to the forecast target. A system that stops after finding only one component can produce a plausible but unsupported adjustment. A useful second round should target the exact missing bridge, not repeat broad search.

### 3.4 Context can actively harm forecasting

Most deep-research outputs in Table 1 make the downstream forecaster worse than no context. Raw supporting documents can also hurt when passed without synthesis. Retrieval recall alone is therefore not the deployment objective; evidence must be converted into a complete, verified, horizon-aligned operation, or the system should abstain.

## 4. Existing system before v004

The strongest frozen policy available in the repository, `retry2_v003`, already had several sound properties:

- five initial Coding hypotheses and two rounds of mutation;
- five causal hindcast folds;
- a numerical-only candidate always remains available;
- evidence adjustments are bounded to one candidate;
- the workflow is `retrieve -> decide -> retrieve -> decide`;
- Decision can only choose already executed trajectories;
- an override requires verified citations;
- an evidence-adjusted candidate must cite every document used to create it.

The remaining problems were in the connections between these components:

1. Retrieval could not see historical numeric values.
2. The quote verifier only required that a quantitative source contain *some digit*. A date could therefore allow an unrelated or hallucinated magnitude to pass.
3. If the first retrieval returned no accepted evidence, the harness did not pass its rejection or missing-information state to the second retrieval.
4. The first Decision's `request_more_retrieval`, rationale, and rejection reason were not passed to the second retrieval.
5. Decision could not see the Retrieval Agent's `sufficient`, missing-information, or verifier-rejection status.
6. The merge function accumulated obsolete fields and could not retract a first-round citation or impact. In an interrupted real run, `task_205` returned an empty corrected snapshot in round two, but append-only merge retained round-one `doc_7551` in the final citations.
7. The prompt encouraged emitting contradictory documents as evidence even after determining that they were irrelevant. Dr-CiK counts such citations as distractors regardless of whether the prose correctly calls them distractors.
8. The repository reported raw MAE/sMAPE and history-scaled proxies, but not the official Dr-CiK scaled metrics.

## 5. v004 design and implementation

The v004 policy preserves every v003 Coding prompt, skill snapshot, search budget, validation budget, workflow length, adjustment budget, and aggregation rule. Only Retrieval/Decision evidence handling and benchmark reporting change.

### 5.1 Numeric-history consistency filter

`ContextTask.retrieval_view()` now includes public historical values alongside timestamps. Future labels remain absent. The Retrieval prompt requires checking:

- entity and target identity;
- numerical scale and trajectory;
- event time relative to the forecast boundary;
- whether a claimed regime matches the observed history;
- whether a separate cited bridge is needed.

This directly addresses the paper's dominant time-series-distractor failure without a new retrieval model.

### 5.2 Lossless evidence instructions

The prompt requires preservation of:

- entity;
- target and unit;
- direction and magnitude;
- exact start and end;
- temporary versus permanent persistence;
- modal and conditional qualifiers;
- exact quote and document ID.

It explicitly forbids rewriting a permanent step into a trend and requires all documents in a multi-hop bridge before declaring the evidence sufficient.

The current dataclasses already represent mechanism layer, temporal relation, direction, permanence, adjustment kind, magnitude, window, exact quote, and source IDs. Reusing these fields avoids a second evidence schema.

### 5.3 Deterministic quantitative-impact verifier

For `multiply` and `add` operations, the host parses the accepted source quote(s) and checks that the proposed signed direction and magnitude are expressed near one another. It supports decimal numbers, comma-separated numbers, `%`, `percent`, and `per cent`; a multiplicative change requires an explicit percentage token and converts it to a fraction, while an additive change rejects percentage tokens. Numbers inside recognized date/time spans, and numbers immediately followed by temporal/count units (seconds through years, intervals, or readings), are ignored so they cannot stand in for a target magnitude. Both proposed boundaries must occur in the accepted source quote(s), using date matching for daily boundaries and date-plus-time matching for intraday boundaries.

If the magnitude does not match:

- the quantitative operation becomes `none`;
- the rejection is recorded as `quantitative_impact_without_matching_magnitude`;
- the retrieval cannot declare the impact sufficient;
- no evidence-adjusted candidate is generated from that operation.

If the magnitude check passes but either boundary is missing or unquoted, the host records `quantitative_impact_without_quoted_window`; this likewise disables the operation and prevents Retrieval from being sufficient.

This is a conservative lexical guard, not full semantic entailment: it does not independently prove target-unit compatibility or causal validity. Those remain prompt-level checks audited by Decision, so an incomplete retrieval must fall back rather than be described as fully host-verified quantitative evidence.

### 5.4 One bounded follow-up with real feedback

The existing two-round workflow is retained. The second Retrieval round now receives:

- the first query;
- selected IDs, accepted claims, missing information, and verifier rejections—even if zero evidence survived;
- the first Decision's selected candidate;
- whether Decision requested more retrieval;
- Decision rationale and rejection reason.

The second round is an authoritative complete snapshot rather than a delta. It must repeat all still-valid prior citations, evidence, and impacts, add corrections or bridge evidence, and omit anything ruled out. The host then uses that verified snapshot directly. This lets round two retract a stale distractor, replace a wrong claim, or supersede an earlier impact; it also prevents a stale first-round impact from consuming v004's single adjustment slot.

### 5.5 Decision receives retrieval status

Decision now sees a structured `retrieval_status` containing:

- `sufficient`;
- unresolved `missing_information`;
- deterministic verifier rejections.

The v004 Decision prompt forbids using unresolved or rejected evidence to justify an override. This is also a deterministic host constraint: a non-default choice is rejected whenever Retrieval is insufficient, has an unresolved gap, or has a verifier rejection.

### 5.6 Do not cite ruled-out counterevidence

v004 still asks Retrieval to consider apparent counterevidence internally. If entity, target, time, or historical-trajectory checks rule a document out, it must not appear in `selected_document_ids`, evidence, impacts, or final citations. Contradictory evidence is emitted only when it genuinely applies to the same target and horizon.

This is aligned with the official metric: correctly identifying a distractor in prose does not prevent the citation itself from lowering Distractor Avoidance.

### 5.7 Official metric reporting

The shared scorer now implements future-value scaling, sample-mean point prediction, empirical CRPS, per-task capping, and mean/SE aggregation. Frozen public runs emit:

- `mean_drcik_smae` and `drcik_smae_se`;
- `mean_drcik_srmse` and `drcik_srmse_se`;
- deterministic `mean_drcik_scrps_deterministic` and `drcik_scrps_deterministic_se` for the current point-valued harness.

The deterministic name is intentional. v004 does not falsely present 100 repeated points as calibrated uncertainty.

### 5.8 Retrieval diagnostic correction

For non-empty exported citation sets, the operational diagnostic now computes `1 - cited_distractors / cited_documents`, matching the official per-task denominator. It remains a local proxy because empty sets receive 1.0 in `ResolvedOutcome` instead of being excluded from aggregation, and the harness does not run the maintained private citation resolver. Supporting recall and evidence-token recall are also local proxies because the official metrics require evidence-item/document mappings and semantic judging not currently present in this harness.

### 5.9 Decision-skill and snapshot-identity guards

Decision may report only skill names present in its validated library; an unknown name now routes the choice to the existing host default instead of accepting an override that claims an unavailable rule. Runtime topology validation also prevents votes from being combined across different Retrieval snapshots and requires a final Decision after any later Retrieval. Multiple Decision calls may still be aggregated when they all see the same snapshot. The frozen v004 workflow already uses `last` aggregation and ends in Decision, so these guards remove invalid evolved topologies without changing its intended path.

### 5.10 Globally unique candidate identity

Library, generated, and mutation programs can independently propose the same name. Coding now assigns stable unique names after all three sources have been combined, and evidence-derived candidates avoid every existing ID. Decision also rejects any duplicate-ID input instead of silently overwriting one candidate in its lookup table. This preserves the actual lowest-hindcast host default and prevents resolved candidate-oracle statistics from dropping distinct trajectories that happened to share a name.

## 6. Why this is conservative and general

The change adds no model, service, agent role, unbounded loop, external database, learned reranker, or dependency. It uses information already legal under Dr-CiK inference and host checks that apply across domains.

The expected effect is asymmetric:

- it should reduce harmful contextual overrides and distractor citations;
- it may lower naive retrieval recall because ruled-out documents are no longer emitted;
- it cannot improve cases where Coding lacks a good numerical candidate;
- it does not solve probabilistic calibration.

The numeric-only candidate and fallback target remain unchanged; the new gates route incomplete evidence to that existing default rather than selecting a new trajectory.

## 7. Verification

The focused metric, harness, frozen-inference, co-evolution, outcome-learning, and Dr-CiK tests pass. End-to-end regression cases verify that:

1. round one returns no accepted evidence and names a missing magnitude;
2. Decision requests that magnitude;
3. round two receives both the failed retrieval and Decision feedback;
4. round two supplies a matching exact quote;
5. stale missing information is cleared;
6. only then is the adjusted trajectory selectable.

Additional cases verify that the second-round snapshot can retract a first-round citation and replace its adjustment, that opposite-signed magnitudes, duration numbers, and unsupported windows are rejected, and that the host refuses a non-default choice while Retrieval remains insufficient. A duplicate-identity regression reproduces two same-name candidates with different forecasts and hindcast scores, then verifies unique IDs, the true lowest-hindcast host default, the selected trajectory, and candidate-oracle scoring.

The focused metric, harness, frozen-inference, co-evolution, outcome-learning, and Dr-CiK regression set completed with 73 passes. The final full repository run completed with 1,040 passes, 14 failures, and one skip. All 14 failures are in newly synchronized TSFM environment-packaging tests: the system Python lacks `ensurepip`, and their trusted-parent guard rejects pytest's `/tmp` path. None imports or executes the changed Dr-CiK modules; no unrelated TSFM setup code was changed.

## 8. Frozen 30-task evaluation

The evaluation applies `retry2_v004` to the historical frozen-30 public task set used for the Setting 1 and Setting 2 v2-v4 recomputation. This task set is disjoint from the archived `fresh30` manifest used for the earlier `retry2_v003` aggregate; all 30 historical IDs are explicitly listed as excluded by that manifest. The run uses `gpt-5.6-sol` at high reasoning effort, the v004 policy derived from v003, 100 identical exported trajectories per task, no outcome learning, and four non-overlapping shards that must merge to exactly 30 unique task IDs. Consequently, paired comparisons are valid against the historical Setting 1/v2-v4 artifacts, not against the archived fresh30-v003 aggregate.

`Setting 2 v4` denotes the archived historical run; `retry2_v004` denotes the new policy revision. They are different artifacts and not a same-code causal ablation. A clean prompt ablation would require rerunning v003 in the same environment.

### 8.1 Artifact gate and execution note

The final artifact passed an independent strict gate over exactly 30 unique tasks in four shards of 8/7/8/7 tasks. Every task has a forecast report, a deep-research report, exactly 100 exported trajectories, and independently recomputed metrics that agree with the stored outcome and shard summary. The final analysis used a separate standard-library implementation of the Dr-CiK formulas rather than importing the repository scorer.

The first process was externally interrupted after 24 complete task rows. Only the exact six missing task IDs were rerun with identical arguments, frozen policy hash `4b6ae9e2f97593491adfee0bf7e728d59f1557c2a862e698e405e6eb85df2f87`, model, reasoning effort, caches, and isolated no-persistence state. The two sets were joined by unique task ID. Because the short deep-research file for the interrupted part had not flushed, its rows were deterministically reserialized from the fully written `run_report` citation and evidence fields; every available original row was checked for exact equality. No forecast or score was reconstructed.

### 8.2 Aggregate result

| System | Raw mean MAE | Median MAE | sMAE | sRMSE | Deterministic sCRPS |
|---|---:|---:|---:|---:|---:|
| Setting 1 | 106.9631 | 7.6947 | 0.5234 +/- 0.1471 | 0.7477 +/- 0.1873 | 0.5234 +/- 0.1471 |
| **Historical Setting 2 v4** | **48.0744** | 8.3134 | **0.4581 +/- 0.1089** | **0.6796 +/- 0.1602** | **0.4581 +/- 0.1089** |
| retry2_v004 final | 218.6567 | **6.7028** | 0.6572 +/- 0.1543 | 0.8985 +/- 0.1867 | 0.6572 +/- 0.1543 |

The new run improves raw median MAE but is not an aggregate improvement. Relative to Setting 1, mean sMAE is 25.58% worse and mean sRMSE is 20.16% worse. Relative to historical Setting 2 v4, mean sMAE is 43.48% worse and mean sRMSE is 32.21% worse. One v004 task reaches the sRMSE cap; no v004 task reaches the sMAE or deterministic-sCRPS cap.

The paired results tell the same story:

| Baseline | Metric | Wins / ties / losses | Strict win rate |
|---|---|---:|---:|
| Setting 1 | raw MAE / sMAE | 14 / 0 / 16 | 46.7% |
| Setting 1 | sRMSE | 11 / 1 / 18 | 36.7% |
| Setting 2 v4 | raw MAE / sMAE | 10 / 0 / 20 | 33.3% |
| Setting 2 v4 | sRMSE | 11 / 0 / 19 | 36.7% |

Thus v004 neither wins consistently nor controls its tail risk. The lower raw median is outweighed by severe regressions, and the official normalized medians (`sMAE 0.4117`, `sRMSE 0.6949`) are also worse than v4 (`0.2507`, `0.3945`).

### 8.3 Retrieval and decision diagnostics

The final snapshot retrieves 2.30 documents, emits 3.23 verified evidence items, and emits 1.70 typed impacts per task on average. Local task-macro diagnostics are retrieval precision 0.536, supporting recall 0.139, and citation-set distractor avoidance 0.669. Four tasks retrieve no document. The deterministic verifier records six rejections across two tasks: four ungrounded quotes and two impacts without a verified citation.

However, none of the 51 final impacts is a verified quantitative `add` or `multiply` operation: 42 are `none` and nine are `preserve`. Consequently:

- zero evidence-adjusted candidates are constructed;
- zero non-default decisions are accepted;
- Retrieval changes zero forecasts;
- retrieval candidate gain is exactly zero on all 30 tasks.

This is the central result of the recheck. v004 makes provenance and abstention safer, but its evidence-to-action path is effectively inactive on this slice. All forecasting differences arise from the newly executed Coding candidates and numerical host selection, not from external knowledge. The run therefore does not demonstrate a forecasting gain from the new Retrieval/Decision prompts.

The numerical path also has substantial selection error. The per-task Coding oracle has mean MAE 105.09 and median MAE 3.29, while the selected forecast has 218.66 and 6.70. Mean decision-selection MAE regret is 113.57, although its median is only 2.80; a few severe ranking failures dominate the mean.

### 8.4 Case studies

**Task 73: a genuine point-forecast success, but not a context-caused success.** The selected `conservative_fold_ensemble` attains MAE 4.2605, sMAE 0.1504, and sRMSE 0.2484. Setting 1 has MAE 11.5081 / sMAE 0.4062, while v4 has 22.8636 / 0.8071. Retrieval identifies two corroborating patch-recovery documents and an underspecified planned shutdown, then correctly refuses to infer a shutdown magnitude or apply an adjustment. The selected candidate is also the Coding oracle, so the gain comes from numerical evolution and sound abstention, not from an evidence override.

**Task 207: precise observation-layer research with conservative abstention.** Four supporting documents establish that a highway temperature sensor was already faulty when logging began, corruption ended before the horizon, and post-replacement readings were stable. Local retrieval precision and distractor avoidance are both 1.0. The numeric `rolling_origin_ensemble` reaches MAE 7.2296 and sMAE 0.0996, improving over Setting 1 (8.1434 / 0.1121) and v4 (11.5525 / 0.1591). Because the documents do not state a post-replacement Fahrenheit baseline, v004 applies no fabricated offset. Again, this is a strong research-and-abstention trace rather than proof that context improved the forecast.

**Task 67: catastrophic numerical selection despite correct research.** Retrieval precisely finds a supporting report that historical traffic jitter ended at the forecast boundary and records an observation-layer impact with no invented magnitude. Nevertheless, the lowest-hindcast `worst_fold_phase_ensemble` has future MAE 3,594.89, versus 197.17 for Setting 1, 163.11 for v4, and 515.22 for another available Coding candidate. Decision-selection MAE regret is 3,079.67. The safety gate prevents an unsupported contextual correction, but the host selector has no robust way to map qualitative remediation evidence to a safer existing candidate.

**Task 53: a plausible distractor is rejected too late to rescue the numeric forecast.** Retrieval cites a vessel-speed plateau document but cannot anchor its “next 120 seconds” statement to the actual forecast timestamps; the document is a labeled distractor on this task, so local precision and avoidance are both zero. The gate correctly creates no adjustment, yet the selected damped trend still yields sMAE 1.2549, compared with 0.2918 for Setting 1 and 0.1375 for v4. Preventing a bad evidence action is necessary but does not compensate for a weak numerical default.

### 8.5 Interpretation

The implementation-level safety changes should be retained: official scoring, label isolation, exact citation derivation, signed-magnitude/window verification, authoritative retrieval snapshots, unique candidate identity, and hard fallback on invalid decisions all close real correctness holes. The frozen `retry2_v004` policy should **not** be promoted as a performance upgrade.

The next bounded experiment should first repair the inactive bridge between qualitative evidence and existing numerical candidates. A conservative option is a typed `select_existing` operation: corroborated observation/regime evidence may falsify a candidate assumption and select another already executed trajectory, but may not create values. It should require target identity, horizon alignment, at least one verified supporting chain, and a predeclared candidate tag or failure condition. Separately, the host selector needs a tail-risk criterion using available fold dispersion or worst-fold loss; Task 67 shows that mean hindcast alone is insufficient. Both changes need a same-code v003/v004 ablation before another untouched evaluation.

## 9. Remaining probabilistic gap

After point selection and evidence action coverage are repaired, the largest benchmark-level gap remains probabilistic forecasting. Setting 2 currently repeats one point trajectory, producing zero interval width and approximately 6% coverage for a nominal 90% interval on the historical 30 tasks.

The next bounded addition should be a history-only residual or block bootstrap applied to the frozen selected point model:

1. obtain residuals only from causal rolling hindcasts;
2. preserve temporal blocks where autocorrelation is material;
3. produce at least 100 trajectories;
4. apply a verified contextual operation consistently to each trajectory, or sample only from an explicitly documented magnitude interval;
5. accept the calibration layer only if development `sCRPS` improves without point-metric or coverage failure;
6. freeze all choices before the next untouched evaluation.

This is a separate scientific change and is intentionally not bundled into v004.

## 10. Sources

- [Dr-CiK paper](https://arxiv.org/abs/2605.27904)
- [Dr-CiK official repository](https://github.com/ServiceNow/Dr-CiK)
- [Dr-CiK submission protocol at the inspected revision](https://github.com/ServiceNow/Dr-CiK/blob/4acbafe11f2e7caec792277caed606001abaf12c/SUBMISSION.md)
- [MoiraiAgent research implementation at the inspected revision](https://github.com/SalesforceAIResearch/uni2ts/tree/cfd46d4510ed8896f263116f32928eede05b0a75/project/moirai-agent)
- [TimeClaw](https://arxiv.org/abs/2606.05404)
