# Toto-Balanced v3 Three-Agent Co-Evolution Review

**Experiment date:** 2026-09-06

**Status:** Design audit and interaction smoke complete; formal and Public-99
results pending.

## Scope and interpretation

This experiment evolves one package containing a Numerical agent, a Retrieval
agent, and a Decision agent. The coordinate order is Numerical → Retrieval →
Decision for at most two cycles. Each formal coordinate proposes three children,
but only one principal component may change in a child. Selection uses the final
point forecast from the complete package rather than a proxy score from the
component being changed.

The registered v3 partition contains 80 Train tasks, internally divided into 64
Build and 16 Calibration tasks, 20 Dev tasks, and 99 Public regression tasks.
The split was stratified with future Toto error, so Public-99 is a frozen,
label-informed regression set. It is useful for balanced paired comparison, but
it is not an untouched estimate of generalization. Only the official hidden test
or a newly collected test set can support that claim.

The package evaluator reports capped and raw sMAE and sRMSE. It does not report
sCRPS, so its results are not a direct reproduction of the distributional ranks
in Dr-CiK Tables 6 and 7.

## Research alignment

### What is well supported

1. **Keep a strong numerical prior recoverable.** The Numerical release retains
   Toto as a protected anchor, and Retrieval/Decision cannot directly replace
   its forecast with unconstrained free-form numbers. This matches the
   context-guided-reviser framing in
   [Rethinking Post-Training Recipes for Multimodal Time-Series Forecasting](https://arxiv.org/abs/2605.29401),
   which finds that an LLM should learn when to revise or preserve a TSFM prior,
   and the immutable-baseline, validated-action design in
   [Bridging the Last Mile of Time Series Forecasting with LLM Agents](https://arxiv.org/abs/2606.02497).

2. **Judge retrieval by downstream forecast utility.** Dr-CiK reports that
   high-quality supporting evidence helps substantially, while current deep
   retrieval agents often recover less than 5% of supporting evidence, cite
   distractors more than 80% of the time, and can make forecasting worse than no
   context. The present pipeline therefore uses complete-package forecast gates,
   target matching, temporal overlap, exact evidence traces, and counterevidence
   search rather than treating topical retrieval as success. This is aligned
   with [Dr-CiK](https://arxiv.org/abs/2605.27904).

3. **Use structured evidence instead of an ever-growing transcript.** The task
   feedback path passes a host-validated projection with stance, target match,
   window relation, magnitude status, mechanism, and decision action. That is
   directionally consistent with the structured linguistic belief state in
   [Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs](https://arxiv.org/abs/2604.18576),
   while remaining specific to continuous time-series decisions.

4. **Learn on historical labels, then freeze before final evaluation.** All
   proposal and feedback activity is restricted to Train and Dev, and the
   Public evaluator accepts only a sealed bundle. This matches the separation in
   [From Long News to Accurate Forecast](https://arxiv.org/abs/2606.03097), where
   forecast-aware filtering and retrieval ranking are trained offline with
   historical outcomes and frozen at inference.

### What is not yet fully supported

1. **Retrieval is forecast-aware at evaluation time, not yet at ranking time.**
   The current Retrieval agent mutates a scoped genome and is rewarded by final
   forecast quality, but it has no learned Train-only model that ranks document
   candidates by expected forecast improvement. The Long News paper gives a
   stronger mechanism: generate a candidate pool, learn an error-conditioned
   process reward model from historical outcomes, and freeze it for deployment.
   Adding a small forecast-utility reranker is the highest-value architectural
   follow-up if this run shows weak or unstable Retrieval gains.

2. **Coordinate ascent can miss complementary changes.** A Retrieval change that
   helps only with a matching Decision policy, or evidence that helps only a new
   Numerical route, may be rejected when tested alone. The one-coordinate rule
   gives clean attribution and replay safety, but it is not evidence that the
   globally best package was found. A future Train-only 2×2 paired ablation
   should compare parent, Retrieval-only, Decision-only, and the paired
   Retrieval+Decision child before any Dev access.

3. **Repeated Dev selection can adapt to a small set.** Two cycles and three
   children cap the search, but Dev-20 is still consulted repeatedly. Reported
   Dev gains are development evidence, not a confidence interval. Nested
   resampling, a larger untouched validation set, or a fixed pre-registered
   finalist budget would make the selection claim stronger.

4. **The current evidence mechanism is intentionally conservative.** Numerical
   sees task-level feedback only from a verified accepted Retrieval lineage. If
   Retrieval is rejected, the next Numerical step receives an explicit empty
   treatment rather than untrusted traces. This avoids feedback leakage, but it
   also prevents potentially useful negative retrieval evidence from being
   learned. A safe extension would expose only host-derived failure categories
   for rejected candidates, never their free-form text or Dev outcomes.

## Design verdict before results

The present design is appropriate as a conservative, auditable research
baseline: it protects Toto, makes evidence falsifiable, scores end-to-end, and
keeps Public outside evolution. It should not be described as a fully learned
three-agent system. In particular, Retrieval remains proposal-driven rather
than trained to rank forecast-useful evidence, and coordinate-wise acceptance
can suppress interactions. The run can determine whether the existing mechanism
finds a safe gain; it cannot establish that retrieval in general is ineffective
if no child is accepted.

## Execution evidence

The successful real-model interaction smoke is
`runs/package_coevolution/smoke_toto_balanced_v3_luna_20260906_r5`.

- Completion status: `complete`; full chain exercised: `true`.
- Coordinate steps: six, ordered Numerical → Retrieval → Decision twice.
- Accepted steps: two Retrieval lineage publications. Both changed only the
  Retrieval principal fingerprint and passed cache-backed replay.
- Rejected steps: both Numerical proposals were structurally invalid on the
  reduced Build set; both Decision proposals ran but failed the reduced Build
  gate. These are valid experimental rejections, not unavailable phases.
- Cycle feedback: generation 3 contains 12 host-constructed task evidence
  cases. Because the published Retrieval identity differs from the evaluated
  prepublication identity, these cases conservatively say the anchor assumption
  could not be verified; they do not invent positive evidence.
- Public access: false in the completion marker and all six trace records.

The smoke required three implementation corrections, each covered by a failing
regression test before the production change: group-safe five-fold smoke
sampling, trying later Numerical recipes after one fit fails, and exposing plus
enforcing the closed Retrieval mutation schema with a validated repair fallback.

Formal evolution and Public-99 evidence will be added only after the formal
bundle is sealed and the separate Public evaluator finishes once.
