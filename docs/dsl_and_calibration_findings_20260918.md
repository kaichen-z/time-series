# Level-0 interaction DSL + data↔text calibration — validated on real dev data

Date: 2026-09-18. Branch: `fix/p2-feasible-child`. Follow-up to
`docs/evidence_headroom_findings_20260918.md`.

## 1. The Level-0 interaction DSL (rules-as-data), and a real bug it exposed

`evolving_loop/adjustment/dsl.py` encodes the text→forecast interaction policy as an
evolvable, auditable rule table (`Predicate` + `Action` = `Rule`; ordered `Policy`).
`apply_policy` interprets it with the invariant **kernel** enforced independently of
any rule: only *grounded* effects fire, and the output always passes
`apply_bounded_delta` (no rule can move the forecast > `max_frac·|base|`).
`policy_to_text` prints a policy as English.

**Bug the cache exposed:** `project_evidence` passed retrieval's direction vocabulary
(`up/down/stable`) through verbatim, but `.actionable` and the DSL gate on
`increase/decrease` — so *no real evidence ever fired* (0/15 in the headroom probe).
Fixed by canonicalizing direction at the projection boundary
(`up→increase`, `down→decrease`).

### Offline eval on dev (`scripts/eval_dsl_offline.py`, no LLM calls)

joint = (sMAE+sRMSE)/2, lower is better; base = cached toto_2_0.

| task | toto | event_scale | relaxed_entity | grounded_event |
|------|------|-------------|----------------|----------------|
| task_152 | 0.46357 | 0.46357 | 0.46357 | **0.32638** |
| all 7 others | = toto | = toto | = toto | = toto |
| MEAN(all) | 0.68980 | 0.68980 | 0.68980 | **0.67265** |

Only `GROUNDED_EVENT` (which relaxes retrieval's over-strict `entity_match` /
`target_match` / `numeric_eligible` judgment flags, keeping grounded + directional +
windowed + quantified) recovers task_152: **−30% joint on the one task with real
discarded signal, zero regression elsewhere** (the kernel guarantees identity when no
rule fires). The strict and entity-only-relaxed policies cannot recover it because
retrieval marked `target_match=False` too.

## 2. Data↔text calibration — the history *explains* the series (validated)

Idea (user): the regularities already visible in the given time series are partly
*caused* by things retrievable from the documents — so the history itself can
calibrate a document event's magnitude, instead of a hand-picked cap.

**Constraint found:** Dr-CiK task histories are short (48–168 steps; task_152 has only
6 days, 2024-06-27..07-03), so there is no *prior occurrence of the same annual event*
to back-test. The workable form is to use a **same-class, document-explained
regularity** observable within the short history.

**task_152 (`scripts/calibrate_probe_task152.py`):** the docs say
"July-4 holiday → business closures → lower road occupancy". In the 6-day history:

- weekday mean occupancy = **3.61**, weekend mean = **2.26** → weekends are **37.4%
  lower** (business activity drops — the same mechanism the doc names).
- Future truth: 7-4 (a Thursday, the holiday) = **2.075**, i.e. a weekday behaving
  like a weekend (≈ −43% vs weekday level); 7-5/7-6 return to normal — matching the
  document's window (the holiday day only).

So the weekend/weekday level ratio (0.626) **calibrates** the holiday magnitude:

| version | joint |
|---------|-------|
| toto base | 0.46357 |
| fixed −30% cap | 0.32638 |
| **history-calibrated (−37.4%)** | **0.30510** |

Calibration beats the fixed cap and lands near the truth (window mean: toto 3.66 →
calibrated 2.29 → truth 2.08). This is genuine **bidirectional** coupling: the
document says *which* future steps should behave like a low-activity regime; the
history says *how much* that regime lowers the series.

## 3. Implemented: `history_calibrated` mode with switchable regime candidates

Added to `evolving_loop/adjustment/dsl.py`. A rule's `Action` can set
`magnitude="history_calibrated"` with a `regime` name; the move size then comes from a
**document-explained historical regime**, not a constant. Estimators are a switchable
registry (`available_regimes()`): `weekend_weekday`, `low_quantile_day`, `hour_of_day`.
Each returns a per-step factor or **None**, where None is the **data-driven qualify
gate** — no observable regime ⇒ drop the effect. There is also a `data↔text
cross-check`: if the regime's direction contradicts the document's direction, drop it.
Everything still passes the kernel (grounded + bounded).

### Offline dev eval of the switchable candidates (`scripts/eval_dsl_offline.py`)

joint = (sMAE+sRMSE)/2; base = cached toto_2_0; only task_152 carries actionable signal.

| policy | task_152 | MEAN(all 8) | note |
|--------|----------|-------------|------|
| toto | 0.46357 | 0.68980 | base |
| grounded_event (fixed −30%) | 0.32638 | 0.67265 | constant cap |
| **cal[weekend_weekday]** | **0.30510** | **0.66999** | regime fits a holiday |
| cal[low_quantile_day] | 0.46357 | 0.68980 | not fired (small-sample quantile ⇒ dir conflict) |
| cal[hour_of_day] | 0.46357 | 0.68980 | not fired (intra-day regime N/A for an all-day drop) |
| **cal[cascade]** | **0.30510** | **0.66999** | weekend→lowq→fixed fallback picks the best |

Takeaways: (1) the weekend regime and the cascade beat both the fixed cap and toto;
(2) **zero regression** — candidates that cannot observe a consistent regime simply do
not fire (the qualify gate), and all 7 no-signal tasks stay identical to toto; (3) this
is the intended shape — several auditable candidates, held-out picks the winner.

Which regime to use (and the cascade order) is now an evolvable, auditable choice.
Next: fold these into the co-evolution loop with stratified (event vs no-signal),
downside-protected fitness and nested holdout; add more regime estimators as needed.
