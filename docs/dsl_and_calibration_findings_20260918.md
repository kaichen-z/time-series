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

## 3. Implication for the evolution space

- Add a `history_calibrated` magnitude mode to the Level-0 DSL: the action's size
  comes from a **historical regime ratio** (weekend/weekday, low-quantile day,
  matching hour-of-day, …) rather than a constant. Which regime estimator to use is
  itself an evolvable, auditable rule kept only if it improves held-out accuracy.
- This doubles as a **data-driven `qualify` gate**: if no document-explained regime is
  observable in the history, the effect is not calibratable → drop it. That replaces
  retrieval's over-strict `numeric_eligible`/`entity_match` text judgment with an
  evidence-from-the-numbers test — using data, not literal string matching.
- Still bounded + grounded by the kernel; still stratified evaluation (event-driven
  vs no-signal) with nested holdout.
