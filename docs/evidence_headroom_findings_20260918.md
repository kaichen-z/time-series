# Evidence-headroom findings — documents carry real signal that the pipeline discards

Date: 2026-09-18. Branch: `fix/p2-feasible-child`. Follow-up to
`docs/evolution_haiku_big_20260917_results.md` (champion ≈ toto_2_0 on test).

## Why this investigation

The evolved champion tied plain `toto_2_0` on the 99 test. We traced the cause to
the pipeline's structure: `evolving_loop/numerical_two_stage.py` guards the output
to `"materialized_ranked_alternatives_only"` (line ~148: the final forecast must
equal a materialized numerical candidate), so the decision stage can only **select**
a TSFM candidate, never **modify** it with document evidence. To know whether that
guard is *hiding value* (documents have signal) or *harmless* (documents have none),
we measured how much **actionable** evidence retrieval actually surfaces.

## Method

`scripts/evidence_headroom_probe.py`: run the champion two-stage pipeline over dev
tasks with a trace capturing each task's retrieval evidence card, project each
`EvidenceChain` to an `EvidenceEffect` (via `evolving_loop/adjustment`), and count
how many are *actionable* = grounded (cites a document) ∧ numeric_eligible ∧
entity_match ∧ target_match ∧ direction∈{up,down} ∧ has magnitude ∧ has time window.
Also score toto vs `reference_future_event_adjust(toto, evidence)` vs truth.

## Result (8 dev tasks, per-criterion pass counts over 15 evidence chains)

| criterion | pass |
|-----------|------|
| grounded (cites a document) | 15/15 |
| has time window | 13/15 |
| has magnitude | 9/15 |
| **numeric_eligible** | **0/15** |
| **entity_match** | **0/15** |
| **direction (up/down)** | **0/15** |
| → **actionable** | **0/15** |

`reference` adjuster deviated from toto on 0/8 tasks; mean joint identical
(toto = identity = reference = 0.68980). So with the current retrieval, **no
document evidence reaches the numbers** — which is exactly why champion ≈ toto.

## The decisive part — the discarded signal is real (task-level inspection)

Reading the actual documents (`.scratch/dev_cards.json` + raw task docs) shows the
`0/15 actionable` splits into **two different causes**:

- **Solar tasks (task_115, task_118 — "Direct Normal Irradiance"):** the 13
  supporting docs are *definitional / measurement-SOP* boilerplate ("what DNI is,
  how it is measured"); the structured-looking `timeseries/temporal/profile` docs
  are **distractors**. There is genuinely **no forecast-relevant signal** —
  retrieval's `numeric_eligible=False` was **correct**.
- **Road-occupancy task (task_152 — entity "Safety Barrier Configuration"):** the
  supporting docs contain a **real, quantifiable, windowed, directional event** —
  a **2024-07-04 Independence Day holiday → business closures → lower road
  occupancy** over the 07/04–07/06 window (docs 5720/5721/5723/5724, the last one
  naming the entity). Retrieval even produced a `direction='down'` chain for it —
  but marked it `entity_match=False`, so it was **discarded**. The entity is an odd
  proxy name ("Safety Barrier Configuration") that the docs do not mention verbatim,
  and the retrieval skill "reject evidence about neighboring entities" gates on a
  strict literal match → a genuine signal is killed.

## Conclusion

The materialized-only guard **is** hiding value, but only on part of the data:

1. **Event-driven tasks** (holidays, closures, maintenance, …) have real document
   signal that the pipeline currently throws away at the retrieval-judgment layer
   (over-strict `entity_match` / `numeric_eligible`), then again at the
   select-only decision. → **headroom exists.**
2. **Definitional-doc tasks** (e.g. solar irradiance) have no document signal;
   document-conditioning cannot help them. → **no headroom.**

So the bottleneck is **not** `post_adjust` (already built + safety-verified in
`evolving_loop/adjustment/`) and **not** universally-empty data — it is (a) retrieval
extraction being too conservative to surface/quantify events, and (b) the
select-only decision being unable to apply them.

## Implications for the evolution-space design

- Evolve the **retrieval extraction** to surface + quantify event effects, and
  relax the over-strict literal `entity_match` (the task_152 failure mode), not just
  evolve `post_adjust`.
- Reconsider the `actionable` gate: requiring `entity_match=True` is too strict for
  proxy-named entities — this cost us the task_152 holiday effect.
- **Evaluate in strata**: separate event-driven tasks (headroom) from
  definitional-doc tasks (none) so a real gain on the former is not averaged away.
- Success target stays: a `post_adjust` (+ improved evidence) champion that beats
  toto on held-out **event-driven** tasks, generalizing.
