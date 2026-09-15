# Decision-owned Dictionary execution

New real P3 runs keep the full P2 Dictionary fixed. Retrieval gathers verified
context; Decision sees the complete method catalog and visible history, asks the
numerical tool to evaluate methods or convex ensembles, reads their historical
hindcast scores, and selects an executed forecast. If a second Retrieval round
provides evidence, Decision can evaluate a different set or combination again.

Decision prompt variants participate in P3 cooperative evolution. Those prompts
control both numerical evaluation requests and final selection. There is no
6–8-method shortlist, mandatory Anchor weight, or 0–2-specialist limit in this
path. Each planning call can request up to four evaluations to bound work; each
evaluation may contain any number of distinct available Dictionary methods.
Weights must be finite, nonnegative, and sum to one. Invalid requests use the
existing baseline fallback when no requested evaluation succeeds.

The current numerical tool uses P2's frozen per-task execution cache. It reuses
individual forecasts and history-only fold predictions, computes the requested
weighted forecast and fold predictions, and scores the combination afresh. It
does not execute new Python source or fit backbones during a Decision call.
Missing or incompatible historical folds are rejected. Future evaluation labels
are not accepted by the tool. This is a cache-backed implementation of the
Decision execution interface, not a fresh-runtime forecasting backend.

The real bridge still uses the existing `p3_dictionary` mode. New releases and
packages identify Decision ownership with `decision_dictionary`; old
`p3_selector` closures remain readable. P4/P5 consume the same sealed bundle
interface. The older standalone Selector functions remain available for
comparison, but the real bridge no longer generates Selector mutations.

Offline checks (including a scripted two-stage end-to-end smoke):

```bash
python -m pytest -q tests/test_decision_dictionary_tools.py tests/test_evolution_v2_p3_dictionary.py tests/test_evolution_v2_real_cooperative.py tests/test_numerical_retrieval_handoff.py
```

Scripted smoke results demonstrate the execution path, not forecasting gains or
a successful live LLM experiment. Real P3 currently evaluates the existing 4/1
projection of the 80/20 split; this change does not expand that experiment.
