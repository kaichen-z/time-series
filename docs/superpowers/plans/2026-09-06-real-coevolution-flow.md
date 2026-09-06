# Real Co-evolution Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace invented stepper payloads with source-backed contracts and provenance-labeled recorded artifacts across all nine co-evolution steps.

**Architecture:** Keep the existing self-contained HTML stepper, but replace its data model with two explicit truth modes: current source contract and recorded artifact. Each stage carries provenance, exact interface content, Host processing, evolution behavior, result, artifacts, and typed downstream handoff; missing historical exchanges render `NOT RECORDED`.

**Tech Stack:** Static HTML/CSS/JavaScript, Python `pytest`, repository Python sources and JSON artifacts.

**Spec:** `docs/superpowers/specs/2026-09-06-real-coevolution-flow-design.md`

## Global Constraints

- Do not modify production runners, model code, or historical artifacts.
- Do not present reconstructed or cross-run content as a recorded exchange.
- Remove `train_method_reports` and the fabricated four-method catalog.
- Keep all nine steps navigable and the repository/Desktop HTML copies byte-identical.
- Label each panel `CURRENT CONTRACT`, `RECORDED RUN`, or `NOT RECORDED` with a source path.

---

### Task 1: Truth-model regression tests

**Files:**
- Modify: `tests/test_coevolution_flow_html.py`
- Test: `tests/test_coevolution_flow_html.py`

**Interfaces:**
- Consumes: rendered HTML source and embedded `orchestrationSteps` data.
- Produces: assertions preventing invented fields and requiring truth/provenance/handoff disclosures.

- [x] **Step 1: Write failing assertions**

Add tests asserting that the HTML omits `train_method_reports` and the fabricated four-item payload, contains `CURRENT CONTRACT`, `RECORDED RUN`, and `NOT RECORDED`, names the 93-method bootstrap artifact, distinguishes old and current report schemas, and exposes exact system/input/raw-output/Host/evolve/result/artifact/handoff labels.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coevolution_flow_html.py -q`

Expected: FAIL on the existing invented payload and missing truth-model disclosures.

- [x] **Step 3: Do not change production HTML in this task**

The failing assertions are the deliverable for the red phase.

---

### Task 2: Replace the stepper with real contracts and artifacts

**Files:**
- Modify: `docs/three-agent-coevolution-flow.html`
- Test: `tests/test_coevolution_flow_html.py`

**Interfaces:**
- Consumes: exact prompt constants and request/response fields from `numerical_agent/evolution/`, `evolving_loop/retrieval_agent/`, `evolving_loop/decision_agent/`, `evolving_loop/co_evolution.py`, plus named artifacts under `/Users/yyoraa/time-series/runs/`.
- Produces: `truthModes: {contract, recorded}` per step and a renderer exposing `provenance`, `systemPrompt`, `input`, `rawOutput`, `hostProcessing`, `evolve`, `result`, `artifacts`, and `handoff`.

- [x] **Step 1: Implement the truth-aware data model**

Replace every illustrative payload with source-derived contract text or exact recorded values. Use `NOT RECORDED` where the historical request/response is absent. Include the real Method flow and 93-method bootstrap provenance; do not relabel the 2026-08-22 legacy metrics as current sMAE/sRMSE metrics.

- [x] **Step 2: Implement contract/recorded controls and disclosures**

Add a mode toggle and collapsible complete-content sections. Every downstream edge must name its Host projection or safety filter instead of claiming unconditional raw copying.

- [x] **Step 3: Run focused tests**

Run: `python -m pytest tests/test_coevolution_flow_html.py -q`

Expected: all tests pass.

---

### Task 3: Deliver and verify the standalone page

**Files:**
- Modify: `/Users/yyoraa/Desktop/Time-Series-CoEvolution-Flow.html`
- Verify: `docs/three-agent-coevolution-flow.html`
- Verify: `tests/test_coevolution_flow_html.py`

**Interfaces:**
- Consumes: verified repository HTML.
- Produces: byte-identical desktop delivery copy.

- [x] **Step 1: Synchronize with `apply_patch`**

Apply the repository-to-Desktop unified diff through `apply_patch`; do not use shell redirection or copying commands to write the file.

- [x] **Step 2: Verify tests, prohibited strings, and identity**

Run:

```bash
python -m pytest tests/test_coevolution_flow_html.py -q
rg -n 'train_method_reports|catalog_candidates.*toto.*seasonal_naive.*linear_trend.*croston' docs/three-agent-coevolution-flow.html
shasum -a 256 docs/three-agent-coevolution-flow.html /Users/yyoraa/Desktop/Time-Series-CoEvolution-Flow.html
git diff --check
```

Expected: tests pass, prohibited scan has no matches, hashes are identical, and diff check reports no whitespace errors.

- [x] **Step 3: Report verification limits**

If the in-app browser runtime remains unavailable, report that visual QA was not executed; do not claim it passed.
