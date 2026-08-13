# PPT update: Coding Agent and co-evolution

## Slide 3 — Three research threads converge in one forecasting harness

**Numerical forecasting**  
Chronos / TimesFM / NEXUS / Last-Mile  
Strong numerical priors and contextual revision, but mostly fixed workflows.

**Evidence retrieval**  
Dr-CiK / ReflectiveRAG / S2G-RAG  
Retrieve sufficient, grounded evidence, but do not evolve executable time-series programs.

**Harness evolution**  
CORAL / Self-Harness / Harness-R1  
Improve agent policies or tools, but are not designed around contextual time-series forecasting.

**Our connection**  
Evolve the numerical hypothesis generator, evidence strategy, and decision policy using resolved forecasting outcomes.

---

## Slide 4 — The Coding Agent turns numbers into falsifiable programs

**Input — historical numbers only**

- Historical timestamps and values
- Sampling frequency and forecast horizon
- Numerical diagnostics derived from history

**Process**

1. Analyze statistical properties.
2. Generate multiple falsifiable assumptions.
3. Translate each assumption into executable Python code.
4. Run every program in a restricted sandbox.
5. Evaluate candidates with rolling historical hindcasting.
6. Revise one failed assumption/program once; retain it only if validation improves.

**Output contract**

```json
{
  "assumption": "The recent 22-step seasonal pattern will persist.",
  "failure_condition": "It fails if the pattern came from a temporary regime.",
  "code": "def forecast(history, horizon, seasonal_period): ...",
  "hindcast_score": 0.42
}
```

**Hard information boundary**

The Coding Agent cannot see documents, retrieved evidence, `gt_evidence`, future values, or Retrieval Agent output.

---

## Slide 5 — Assumptions drive retrieval; delayed outcomes drive co-evolution

### Inference: no labels or future values

```text
Historical numbers
    -> Coding Agent
    -> assumptions + executable candidate forecasts
                          |
Document corpus           v
    -> Retrieval Agent searches for evidence that distinguishes the assumptions
                          |
                          v
    -> verified evidence + candidate forecasts
    -> Decision Agent cross-checks, selects, or requests one revision
    -> probabilistic final forecast
```

### Training: labels are used only after the future resolves

```text
Resolved future + retrieval labels
    -> Coding coverage: did the candidate set contain a good future?
    -> Retrieval quality: did evidence retrieval find support and avoid distractors?
    -> Decision regret: was a good candidate available but not selected?
    -> Failure attribution selects Coding OR Retrieval OR Decision
    -> Evolver changes exactly one eligible prompt
    -> population evaluation on training tasks
    -> dev validation
    -> retain only a validated bundle for the next generation
```

**Two nested evolution timescales**

- Inner loop: task-level assumption/program revision using history-only hindcasting.
- Outer loop: cross-task evolution of the reusable Coding, Retrieval, and Decision policies.

The current implementation evolves prompts, not neural weights or graph topology.
