# Champion–Challenger Numerical Evolution Design

Date: 2026-09-01
Status: proposed for implementation
Scope: Numerical Agent only; Retrieval and final Decision consume its frozen output later

## 1. Goal

Turn the existing method, dictionary, screening, Combined, morphology, and
Toto-safe experiments into one reusable Numerical self-evolution workflow.
The workflow must improve the current accepted forecasting policy without
making Toto the permanent center of the architecture.

The system starts with a strong Champion, generates structurally different
Challengers from the executable Dictionary, and promotes a Child only when
trusted Train and Dev evaluation proves it safer and better. The initial
Champion may be Toto, but any accepted single model, Combined policy, router,
or segmented forecast may become the next Champion.

## 2. Non-goals

- Do not train or merge TSFM model weights.
- Do not let an LLM choose numeric thresholds, weights, scores, or acceptance
  decisions.
- Do not require Toto to be a parent of every Combined method.
- Do not globally discard a method only because it is weak outside its stated
  applicability region.
- Do not use Dev, Public-99, hidden labels, documents, or Retrieval output to
  propose Numerical Children.
- Do not reinterpret the already-consumed Public-99 regression set as an
  unseen test set.

## 3. Existing assets

The implementation must reuse the current boundaries rather than create a
second forecasting stack:

- `methods.py`: executable Statistical methods evolved through typed Git
  operations.
- `policies.py`: reviewed TSFM bindings and executable Combined policies.
- `dictionary.py`: keep/specialized/repair/quarantine status and applicability.
- `skills.py`: reviewed history-only morphology primitives.
- `numerical_agent.evolution.screening`: task-conditioned active Dictionary.
- `numerical_agent.evolution.combined_evolution`: typed multi-parent Combined
  proposal boundary.
- `numerical_agent.evolution.numerical_loop`: history-only materialization,
  hindcasts, assumptions, protected selection, and forecast packaging.
- canonical Dr-CiK capped sMAE/sRMSE metrics and raw-tail reporting.

The exploratory adaptive Toto v1–v7 artifacts remain report-only evidence.
They cannot seed or resume a formal active run without passing the new schema
and source bindings.

## 4. Vocabulary and ownership

### 4.1 Executable Dictionary

The Dictionary is the complete candidate supply:

- Statistical leaves;
- reviewed TSFM leaves;
- Statistical–Statistical Combined policies;
- TSFM–Statistical Combined policies;
- TSFM–TSFM Combined policies.

Dictionary evolution creates, repairs, specializes, quarantines, and composes
tools. It does not select a final forecast by hindsight.

### 4.2 Champion

The Champion is the exact currently accepted Numerical policy. It includes its
candidate recipe, assumptions, host-fitted numeric parameters, source hashes,
metric-policy fingerprint, and lineage. A failed Child never mutates it.

### 4.3 Challenger and Child

A Challenger is a typed structural proposal. After trusted host expansion and
evaluation it becomes a Child. A Child may be:

1. a single candidate;
2. a fixed Combined forecast;
3. a task-conditioned route;
4. a horizon-segment route;
5. a bounded residual overlay;
6. a top-two diversity ensemble.

No candidate kind is required to contain Toto.

### 4.4 Safe fallback

The fallback is owned by the accepted Champion policy. Toto is the initial
strong fallback when configured, but a later accepted Champion may use another
reviewed stable forecast. A Challenger cannot silently replace the fallback.

### 4.5 Assumption

Every nontrivial Challenger carries one or more typed, history-only
assumptions. The LLM proposes structure; trusted code validates and fits it.

```python
@dataclass(frozen=True)
class EvolutionAssumption:
    assumption_id: str
    candidate_name: str
    feature: str
    direction: Literal["above", "below"]
    horizon_region: Literal["early", "late", "full"]
    operator: Literal[
        "select", "route", "horizon_route", "weighted", "median",
        "bounded_overlay"
    ]
    rationale: str
    failure_condition: str
```

Numeric thresholds, blend weights, overlay strengths, and correction caps are
absent from the LLM schema. The host derives them from Build data.

## 5. Candidate supply

The proposal inventory must expose all successful active leaves and all
reviewed Combined policies. It must also expose sanitized group evidence for:

- periodicity strength;
- trend strength;
- intermittency and zero fraction;
- recent regime confidence;
- outlier and noise scale;
- history length;
- horizon and horizon/history ratio;
- frequency.

The proposer may reference only supplied candidate names, features, operators,
and horizon regions. It cannot emit Python code in this phase. Method-code
evolution remains a separate upstream coordinate with its own identity checks.

Each generation should propose five through ten structurally diverse
Challengers. The host preserves at least three lineages when available:

- one independent Combined policy;
- one conditional route or segmented forecast;
- one bounded correction or diversity ensemble.

This archive prevents evolution from collapsing to one trivial safe method.
Only one Champion is released; the archive is proposal state, not runtime
authority.

## 6. Formal data discipline

The existing external split remains 80 Train / 20 Dev / 99 Public Regression.
The 80 Train tasks receive a deterministic, entity-disjoint internal split:

- Build-64: iterative proposals, numeric expansion, successive halving, and
  aggregate feedback;
- Calibration-16: one read-only comparison of the final Build shortlist;
- Dev-20: one read-only acceptance evaluation after structure freeze;
- Public-99: one frozen regression evaluation after release acceptance.

Build feedback may be reused across generations and must be labeled
`adaptive_train_build_diagnostic`. It is not an independent generalization
estimate.

Calibration results are never returned to the proposer. Once Calibration-16
is consumed for a formal run, additional structural iteration requires a new
run manifest and a newly pre-registered internal split. Dev results never
enter proposal, fitting, filtering, or assumption construction.

After a Child wins Calibration, its structure is frozen. Numeric parameters
may be refit on all 80 Train tasks before the one Dev evaluation; no candidate,
feature, operator, or horizon-region change is permitted during refit.

## 7. Evolution cycle

### 7.1 Build materialization

For every Build task:

1. compute the deterministic TaskProfile;
2. materialize each eligible leaf exactly once;
3. materialize each eligible Combined candidate only from successful cached
   leaves;
4. compute rolling history-only hindcasts;
5. store canonical sMAE/sRMSE diagnostics and raw tails;
6. aggregate anonymous evidence by morphology group.

The proposer receives no task IDs, future arrays, raw per-task forecasts, Dev
rows, Public rows, documents, Retrieval evidence, or hidden labels.

### 7.2 Structural proposal

The LLM receives:

- canonical Dictionary cards;
- the exact Parent/Champion recipe;
- sanitized group-level Build evidence;
- accepted and rejected prior structural attempts;
- bounded failure and regret summaries.

It returns typed assumptions and candidate recipes only. Strict parsing rejects
unknown fields, duplicate keys, nonfinite numbers, unknown candidates,
unsupported operators, and incompatible horizon regions.

### 7.3 Host expansion

Trusted code expands each structure across bounded grids derived from Build:

- feature thresholds from observed quantiles;
- normalized nonnegative blend weights;
- overlay alpha and correction caps;
- early/late split positions;
- local hindcast margins and tolerances.

Every expanded forecast must be reproducible from already materialized leaves.
No LLM call is allowed in expansion or scoring.

### 7.4 Successive halving

The default Build schedule is:

1. all Children on 8 fixed screen tasks;
2. survivors on 32 Build tasks;
3. at most three Children on all 64 Build tasks;
4. retain at most three structurally different finalists.

The screen order and task membership are fingerprinted before any proposal.
A stage can remove a clearly unsafe Child but cannot promote it into the active
Parent.

### 7.5 Parent preservation

The exact Parent remains active throughout Build. Rejected Child evidence may
inform a later proposal, but the later proposal always names the unchanged
active Parent. Only a Calibration-accepted Child becomes the Train winner.
Only a Dev-accepted Train winner becomes a release Champion.

## 8. Scoring and acceptance

### 8.1 Primary metrics

The authoritative pair is the Dr-CiK point-forecast metric policy:

- mean capped sMAE;
- mean capped sRMSE.

Joint scaled error is used for deterministic ranking after pairwise safety, not
as a substitute for either metric.

### 8.2 Safety metrics

Every comparison also reports and gates:

- coverage and failure rate;
- raw and capped P90/P95 sMAE;
- raw and capped P90/P95 sRMSE;
- clipped-task counts for both metrics;
- maximum per-task regret;
- fold stability;
- paired win/tie/loss counts.

Win count is diagnostic, not an unconditional rejection rule. A policy may
produce more small losses than wins while still improving both means and
remaining tail-safe.

### 8.3 Two-level target

Before a new formal run, its manifest fixes two thresholds:

- Candidate threshold: default 0.5% joint improvement, strict non-regression
  on both primary metrics, at least four of five Build folds improved, and all
  safety gates passing;
- Research target: 5% joint improvement with the same pair and safety gates.

The 5% value is a success target, not the sole permission to inspect Dev. A
safe 0.5–5% Child may enter Calibration and Dev when the manifest authorized
that rule before the run. Thresholds cannot be changed after results appear.

### 8.4 Calibration and Dev

Calibration compares the Build shortlist against the exact active Parent. A
Child must Pareto-improve capped mean sMAE/sRMSE and pass the full safety set.
If no Child passes, the exact Parent is retained and Dev stays closed.

Dev compares the refit frozen Child against the release Parent exactly once.
Dev failure retains the old release byte-for-byte. Dev acceptance publishes a
new immutable release with its complete evidence and lineage.

## 9. Runtime forecast path

Inference remains history-only:

```text
history/frequency/horizon
        ↓
deterministic TaskProfile
        ↓
task-conditioned active Dictionary
        ↓
materialize leaves once and build eligible Combined forecasts
        ↓
rolling hindcast diagnostics
        ↓
validate typed assumptions
        ↓
execute frozen Champion recipe
        ↓
NumericalForecastPackage
```

The runtime Champion may choose an independent Challenger directly, route
between methods, use different methods across the horizon, or apply a bounded
correction. If a condition, forecast, diagnostic, or assumption is invalid, the
exact configured fallback is returned.

Retrieval and final Decision operate after this package is produced. They may
rank or select only already materialized Numerical forecasts under their own
verified evidence boundaries. They cannot create new numeric values or feed
Public/hidden outcomes back into Numerical evolution.

## 10. Durable artifacts

The formal runner writes:

- `run_manifest.json`: source, split, task, metric, runtime, model, Dictionary,
  threshold, and schedule fingerprints;
- `checkpoint.json`: schema-versioned active Parent, completed stage, proposal
  archive, and sanitized feedback;
- `generation_NNN_proposals.json`: typed structural attempts;
- `generation_NNN_build_report.json`: adaptive Build diagnostics;
- `calibration_report.json`: one-shot shortlist comparison;
- `dev_report.json`: optional one-shot read-only acceptance;
- `champion_release.json`: immutable accepted recipe and lineage;
- `public_regression_report.json`: optional frozen Public-99 evaluation that is
  never accepted as evolution input.

The executable Dictionary remains Python in its Git repository. JSON records
bind lifecycle state and evidence; they do not replace executable methods.

Checkpoints fail closed on schema drift, source changes, split changes, task or
forecast changes, Dictionary changes, metric-policy changes, runtime identity
changes, or threshold changes.

## 11. CLI boundary

Add a production runner rather than retaining a `runs/` script:

```bash
python -m numerical_agent.run_champion_evolution \
  --repo runs/method_evolution/v001 \
  --split-file splits/drcik_public_80_20_99_v1.json \
  --tasks-file external/Dr-CiK/full-download/Dr-CiK_public/tasks \
  --generations 3 \
  --proposer-model gpt-5.6-luna \
  --proposer-reasoning-effort low \
  --candidate-minimum-gain 0.005 \
  --research-target-gain 0.05 \
  --output-dir runs/numerical_champion/v001
```

Public evaluation remains a separate frozen command. The evolution command has
no flag capable of loading Public task bodies or labels.

## 12. Failure behavior

- LLM schema failure: reject the generation; keep exact Parent.
- Missing or failed leaf: mark unavailable for that task; never fabricate a
  forecast.
- Invalid Combined recipe: reject before execution.
- Insufficient assumption support: return fallback for that task.
- Build gate failure: retain Parent; allow sanitized feedback only.
- Calibration failure: retain Parent; do not open Dev.
- Dev failure: retain prior release; do not evaluate Public.
- Resume mismatch: fail closed before model or task execution.
- Interrupted publication: preserve the last accepted immutable release.

## 13. Testing strategy

Implementation follows strict RED–GREEN TDD:

1. unit tests for typed assumptions, candidate kinds, strict schemas, and
   Parent identity;
2. adversarial tests for metric substitution, cached-pass forgery, split
   leakage, invalid forecasts, fallback identity, and checkpoint drift;
3. deterministic fake 8 Build / 2 Calibration smoke covering multiple
   generations and exact rejection semantics;
4. deterministic 64/16/20 lifecycle test with cached forecasts and zero real
   model downloads;
5. real small-task smoke only when reviewed checkpoint attestations exist;
6. full repository regression suite;
7. independent code review before integration.

## 14. Acceptance criteria

The implementation is complete only when:

- one command runs Dictionary-backed Champion–Challenger evolution;
- Toto is neither mandatory in a Child nor hard-coded as the permanent
  Champion;
- Statistical, TSFM, and Combined candidates are all eligible;
- assumptions are typed, history-only, and host-validated;
- failed Children preserve the exact Parent;
- Build, Calibration, Dev, and Public boundaries are mechanically enforced;
- sMAE and sRMSE remain paired authorities at every acceptance stage;
- raw tails, clipping, coverage, and regret are gated;
- the frozen runtime can reproduce the accepted Champion without an LLM;
- Public-99 output cannot be consumed by any evolution API.
