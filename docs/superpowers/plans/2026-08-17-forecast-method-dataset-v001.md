# Forecast Method Dataset v001 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a reproducible, provenance-grounded dataset containing every distinct verified statistical, TSFM, and combined forecasting method found before systematic source collection reaches the defined coverage and saturation criteria.

**Architecture:** Human- or research-agent-produced source and method manifests are treated as untrusted collection inputs. A deterministic Python pipeline validates provenance, normalizes stable identities, reports possible duplicates, audits taxonomy coverage and saturation, and builds one canonical JSON release. The release is independent of Dr-CiK labels and documents and becomes the immutable seed for later implementation and selector evolution.

**Tech Stack:** Python 3.10+, standard library, dataclasses, JSON/JSONL, pytest, existing `numerical_agent` CLI patterns.

**Spec:** `docs/superpowers/specs/2026-08-17-evolving-method-dictionary-selector-design.md`

## Global Constraints

- The collection cutoff for v001 is `2026-08-17`.
- Method count has no target and no upper limit. Passing 100, 200, or 300 records never stops collection; only coverage and saturation do.
- Families are exactly `statistical`, `foundation`, and `combined`.
- Every verified method must have an immutable `method_uid`, at least one authoritative definition source, explicit assumptions, explicit failure conditions, applicability metadata, and implementation-availability metadata.
- Every scientific definition must be paraphrased; do not copy long copyrighted passages.
- Unofficial websites may identify candidates but cannot independently verify a method.
- Surveys and benchmarks may audit coverage but cannot replace an authoritative definition source.
- Phase 0 collection must never consume Dr-CiK `future_values`, `gt_evidence`, document roles, document subtypes, or task-specific documents.
- A released dataset is immutable. Corrections create a new dataset version.
- The generated canonical JSON must be deterministic: stable sorting, stable serialization, and a SHA-256 content hash.
- No method implementation or forecast performance is required for v001; this plan builds the method-knowledge dataset consumed by later curation.

## File Structure

```text
numerical_agent/
  collection/
    __init__.py             Public collection interfaces
    contracts.py            SourceRecord, MethodCard, DatasetRelease schemas
    registry.py             JSONL loading and deterministic release writing
    normalization.py        Alias normalization and duplicate-candidate reporting
    verification.py         Provenance and method verification gates
    coverage.py             Taxonomy coverage and saturation calculations
  datasets/
    source_registry_v001.jsonl
    method_candidates_v001.jsonl
    collection_journal_v001.json
    forecast_method_dataset_v001.json
    forecast_method_dataset_v001.sha256
    collection_queries_v001.json
    collection_audit_v001.json
numerical_agent/main.py      Add collect-methods, verify-methods, build-dataset commands
numerical_agent/README.md    Document dataset build and validation workflow
scripts/build_method_dataset.sh
tests/fixtures/method_collection/
  valid_sources.jsonl
  valid_methods.jsonl
  duplicate_methods.jsonl
tests/test_collection_contracts.py
tests/test_collection_registry.py
tests/test_collection_normalization.py
tests/test_collection_verification.py
tests/test_collection_coverage.py
tests/test_method_dataset_cli.py
```

---

### Task 1: Define source, method-card, and release contracts

**Files:**
- Create: `numerical_agent/collection/__init__.py`
- Create: `numerical_agent/collection/contracts.py`
- Test: `tests/test_collection_contracts.py`

**Interfaces:**
- Produces: `SourceRecord.from_payload()`, `MethodCard.from_payload()`, `DatasetRelease.from_payload()`, and deterministic `to_payload()` methods.
- Consumes: existing family names from `numerical_agent.config.ALLOWED_FAMILIES`.

- [ ] **Step 1: Write failing source-record tests**

```python
def test_source_record_requires_authoritative_locator():
    payload = {
        "source_id": "source_000001",
        "title": "Forecasting source",
        "authors": ["A. Author"],
        "year": 2024,
        "source_type": "paper",
        "url": "https://example.org/paper",
        "retrieved_at": "2026-08-17",
        "primary": True,
        "review_status": "verified",
    }
    source = SourceRecord.from_payload(payload)
    assert source.source_id == "source_000001"


def test_unverified_source_cannot_claim_verified_review():
    payload = valid_source_payload()
    payload["url"] = ""
    with pytest.raises(ValueError, match="authoritative locator"):
        SourceRecord.from_payload(payload)
```

- [ ] **Step 2: Run the source-record tests and verify they fail**

Run: `pytest tests/test_collection_contracts.py -v`

Expected: FAIL because `numerical_agent.collection.contracts` does not exist.

- [ ] **Step 3: Implement `SourceRecord`**

Use a frozen dataclass with these exact fields:

```python
@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    source_type: Literal[
        "paper", "textbook", "official_docs", "model_card",
        "official_repo", "survey", "benchmark"
    ]
    url: str
    doi: str = ""
    isbn: str = ""
    retrieved_at: str = "2026-08-17"
    primary: bool = False
    review_status: Literal["candidate", "verified", "rejected"] = "candidate"
```

Require a non-empty HTTPS URL or a DOI/ISBN. Parse dates with `datetime.date.fromisoformat` and
reject years outside `1600..2026`.

- [ ] **Step 4: Write failing method-card and release tests**

```python
def test_verified_method_requires_definition_source_and_complete_behavior():
    card = MethodCard.from_payload(valid_method_payload())
    assert card.method_uid == "method_000001"
    assert card.verification_status == "verified"


def test_release_rejects_duplicate_method_uids():
    payload = valid_release_payload()
    payload["methods"].append(dict(payload["methods"][0]))
    with pytest.raises(ValueError, match="duplicate method_uid"):
        DatasetRelease.from_payload(payload)
```

- [ ] **Step 5: Implement `MethodCard` and `DatasetRelease`**

`MethodCard` uses the exact fields from the design plus:

```python
method_uid: str
definition_version: int
canonical_name: str
aliases: tuple[str, ...]
family: Literal["statistical", "foundation", "combined"]
category: str
description: str
assumptions: tuple[str, ...]
failure_conditions: tuple[str, ...]
applicability: Mapping[str, object]
hyperparameters: tuple[str, ...]
definition_source_ids: tuple[str, ...]
implementation_source_ids: tuple[str, ...]
implementation_availability: Literal["available", "partial", "unavailable", "unknown"]
verification_status: Literal["unverified", "verified", "rejected"]
lineage: Mapping[str, object]
```

Foundation methods additionally require a `foundation_metadata` object containing
`checkpoint_or_api`, `release_version`, `context_length`, `prediction_length`, `inference_mode`,
`probabilistic_output`, `covariate_support`, `device_requirements`, `license`, `weights_available`,
and `code_available`. Statistical and combined methods store an empty object.

`DatasetRelease` contains `schema_version`, `dataset_id`, `release_date`, `collection_cutoff`,
`sources`, `methods`, `taxonomy`, `collection_batches`, and `content_hash`.

- [ ] **Step 6: Run contract tests**

Run: `pytest tests/test_collection_contracts.py -v`

Expected: PASS.

- [ ] **Step 7: Commit contracts**

```bash
git add numerical_agent/collection tests/test_collection_contracts.py
git commit -m "feat(dataset): add method collection contracts"
```

---

### Task 2: Load untrusted collection manifests and write deterministic releases

**Files:**
- Create: `numerical_agent/collection/registry.py`
- Create: `tests/fixtures/method_collection/valid_sources.jsonl`
- Create: `tests/fixtures/method_collection/valid_methods.jsonl`
- Test: `tests/test_collection_registry.py`

**Interfaces:**
- Consumes: `SourceRecord`, `MethodCard`, and `DatasetRelease` from Task 1.
- Produces: `load_source_records(path)`, `load_method_cards(path)`, `build_release(...)`, and `write_release(...)`.

- [ ] **Step 1: Write failing JSONL loader tests**

```python
def test_registry_loads_jsonl_with_line_numbered_errors(tmp_path):
    path = tmp_path / "sources.jsonl"
    path.write_text('{"source_id":"broken"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"sources.jsonl:1"):
        load_source_records(path)


def test_release_writer_is_byte_deterministic(tmp_path):
    release = build_release(load_sources(), load_methods(), release_metadata())
    first = write_release(release, tmp_path / "one.json")
    second = write_release(release, tmp_path / "two.json")
    assert first.read_bytes() == second.read_bytes()
```

- [ ] **Step 2: Run registry tests and verify they fail**

Run: `pytest tests/test_collection_registry.py -v`

Expected: FAIL because registry functions do not exist.

- [ ] **Step 3: Implement JSONL loading**

Read one JSON object per non-empty line. Reject arrays, duplicate source IDs, duplicate method UIDs,
and malformed UTF-8. Include the path and one-based line number in every error.

- [ ] **Step 4: Implement deterministic release construction**

Sort sources by `source_id`, methods by `method_uid`, dictionary keys with `sort_keys=True`, and
serialize with `ensure_ascii=False`, `indent=2`, and a final newline. Compute `content_hash` from
the same payload with `content_hash` set to an empty string, then emit `sha256:<hex>`.
The SHA-256 sidecar format is `<64 lowercase hex characters>  <release basename>\n` so it remains
portable across directories.

- [ ] **Step 5: Run registry tests**

Run: `pytest tests/test_collection_registry.py -v`

Expected: PASS.

- [ ] **Step 6: Commit registry support**

```bash
git add numerical_agent/collection/registry.py tests/fixtures/method_collection tests/test_collection_registry.py
git commit -m "feat(dataset): add deterministic registry build"
```

---

### Task 3: Normalize identities and report possible duplicates

**Files:**
- Create: `numerical_agent/collection/normalization.py`
- Create: `tests/fixtures/method_collection/duplicate_methods.jsonl`
- Test: `tests/test_collection_normalization.py`

**Interfaces:**
- Consumes: `Sequence[MethodCard]`.
- Produces: `normalize_name(name) -> str`, `DuplicateCandidate`, and `find_duplicate_candidates(methods) -> tuple[DuplicateCandidate, ...]`.

- [ ] **Step 1: Write failing normalization tests**

```python
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Auto-ARIMA", "auto arima"),
        ("Damped Holt's trend", "damped holts trend"),
    ],
)
def test_normalized_aliases_match(left, right):
    assert normalize_name(left) == normalize_name(right)


def test_wrapper_and_underlying_model_are_not_auto_merged():
    methods = load_duplicate_fixture("auto_arima", "arima")
    candidates = find_duplicate_candidates(methods)
    assert candidates[0].requires_manual_review is True
    assert methods[0].method_uid != methods[1].method_uid
```

- [ ] **Step 2: Run normalization tests and verify they fail**

Run: `pytest tests/test_collection_normalization.py -v`

Expected: FAIL because normalization functions do not exist.

- [ ] **Step 3: Implement conservative normalization**

Apply Unicode NFKC normalization, lowercase, remove punctuation, collapse whitespace, and normalize
possessive apostrophes. Report collisions across canonical names and aliases. Never merge records
automatically. Include reasons `canonical_collision`, `alias_collision`, or `shared_source_claim`.

- [ ] **Step 4: Run normalization tests**

Run: `pytest tests/test_collection_normalization.py -v`

Expected: PASS.

- [ ] **Step 5: Commit duplicate reporting**

```bash
git add numerical_agent/collection/normalization.py tests/fixtures/method_collection/duplicate_methods.jsonl tests/test_collection_normalization.py
git commit -m "feat(dataset): report duplicate method identities"
```

---

### Task 4: Enforce provenance and verification gates

**Files:**
- Create: `numerical_agent/collection/verification.py`
- Test: `tests/test_collection_verification.py`

**Interfaces:**
- Consumes: source and method sequences.
- Produces: `VerificationIssue`, `VerificationReport`, and `verify_registry(sources, methods)`.

- [ ] **Step 1: Write failing verification tests**

```python
def test_verified_method_needs_authoritative_definition_source():
    sources = [source(source_type="survey", primary=False)]
    methods = [method(definition_source_ids=(sources[0].source_id,))]
    report = verify_registry(sources, methods)
    assert report.is_publishable is False
    assert "missing_authoritative_definition" in report.issue_codes


def test_foundation_method_requires_release_and_license_metadata():
    sources = valid_sources()
    methods = [foundation_method(foundation_metadata={})]
    report = verify_registry(sources, methods)
    assert "incomplete_foundation_metadata" in report.issue_codes
```

- [ ] **Step 2: Run verification tests and verify they fail**

Run: `pytest tests/test_collection_verification.py -v`

Expected: FAIL because verification functions do not exist.

- [ ] **Step 3: Implement source-reference validation**

Reject unknown source IDs. Definition sources must include at least one verified `paper`,
`textbook`, `official_docs`, or `model_card` source marked primary. Official repositories can prove
implementation availability but cannot be the only scientific definition source unless the method
is explicitly a software/API forecasting service.

- [ ] **Step 4: Implement method-completeness validation**

Require non-empty descriptions, assumptions, failure conditions, and applicability. Foundation
metadata uses the Task 1 field list. Combined methods require at least two parent UIDs in lineage.
Reject self-parenting, unknown parents, cyclic lineage, and unsupported family values.

- [ ] **Step 5: Run verification tests**

Run: `pytest tests/test_collection_verification.py -v`

Expected: PASS.

- [ ] **Step 6: Commit provenance gates**

```bash
git add numerical_agent/collection/verification.py tests/test_collection_verification.py
git commit -m "feat(dataset): enforce method provenance gates"
```

---

### Task 5: Audit taxonomy coverage and collection saturation

**Files:**
- Create: `numerical_agent/collection/coverage.py`
- Create: `numerical_agent/datasets/collection_queries_v001.json`
- Test: `tests/test_collection_coverage.py`

**Interfaces:**
- Produces: `CoverageReport`, `SaturationReport`, `audit_coverage(methods, query_manifest)`, and `audit_saturation(batch_counts)`.

- [ ] **Step 1: Write failing coverage tests**

```python
def test_coverage_reports_empty_taxonomy_cells():
    report = audit_coverage([statistical_method()], query_manifest())
    assert "foundation.zero_shot" in report.empty_cells


def test_saturation_requires_three_low_yield_batches():
    assert audit_saturation([120, 4, 2, 1], base_count=120).saturated is False
    assert audit_saturation([120, 2, 1, 1], base_count=120).saturated is True
```

- [ ] **Step 2: Run coverage tests and verify they fail**

Run: `pytest tests/test_collection_coverage.py -v`

Expected: FAIL because coverage functions do not exist.

- [ ] **Step 3: Add the fixed v001 query matrix**

The manifest contains source tiers and these exact taxonomy groups:

```json
{
  "classical_level_trend": ["naive", "moving_average", "exponential_smoothing", "damped_trend"],
  "autoregressive_state_space": ["ar", "ma", "arima", "structural", "state_space"],
  "seasonal_spectral_decomposition": ["seasonal", "fourier", "spectral", "decomposition"],
  "intermittent_count": ["intermittent_demand", "count_forecasting"],
  "robust_regime": ["robust", "outlier", "change_point", "regime_switching", "analogue"],
  "machine_learning": ["regression", "tree", "kernel", "neural"],
  "probabilistic_hierarchical": ["probabilistic", "calibration", "reconciliation"],
  "foundation": ["zero_shot", "fine_tuned", "probabilistic_tsfm", "covariate_tsfm"],
  "combined": ["ensemble", "selector", "residual_correction", "fallback"]
}
```

For every term, store these five exact query templates after substituting `{term}`:
`time series forecasting {term} original paper`, `time series forecasting {term} textbook`,
`time series forecasting {term} official documentation`, `time series forecasting {term} model
card`, and `time series forecasting {term} official repository`.

- [ ] **Step 4: Implement coverage and saturation reports**

Coverage reports counts by family, category, source tier, verification state, implementation
availability, license presence, and empty taxonomy cells. Saturation follows the design rule:
three consecutive batches each add fewer than two percent new canonical methods relative to the
pre-batch verified count.

- [ ] **Step 5: Run coverage tests**

Run: `pytest tests/test_collection_coverage.py -v`

Expected: PASS.

- [ ] **Step 6: Commit coverage auditing**

```bash
git add numerical_agent/collection/coverage.py numerical_agent/datasets/collection_queries_v001.json tests/test_collection_coverage.py
git commit -m "feat(dataset): audit taxonomy and saturation"
```

---

### Task 6: Add reproducible dataset CLI commands

**Files:**
- Modify: `numerical_agent/main.py`
- Create: `tests/test_method_dataset_cli.py`

**Interfaces:**
- Produces CLI commands `collect-methods`, `verify-methods`, and `build-dataset`.
- Consumes all collection modules from Tasks 1-5.

- [ ] **Step 1: Write failing parser tests**

```python
def test_build_dataset_cli_requires_source_and_method_manifests():
    parser = build_parser()
    args = parser.parse_args([
        "build-dataset",
        "--sources", "sources.jsonl",
        "--methods", "methods.jsonl",
        "--queries", "queries.json",
        "--output", "release.json",
        "--audit-output", "audit.json",
    ])
    assert args.command == "build-dataset"
```

- [ ] **Step 2: Run CLI tests and verify they fail**

Run: `pytest tests/test_method_dataset_cli.py -v`

Expected: FAIL because the subcommands do not exist.

- [ ] **Step 3: Implement `collect-methods`**

This command imports and normalizes externally collected JSONL records, writes a raw registry, and
writes a duplicate-candidate report. It does not browse the internet itself and does not mark
records verified.

- [ ] **Step 4: Implement `verify-methods`**

This command produces a machine-readable verification report. Return exit code 2 when the registry
is not publishable; do not write a verified release.

- [ ] **Step 5: Implement `build-dataset`**

Run loading, duplicate reporting, verification, coverage, saturation, deterministic release
writing, and SHA-256 sidecar generation. Reject releases with unresolved duplicate candidates,
empty required taxonomy groups, unsatisfied saturation, or failed provenance checks. Never reject
or truncate a release because its verified method count is above a round-number milestone.
`build-dataset` requires `--collection-journal`; this immutable input contains collection-batch
counts and manual duplicate resolutions. `--audit-output` is generated and is never reused as an
input.

- [ ] **Step 6: Run CLI tests**

Run: `pytest tests/test_method_dataset_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Commit CLI commands**

```bash
git add numerical_agent/main.py tests/test_method_dataset_cli.py
git commit -m "feat(dataset): add reproducible build commands"
```

---

### Task 7: Migrate the existing 41-method seed without inventing provenance

**Files:**
- Create: `numerical_agent/datasets/source_registry_v001.jsonl`
- Create: `numerical_agent/datasets/method_candidates_v001.jsonl`
- Test: `tests/test_seed_method_migration.py`

**Interfaces:**
- Consumes: `numerical_agent/dictionaries/statistical_base_methods_v000.json`.
- Produces: initial unverified v001 source/method manifests with stable UIDs.

- [ ] **Step 1: Write a failing seed-migration test**

```python
def test_every_legacy_method_has_one_stable_candidate_uid():
    legacy = load_legacy_methods()
    candidates = load_method_cards(CANDIDATE_PATH)
    assert len(legacy) == 41
    assert len({card.method_uid for card in candidates}) == 41
    assert {card.canonical_name for card in candidates} == {
        method["method_id"] for method in legacy
    }
```

- [ ] **Step 2: Run the seed-migration test and verify it fails**

Run: `pytest tests/test_seed_method_migration.py -v`

Expected: FAIL because the v001 manifests do not exist.

- [ ] **Step 3: Migrate seed definitions**

Assign deterministic UIDs `method_seed_0001` through `method_seed_0041`, preserve the old method ID
as an alias, copy descriptions, assumptions, and failure conditions, and set
`verification_status = "unverified"`. Do not add a source or claim verification until Task 8 finds
and reviews an authoritative source.

- [ ] **Step 4: Run the seed-migration test**

Run: `pytest tests/test_seed_method_migration.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the provenance-safe seed migration**

```bash
git add numerical_agent/datasets/source_registry_v001.jsonl numerical_agent/datasets/method_candidates_v001.jsonl tests/test_seed_method_migration.py
git commit -m "data(dataset): migrate statistical seed cards"
```

---

### Task 8: Systematically collect and verify the real v001 dataset

**Files:**
- Modify: `numerical_agent/datasets/source_registry_v001.jsonl`
- Modify: `numerical_agent/datasets/method_candidates_v001.jsonl`
- Create: `numerical_agent/datasets/collection_journal_v001.json`
- Create: `numerical_agent/datasets/collection_audit_v001.json`
- Create: `numerical_agent/datasets/forecast_method_dataset_v001.json`
- Create: `numerical_agent/datasets/forecast_method_dataset_v001.sha256`

**Interfaces:**
- Consumes: the query matrix and all collection validators.
- Produces: the first verified dataset release and collection audit.

- [ ] **Step 1: Run the initial audit and freeze the gap report**

Run:

```bash
python -m numerical_agent verify-methods \
  --sources numerical_agent/datasets/source_registry_v001.jsonl \
  --methods numerical_agent/datasets/method_candidates_v001.jsonl \
  --queries numerical_agent/datasets/collection_queries_v001.json \
  --output numerical_agent/datasets/collection_audit_v001.json
```

Expected: exit 2 with all 41 seed methods reported as unverified and multiple empty taxonomy cells.

- [ ] **Step 2: Collect textbook and classical-statistical sources**

Search the `classical_level_trend`, `autoregressive_state_space`,
`seasonal_spectral_decomposition`, and `intermittent_count` query groups. Prefer textbook publisher
pages, DOI landing pages, original papers, and official statistical-software documentation. Add
source metadata and paraphrased method cards; connect aliases rather than duplicating concepts.
Run verification after each batch.

- [ ] **Step 3: Collect robust, regime, machine-learning, and probabilistic methods**

Search `robust_regime`, `machine_learning`, and `probabilistic_hierarchical`. Use primary papers and
official implementations. Record applicability, failure conditions, output type, and license or
implementation availability. Run verification after each batch.

- [ ] **Step 4: Collect TSFM methods and executable variants**

Search the `foundation` group using current primary papers, official model cards, and official
repositories. Record scientific method concepts separately from checkpoint variants. Populate all
foundation metadata fields and record whether weights and code are public. Run verification after
each batch.

- [ ] **Step 5: Collect combined, selector, residual-correction, and fallback methods**

Search the `combined` group. Every combined method must reference at least two known parent UIDs or
an authoritative source defining the composition. Run verification after each batch.

- [ ] **Step 6: Resolve duplicate candidates manually and record decisions**

For every duplicate-candidate pair, set one of `same_concept`, `distinct_wrapper`,
`distinct_checkpoint_variant`, or `not_duplicate` in `collection_journal_v001.json`. Verified output
must contain no unresolved pair.

- [ ] **Step 7: Run three saturation batches**

Repeat the lowest-coverage query cells with different authoritative source tiers. Record each
batch's reviewed-source count, candidate count, new canonical-method count, duplicate count, and
rejected count in `collection_journal_v001.json`. Stop only after three consecutive batches each
add fewer than two percent new canonical methods and no required taxonomy group is empty.

- [ ] **Step 8: Build the release**

Run:

```bash
python -m numerical_agent build-dataset \
  --sources numerical_agent/datasets/source_registry_v001.jsonl \
  --methods numerical_agent/datasets/method_candidates_v001.jsonl \
  --queries numerical_agent/datasets/collection_queries_v001.json \
  --collection-journal numerical_agent/datasets/collection_journal_v001.json \
  --output numerical_agent/datasets/forecast_method_dataset_v001.json \
  --audit-output numerical_agent/datasets/collection_audit_v001.json \
  --sha256-output numerical_agent/datasets/forecast_method_dataset_v001.sha256
```

Expected: exit 0, every verified non-duplicate method retained, all required taxonomy groups
covered, saturation passed, no unresolved duplicates, and a matching SHA-256 sidecar.

- [ ] **Step 9: Rebuild and prove determinism**

Run the same command with output paths under a task-specific temporary directory and compare:

```bash
dataset_verify_dir=$(mktemp -d)
python -m numerical_agent build-dataset \
  --sources numerical_agent/datasets/source_registry_v001.jsonl \
  --methods numerical_agent/datasets/method_candidates_v001.jsonl \
  --queries numerical_agent/datasets/collection_queries_v001.json \
  --collection-journal numerical_agent/datasets/collection_journal_v001.json \
  --output "$dataset_verify_dir/forecast_method_dataset_v001.json" \
  --audit-output "$dataset_verify_dir/collection_audit_v001.json" \
  --sha256-output "$dataset_verify_dir/forecast_method_dataset_v001.sha256"
cmp numerical_agent/datasets/forecast_method_dataset_v001.json "$dataset_verify_dir/forecast_method_dataset_v001.json"
cmp numerical_agent/datasets/forecast_method_dataset_v001.sha256 "$dataset_verify_dir/forecast_method_dataset_v001.sha256"
```

Expected: both comparisons exit 0.

- [ ] **Step 10: Commit the verified dataset**

```bash
git add numerical_agent/datasets
git commit -m "data(dataset): publish forecast methods v001"
```

---

### Task 9: Document and automate the dataset release workflow

**Files:**
- Modify: `numerical_agent/README.md`
- Create: `scripts/build_method_dataset.sh`

**Interfaces:**
- Produces: one-command local verification and release rebuild.

- [ ] **Step 1: Add the build script**

The script uses `set -euo pipefail`, resolves the repository root relative to the script, invokes
`python -m numerical_agent build-dataset` with the v001 paths from Task 8, runs the collection test
files, and prints the release hash and method count.

- [ ] **Step 2: Document dataset semantics**

Explain raw candidates versus verified release, source requirements, family definitions,
deduplication decisions, saturation, immutable versioning, how to add a source/method, how to build
the release, and why implementation performance is not part of v001.

- [ ] **Step 3: Run the script**

Run: `bash scripts/build_method_dataset.sh`

Expected: exit 0 and print `forecast_method_dataset_v001` with its uncapped verified method count.

- [ ] **Step 4: Commit documentation and automation**

```bash
git add numerical_agent/README.md scripts/build_method_dataset.sh
git commit -m "docs(dataset): document release workflow"
```

---

### Task 10: Final dataset verification

**Files:**
- Verify only; no source changes unless a check exposes a defect.

**Interfaces:**
- Consumes the complete branch.
- Produces verification evidence for review.

- [ ] **Step 1: Run focused collection tests**

Run:

```bash
pytest -q \
  tests/test_collection_contracts.py \
  tests/test_collection_registry.py \
  tests/test_collection_normalization.py \
  tests/test_collection_verification.py \
  tests/test_collection_coverage.py \
  tests/test_method_dataset_cli.py \
  tests/test_seed_method_migration.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete regression suite**

Run: `pytest -q`

Expected: all tests pass, including the original 293 tests.

- [ ] **Step 3: Verify repository and dataset integrity**

Run:

```bash
git diff --check origin/main...HEAD
(cd numerical_agent/datasets && shasum -a 256 -c forecast_method_dataset_v001.sha256)
python -m numerical_agent verify-methods \
  --sources numerical_agent/datasets/source_registry_v001.jsonl \
  --methods numerical_agent/datasets/method_candidates_v001.jsonl \
  --queries numerical_agent/datasets/collection_queries_v001.json \
  --output /tmp/forecast_method_collection_audit.json
```

Expected: no whitespace errors, checksum passes, verification exits 0, all saturated-search
methods retained, no unresolved duplicate candidates, and no empty required taxonomy groups.

- [ ] **Step 4: Record final commit state**

Run: `git status --short --branch && git log --oneline origin/main..HEAD`

Expected: a clean feature branch containing the dataset implementation and release commits.
