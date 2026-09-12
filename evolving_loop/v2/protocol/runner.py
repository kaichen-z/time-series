"""Closed-boundary publication and replay for L1 infrastructure protocols."""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from evolving_loop.v2.budget import BudgetLedger, BudgetPlan, ResourceUse
from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from evolving_loop.v2.store import _atomic_write, append_jsonl, write_once_json

from .compatibility import CompatibilityCorpusV2, CompatibilityEvidenceV2, CompatibilityHostInputsV2, check_compatibility, decide_protocol
from .contracts import InfrastructureProtocolV2, ProtocolComponentV2, ProtocolProposalV2, ProtocolReleaseV2
from .runtime import ProtocolRuntimeRegistry


_CONFIG_FIELDS = {"schema_version", "profile", "seed", "max_proposals", "hard_limit_seconds"}
_MANIFEST_FIELDS = {"schema_version", "l0_commitment", "runtime_fingerprint", "corpus", "seed_protocol", "host_input_files", "frozen_bundle_sha256", "replacement_templates"}


def _read(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_v2_bytes(value):
        raise ValueError(f"expected canonical object: {path}")
    return value


def _write_mutable(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(path, canonical_v2_bytes(payload))


def _validate_config(config: Mapping[str, object]) -> None:
    if set(config) != _CONFIG_FIELDS or config.get("schema_version") != 1 or type(config.get("seed")) is not int:
        raise ValueError("protocol config must use the exact schema")
    if type(config.get("max_proposals")) is not int or config["max_proposals"] <= 0 or type(config.get("hard_limit_seconds")) is not int or config["hard_limit_seconds"] <= 0:
        raise ValueError("protocol config limits must be positive integers")


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if set(manifest) != _MANIFEST_FIELDS or manifest.get("schema_version") != 1:
        raise ValueError("protocol input manifest must use the exact schema")
    require_sha256(manifest["l0_commitment"], "l0_commitment")
    require_sha256(manifest["runtime_fingerprint"], "runtime_fingerprint")


def _release_payload(release: ProtocolReleaseV2) -> dict[str, object]:
    return release.to_payload()


def _budget_plan(config: Mapping[str, object]) -> BudgetPlan:
    return BudgetPlan(
        int(config["hard_limit_seconds"]),
        0.0,
        ResourceUse(
            wall_seconds=float(config["hard_limit_seconds"]),
        ),
    )


def _progress_fingerprint(root: Path) -> str:
    progress = root / "progress.jsonl"
    return fingerprint_payload({"rows": progress.read_text() if progress.exists() else ""})


def _validate_closed_prefix(root: Path, checkpoint: Mapping[str, object], seed_sha: str) -> None:
    completed = checkpoint.get("completed")
    if not isinstance(completed, list) or checkpoint.get("next_index") != len(completed):
        raise ValueError("checkpoint completed prefix is malformed")
    expected = {"protocols": {seed_sha}, "proposals": set(), "evidence": set(), "decisions": set(), "releases": set(), "sealed": set(), "migrations": set()}
    previous = seed_sha
    for index, item in enumerate(completed):
        if not isinstance(item, Mapping) or item.get("index") != index or item.get("before_protocol_sha256") != previous:
            raise ValueError("checkpoint progress continuity is invalid")
        for field, directory in (("proposal_sha256", "proposals"), ("child_protocol_sha256", "protocols"), ("evidence_sha256", "evidence"), ("decision_sha256", "decisions")):
            identity = require_sha256(item.get(field), field)
            expected[directory].add(identity)
        for field, directory in (("release_sha256", "releases"), ("sealed_dev_sha256", "sealed")):
            identity = item.get(field)
            if identity is not None:
                expected[directory].add(require_sha256(identity, field))
        evidence = CompatibilityEvidenceV2.from_payload(_read(root / "evidence" / f"{item['evidence_sha256']}.json"))
        for mapping in evidence.migration_mapping:
            expected["migrations"].add(require_sha256(mapping["new_envelope_sha256"], "new_envelope_sha256"))
        previous = require_sha256(item.get("after_protocol_sha256"), "after_protocol_sha256")
    if checkpoint.get("active_protocol_sha256") != previous:
        raise ValueError("checkpoint active protocol does not match progress")
    for directory, identities in expected.items():
        path = root / directory
        actual = {item.stem for item in path.glob("*.json")} if path.exists() else set()
        if actual != identities:
            raise ValueError("incomplete writes exist outside the sealed checkpoint prefix")
    for identity in expected["protocols"]:
        if InfrastructureProtocolV2.from_payload(_read(root / "protocols" / f"{identity}.json")).fingerprint() != identity:
            raise ValueError("stored protocol identity does not match checkpoint")
    for identity in expected["proposals"]:
        if ProtocolProposalV2.from_payload(_read(root / "proposals" / f"{identity}.json")).fingerprint() != identity:
            raise ValueError("stored proposal identity does not match checkpoint")
    for identity in expected["evidence"]:
        if CompatibilityEvidenceV2.from_payload(_read(root / "evidence" / f"{identity}.json")).fingerprint() != identity:
            raise ValueError("stored evidence identity does not match checkpoint")
    for identity in expected["decisions"]:
        if fingerprint_payload(_read(root / "decisions" / f"{identity}.json")) != identity:
            raise ValueError("stored decision identity does not match checkpoint")
    for identity in expected["releases"]:
        if ProtocolReleaseV2.from_payload(_read(root / "releases" / f"{identity}.json")).fingerprint() != identity:
            raise ValueError("stored release identity does not match checkpoint")
    for identity in expected["sealed"]:
        if fingerprint_payload(_read(root / "sealed" / f"{identity}.json")) != identity:
            raise ValueError("sealed Dev identity does not match checkpoint")
    for identity in expected["migrations"]:
        if fingerprint_payload(_read(root / "migrations" / f"{identity}.json")) != identity:
            raise ValueError("migrated envelope identity does not match checkpoint")
    progress = root / "progress.jsonl"
    lines = progress.read_bytes().splitlines() if progress.exists() else []
    if len(lines) != len(completed):
        raise ValueError("progress rows do not match sealed checkpoint prefix")
    for index, line in enumerate(lines):
        row = json.loads(line)
        if canonical_v2_bytes(row).rstrip(b"\n") != line or row.get("index") != index or row.get("proposal_sha256") != completed[index]["proposal_sha256"]:
            raise ValueError("progress row is not canonical or continuous")


def _handoff_resolver(root: Path, release: ProtocolReleaseV2, bundle: EvolutionBundleV2, host_inputs: CompatibilityHostInputsV2) -> Callable[[str], dict]:
    """Return the exact closed authority set needed by the frozen handoff."""
    evidence_map = {path.stem: _read(path) for path in (root / "evidence").glob("*.json")}
    protocol_map = {path.stem: _read(path) for path in (root / "protocols").glob("*.json")}
    bundle_map = {bundle.fingerprint(): bundle.to_payload(), release.fingerprint(): release.to_payload()}
    if bundle.acceptance_evidence_sha256 is None or host_inputs.bundle_acceptance_evidence is None:
        raise ValueError("Host input is missing frozen Bundle acceptance evidence")
    try:
        bundle_evidence = host_inputs.bundle_acceptance_evidence[bundle.acceptance_evidence_sha256]
    except KeyError as error:
        raise ValueError("Host input does not close frozen Bundle acceptance evidence") from error
    if fingerprint_payload(bundle_evidence) != bundle.acceptance_evidence_sha256:
        raise ValueError("Host Bundle acceptance evidence does not match committed SHA")
    bundle_map[bundle.acceptance_evidence_sha256] = dict(bundle_evidence)

    def resolve(identity: str) -> dict:
        if identity in bundle_map:
            return bundle_map[identity]
        if identity in evidence_map:
            return evidence_map[identity]
        if identity in protocol_map:
            return protocol_map[identity]
        raise ValueError("unresolved handoff reference")
    return resolve


def _validate_completed_run(root: Path, checkpoint: Mapping[str, object], *, host_inputs: CompatibilityHostInputsV2, seed_bundle_sha: str) -> dict[str, object]:
    """Read-only completion must still prove the final publication linkage."""
    completion = _read(root / "completion.json")
    active_sha = require_sha256(checkpoint.get("active_protocol_sha256"), "active protocol")
    active = InfrastructureProtocolV2.from_payload(_read(root / "protocols" / f"{active_sha}.json"))
    if active.fingerprint() != active_sha or (root / "active_protocol.json").read_bytes() != canonical_v2_bytes(active.to_payload()):
        raise ValueError("active protocol pointer does not match completed checkpoint")
    release_sha = require_sha256(checkpoint.get("active_release_sha256"), "active release")
    release = ProtocolReleaseV2.from_payload(_read(root / "releases" / f"{release_sha}.json"))
    evidence = CompatibilityEvidenceV2.from_payload(_read(root / "evidence" / f"{release.evidence_sha256}.json"))
    if release.fingerprint() != release_sha or release.protocol_sha256 != active_sha or evidence.fingerprint() != release.evidence_sha256 or evidence.proposed_protocol_sha256 != active_sha:
        raise ValueError("completed release/evidence/protocol linkage is invalid")
    if release.l0_commitment_sha256 != active.l0_commitment_sha256 or release.runtime_fingerprint != host_inputs.runtime_fingerprint or release.frozen_bundle_sha256 != seed_bundle_sha:
        raise ValueError("completed release runtime/L0/Bundle binding is invalid")
    bundle = host_inputs.archive_bundles.get(seed_bundle_sha)
    if bundle is None:
        raise ValueError("completed frozen Bundle is unavailable from Host")
    expected_handoff = freeze_protocol_handoff(release, bundle, resolve=_handoff_resolver(root, release, bundle, host_inputs))
    handoff = _read(root / "frozen_protocol_handoff.json")
    handoff_sha = fingerprint_payload(handoff)
    completed = checkpoint["completed"]
    expected = {
        "schema_version": 1, "status": "protocol_evolution_complete", "active_protocol_sha256": active_sha,
        "active_release_sha256": release_sha, "accepted": checkpoint["accepted"], "rejected": checkpoint["rejected"],
        "completed_proposal_sha256s": [item["proposal_sha256"] for item in completed], "frozen_handoff_sha256": handoff_sha,
        "public_test_accessed": False,
    }
    if completion != expected or handoff != expected_handoff:
        raise ValueError("completed handoff or semantic completion is invalid")
    return completion


def freeze_protocol_handoff(release: ProtocolReleaseV2, bundle: EvolutionBundleV2, *, resolve: Callable[[str], dict]) -> dict[str, object]:
    """Resolve an accepted release and Bundle into a no-Public frozen handoff."""
    if not isinstance(release, ProtocolReleaseV2) or not isinstance(bundle, EvolutionBundleV2):
        raise ValueError("frozen handoff requires a release and frozen Bundle")
    if bundle.fingerprint() != release.frozen_bundle_sha256:
        raise ValueError("frozen Bundle does not match release binding")
    resolved_release = ProtocolReleaseV2.from_payload(resolve(release.fingerprint()))
    if resolved_release != release:
        raise ValueError("release resolution does not match release SHA")
    evidence = CompatibilityEvidenceV2.from_payload(resolve(release.evidence_sha256))
    if evidence.fingerprint() != release.evidence_sha256 or not all(evidence.checks.values()):
        raise ValueError("release does not resolve accepted compatibility evidence")
    protocol = InfrastructureProtocolV2.from_payload(resolve(release.protocol_sha256))
    if protocol.fingerprint() != release.protocol_sha256:
        raise ValueError("release protocol resolution does not match protocol SHA")
    resolved_bundle = EvolutionBundleV2.from_payload(resolve(bundle.fingerprint()))
    if resolved_bundle != bundle:
        raise ValueError("frozen Bundle resolution does not match Bundle SHA")
    if bundle.acceptance_evidence_sha256 is None:
        raise ValueError("frozen Bundle must have acceptance evidence")
    bundle_evidence = resolve(bundle.acceptance_evidence_sha256)
    if fingerprint_payload(bundle_evidence) != bundle.acceptance_evidence_sha256:
        raise ValueError("Bundle acceptance evidence resolution does not match SHA")
    if bundle.protocol_fingerprint != release.l0_commitment_sha256:
        raise ValueError("Bundle and release L0 commitments do not match")
    if release.protocol_sha256 != evidence.proposed_protocol_sha256:
        raise ValueError("release protocol does not match evidence")
    return {
        "schema_version": 1,
        "bundle_sha256": bundle.fingerprint(),
        "l0_commitment_sha256": release.l0_commitment_sha256,
        "protocol_sha256": release.protocol_sha256,
        "release_sha256": release.fingerprint(),
        "acceptance_evidence_sha256": release.evidence_sha256,
        "runtime_fingerprint": release.runtime_fingerprint,
        "public_test_accessed": False,
    }


def run_protocol_evolution(output_dir: Path, config: dict, input_manifest: dict, registry: ProtocolRuntimeRegistry, *, host_inputs: CompatibilityHostInputsV2, stop_after: int | None = None, monotonic: Callable[[], float] | None = None) -> dict:
    """Evaluate precommitted replacements and resume only sealed closed steps."""
    _validate_config(config)
    _validate_manifest(input_manifest)
    if not isinstance(registry, ProtocolRuntimeRegistry) or not isinstance(host_inputs, CompatibilityHostInputsV2):
        raise TypeError("protocol evolution requires Host runtime authorities")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a non-negative closed-step count")
    clock = time.monotonic if monotonic is None else monotonic
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_sha, manifest_sha = fingerprint_payload(config), fingerprint_payload(input_manifest)
    corpus = CompatibilityCorpusV2.from_payload(input_manifest["corpus"])
    active = InfrastructureProtocolV2.from_payload(input_manifest["seed_protocol"])
    if active.l0_commitment_sha256 != input_manifest["l0_commitment"] or corpus.l0_commitment_sha256 != active.l0_commitment_sha256:
        raise ValueError("input manifest L0 commitments do not match")
    if input_manifest["runtime_fingerprint"] != host_inputs.runtime_fingerprint:
        raise ValueError("input manifest runtime commitment does not match Host")
    templates = tuple(ProtocolComponentV2.from_payload(item) for item in input_manifest["replacement_templates"])
    if len(templates) != config["max_proposals"]:
        raise ValueError("replacement templates must match max_proposals")
    seed_bundle_sha = require_sha256(input_manifest["frozen_bundle_sha256"], "frozen_bundle_sha256")
    if seed_bundle_sha not in host_inputs.archive_bundles:
        raise ValueError("frozen incumbent Bundle is missing")
    run_manifest = {"schema_version": 1, "system": "evolution_v2_protocol", "config_sha256": config_sha, "input_manifest_sha256": manifest_sha, "runtime_fingerprint": host_inputs.runtime_fingerprint, "l0_commitment_sha256": active.l0_commitment_sha256, "corpus_sha256": corpus.fingerprint()}
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        if _read(manifest_path) != run_manifest:
            raise ValueError("resume input/config/runtime commitments do not match")
    else:
        write_once_json(manifest_path, run_manifest)
    checkpoint_path = root / "protocol_checkpoint.json"
    plan = _budget_plan(config)
    completion_path = root / "completion.json"
    if completion_path.exists():
        checkpoint = _read(root / "protocol_checkpoint.json")
        if checkpoint.get("config_sha256") != config_sha or checkpoint.get("input_manifest_sha256") != manifest_sha or checkpoint.get("progress_prefix_sha256") != _progress_fingerprint(root) or checkpoint.get("budget_plan_sha256") != plan.fingerprint():
            raise ValueError("completed run commitments do not match")
        _validate_closed_prefix(root, checkpoint, active.fingerprint())
        return _validate_completed_run(root, checkpoint, host_inputs=host_inputs, seed_bundle_sha=seed_bundle_sha)
    seed_sha = active.fingerprint()
    if checkpoint_path.exists():
        checkpoint = _read(checkpoint_path)
        if checkpoint.get("config_sha256") != config_sha or checkpoint.get("input_manifest_sha256") != manifest_sha:
            raise ValueError("checkpoint commitment mismatch")
        if checkpoint.get("runtime_fingerprint") != host_inputs.runtime_fingerprint or checkpoint.get("l0_commitment_sha256") != corpus.l0_commitment_sha256 or checkpoint.get("corpus_sha256") != corpus.fingerprint() or checkpoint.get("budget_plan_sha256") != plan.fingerprint():
            raise ValueError("checkpoint runtime/L0/corpus/budget commitments do not match")
        if checkpoint.get("progress_prefix_sha256") != _progress_fingerprint(root):
            raise ValueError("checkpoint progress prefix does not match sealed writes")
        _validate_closed_prefix(root, checkpoint, seed_sha)
        next_index = checkpoint["next_index"]
        active = InfrastructureProtocolV2.from_payload(_read(root / "protocols" / f"{checkpoint['active_protocol_sha256']}.json"))
        if (root / "active_protocol.json").read_bytes() != canonical_v2_bytes(active.to_payload()):
            raise ValueError("active protocol pointer does not match sealed checkpoint")
        accepted, rejected = checkpoint["accepted"], checkpoint["rejected"]
        completed = list(checkpoint["completed"])
        active_release_sha = checkpoint.get("active_release_sha256")
        ledger = BudgetLedger.resume(plan, checkpoint["budget_ledger"], monotonic=clock)
    else:
        next_index, accepted, rejected, completed, active_release_sha = 0, 0, 0, [], None
        ledger = BudgetLedger(plan, monotonic=clock)
        write_once_json(root / "protocols" / f"{active.fingerprint()}.json", active.to_payload())
        _write_mutable(root / "active_protocol.json", active.to_payload())
    for index in range(next_index, len(templates)):
        if stop_after is not None and index >= stop_after:
            break
        replacement = templates[index]
        proposal = ProtocolProposalV2(1, active.fingerprint(), replacement.kind, replacement, corpus.fingerprint())
        child = proposal.to_child(active)
        proposal_sha, child_sha = proposal.fingerprint(), child.fingerprint()
        estimate = ResourceUse(wall_seconds=float(config["hard_limit_seconds"]) / int(config["max_proposals"]))
        permit = ledger.reserve_stage(proposal_sha, estimate)
        if not permit.allowed:
            raise ValueError(f"protocol budget denied proposal: {permit.reason}")
        started = clock()
        write_once_json(root / "proposals" / f"{proposal_sha}.json", proposal.to_payload())
        write_once_json(root / "protocols" / f"{child_sha}.json", child.to_payload())
        evidence = check_compatibility(active, child, corpus, registry, host_inputs=host_inputs, sealed_store=root)
        evidence_sha = evidence.fingerprint()
        evidence_path = write_once_json(root / "evidence" / f"{evidence_sha}.json", evidence.to_payload())
        if CompatibilityEvidenceV2.from_payload(_read(evidence_path)) != evidence:
            raise ValueError("stored compatibility evidence failed readback")
        decision = decide_protocol(evidence)
        decision_sha = decision.fingerprint()
        write_once_json(root / "decisions" / f"{decision_sha}.json", decision.to_payload())
        before_sha = active.fingerprint()
        elapsed = clock() - started
        if elapsed < 0.0:
            raise ValueError("monotonic clock moved backwards during protocol proposal")
        closed = ledger.close_stage(permit, ResourceUse(wall_seconds=elapsed))
        if not closed.allowed:
            raise ValueError(f"protocol budget close failed: {closed.reason}")
        if decision.decision == "accept":
            release = ProtocolReleaseV2(1, child_sha, child.l0_commitment_sha256, evidence_sha, seed_bundle_sha, host_inputs.runtime_fingerprint)
            release_sha = release.fingerprint()
            write_once_json(root / "releases" / f"{release_sha}.json", _release_payload(release))
            if ProtocolReleaseV2.from_payload(_read(root / "releases" / f"{release_sha}.json")) != release:
                raise ValueError("stored protocol release failed readback")
            _write_mutable(root / "active_protocol.json", child.to_payload())
            active, active_release_sha, accepted = child, release_sha, accepted + 1
        else:
            rejected += 1
        row = {"schema_version": 1, "index": index, "proposal_sha256": proposal_sha, "before_protocol_sha256": before_sha, "after_protocol_sha256": active.fingerprint(), "decision": decision.decision, "reason_codes": list(decision.reason_codes)}
        append_jsonl(root / "progress.jsonl", row)
        completed.append({"index": index, "proposal_sha256": proposal_sha, "child_protocol_sha256": child_sha, "evidence_sha256": evidence_sha, "decision_sha256": decision_sha, "release_sha256": active_release_sha if decision.decision == "accept" else None, "sealed_dev_sha256": evidence.sealed_dev_sha256, "before_protocol_sha256": before_sha, "after_protocol_sha256": active.fingerprint()})
        next_index = index + 1
        checkpoint = {"schema_version": 1, "config_sha256": config_sha, "input_manifest_sha256": manifest_sha, "runtime_fingerprint": host_inputs.runtime_fingerprint, "l0_commitment_sha256": active.l0_commitment_sha256, "corpus_sha256": corpus.fingerprint(), "next_index": next_index, "active_protocol_sha256": active.fingerprint(), "active_release_sha256": active_release_sha, "accepted": accepted, "rejected": rejected, "completed": completed, "progress_prefix_sha256": _progress_fingerprint(root), "budget_plan_sha256": plan.fingerprint(), "budget_ledger": ledger.checkpoint()}
        _write_mutable(checkpoint_path, checkpoint)
    if next_index != len(templates):
        return {"schema_version": 1, "status": "protocol_evolution_incomplete", "next_index": next_index, "accepted": accepted, "rejected": rejected, "public_test_accessed": False}
    if active_release_sha is None:
        raise ValueError("protocol evolution completed without an accepted release")
    release = ProtocolReleaseV2.from_payload(_read(root / "releases" / f"{active_release_sha}.json"))
    bundle = host_inputs.archive_bundles[seed_bundle_sha]
    handoff = freeze_protocol_handoff(release, bundle, resolve=_handoff_resolver(root, release, bundle, host_inputs))
    handoff_sha = fingerprint_payload(handoff)
    write_once_json(root / "frozen_protocol_handoff.json", handoff)
    completion = {"schema_version": 1, "status": "protocol_evolution_complete", "active_protocol_sha256": active.fingerprint(), "active_release_sha256": active_release_sha, "accepted": accepted, "rejected": rejected, "completed_proposal_sha256s": [item["proposal_sha256"] for item in completed], "frozen_handoff_sha256": handoff_sha, "public_test_accessed": False}
    write_once_json(completion_path, completion)
    return completion
