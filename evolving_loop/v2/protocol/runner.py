"""Closed-boundary publication and replay for L1 infrastructure protocols."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from evolving_loop.v2.bundle import EvolutionBundleV2
from evolving_loop.v2.contracts import canonical_v2_bytes, fingerprint_payload, require_sha256
from evolving_loop.v2.store import _atomic_write, append_jsonl, write_once_json

from .compatibility import CompatibilityCorpusV2, CompatibilityEvidenceV2, CompatibilityHostInputsV2, check_compatibility, decide_protocol
from .contracts import InfrastructureProtocolV2, ProtocolComponentV2, ProtocolProposalV2, ProtocolReleaseV2
from .runtime import ProtocolRuntimeRegistry


_CONFIG_FIELDS = {"schema_version", "profile", "seed", "max_proposals", "hard_limit_seconds"}
_MANIFEST_REQUIRED = {"schema_version", "l0_commitment", "runtime_fingerprint", "corpus", "seed_protocol", "frozen_bundle_sha256", "replacement_templates"}


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
    if not _MANIFEST_REQUIRED <= set(manifest) or manifest.get("schema_version") != 1:
        raise ValueError("protocol input manifest is incomplete")
    require_sha256(manifest["l0_commitment"], "l0_commitment")
    require_sha256(manifest["runtime_fingerprint"], "runtime_fingerprint")


def _release_payload(release: ProtocolReleaseV2) -> dict[str, object]:
    return release.to_payload()


def freeze_protocol_handoff(release: ProtocolReleaseV2, bundle: EvolutionBundleV2, *, resolve: Callable[[str], dict]) -> dict[str, object]:
    """Resolve an accepted release and Bundle into a no-Public frozen handoff."""
    if not isinstance(release, ProtocolReleaseV2) or not isinstance(bundle, EvolutionBundleV2):
        raise ValueError("frozen handoff requires a release and frozen Bundle")
    resolved_release = ProtocolReleaseV2.from_payload(resolve(release.fingerprint()))
    if resolved_release != release:
        raise ValueError("release resolution does not match release SHA")
    evidence = CompatibilityEvidenceV2.from_payload(resolve(release.evidence_sha256))
    if evidence.fingerprint() != release.evidence_sha256 or not all(evidence.checks.values()):
        raise ValueError("release does not resolve accepted compatibility evidence")
    if bundle.acceptance_evidence_sha256 is None:
        raise ValueError("frozen Bundle must have acceptance evidence")
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


def run_protocol_evolution(output_dir: Path, config: dict, input_manifest: dict, registry: ProtocolRuntimeRegistry, *, host_inputs: CompatibilityHostInputsV2, stop_after: int | None = None) -> dict:
    """Evaluate precommitted replacements and resume only sealed closed steps."""
    _validate_config(config)
    _validate_manifest(input_manifest)
    if not isinstance(registry, ProtocolRuntimeRegistry) or not isinstance(host_inputs, CompatibilityHostInputsV2):
        raise TypeError("protocol evolution requires Host runtime authorities")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a non-negative closed-step count")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_sha, manifest_sha = fingerprint_payload(config), fingerprint_payload(input_manifest)
    corpus = CompatibilityCorpusV2.from_payload(input_manifest["corpus"])
    active = InfrastructureProtocolV2.from_payload(input_manifest["seed_protocol"])
    if active.l0_commitment_sha256 != input_manifest["l0_commitment"] or corpus.l0_commitment_sha256 != active.l0_commitment_sha256:
        raise ValueError("input manifest L0 commitments do not match")
    if input_manifest["runtime_fingerprint"] != host_inputs.runtime_fingerprint:
        raise ValueError("input manifest runtime fingerprint does not match Host")
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
    completion_path = root / "completion.json"
    if completion_path.exists():
        return _read(completion_path)
    checkpoint_path = root / "protocol_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = _read(checkpoint_path)
        if checkpoint.get("config_sha256") != config_sha or checkpoint.get("input_manifest_sha256") != manifest_sha:
            raise ValueError("checkpoint commitment mismatch")
        progress_path = root / "progress.jsonl"
        actual_prefix = fingerprint_payload({"rows": progress_path.read_text() if progress_path.exists() else ""})
        if checkpoint.get("progress_prefix_sha256") != actual_prefix:
            raise ValueError("checkpoint progress prefix does not match sealed writes")
        next_index = checkpoint["next_index"]
        active = InfrastructureProtocolV2.from_payload(_read(root / "protocols" / f"{checkpoint['active_protocol_sha256']}.json"))
        accepted, rejected = checkpoint["accepted"], checkpoint["rejected"]
        completed = list(checkpoint["completed_proposal_sha256s"])
        active_release_sha = checkpoint.get("active_release_sha256")
    else:
        next_index, accepted, rejected, completed, active_release_sha = 0, 0, 0, [], None
        write_once_json(root / "protocols" / f"{active.fingerprint()}.json", active.to_payload())
        _write_mutable(root / "active_protocol.json", active.to_payload())
    for index in range(next_index, len(templates)):
        if stop_after is not None and index >= stop_after:
            break
        replacement = templates[index]
        proposal = ProtocolProposalV2(1, active.fingerprint(), replacement.kind, replacement, corpus.fingerprint())
        child = proposal.to_child(active)
        proposal_sha, child_sha = proposal.fingerprint(), child.fingerprint()
        write_once_json(root / "proposals" / f"{proposal_sha}.json", proposal.to_payload())
        write_once_json(root / "protocols" / f"{child_sha}.json", child.to_payload())
        evidence = check_compatibility(active, child, corpus, registry, host_inputs=host_inputs, sealed_store=root)
        evidence_sha = evidence.fingerprint()
        evidence_path = write_once_json(root / "evidence" / f"{evidence_sha}.json", evidence.to_payload())
        if CompatibilityEvidenceV2.from_payload(_read(evidence_path)) != evidence:
            raise ValueError("stored compatibility evidence failed readback")
        decision = decide_protocol(evidence)
        write_once_json(root / "decisions" / f"{proposal_sha}.json", decision.to_payload())
        before_sha = active.fingerprint()
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
        completed.append(proposal_sha)
        next_index = index + 1
        checkpoint = {"schema_version": 1, "config_sha256": config_sha, "input_manifest_sha256": manifest_sha, "runtime_fingerprint": host_inputs.runtime_fingerprint, "l0_commitment_sha256": active.l0_commitment_sha256, "corpus_sha256": corpus.fingerprint(), "next_index": next_index, "active_protocol_sha256": active.fingerprint(), "active_release_sha256": active_release_sha, "accepted": accepted, "rejected": rejected, "completed_proposal_sha256s": completed, "progress_prefix_sha256": fingerprint_payload({"rows": (root / "progress.jsonl").read_text()}), "budget_counters": {"closed_proposals": next_index}}
        _write_mutable(checkpoint_path, checkpoint)
    if next_index != len(templates):
        return {"schema_version": 1, "status": "protocol_evolution_incomplete", "next_index": next_index, "accepted": accepted, "rejected": rejected, "public_test_accessed": False}
    if active_release_sha is None:
        raise ValueError("protocol evolution completed without an accepted release")
    release = ProtocolReleaseV2.from_payload(_read(root / "releases" / f"{active_release_sha}.json"))
    evidence_map = {path.stem: _read(path) for path in (root / "evidence").glob("*.json")}
    release_map = {release.fingerprint(): release.to_payload()}
    def resolve(identity: str) -> dict:
        if identity in release_map:
            return release_map[identity]
        if identity in evidence_map:
            return evidence_map[identity]
        raise ValueError("unresolved handoff reference")
    handoff = freeze_protocol_handoff(release, host_inputs.archive_bundles[seed_bundle_sha], resolve=resolve)
    handoff_sha = fingerprint_payload(handoff)
    write_once_json(root / "frozen_protocol_handoff.json", handoff)
    completion = {"schema_version": 1, "status": "protocol_evolution_complete", "active_protocol_sha256": active.fingerprint(), "active_release_sha256": active_release_sha, "accepted": accepted, "rejected": rejected, "completed_proposal_sha256s": completed, "frozen_handoff_sha256": handoff_sha, "public_test_accessed": False}
    write_once_json(completion_path, completion)
    return completion
