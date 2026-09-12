import json

import pytest

from common.payload import strict_json_loads
from evolving_loop.v2.contracts import canonical_v2_bytes
from evolving_loop.v2.real.contracts import (
    PROFILE_SCHEDULES,
    RealEvolutionManifestV2,
    RealModelBindingV2,
)


def manifest_payload():
    return {
        "schema_version": 1, "profile": "real-30m",
        "model": {"schema_version": 1, "name": "gpt-5.6-luna", "reasoning_effort": "medium"},
        "files": [
            {"role": "split", "relative_path": "splits/a.json", "sha256": "a" * 64},
            {"role": "tasks", "relative_path": "external/tasks", "sha256": "b" * 64},
        ],
        "runtime_locations": [{"role": "python", "relative_path": "runs/runtime", "identity_sha256": "c" * 64}],
        "l0_fingerprints": {"metric": "d" * 64},
    }


def test_approved_profiles_have_exact_total_and_reserve():
    assert PROFILE_SCHEDULES["real-30m"].allocations == {
        "p2": 840, "p3": 360, "p4": 120, "p5": 120, "finalization": 360,
    }
    assert PROFILE_SCHEDULES["real-1h"].total_seconds == 3600
    assert sum(PROFILE_SCHEDULES["real-1h"].allocations.values()) == 3600


def test_manifest_rejects_model_or_path_drift():
    payload = manifest_payload()
    payload["model"]["reasoning_effort"] = "high"
    with pytest.raises(ValueError, match="gpt-5.6-luna/medium"):
        RealEvolutionManifestV2.from_payload(payload)
    payload["model"]["reasoning_effort"] = "medium"
    payload["files"][0]["relative_path"] = "../secret"
    with pytest.raises(ValueError, match="relative"):
        RealEvolutionManifestV2.from_payload(payload)


def test_manifest_round_trip_is_canonical_and_strict():
    manifest = RealEvolutionManifestV2.from_payload(manifest_payload())
    assert RealEvolutionManifestV2.from_payload(manifest.to_payload()) == manifest
    assert manifest.fingerprint() == RealEvolutionManifestV2.from_payload(
        json.loads(manifest.canonical_bytes())
    ).fingerprint()
    with pytest.raises(ValueError):
        RealEvolutionManifestV2.from_payload(manifest.to_payload() | {"extra": 1})


def test_model_binding_is_fixed():
    binding = RealModelBindingV2.from_payload(
        {"schema_version": 1, "name": "gpt-5.6-luna", "reasoning_effort": "medium"}
    )
    assert binding.canonical_bytes() == canonical_v2_bytes(binding.to_payload())
    with pytest.raises(ValueError, match="gpt-5.6-luna/medium"):
        RealModelBindingV2.from_payload(
            {"schema_version": 1, "name": "other", "reasoning_effort": "medium"}
        )


def test_profiles_are_canonical_json():
    for name, schedule in PROFILE_SCHEDULES.items():
        path = __import__("pathlib").Path("configs/evolution_v2/real") / f"{name}.json"
        payload = strict_json_loads(path.read_text(), context=str(path))
        assert canonical_v2_bytes(payload) == schedule.canonical_bytes()
