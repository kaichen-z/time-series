from __future__ import annotations

import subprocess

import pytest

from evolving_loop.v2.source import (
    SourceRequestV2,
    SourceVariantV2,
    audit_source,
    run_policy,
)


def _variant(text: str) -> SourceVariantV2:
    return SourceVariantV2.seed(text, "a" * 64, "b" * 64)


def _request() -> SourceRequestV2:
    return SourceRequestV2(
        1,
        ("numerical", "decision"),
        0,
        17,
        {"numerical": 0.0, "decision": 0.1},
    )


def test_policy_executes_actual_branch_in_child_process() -> None:
    text = 'def choose_arm(request):\n    return request["enabled_arms"][1]\n'
    variant = _variant(text)

    assert run_policy(variant, _request()) == "decision"
    assert SourceVariantV2.from_payload(variant.to_payload()) == variant


def test_child_copies_commitments_and_binds_actual_parent() -> None:
    parent = _variant('def choose_arm(request):\n    return "numerical"\n')
    child = SourceVariantV2.child(
        parent, 'def choose_arm(request):\n    return "decision"\n', "replace-arm"
    )

    assert child.parent_source_sha256 == parent.fingerprint()
    assert child.protocol_fingerprint == parent.protocol_fingerprint
    assert child.runtime_fingerprint == parent.runtime_fingerprint


@pytest.mark.parametrize(
    ("parent_source_sha256", "operator"),
    [("a" * 64, "seed"), (None, "mutation")],
)
def test_lineage_requires_seed_and_parent_to_bind_each_other(
    parent_source_sha256: str | None, operator: str
) -> None:
    with pytest.raises(ValueError):
        SourceVariantV2(
            1,
            parent_source_sha256,
            operator,
            {"policy.py": 'def choose_arm(request):\n    return "decision"\n'},
            "a" * 64,
            "b" * 64,
        )


def test_kernel_path_edit_is_rejected() -> None:
    payload = _variant('def choose_arm(request):\n    return "decision"\n').to_payload()
    payload["files"]["../kernel.py"] = "changed = True\n"  # type: ignore[index]

    with pytest.raises(ValueError):
        SourceVariantV2.from_payload(payload)


@pytest.mark.parametrize(
    "text",
    [
        'import os\ndef choose_arm(request):\n    return "decision"\n',
        'def choose_arm(request):\n    return request.__class__\n',
        'def choose_arm(request):\n    return open("active_source.json")\n',
    ],
)
def test_ordinary_authority_access_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        audit_source(_variant(text))


def test_request_rejects_unknown_fields_and_noncanonical_arm_data() -> None:
    payload = _request().to_payload()
    payload["dev_reward"] = 0.2
    with pytest.raises(ValueError):
        SourceRequestV2.from_payload(payload)

    with pytest.raises(ValueError):
        SourceRequestV2(1, ("decision", "numerical"), 0, 17, {"decision": 0.1, "numerical": 0.0})


def test_source_rejects_oversized_policy() -> None:
    with pytest.raises(ValueError):
        _variant("#" * 8193)


def test_host_rejects_policy_output_outside_enabled_arms() -> None:
    variant = _variant('def choose_arm(request):\n    return "retrieval"\n')
    with pytest.raises(ValueError, match="enabled arms"):
        run_policy(variant, _request())


def test_host_rejects_worker_exception() -> None:
    variant = _variant('def choose_arm(request):\n    return request["missing"]\n')
    with pytest.raises(ValueError, match="worker failed"):
        run_policy(variant, _request())


def test_timeout_from_subprocess_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def expired(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(["python"], 0.01)

    monkeypatch.setattr("evolving_loop.v2.source.runtime.subprocess.run", expired)
    with pytest.raises(ValueError, match="timed out"):
        run_policy(_variant('def choose_arm(request):\n    return "decision"\n'), _request())
