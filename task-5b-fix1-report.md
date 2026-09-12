# Task 5B fix 1: schema-v2 registry digest

Removed the schema-v1-only guard from Kernel's registry-manifest digest check.
Both envelope versions now prove that `registry_sha256` matches the exact
package set, using the manifest produced by `FrozenNumericalPackageRegistry`.

Added a focused regression alongside the valid v2 Kernel replay: change only
the non-winner Anchor forecast in one task package, reseal its package envelope
and frozen pair, preserve all local-evidence references and winner rows, and
retain the original Bundle/registry SHA. Kernel must reject the registry claim.

TDD RED reproduced the issue before the fix: the regression failed with
`DID NOT RAISE KernelAuthorityError` (14.35 seconds). The implementation change
is one removed version guard. Protected bridge and cooperative files were not
modified or staged.

GREEN: `python -m pytest -q tests/test_evolution_v2_frozen_shortlist.py
tests/test_evolution_v2_kernel.py --tb=short --maxfail=1` completed with
**98 passed in 114.14 seconds**. `git diff --check` is clean.
