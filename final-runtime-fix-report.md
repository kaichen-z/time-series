# Final runtime fixes: C1, C2, C3, I1

Implemented production P2 evidence intake through `--task-local-evidence` and
`--task-local-dictionary`, with matching Host API keywords. The constructor
loads canonical Task 4 bytes, verifies the actual Dictionary identity and exact
Host task/history universe, and binds the index into immutable adapter identity.
Schema-2 CLI and direct Host runs require that authority unless
`legacy_bootstrap=True` / `--legacy-bootstrap` is explicit.

Evidence-aware seed and child package execution resolves the shortlist before
full forecasts. Only shortlisted executable methods run; derived recipes use
already-shortlisted parent forecasts. Catalog metadata remains complete and
separate from execution membership. Sealed Task 4 diagnostics are reused
without another hindcast. Independent Train recipe cross-fit fitting retains
its separate verified-source forecast stage.

The schema-2 binder requires diagnostic content, compares it to package
diagnostics, executes the existing Anchor-heavy tournament, and preserves its
specialists, weights, forecast and fallback. Package result fingerprints bind
selection and diagnostic identity. Frozen byte-only validation compares exact
diagnostic contents and result identity; resealing different package diagnostic
contents does not pass. Missing observations have deterministic ineligible
markers and retain exact Anchor fallback.

Schema-1 projection again exports only selected specifications in original
selection order. Both schemas retain the exact parent Anchor after checking
the child's Anchor forecast. Schema 2 merges catalog alternatives, allowing a
selected policy rebind only from the verified direct parent; unchanged
alternative materializations still undergo diagnostic conflict checks.

The integrated regression uses the production constructor, 14 alternatives,
two history-specific eight-member shortlists, actual seed and child execution,
a nonempty archive and required winner, activated specialists, a completed
bounded P2 run, and `load_active_frozen_pair` with forecasts disabled. Exact
selection and `(12.0, 12.0)` output survive restoration.

Iteration boundary: newly invented IDs absent from sealed Task 4 membership
remain catalog-only until Task 4 evidence is refreshed. The exact-winner guard
remains active. The evidence runtime currently requires a directly executable
protected Anchor; it rejects composite Anchor policies rather than replacing
their protected output with a fallback parent.

Validation:

- TDD reproduced both original projection failures unchanged before fixing.
- Specialist preservation, unshortlisted seed execution, resealed diagnostic
  mismatch, CLI missing-evidence and Host missing-evidence checks were observed
  failing before their fixes.
- CLI/package/integrated suite: 56 passed in 258.09s.
- Completed-run integrated freeze/reload test: 1 passed in 97.77s.
- Earlier projection + integrated focused gate: 4 passed in 47.55s.
- Final fresh essential gate: 6 passed in 76.40s, including both original
  projection tests unchanged, the schema-2 Anchor-cache variant, both
  production evidence guards and the integrated completed-run reload.
- `git diff --check` passed. A supplemental full package/adapter/frozen gate
  was still running at commit time; its final outcome is reported separately.

The protected P3 files were neither edited nor staged by this task.
