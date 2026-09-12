# Final runtime re-review fix

Addressed the two findings in `final-rereview-report.md` without introducing a
derived-evidence authority.

Changed shortlisted policies are rejected before local package execution and
again at freeze. The comparison covers the complete parent alternative spec,
including its recipe, full policy and five cross-fit policies. Source bytes
must remain in the original Host-authorized source snapshot, and runtime
commitments must remain unchanged. Revisions require refreshed Task 4 evidence;
the required-winner guard is unchanged. The integrated fixture now authorizes
the exact pre-fitted policy replayed by its unchanged child.

Direct and derived forecasts are guarded per candidate. Failed or non-finite
specialists and unresolved derived dependencies are omitted from materialized
forecasts and recorded as `shortlisted_runtime_failure` in the typed selection
result. Any such failure forces the exact successful Anchor forecast. Sealed
historical diagnostics remain unchanged; runtime failure records make failed
members unavailable for selection. Freeze retains those records, and byte-only
restore permits omissions only when the result records the corresponding
failure and exact Anchor fallback. Anchor execution failure without a verified
parent forecast remains a hard failure; no synthetic Anchor is created.

Validation: both original failures were reproduced before their fixes. The
specialist-failure regression verifies exact `(14.0, 14.0)` Anchor fallback and
restoration. The policy regression redirects an existing specialist ID to a
different executable parent while retaining its old folds; it is rejected
before local execution. The integrated unchanged-policy/nonempty-winner case
retains specialist activation and completed-run frozen reload.

Final essential gate: **27 passed in 70.92s**, covering both new regressions,
the integrated unchanged-policy path, both original projection tests unchanged,
the schema-2 Anchor-cache projection, diagnostic-content rejection, and the
package supply suite. `git diff --check` passed.
Protected P3 files were not edited or staged.
