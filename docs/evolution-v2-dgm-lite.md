# Evolution V2 DGM-lite source smoke

This offline research prototype evolves only one audited `policy.py` function.
It is not a claim of real-dataset improvement or production hostile-code
isolation.

```bash
python -m evolving_loop.v2.source make-smoke-inputs --output-dir /tmp/source-inputs
python -m evolving_loop.v2.source evolve \
  --config configs/evolution_v2/source/smoke.json \
  --input-manifest /tmp/source-inputs/manifest.json \
  --output-dir /tmp/source-run
```

The editable boundary is exactly `choose_arm(request)` in the generated source
variants.  Inputs are frozen Train4/Dev1 fixture commitments with two Train
folds. Public data is never supplied. The manifest hashes each input before
execution, and its directory cannot overlap the output directory.

The Host stores the branching source archive in `source_archive/`, sealed
validation/canary evidence in `sealed/` and `authority/sealed/`, the active
pointer in `authority/active_source.json`, and terminal semantic output in
`evaluation_complete.json`. A passed canary activates automatically; a failed
canary writes rollback history and restores the exact previous pointer bytes.
Re-running the same command verifies commitments and performs a read-only
completed resume.

This is limited to the deterministic offline fixture and its fixed one-epoch
source policy experiment. OS sandboxing, generic Harness rewriting, and
general/dataset-scale evaluation remain deferred.
