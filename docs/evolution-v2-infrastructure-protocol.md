# Evolution V2 infrastructure protocol prototype

This offline research prototype evolves a typed L1 infrastructure manifest while
leaving the L0 commitment and frozen Bundle objects unchanged.  A proposal
replaces exactly one of five Host-resolved kinds: backbone, loader, verifier
strategy, diagnostic metric, or migration envelope.

```bash
python -m evolving_loop.v2 protocol-make-smoke-inputs --output-dir /tmp/protocol-inputs
python -m evolving_loop.v2 protocol-evolve \
  --config configs/evolution_v2/protocol/smoke.json \
  --input-manifest /tmp/protocol-inputs/input_manifest.json \
  --output-dir /tmp/protocol-run
```

The smoke profile is fixed at schema version 1, seed 17, six precommitted
proposals, and a 120-second limit. It evaluates two frozen archive Bundles on
four committed Train tasks, then seals the one-task Dev comparison. The five
compatible changes publish releases; the changed-history loader is retained as
a deliberate compatibility rejection. Repeating the command resumes only
closed checkpoints, and completed runs are read-only.

`active_protocol.json` changes only after accepted evidence is reread. Old
protocols, envelope originals, Bundle objects, and their L0 binding remain
addressable. Dev aggregates are written below the sealed store and the public
evidence carries only their hash. `frozen_protocol_handoff.json` pins the
accepted release, evidence, Bundle, L0 commitment, and runtime fingerprint;
it always records `public_test_accessed: false`.

This is not a Public evaluator, arbitrary code-execution system, production
model run, or change to the existing validation-only `public-evaluate` command.
