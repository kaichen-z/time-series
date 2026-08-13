# Historical `before` snapshot

This directory preserves work that existed locally before the repository was
realigned with the rewritten `main` history on 2026-08-14. It is an archive,
not the active implementation.

## Contents

### `local_uncommitted_snapshot/`

Snapshot of the uncommitted files from the older local `main` worktree at
commit `75743d4a2a5b6467ff2b6598ef54c518fd640252`.

It includes:

- the earlier prompt-bundle co-evolution implementation;
- the numbers-only Coding Agent policy changes;
- configurable Coding, Retrieval, and Decision prompts;
- the retrieval-state reset that prevents evidence impacts from leaking
  between tasks;
- the `--agent-bundle` CLI integration;
- co-evolution and Codex-triad tests (stored with a `.py.snapshot` suffix so
  the active `pytest` suite does not collect them);
- presentation notes; and
- the local Codex response cache under `runs/`.

The files retain their original repository-relative layout beneath this
directory. They are preserved for comparison and selective migration. The
active implementation remains at the repository root.

### `experiment_runs_snapshot/`

Point-in-time copy of uncommitted experiment artifacts from the integration
worktree. It contains Task 42 smoke runs, sample prompt evolution, open-genome
evolution, source-evolution smoke artifacts, and partially populated LLM-only
pilot/full-run directories.

These directories include cached model responses and split manifests. A file
count is not evidence that an experiment completed, and incomplete directories
must not be reported as final benchmark results.

The original `subset_tasks/` entries were absolute symbolic links into a local
Dr-CiK checkout, so they were intentionally not copied. The selected public
task IDs are retained in `subset_manifest.json`; users should obtain Dr-CiK
from its official repository.

## Safety and use

- Do not import code from this directory in production runs.
- Do not run archived tests as part of the current test suite.
- Compare archived code with the active implementation before reusing it.
- Public benchmark labels and cached research outputs are historical artifacts,
  not hidden-test data.
- No API keys, tokens, passwords, or environment files are included.
