# Protocol v5 all-Train launcher

This directory contains the reviewed persistent launcher for the TimesX and
Time-MMD protocol-v5 run. It evolves on all Train tasks, opens Dev once for
host-private final selection, locks the selected program, and does not open
official Test.

The frozen package is not committed because it is 173 MB and exceeds GitHub's
100 MB per-file limit. Obtain the separately distributed
`drcik_aligned_evolve_v5.tgz` and verify:

```text
6392b172d6cd987e795d0d0d43ef7b3d790523bf294b1db9a9e9d620cd570137
```

Run from a Linux host with Python 3, NumPy, bubblewrap, tmux, and an
authenticated Codex CLI:

```bash
cd handoff/drcik_aligned_evolve_v5
export CODEX_BIN="$(command -v codex)"
bash run_formal_v5_tmux.sh /absolute/path/drcik_aligned_evolve_v5.tgz /absolute/path/ts_v5_formal
```

The launcher refuses to overwrite the work directory, verifies the package
SHA, starts each orchestrator as the foreground process of a distinct tmux
session, waits three seconds, and fails unless both sessions and PIDs remain
alive. The PID/session/log ledger is written to `WORK_DIR/pids.tsv`.

This launcher was checked with `bash -n` and a no-model tmux lifecycle smoke
test that created both sessions, verified persistence after launcher return,
and terminated the test sessions. The smoke test made no Codex/model call.
