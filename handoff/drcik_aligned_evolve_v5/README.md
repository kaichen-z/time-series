# Protocol v5 all-Train launcher

This directory contains the reviewed persistent launcher for the TimesX and
Time-MMD protocol-v5 run. It evolves on all Train tasks, opens Dev once for
host-private final selection, locks the selected program, and does not open
official Test.

The frozen 173 MB package is committed as two byte-for-byte split parts because
GitHub limits individual files to 100 MB. Reassemble and verify it before use:

```bash
sha256sum -c parts.sha256
cat drcik_aligned_evolve_v5.tgz.part00 \
    drcik_aligned_evolve_v5.tgz.part01 > drcik_aligned_evolve_v5.tgz
echo "6392b172d6cd987e795d0d0d43ef7b3d790523bf294b1db9a9e9d620cd570137  drcik_aligned_evolve_v5.tgz" | sha256sum -c -
```

Expected part sizes and SHA256 values:

| File | Bytes | SHA256 |
|---|---:|---|
| `drcik_aligned_evolve_v5.tgz.part00` | 94,371,840 | `de0f4d95fdc5032a5e3642531b4c88a1a5ffe232407ec3847b74ee541826e498` |
| `drcik_aligned_evolve_v5.tgz.part01` | 78,205,531 | `89206f95401aa7953f7e4322e34fc191b1eec64ef62a0a44bde418a8607559c9` |

Run from a Linux host with Python 3, NumPy, bubblewrap, tmux, and an
authenticated Codex CLI:

```bash
cd handoff/drcik_aligned_evolve_v5
export CODEX_BIN="$(command -v codex)"
bash run_formal_v5_tmux.sh "$PWD/drcik_aligned_evolve_v5.tgz" /absolute/path/ts_v5_formal
```

The launcher refuses to overwrite the work directory, verifies the package
SHA, starts each orchestrator as the foreground process of a distinct tmux
session, waits three seconds, and fails unless both sessions and PIDs remain
alive. The PID/session/log ledger is written to `WORK_DIR/pids.tsv`.

This launcher was checked with `bash -n` and a no-model tmux lifecycle smoke
test that created both sessions, verified persistence after launcher return,
and terminated the test sessions. The smoke test made no Codex/model call.
