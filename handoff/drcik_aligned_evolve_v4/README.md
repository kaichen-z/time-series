# Reviewed TimesX / Time-MMD evolution package (protocol v3, package v4)

- `drcik_aligned_evolve_v4.tgz` — complete package incl. frozen task packs (SHA256 `3b9403f4ac904dac0971ede586a8e1941d927ed62f7b72ce870a06e1bb84c025`), independently reviewed (17/17 model-free checks).
- `run_formal_v4_reviewed.sh` — one-shot launcher (SHA256 `68c239d04055ce21e23ada303b47bcc961fddd72ad0be07c874def543cad75e1`).

```bash
git clone -b official-ts-llm-handoff-20261009 https://github.com/kaichen-z/time-series.git
cd time-series/handoff/drcik_aligned_evolve_v4
bash run_formal_v4_reviewed.sh drcik_aligned_evolve_v4.tgz            # real model calls: <= 80 episodes / 400 submissions per dataset
# optional model-free self-check first:
#   mkdir -m 700 /tmp/v4 && tar xzf drcik_aligned_evolve_v4.tgz -C /tmp/v4 && (cd /tmp/v4/drcik_aligned_evolve && python3 -I -S -B run_tests.py)
```
Needs Linux, python3, bubblewrap (`bwrap`), a logged-in `codex` CLI (set `CODEX_BIN=/abs/path/codex` if not auto-detected).
Results: `ts_v4_formal/runs/<dataset>/final/LOCK.json` (chosen program) and `F2_final_test.json` (one-time final test).
