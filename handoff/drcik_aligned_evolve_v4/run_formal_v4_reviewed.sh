#!/usr/bin/env bash
# Formal protocol-v3 run for the reviewed TimesX + Time-MMD v4 package.
# Usage: bash run_formal_v4_reviewed.sh drcik_aligned_evolve_v4.tgz [WORK_DIR]
set -euo pipefail

TGZ_INPUT=${1:?usage: bash run_formal_v4_reviewed.sh drcik_aligned_evolve_v4.tgz [WORK_DIR]}
WORK_INPUT=${2:-./ts_v4_formal}
EXPECTED=3b9403f4ac904dac0971ede586a8e1941d927ed62f7b72ce870a06e1bb84c025

for cmd in python3 bwrap sha256sum realpath tar nohup; do
  command -v "$cmd" >/dev/null || { echo "missing command: $cmd" >&2; exit 1; }
done

TGZ=$(realpath "$TGZ_INPUT")
WORK=$(realpath -m "$WORK_INPUT")
printf '%s  %s\n' "$EXPECTED" "$TGZ" | sha256sum -c -

if [[ -n ${CODEX_BIN:-} ]]; then
  CODEX_BIN=$(realpath "$CODEX_BIN")
elif command -v npm >/dev/null 2>&1; then
  NPM_ROOT=$(npm root -g)
  CODEX_BIN=$(realpath "$NPM_ROOT/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex")
elif command -v codex >/dev/null 2>&1; then
  CODEX_BIN=$(realpath "$(command -v codex)")
else
  echo "codex not found; set CODEX_BIN=/absolute/path/to/codex" >&2
  exit 1
fi
[[ -x "$CODEX_BIN" ]] || { echo "not executable: $CODEX_BIN" >&2; exit 1; }

[[ ! -e "$WORK" ]] || { echo "refusing to overwrite existing path: $WORK" >&2; exit 1; }
mkdir -m 700 -p "$WORK"
tar -xzf "$TGZ" -C "$WORK"
ROOT="$WORK/drcik_aligned_evolve"
[[ -f "$ROOT/orchestrate.py" ]] || { echo "missing orchestrate.py after extraction" >&2; exit 1; }

: > "$WORK/pids.tsv"
for ds in timesx time_mmd; do
  log="$WORK/$ds.log"
  nohup python3 "$ROOT/orchestrate.py" \
    --dataset "$ds" \
    --pack "$ROOT/packs/$ds" \
    --out "$WORK/runs/$ds" \
    --real-codex "$CODEX_BIN" \
    >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\t%s\n' "$ds" "$pid" >> "$WORK/pids.tsv"
  echo "$ds started: pid=$pid log=$log"
done

echo "PID ledger: $WORK/pids.tsv"
echo "Results: $WORK/runs/<dataset>/final/{LOCK.json,F2_final_test.json}"
