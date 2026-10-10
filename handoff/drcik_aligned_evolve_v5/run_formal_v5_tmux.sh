#!/usr/bin/env bash
# Formal protocol-v5 (all_train) run for TimesX + Time-MMD: evolve on ALL official Train (TimesX 2,064 after purge,
# Time-MMD 45,948), select the final program once on Dev (550 / 5,780), lock. The final Test check is NOT run here
# (separate approval). Real model calls (gpt-5.6-sol, high) - run only after explicit approval.
# Usage: bash run_formal_v5.sh drcik_aligned_evolve_v5.tgz [WORK_DIR]
set -euo pipefail

TGZ_INPUT=${1:?usage: bash run_formal_v5.sh drcik_aligned_evolve_v5.tgz [WORK_DIR]}
WORK_INPUT=${2:-./ts_v5_formal}
EXPECTED=6392b172d6cd987e795d0d0d43ef7b3d790523bf294b1db9a9e9d620cd570137

for cmd in python3 bwrap sha256sum realpath tar tmux; do
  command -v "$cmd" >/dev/null || { echo "missing command: $cmd" >&2; exit 1; }
done
python3 -c "import numpy" || { echo "python3 needs numpy" >&2; exit 1; }

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
ROOT="$WORK/drcik_aligned_evolve_v5"
[[ -f "$ROOT/orchestrate.py" ]] || { echo "missing orchestrate.py after extraction" >&2; exit 1; }
avail=$(df -Pk "$WORK" | awk 'NR==2{print $4}')
(( avail > 15*1024*1024 )) || { echo "need >= 15 GB free under $WORK (Time-MMD run dirs ~0.35 GB each)" >&2; exit 1; }

: > "$WORK/pids.tsv"
tag=$(printf '%s' "$WORK" | sha256sum | cut -c1-10)
for ds in timesx time_mmd; do
  log="$WORK/$ds.log"
  session="ts_v5_${ds}_${tag}"
  printf -v launch 'exec %q %q --dataset %q --pack %q --out %q --real-codex %q >%q 2>&1' \
    python3 "$ROOT/orchestrate.py" "$ds" "$ROOT/packs/${ds}_all_train" "$WORK/runs/$ds" "$CODEX_BIN" "$log"
  tmux new-session -d -s "$session" "$launch"
  pid=$(tmux list-panes -t "$session" -F '#{pane_pid}')
  printf '%s\t%s\t%s\t%s\n' "$ds" "$session" "$pid" "$log" >> "$WORK/pids.tsv"
done

# A successful tmux create is not enough: verify that both foreground
# orchestrators survive after the launcher has returned control.
sleep 3
while IFS=$'\t' read -r ds session pid log; do
  if ! tmux has-session -t "$session" 2>/dev/null || ! kill -0 "$pid" 2>/dev/null; then
    echo "$ds failed to remain alive; log=$log" >&2
    sed -n '1,120p' "$log" >&2 || true
    exit 1
  fi
  echo "$ds started persistently: session=$session pid=$pid log=$log"
done < "$WORK/pids.tsv"

echo "PID ledger: $WORK/pids.tsv"
echo "Monitor: tmux list-sessions | grep ts_v5_"
echo "Results: $WORK/runs/<dataset>/final/{LOCK.json,FINAL_PENDING.json,access_log.json}"
