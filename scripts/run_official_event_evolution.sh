#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TASKS_PATH OUTPUT_DIR" >&2
  exit 2
fi

tasks_path=$1
output_dir=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/.." && pwd -P)
seed_policy="$repo_root/runs/meta_harness_v2/real_8_2_gpt56sol/best_policy.json"

if [[ ! -f "$tasks_path" ]]; then
  echo "task source does not exist: $tasks_path" >&2
  exit 2
fi
if [[ ! -f "$seed_policy" ]]; then
  echo "accepted seed policy does not exist: $seed_policy" >&2
  exit 2
fi

mkdir -p "$output_dir"
cd "$repo_root"

python -m evolving_loop.cli evolve \
  --meta-harness-v2 \
  --tasks-file "$tasks_path" \
  --setting statistics \
  --llm-backend codex \
  --codex-model gpt-5.6-sol \
  --codex-reasoning-effort high \
  --codex-timeout 900 \
  --codex-cache-dir "$output_dir/codex-cache" \
  --seed-policy-path "$seed_policy" \
  --coding-initial-programs 3 \
  --coding-mutations 1 \
  --coding-validation-folds 2 \
  --generations 1 \
  --children 4 \
  --successive-halving \
  --screen-train-tasks 6 \
  --screen-promote 2 \
  --screen-tolerance 0.01 \
  --seed 7 \
  --dev-fraction 0.2 \
  --holdout-fraction 0 \
  --library-path "$output_dir/coding_skills.json" \
  --retrieval-library-path "$output_dir/retrieval_skills.json" \
  --decision-library-path "$output_dir/decision_skills.json" \
  --split-manifest-path "$output_dir/split_manifest.json" \
  --policy-path "$output_dir/best_policy.json" \
  --trace-path "$output_dir/evolution_trace.json" \
  --checkpoint-path "$output_dir/checkpoint.json" \
  --progress-path "$output_dir/progress.jsonl" \
  | tee "$output_dir/result.json"
