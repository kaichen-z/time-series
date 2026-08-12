#!/usr/bin/env bash
# Shared configuration for every run script. Source this, don't execute it.
#
# Everything here is overridable from your shell, e.g.:
#   EA_OUT_ROOT=/tmp/myrun ./scripts/01_evolve_coding.sh
#
# Why the defaults point at /raid: the root filesystem on this box is effectively full
# (a few GB free, shared with other users), while /raid has hundreds of GB. Model
# downloads, LLM caches, checkpoints and run logs all live on /raid by default so a
# long run cannot fill up / and take other people's jobs down with it.

set -euo pipefail

# ---------------------------------------------------------------- repo layout
EA_REPO_ROOT="${EA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export EA_REPO_ROOT
cd "$EA_REPO_ROOT"

# ---------------------------------------------------------------- data on raid
export DRCIK_DATA_DIR="${DRCIK_DATA_DIR:-/raid/home/air/khoutaibi/time_series_dataset/Dr-CiK}"
export DRCIK_SAMPLE_DIR="${DRCIK_SAMPLE_DIR:-/raid/home/air/khoutaibi/external/Dr-CiK/sample}"

# ------------------------------------------------- model + HF caches on raid
# HF_HOME matters most: the evolver checkpoint is ~35GB and would not fit on /.
export EA_MODEL_CACHE="${EA_MODEL_CACHE:-/raid/home/air/khoutaibi/models}"
export HF_HOME="${HF_HOME:-$EA_MODEL_CACHE}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$EA_MODEL_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$EA_MODEL_CACHE}"

# ------------------------------------------------------------ run outputs
EA_OUT_ROOT="${EA_OUT_ROOT:-/raid/home/air/khoutaibi/evolving_agents_out}"
export EA_OUT_ROOT
export EA_RUNS_DIR="${EA_RUNS_DIR:-$EA_OUT_ROOT/runs}"
export EA_CACHE_DIR="${EA_CACHE_DIR:-$EA_OUT_ROOT/cache/llm}"
export EA_CKPT_ROOT="${EA_CKPT_ROOT:-$EA_OUT_ROOT/checkpoints}"
export EA_BUNDLES_DIR="${EA_BUNDLES_DIR:-$EA_OUT_ROOT/bundles}"
export EA_RESULTS_DIR="${EA_RESULTS_DIR:-$EA_OUT_ROOT/results}"

# ---------------------------------------------------------------- models
export EA_WORKER_MODEL="${EA_WORKER_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
export EA_EVOLVER_MODEL="${EA_EVOLVER_MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"

# ---------------------------------------------------------------- budget
export EA_GENERATIONS="${EA_GENERATIONS:-10}"
export EA_POPULATION="${EA_POPULATION:-6}"
export EA_KEEP_ELITE="${EA_KEEP_ELITE:-2}"
export EA_MINIBATCH="${EA_MINIBATCH:-20}"
export EA_STALL_PATIENCE="${EA_STALL_PATIENCE:-3}"
export EA_SEED="${EA_SEED:-7}"
export EA_TRACE_LEVEL="${EA_TRACE_LEVEL:-summary}"

export EA_LOG_DIR="${EA_LOG_DIR:-$EA_REPO_ROOT/logs}"
export EA_CONSOLE_LEVEL="${EA_CONSOLE_LEVEL:-INFO}"
export EA_LOG_LEVEL="${EA_LOG_LEVEL:-INFO}"
# Off by default: the per-call trace is thousands of lines and always lands in the log file.
export EA_TRACE_CONSOLE="${EA_TRACE_CONSOLE:-0}"

ea_die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ea_info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

# Every script logs to its own timestamped file so reruns never overwrite each other.
ea_log_path() {
  local run_name="$1"
  mkdir -p "$EA_LOG_DIR"
  echo "$EA_LOG_DIR/${run_name}_$(date +%Y%m%d_%H%M%S).log"
}

ea_log_flags() {
  local log_file="$1"
  printf -- '--log-file %s --log-level %s --console-level %s' "$log_file" "$EA_LOG_LEVEL" "$EA_CONSOLE_LEVEL"
  [[ "$EA_TRACE_CONSOLE" == "1" ]] && printf -- ' --trace-console'
}

ea_report_log() {
  local log_file="$1"
  echo
  ea_info "full log (every prompt, tool call and reasoning block):"
  echo "    $log_file"
  ea_info "watch it live in another terminal with:"
  echo "    tail -f $log_file"
}

# Pick the N GPUs with the most free memory right now. This box is shared and its
# load moves around, so devices are chosen at launch rather than hardcoded.
ea_pick_gpus() {
  local want="${1:-2}"
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2 -rn | head -n "$want" | awk -F', *' '{printf "cuda:%s\n", $1}'
}

# Worker and evolver go on separate GPUs: both auto-picking at once can land on one card.
ea_resolve_devices() {
  if [[ -n "${EA_WORKER_DEVICE:-}" && -n "${EA_EVOLVER_DEVICE:-}" ]]; then return; fi
  local picked
  mapfile -t picked < <(ea_pick_gpus 2)
  export EA_WORKER_DEVICE="${EA_WORKER_DEVICE:-${picked[0]:-}}"
  export EA_EVOLVER_DEVICE="${EA_EVOLVER_DEVICE:-${picked[1]:-${picked[0]:-}}}"
}

# Refuse to start a long run that will die halfway through on a full disk.
ea_check_disk() {
  local path="$1" need_gb="${2:-20}" avail_gb
  mkdir -p "$path"
  avail_gb=$(df -BG --output=avail "$path" | tail -1 | tr -dc '0-9')
  if (( avail_gb < need_gb )); then
    ea_die "only ${avail_gb}GB free on $path, need ~${need_gb}GB. Set EA_OUT_ROOT to a roomier disk."
  fi
  ea_info "disk ok: ${avail_gb}GB free on $path"
}

# The task source every script accepts: --sample (fast, 3 tasks) or the full dataset.
ea_task_source() {
  if [[ "${EA_USE_SAMPLE:-0}" == "1" ]]; then
    [[ -d "$DRCIK_SAMPLE_DIR" ]] || ea_die "sample dir not found: $DRCIK_SAMPLE_DIR"
    echo "--sample-dir $DRCIK_SAMPLE_DIR"
  else
    [[ -d "$DRCIK_DATA_DIR" ]] || ea_die "data dir not found: $DRCIK_DATA_DIR"
    echo "--data-dir $DRCIK_DATA_DIR"
  fi
}

ea_show_config() {
  ea_info "configuration"
  cat <<EOF
  repo            $EA_REPO_ROOT
  task source     $(ea_task_source)
  worker model    $EA_WORKER_MODEL  (${EA_WORKER_DEVICE:-auto})
  evolver model   $EA_EVOLVER_MODEL  (${EA_EVOLVER_DEVICE:-auto})
  model cache     $EA_MODEL_CACHE
  out root        $EA_OUT_ROOT
    runs          $EA_RUNS_DIR
    llm cache     $EA_CACHE_DIR
    checkpoints   $EA_CKPT_ROOT
    bundles       $EA_BUNDLES_DIR
  budget          G=$EA_GENERATIONS P=$EA_POPULATION K=$EA_KEEP_ELITE batch=$EA_MINIBATCH seed=$EA_SEED
  trace level     $EA_TRACE_LEVEL
EOF
}

mkdir -p "$EA_RUNS_DIR" "$EA_CACHE_DIR" "$EA_CKPT_ROOT" "$EA_BUNDLES_DIR" "$EA_RESULTS_DIR" "$EA_REPO_ROOT/logs"
