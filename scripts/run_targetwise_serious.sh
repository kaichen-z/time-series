#!/usr/bin/env bash
# Run ten target-wise generations with ten children and six full evaluations.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-targetwise10_top6_temp03_10gen_20260830}"
RUN_REPO="$ROOT/runs/method_evolution/$RUN_NAME"
SEED_REPO="${SEED_REPO:-$ROOT/runs/method_evolution/targetwise8_temp03_10gen_20260829}"
SEED_COMMIT="${SEED_COMMIT:-01ee5c8}"
CARDS="${CARDS:-2,5}"
TARGET_GENERATIONS="${TARGET_GENERATIONS:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
MAX_STALLS="${MAX_STALLS:-6}"

cd "$ROOT"

if [[ ! -d "$RUN_REPO/.git" ]]; then
    [[ ! -e "$RUN_REPO" ]] || {
        echo "error: $RUN_REPO exists but is not a git repository" >&2
        exit 2
    }
    [[ -d "$SEED_REPO/.git" ]] || {
        echo "error: seed repository does not exist: $SEED_REPO" >&2
        exit 2
    }
    git clone --no-hardlinks "$SEED_REPO" "$RUN_REPO"
    git -C "$RUN_REPO" switch --detach "$SEED_COMMIT"
    git -C "$RUN_REPO" switch -c serious-evolution
fi

attempt=0
stalls=0
while (( attempt < MAX_ATTEMPTS )); do
    completed="$(git -C "$RUN_REPO" log --format=%s | grep -c '^generation ')"
    remaining=$((TARGET_GENERATIONS - completed))
    if (( remaining <= 0 )); then
        echo "supervisor: $completed/$TARGET_GENERATIONS generations complete"
        break
    fi

    echo "supervisor: $completed/$TARGET_GENERATIONS complete; attempt $((attempt + 1))"
    echo "supervisor: launching on GPUs $CARDS"

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_HOME=/raid/home/air/khoutaibi/hf_cache \
    CUDA_VISIBLE_DEVICES="$CARDS" \
    ME_REPO="$RUN_REPO" \
    ME_GENERATIONS="$remaining" \
    ME_LLM_BACKEND=qwen \
    ME_MODEL_ID=Qwen/Qwen3.5-27B \
    ME_MEMORY_DEVICE=cuda:1 \
    ME_EVOLUTION_STRATEGY=targetwise \
    ME_MAX_TARGETS=10 \
    ME_SCREEN_TASKS=4 \
    ME_FULL_EVALUATION_CANDIDATES=6 \
    ME_MUTATION_TEMPERATURE=0.3 \
    ME_SELECTION_TEMPERATURE=0.3 \
    ME_MAX_BARREN_GENERATIONS=4 \
    ME_FAILURE_JUDGE=0 \
    ME_TRAIN_LIMIT=100 \
    ME_VALIDATION_TAIL=20 \
        bash scripts/run_method_evolution.sh >> "$RUN_REPO/run.log" 2>&1
    status=$?

    after="$(git -C "$RUN_REPO" log --format=%s | grep -c '^generation ')"
    if (( status == 0 && after == completed )); then
        stalls=$((stalls + 1))
        echo "supervisor: no promotion ($stalls/$MAX_STALLS consecutive stalls)"
        if (( stalls >= MAX_STALLS )); then
            echo "supervisor: stopping after $MAX_STALLS consecutive stalls"
            break
        fi
    else
        stalls=0
    fi
    attempt=$((attempt + 1))
    sleep 5
done

completed="$(git -C "$RUN_REPO" log --format=%s | grep -c '^generation ')"
echo "supervisor: final state $completed/$TARGET_GENERATIONS generations"
echo "supervisor: log $RUN_REPO/run.log"
