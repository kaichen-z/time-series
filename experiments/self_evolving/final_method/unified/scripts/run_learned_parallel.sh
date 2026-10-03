#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
S=$ROOT/experiments/self_evolving/final_method/unified/scripts
PY=${PY:-$ROOT/.venv/learned/bin/python}
GPUS=${GPUS:-0,1,2,3}
DATASETS=${DATASETS:-drcik,tmmd,timesx}
MODELS=${MODELS:-posttime,time_llm,mm_tsflib}
FORCE=${FORCE:-0}
LOGS=$ROOT/runs/learned_baselines/logs

cd "$ROOT"
mkdir -p "$LOGS"
PYTHONPATH=$ROOT "$PY" "$S/learned_data.py"
IFS=, read -ra gpus <<< "$GPUS"
IFS=, read -ra datasets <<< "$DATASETS"
IFS=, read -ra models <<< "$MODELS"
if [ ${#gpus[@]} -eq 0 ]; then
    echo "no GPUs selected" >&2
    exit 2
fi

run_one() {
    local gpu=$1 model=$2 dataset=$3 out=$ROOT/runs/learned_baselines/$model/$dataset
    export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$ROOT
    echo "gpu=$gpu model=$model dataset=$dataset start=$(date -Is)"
    if [ "$model" = posttime ]; then
        "$PY" "$S/posttime.py" prior "$dataset"
        if [ "$FORCE" = 1 ] || [ ! -d "$out/sft" ]; then "$PY" "$S/posttime.py" sft "$dataset"; fi
        if [ "$FORCE" = 1 ] || [ ! -d "$out/rlvr" ]; then "$PY" "$S/posttime.py" rlvr "$dataset"; fi
    elif [ "$FORCE" = 1 ] || [ ! -f "$out/best.pt" ]; then
        "$PY" "$S/$model.py" train "$dataset"
    fi
    "$PY" "$S/$model.py" predict "$dataset"
    "$PY" "$S/$model.py" score "$dataset"
    echo "gpu=$gpu model=$model dataset=$dataset end=$(date -Is)"
}

jobs=()
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do jobs+=("$model:$dataset"); done
done
free=("${gpus[@]}")
declare -A pid_gpu pid_job
next=0
active=0
failed=0
trap 'kill $(jobs -pr) 2>/dev/null || true' INT TERM

while [ $next -lt ${#jobs[@]} ] || [ $active -gt 0 ]; do
    while [ ${#free[@]} -gt 0 ] && [ $next -lt ${#jobs[@]} ]; do
        gpu=${free[0]}; free=("${free[@]:1}")
        job=${jobs[$next]}; model=${job%%:*}; dataset=${job#*:}; next=$((next + 1))
        log=$LOGS/${model}_${dataset}.log
        run_one "$gpu" "$model" "$dataset" >"$log" 2>&1 &
        pid=$!; pid_gpu[$pid]=$gpu; pid_job[$pid]=$job
        active=$((active + 1))
        echo "launched $job on gpu $gpu pid $pid log $log"
    done
    if wait -n -p pid; then status=0; else status=$?; fi
    gpu=${pid_gpu[$pid]}; job=${pid_job[$pid]}
    unset 'pid_gpu[$pid]' 'pid_job[$pid]'; free+=("$gpu"); active=$((active - 1))
    if [ $status -eq 0 ]; then
        echo "finished $job on gpu $gpu"
    else
        echo "failed $job on gpu $gpu status $status; see $LOGS/${job/:/_}.log" >&2
        failed=1
    fi
done

if [ $failed -ne 0 ]; then exit 1; fi
PYTHONPATH=$ROOT "$PY" "$S/learned_score.py"
