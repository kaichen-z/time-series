#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
S=$ROOT/experiments/self_evolving/final_method/unified/scripts
PY=${PY:-$ROOT/.venv/learned/bin/python}
DATASETS=${DATASETS:-drcik,tmmd,timesx}
MODELS=${MODELS:-time_llm,mm_tsflib,posttime}

cd "$ROOT"
PYTHONPATH=$ROOT "$PY" "$S/learned_data.py"
IFS=, read -ra datasets <<< "$DATASETS"
IFS=, read -ra models <<< "$MODELS"
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        if [ "$model" = posttime ]; then
            PYTHONPATH=$ROOT "$PY" "$S/posttime.py" prior "$dataset"
            PYTHONPATH=$ROOT "$PY" "$S/posttime.py" sft "$dataset"
            PYTHONPATH=$ROOT "$PY" "$S/posttime.py" rlvr "$dataset"
        else
            PYTHONPATH=$ROOT "$PY" "$S/$model.py" train "$dataset"
        fi
        PYTHONPATH=$ROOT "$PY" "$S/$model.py" predict "$dataset"
        PYTHONPATH=$ROOT "$PY" "$S/$model.py" score "$dataset"
    done
done
PYTHONPATH=$ROOT "$PY" "$S/learned_score.py"
