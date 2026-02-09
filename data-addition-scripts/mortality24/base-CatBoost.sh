#!/bin/bash
set -euo pipefail

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Use all hospital IDs from the file
hospital_ids_subset=("${hospital_ids[@]}")

printf -v joined '%s-' "${hospital_ids_subset[@]}"
joined="${joined%-}"

BASE_LOG_DIR="/home/tane/YAIB/yaib_logs"
DATA_DIR="/home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu"
LOG_SUFFIX="_curve"
WANDB_RUN_ID="${WANDB_RUN_ID:-catboost_$(date +%Y%m%dT%H%M%S)_$RANDOM}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

echo "Training single CatBoost model on all hospitals..."
echo "Using shared W&B run id: $WANDB_RUN_ID"
WANDB_PROJECT=defensive-forecasting WANDB_ENTITY=erintan \
WANDB_RUN_ID="$WANDB_RUN_ID" WANDB_RESUME="$WANDB_RESUME" YAIB_WANDB_METRIC_PREFIX="train" \
CUDA_VISIBLE_DEVICES= PATH="$PWD/.venv/bin:$PATH" uv run --python 3.11 python -m icu_benchmarks.run \
    -d "$DATA_DIR" \
    -n eicu \
    -t BinaryClassification \
    -tn Mortality24 \
    -hi "$joined" \
    --complete-train \
    -m CatBoostClassifier \
    -ls "$LOG_SUFFIX" \
    -l "$BASE_LOG_DIR" \
    -wd

MODEL_LOG_PARENT="$BASE_LOG_DIR/eicu/Mortality24/CatBoostClassifier/train${joined}${LOG_SUFFIX}"
RUN_DIR="$(ls -1dt "$MODEL_LOG_PARENT"/*/ 2>/dev/null | head -n 1)"

if [[ -z "$RUN_DIR" ]]; then
    echo "Could not find latest run directory under $MODEL_LOG_PARENT"
    exit 1
fi

echo "Using trained model from: $RUN_DIR"
for hospital1 in "${hospital_ids_subset[@]}"; do
    echo "Evaluating model on hospital ID: $hospital1"
    WANDB_PROJECT=defensive-forecasting WANDB_ENTITY=erintan \
    WANDB_RUN_ID="$WANDB_RUN_ID" WANDB_RESUME="$WANDB_RESUME" YAIB_WANDB_METRIC_PREFIX="eval/hospital_${hospital1}" \
    CUDA_VISIBLE_DEVICES= PATH="$PWD/.venv/bin:$PATH" uv run --python 3.11 python -m icu_benchmarks.run \
        -d "$DATA_DIR" \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "$joined" \
        -hit "$hospital1" \
        --eval \
        --complete-train \
        --source-name eicu \
        --source-dir "$RUN_DIR" \
        -m CatBoostClassifier \
        -ls "$LOG_SUFFIX" \
        -l "$BASE_LOG_DIR" \
        -wd
done
