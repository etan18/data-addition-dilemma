#!/bin/bash
set -euo pipefail

# Track child PIDs to ensure clean termination
declare -a PIDS=()

# Clean up any background jobs if the script exits or is interrupted
cleanup() {
    echo "[all_data-K29] Cleaning up background jobs..."
    # Terminate tracked PIDs and their process groups
    for pid in "${PIDS[@]}"; do
        [[ -z "${pid}" ]] && continue
        # First try graceful termination
        kill -TERM "${pid}" 2>/dev/null || true
        # Also attempt killing the whole process group if supported
        kill -TERM -- -"${pid}" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -KILL "${pid}" 2>/dev/null || true
        kill -KILL -- -"${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

printf -v joined '%s-' "${hospital_ids_subset[@]}"
joined="${joined%-}"

# Parallelism settings
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Hide GPUs entirely for CPU-only runs to prevent accidental CUDA contexts
export CUDA_VISIBLE_DEVICES=""
# Force CPU so frameworks don't attempt CUDA
export PYTORCH_NO_CUDA="1"
# Ensure single device when running on CPU
export YAIB_TRAINER_DEVICES="${YAIB_TRAINER_DEVICES:-1}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Disable wandb logging for batch runs to avoid wandb.log errors.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

# Base directory for logs (matches -l in tune-K29.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${LOG_BASE:-${SCRIPT_DIR}/../../yaib_logs}"

run_hospital() {
    local hospital1="$1"
    echo "Training model for hospital ID: $hospital1 testing for hospital $hospital1"

    # Call the CLI via the module to force uv to use the project env (avoids missing binary on PATH)
    PATH="$PWD/.venv/bin:$PATH" uv run --python 3.11 python -m icu_benchmarks.run \
        -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "$joined" \
        -hit "$hospital1" \
        --complete-train \
        -m K29 \
        --cpu \
        -ls _mondrian
}

for hospital1 in "${hospital_ids_subset[@]}"; do
    # Keep only JOBS concurrent training processes.
    while [[ $(jobs -r -p | wc -l) -ge "${JOBS}" ]]; do
        wait -n
    done
    run_hospital "$hospital1" &
    PIDS+=("$!")
done

wait
