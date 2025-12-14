#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

printf -v joined '%s-' "${hospital_ids_subset[@]}"
joined="${joined%-}"

# Parallelism settings
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Base directory for logs (matches -l in tune-K29.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${LOG_BASE:-${SCRIPT_DIR}/../../yaib_logs}"

# Set this to the run directory produced by tune-K29.sh (contains hyperparameter_tuning_logs.json).
# If unset, we will pick the most recent tuning run automatically.
TUNING_RUN_DIR="${TUNING_RUN_DIR:-""}"

if [[ -z "${TUNING_RUN_DIR}" ]]; then
    TUNING_RUN_DIR="$(LOG_BASE="${LOG_BASE}" uv run python - <<'PY'
import os, glob
base = os.path.abspath(os.environ.get("LOG_BASE", "."))
paths = glob.glob(os.path.join(base, "**", "hyperparameter_tuning_logs.json"), recursive=True)
# Prefer runs containing K29 and tune in their path
paths = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)
for path in paths:
    lower = path.lower()
    if "k29" in lower and "tune" in lower:
        print(os.path.dirname(path))
        break
else:
    # fallback to most recent irrespective of name
    if paths:
        print(os.path.dirname(paths[0]))
PY
)"
fi

if [[ -z "${TUNING_RUN_DIR}" ]]; then
    echo "Could not auto-detect TUNING_RUN_DIR; please export TUNING_RUN_DIR manually."
    exit 1
fi

readarray -t BEST_HP < <(TUNING_RUN_DIR="${TUNING_RUN_DIR}" uv run python - <<'PY'
import json, os, sys, math
run_dir = os.environ.get("TUNING_RUN_DIR")
path = os.path.join(run_dir, "hyperparameter_tuning_logs.json")
if not os.path.isfile(path):
    sys.exit(f"hyperparameter_tuning_logs.json not found at {path}")
with open(path) as f:
    data = json.load(f)
x_iters = data.get("x_iters", [])
func_vals = data.get("func_vals", [])
if not x_iters or not func_vals:
    sys.exit("No tuning history found in hyperparameter_tuning_logs.json")
best_idx = min(range(len(func_vals)), key=lambda i: func_vals[i])
best = x_iters[best_idx]
# Order matches K29.gin bounds: n_rff_features, gamma, categorical_index
keys = ["model/hyperparameter.n_rff_features", "model/hyperparameter.gamma", "model/hyperparameter.categorical_index"]
for k, v in zip(keys, best):
    # Print as key=value for easy capture
    print(f"{k}={v}")
PY
)

if [[ ${#BEST_HP[@]} -lt 2 ]]; then
    echo "Failed to load tuned hyperparameters from ${TUNING_RUN_DIR}"
    exit 1
fi

echo "Using tuned hyperparameters from ${TUNING_RUN_DIR}:"
printf '  %s\n' "${BEST_HP[@]}"

run_hospital() {
    local hospital1="$1"
    echo "Training model for hospital ID: $hospital1 testing for hospital $hospital1"

    # Call the CLI via the module to force uv to use the project env (avoids missing binary on PATH)
   PATH="$PWD/.venv/bin:$PATH" NVIDIA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" uv run --python 3.11 python -m icu_benchmarks.run \
        -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "$joined" \
        -hit "$hospital1" \
        --complete-train \
        -m K29 \
        -hp "${BEST_HP[@]}" 
            # execute_repeated_cv.cv_folds_to_train=1 \
            # execute_repeated_cv.cv_repetitions_to_train=1
}

for hospital1 in "${hospital_ids_subset[@]}"; do
    # Keep only JOBS concurrent training processes.
    while [[ $(jobs -r -p | wc -l) -ge "${JOBS}" ]]; do
        wait -n
    done
    run_hospital "$hospital1" &
done

wait
