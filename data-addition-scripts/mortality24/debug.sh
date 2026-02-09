#!/bin/bash
set -euo pipefail

# Mimic all_data-K29.sh but with fixed test hospital and a loop over n_rff_features.

DATA_DIR="/home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu"
TASK="BinaryClassification"
TASK_NAME="Mortality24"
MODEL="K29"
NAME="eicu"

HOSPITAL_LIST_FILE="/home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt"
mapfile -t H_IDS < "${HOSPITAL_LIST_FILE}"
JOINED="$(printf '%s-' "${H_IDS[@]}")"
JOINED="${JOINED%-}"

TEST_HOSPITAL="252"

N_RFF_FEATURES_LIST=(64 128 256 512 1024 2048)

for n_rff in "${N_RFF_FEATURES_LIST[@]}"; do
  echo "Running K29 with n_rff_features=${n_rff} (test hospital ${TEST_HOSPITAL})"
  # Call the CLI via the module to force uv to use the project env (avoids missing binary on PATH)
  PATH="$PWD/.venv/bin:$PATH" uv run --python 3.11 python -m icu_benchmarks.run \
    -d "${DATA_DIR}" \
    -n "${NAME}" \
    -t "${TASK}" \
    -tn "${TASK_NAME}" \
    -hi "${JOINED}" \
    -hit "${TEST_HOSPITAL}" \
    --complete-train \
    -m "${MODEL}" \
    --cpu \
    --hyperparams "model/hyperparameter.n_rff_features=${n_rff}" \
    -ls "_n_rff_${n_rff}"
done
