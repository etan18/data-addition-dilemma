#!/bin/bash
set -euo pipefail

# Tune K29 on a controlled subset: train on a joined set of hospitals, test on one hospital.
# Train samples per hospital are capped (via --max_train) to keep tuning fast.
# After this finishes, point TUNING_RUN_DIR in all_data-K29.sh to the created run directory.

DATA_DIR="/home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu"
LOG_DIR="../yaib_logs"
TASK="BinaryClassification"
TASK_NAME="Mortality24"
MODEL="K29"
NAME="eicu_k29_tune_joined"

# Build joined train hospitals and pick first hospital as test.
HOSPITAL_LIST_FILE="/home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt"
mapfile -t H_IDS < "${HOSPITAL_LIST_FILE}"
JOINED="$(printf '%s-' "${H_IDS[@]}")"
JOINED="${JOINED%-}"
TEST_HOSPITAL="${H_IDS[4]}"
# MAX_TRAIN_PER_HOSPITAL=5000

echo "Tuning ${MODEL} on train hospitals ${JOINED} with test hospital ${TEST_HOSPITAL}" 
uv run --python 3.11 python -m icu_benchmarks.run \
  -d "${DATA_DIR}" \
  -n "${NAME}" \
  -t "${TASK}" \
  -tn "${TASK_NAME}" \
  -m "${MODEL}" \
  -hi "${JOINED}" \
  -hit "${TEST_HOSPITAL}" \
  --tune \
  -l "${LOG_DIR}" \
  -ls "_k29_tune"

# Optional: cap per-hospital training samples to speed up tuning.
#   --max_train "${MAX_TRAIN_PER_HOSPITAL}" \
