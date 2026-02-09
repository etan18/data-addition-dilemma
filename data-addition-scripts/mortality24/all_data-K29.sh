#!/bin/bash
set -euo pipefail

SCRIPT_NAME="all_data-K29"
DATA_DIR="/home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu"
HOSPITAL_LIST_FILE="${DATA_DIR}/above2000.txt"
WANDB_PROJECT="${WANDB_PROJECT:-defensive-forecasting}"
WANDB_ENTITY="${WANDB_ENTITY:-erintan}"
WANDB_RUN_ID="${WANDB_RUN_ID:-k29_$(date +%Y%m%dT%H%M%S)_$RANDOM}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

die() {
    echo "[${SCRIPT_NAME}] ERROR: $*" >&2
    exit 1
}

warn() {
    echo "[${SCRIPT_NAME}] WARN: $*"
}

run_icu_benchmarks() {
    PATH="$PWD/.venv/bin:$PATH" uv run --python 3.11 python -m icu_benchmarks.run "$@"
}

find_latest_run_dir() {
    local base_dir="$1"
    local latest
    latest="$(ls -1dt "${base_dir}"/20* 2>/dev/null | head -n1 || true)"
    [[ -n "${latest}" ]] || return 1
    printf '%s\n' "${latest}"
}

find_source_dir_with_config() {
    local search_root="$1"
    local source_config
    source_config="$(find "${search_root}" -type f -name "train_config.gin" | head -n1 || true)"
    [[ -n "${source_config}" ]] || return 1
    dirname "${source_config}"
}

get_tuned_n_rff() {
    local log_base="$1"
    python3 - <<'PY' "${log_base}"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
candidates = []
for p in base.rglob("hyperparameter_tuning_logs.json"):
    parts = set(p.parts)
    if "Mortality24" in parts and "K29" in parts:
        candidates.append(p)
candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    print("")
    raise SystemExit(0)

data = json.loads(candidates[0].read_text())
x_iters = data.get("x_iters", [])
func_vals = data.get("func_vals", [])
if not x_iters or not func_vals:
    print("")
    raise SystemExit(0)
best_idx = min(range(len(func_vals)), key=lambda i: func_vals[i])
best = x_iters[best_idx]
print(int(best[0]) if best else "")
PY
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_NO_CUDA="1"
export YAIB_TRAINER_DEVICES="${YAIB_TRAINER_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${LOG_BASE:-${SCRIPT_DIR}/../../../yaib_logs}"
MODE="${MODE:-train_eval}"            # train_eval | eval_only
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
    MODE="eval_only"
fi
TRAIN_SUFFIX="${TRAIN_SUFFIX:-_all_data_train}"
EVAL_SUFFIX="${EVAL_SUFFIX:-_all_data_eval}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"    # Optional: existing timestamp run dir for eval_only
SOURCE_DIR="${SOURCE_DIR:-}"          # Optional: dir containing train_config.gin (or its parent tree)

if [[ "${MODE}" != "train_eval" && "${MODE}" != "eval_only" ]]; then
    die "MODE must be train_eval or eval_only (got ${MODE})"
fi
[[ -f "${HOSPITAL_LIST_FILE}" ]] || die "Hospital list file missing: ${HOSPITAL_LIST_FILE}"

mapfile -t hospital_ids < "${HOSPITAL_LIST_FILE}"
[[ "${#hospital_ids[@]}" -gt 0 ]] || die "Hospital list file is empty: ${HOSPITAL_LIST_FILE}"
hospital_ids_subset=("${hospital_ids[@]}")

printf -v joined '%s-' "${hospital_ids_subset[@]}"
joined="${joined%-}"

RUN_DIR=""

if [[ "${MODE}" == "train_eval" ]]; then
    echo "[${SCRIPT_NAME}] Mode: train_eval"
    echo "[${SCRIPT_NAME}] Training once on all hospitals..."
    echo "[${SCRIPT_NAME}] Using shared W&B run id: ${WANDB_RUN_ID}"
    TUNED_N_RFF="$(get_tuned_n_rff "${LOG_BASE}")"
    if [[ -n "${TUNED_N_RFF}" ]]; then
        echo "[${SCRIPT_NAME}] Tuned n_rff_features: ${TUNED_N_RFF}"
    else
        warn "Could not locate tuned n_rff_features; training will use default config."
    fi

    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_RUN_ID="${WANDB_RUN_ID}" WANDB_RESUME="${WANDB_RESUME}" YAIB_WANDB_METRIC_PREFIX="train" \
    run_icu_benchmarks \
        -d "${DATA_DIR}" \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "${joined}" \
        --complete-train \
        -m K29 \
        --cpu \
        -l "${LOG_BASE}" \
        -ls "${TRAIN_SUFFIX}" \
        -wd \
        ${TUNED_N_RFF:+--hyperparams "model/hyperparameter.n_rff_features=${TUNED_N_RFF}"}

    BASE_LOG_DIR="${LOG_BASE}/eicu/Mortality24/K29/train${joined}${TRAIN_SUFFIX}"
    RUN_DIR="$(find_latest_run_dir "${BASE_LOG_DIR}" || true)"
    [[ -n "${RUN_DIR}" ]] || die "Could not locate training run dir under ${BASE_LOG_DIR}"
    echo "[${SCRIPT_NAME}] Using trained run dir: ${RUN_DIR}"
else
    echo "[${SCRIPT_NAME}] Mode: eval_only"
    echo "[${SCRIPT_NAME}] Using shared W&B run id: ${WANDB_RUN_ID}"
    if [[ -n "${TRAIN_RUN_DIR}" ]]; then
        RUN_DIR="${TRAIN_RUN_DIR}"
        echo "[${SCRIPT_NAME}] Using provided TRAIN_RUN_DIR: ${RUN_DIR}"
    fi
fi

if [[ -n "${SOURCE_DIR}" && ! -f "${SOURCE_DIR}/train_config.gin" ]]; then
    SOURCE_DIR="$(find_source_dir_with_config "${SOURCE_DIR}" || true)"
fi

if [[ -z "${SOURCE_DIR}" ]]; then
    [[ -n "${RUN_DIR}" ]] || die "eval_only requires SOURCE_DIR or TRAIN_RUN_DIR."
    if [[ -f "${RUN_DIR}/train_config.gin" && -f "${RUN_DIR}/model.joblib" ]]; then
        SOURCE_DIR="${RUN_DIR}"
    fi
fi

if [[ -z "${SOURCE_DIR}" ]]; then
    [[ -n "${RUN_DIR}" ]] || die "eval_only requires SOURCE_DIR or TRAIN_RUN_DIR."
    SOURCE_DIR="$(find_source_dir_with_config "${RUN_DIR}" || true)"
    [[ -n "${SOURCE_DIR}" ]] || die "Could not find train_config.gin under ${RUN_DIR}"
fi

[[ -f "${SOURCE_DIR}/train_config.gin" ]] || die "SOURCE_DIR does not contain train_config.gin: ${SOURCE_DIR}"
echo "[${SCRIPT_NAME}] Using source dir for eval: ${SOURCE_DIR}"

if [[ -z "${RUN_DIR}" ]]; then
    # source dir is typically <run_dir>/repetition_0/fold_0
    RUN_DIR="$(dirname "$(dirname "${SOURCE_DIR}")")"
fi

declare -a EVAL_DIRS=()
for hospital1 in "${hospital_ids_subset[@]}"; do
    echo "[${SCRIPT_NAME}] Evaluating on hospital ID: ${hospital1}"
    # K29 is non-parametric; we evaluate the trained model without re-training.
    WANDB_PROJECT="${WANDB_PROJECT}" WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_RUN_ID="${WANDB_RUN_ID}" WANDB_RESUME="${WANDB_RESUME}" YAIB_WANDB_METRIC_PREFIX="eval/hospital_${hospital1}" \
    run_icu_benchmarks \
        -d "${DATA_DIR}" \
        -n eicu \
        -t BinaryClassification \
        -tn Mortality24 \
        -hi "${joined}" \
        -hit "${hospital1}" \
        --complete-train \
        -m K29 \
        --cpu \
        -l "${LOG_BASE}" \
        -ls "${EVAL_SUFFIX}" \
        --eval \
        --source-name eicu \
        --source-dir "${SOURCE_DIR}" \
        -wd

    HOSP_LOG_DIR="${LOG_BASE}/eicu/Mortality24/K29/train${joined}-test${hospital1}${EVAL_SUFFIX}"
    EVAL_RUN_DIR="$(find_latest_run_dir "${HOSP_LOG_DIR}" || true)"
    if [[ -n "${EVAL_RUN_DIR}" ]]; then
        EVAL_DIRS+=("${EVAL_RUN_DIR}")
    else
        warn "Could not locate eval run dir for hospital ${hospital1} under ${HOSP_LOG_DIR}"
    fi
done

if [[ "${#EVAL_DIRS[@]}" -gt 0 ]]; then
    echo "[${SCRIPT_NAME}] Computing macro-average metrics across hospitals..."
    python3 - <<'PY' "${RUN_DIR}" "${EVAL_DIRS[@]}"
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("[all_data-K29] WARN: Missing RUN_DIR argument; skipping macro-average.")
    sys.exit(0)

run_dir = Path(sys.argv[1])
eval_dirs = [Path(p) for p in sys.argv[2:]]
metrics = {}
count = 0
for d in eval_dirs:
    f = d / "accumulated_test_metrics.json"
    if not f.exists():
        continue
    data = json.loads(f.read_text())
    avg = data.get("avg", {})
    for k, v in avg.items():
        if isinstance(v, (int, float)):
            metrics.setdefault(k, 0.0)
            metrics[k] += v
    count += 1

if count == 0:
    print("[all_data-K29] WARN: No accumulated_test_metrics.json found; skipping macro-average.")
    sys.exit(0)

macro = {k: v / count for k, v in metrics.items()}
out = {
    "macro_avg_over_hospitals": macro,
    "num_hospitals": count,
}
out_path = run_dir / "overall_macro_metrics.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"[all_data-K29] Wrote macro-average metrics to {out_path}")
PY
else
    warn "No eval dirs collected; skipping macro-average."
fi
