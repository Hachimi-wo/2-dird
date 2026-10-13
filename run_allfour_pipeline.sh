#!/usr/bin/env bash
set -euo pipefail

# Run the all-four-labels-zero negative pipeline for all endpoints.
# The split script in this folder uses the all-four-zero policy by default.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_JOBS="${N_JOBS:-4}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT="$N_JOBS"

cd "$PROJECT_ROOT"

labels=(Label_ILD Label_Edema Label_Pleural Label_Embolism)

for label in "${labels[@]}"; do
    echo "============================================"
    echo ">>> Starting all-four-zero pipeline: ${label}"
    echo ">>> $(date)"
    echo "============================================"

    echo "--- Step 1/4: Data splitting ---"
    "$PYTHON_BIN" ml_experiments/01_data_splitting.py \
        --input both \
        --label-col "$label" \
        --n-seeds 5

    echo "--- Step 2/4: Hyperparameter tuning ---"
    "$PYTHON_BIN" ml_experiments/02_hyperparameter_tuning.py \
        --metric auroc \
        --label-col "$label"

    echo "--- Step 3/4: Model refitting and evaluation ---"
    "$PYTHON_BIN" ml_experiments/03_model_refitting_and_evaluation.py \
        --metric auroc \
        --label-col "$label"

    echo "--- Step 4/4: Statistical analysis ---"
    "$PYTHON_BIN" ml_experiments/04_statistical_analysis.py \
        --label-col "$label"

    echo "============================================"
    echo ">>> DONE: ${label} — $(date)"
    echo "============================================"
    echo
 done

echo "ALL FOUR LABELS COMPLETE"
