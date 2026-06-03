#!/usr/bin/env bash
set -euo pipefail

# Run this file AFTER logging into the ELSC login server:
#   ssh michal.rubin1@loginserver.elsc.huji.ac.il
#   cd /path/to/cal_vol_soma
#   bash run_cascade_training.sh

# Edit these cluster paths before submitting.
REPO_DIR="/path/to/cal_vol_soma"
CSV_PATH="/path/to/PyrLowFR.csv"
BASE_OUT="/path/to/cascade_training"
MODEL_OUT="${BASE_OUT}/models_leave_one_cell"
REPORT_OUT="${BASE_OUT}/reports_leave_one_cell"
CONDA_ENV="Cascade38"

# First test only one variant. After it works, change this to all variants:
# VARIANTS="f01_p8,f02_p10,f03_p15,f04_p20,f05_p8_rw6s,f06_p8_rw15s"
VARIANTS="f01_p8"

bash scripts/submit_cascade_training_slurm.sh \
  --partition gpu.q \
  --gpu 1 \
  --nodes 1 \
  --cpus-per-task 16 \
  --mem 128G \
  --time 6-00:00:00 \
  -- \
  --repo "${REPO_DIR}" \
  --cascade-repo "${REPO_DIR}/_cascade_ref_repo" \
  --csv "${CSV_PATH}" \
  --base-out "${BASE_OUT}" \
  --model-out "${MODEL_OUT}" \
  --report-out "${REPORT_OUT}" \
  --model-family GC8m_EX_30hz_smothing50ms_CA1 \
  --variants "${VARIANTS}" \
  --noise-levels 2,4,6,8,10,14,19 \
  --ensemble-size 5 \
  --conda-env "${CONDA_ENV}"
