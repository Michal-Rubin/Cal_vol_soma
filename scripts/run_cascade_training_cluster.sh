#!/usr/bin/env bash
set -euo pipefail

# Run cascade_training.ipynb non-interactively on an SSH/HPC Linux machine.
# Do NOT put passwords in this file. Log in with:
#   ssh michal.rubin1@loginserver.elsc.huji.ac.il
# Prefer SSH keys or the cluster password prompt.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_cascade_training_cluster.sh [options]

Main options:
  --repo DIR                 Path to cal_vol_soma repo on the cluster.
  --notebook FILE            Notebook to execute. Default: $REPO_DIR/cascade_training.ipynb
  --cascade-repo DIR         Path to CASCADE repo containing cascade2p.
  --csv FILE                 CSV/database with cell links used to build CASCADE .mat files.
  --base-out DIR             Root folder for prepared variants/splits.
  --model-out DIR            Root folder for trained model folders.
  --report-out DIR           Root folder for CSV reports and summary figures.
  --model-family NAME        Prefix/name for model outputs.
  --run-name NAME            Name used only for executed-notebook/log filenames.
  --variants LIST            Comma-separated variants to train/evaluate.
  --noise-levels LIST        Comma-separated noise levels. Default: 2,4,6,8,10,14,19
  --ensemble-size N          Ensemble size. Default: 5
  --kernel NAME              Jupyter kernel name. Default: python3
  --conda-env NAME           Optional conda env to activate before running.
  --force-retrain            Retrain even if model files already exist.
  --force-reeval             Recompute evaluation CSVs even if they already exist.
  --dry-run                  Print resolved settings and exit.

Example:
  bash scripts/run_cascade_training_cluster.sh \
    --repo /home/michal.rubin1/cal_vol_soma \
    --cascade-repo /home/michal.rubin1/cal_vol_soma/_cascade_ref_repo \
    --csv /data/Michal_Rubin/Dendrites/PyrLowFR.csv \
    --base-out /data/Michal_Rubin/cascade_training \
    --model-out /data/Michal_Rubin/cascade_training/models_leave_one_cell \
    --report-out /data/Michal_Rubin/cascade_training/reports_leave_one_cell \
    --model-family GC8m_EX_30hz_smothing50ms_CA1 \
    --variants f01_p8 \
    --conda-env Cascade38
EOF
}

# Defaults can be overridden by CLI flags or environment variables.
REPO_DIR="${REPO_DIR:-$(pwd)}"
NOTEBOOK="${NOTEBOOK:-}"
CASCADE_REPO="${CASCADE_REPO:-}"
CSV_LINKS_PATH="${CSV_LINKS_PATH:-}"
BASE_OUT="${CASCADE_TRAINING_BASE_OUT:-${BASE_OUT:-}}"
MODEL_OUTPUT_ROOT="${CASCADE_MODEL_OUTPUT_ROOT:-${MODEL_OUTPUT_ROOT:-}}"
REPORT_OUTPUT_ROOT="${CASCADE_REPORT_OUTPUT_ROOT:-${REPORT_OUTPUT_ROOT:-}}"
MODEL_FAMILY="${CASCADE_MODEL_FAMILY:-GC8m_EX_30hz_smothing50ms_CA1}"
RUN_NAME="${RUN_NAME:-cascade_training}"
VARIANTS_TO_RUN="${CASCADE_VARIANTS_TO_RUN:-f01_p8,f02_p10,f03_p15,f04_p20,f05_p8_rw6s,f06_p8_rw15s}"
NOISE_LEVELS="${CASCADE_NOISE_LEVELS:-2,4,6,8,10,14,19}"
ENSEMBLE_SIZE="${CASCADE_ENSEMBLE_SIZE:-5}"
SAMPLING_RATE_HZ="${CASCADE_SAMPLING_RATE_HZ:-30.0}"
SMOOTHING_S="${CASCADE_SMOOTHING_S:-0.05}"
KERNEL_NAME="${JUPYTER_KERNEL:-python3}"
CONDA_ENV="${CONDA_ENV:-}"
FORCE_RETRAIN="${CASCADE_FORCE_RETRAIN:-0}"
FORCE_REEVAL="${CASCADE_FORCE_REEVAL:-0}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_DIR="$2"; shift 2 ;;
    --notebook) NOTEBOOK="$2"; shift 2 ;;
    --cascade-repo) CASCADE_REPO="$2"; shift 2 ;;
    --csv|--database) CSV_LINKS_PATH="$2"; shift 2 ;;
    --base-out) BASE_OUT="$2"; shift 2 ;;
    --model-out) MODEL_OUTPUT_ROOT="$2"; shift 2 ;;
    --report-out|--output-path) REPORT_OUTPUT_ROOT="$2"; shift 2 ;;
    --model-family|--output-name) MODEL_FAMILY="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --variants) VARIANTS_TO_RUN="$2"; shift 2 ;;
    --noise-levels) NOISE_LEVELS="$2"; shift 2 ;;
    --ensemble-size) ENSEMBLE_SIZE="$2"; shift 2 ;;
    --sampling-rate-hz) SAMPLING_RATE_HZ="$2"; shift 2 ;;
    --smoothing-s) SMOOTHING_S="$2"; shift 2 ;;
    --kernel) KERNEL_NAME="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --force-retrain) FORCE_RETRAIN=1; shift ;;
    --force-reeval) FORCE_REEVAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$NOTEBOOK" ]]; then
  NOTEBOOK="$REPO_DIR/cascade_training.ipynb"
fi
if [[ -z "$CASCADE_REPO" ]]; then
  CASCADE_REPO="$REPO_DIR/_cascade_ref_repo"
fi
if [[ -z "$CSV_LINKS_PATH" ]]; then
  echo "ERROR: set --csv /path/to/link_table.csv" >&2
  exit 2
fi
if [[ -z "$BASE_OUT" ]]; then
  echo "ERROR: set --base-out /path/to/cascade_training_root" >&2
  exit 2
fi
if [[ -z "$MODEL_OUTPUT_ROOT" ]]; then
  MODEL_OUTPUT_ROOT="$BASE_OUT/models_leave_one_cell"
fi
if [[ -z "$REPORT_OUTPUT_ROOT" ]]; then
  REPORT_OUTPUT_ROOT="$BASE_OUT/reports_leave_one_cell"
fi

mkdir -p "$MODEL_OUTPUT_ROOT" "$REPORT_OUTPUT_ROOT" "$REPORT_OUTPUT_ROOT/notebooks" "$REPORT_OUTPUT_ROOT/logs"

export CASCADE_TRAINING_CSV_LINKS_PATH="$CSV_LINKS_PATH"
export CASCADE_TRAINING_BASE_OUT="$BASE_OUT"
export CASCADE_REPO="$CASCADE_REPO"
export CASCADE_MODEL_OUTPUT_ROOT="$MODEL_OUTPUT_ROOT"
export CASCADE_REPORT_OUTPUT_ROOT="$REPORT_OUTPUT_ROOT"
export CASCADE_MODEL_FAMILY="$MODEL_FAMILY"
export CASCADE_VARIANTS_TO_RUN="$VARIANTS_TO_RUN"
export CASCADE_NOISE_LEVELS="$NOISE_LEVELS"
export CASCADE_ENSEMBLE_SIZE="$ENSEMBLE_SIZE"
export CASCADE_SAMPLING_RATE_HZ="$SAMPLING_RATE_HZ"
export CASCADE_SMOOTHING_S="$SMOOTHING_S"
export CASCADE_FORCE_RETRAIN="$FORCE_RETRAIN"
export CASCADE_FORCE_REEVAL="$FORCE_REEVAL"

STAMP="$(date +%Y%m%d_%H%M%S)"
EXEC_NOTEBOOK="$REPORT_OUTPUT_ROOT/notebooks/${RUN_NAME}_${STAMP}.ipynb"
LOG_FILE="$REPORT_OUTPUT_ROOT/logs/${RUN_NAME}_${STAMP}.log"

cat <<EOF
Resolved CASCADE training run:
  repo:          $REPO_DIR
  notebook:      $NOTEBOOK
  cascade repo:  $CASCADE_REPO
  csv/database:  $CSV_LINKS_PATH
  base out:      $BASE_OUT
  model out:     $MODEL_OUTPUT_ROOT
  report out:    $REPORT_OUTPUT_ROOT
  model family:  $MODEL_FAMILY
  variants:      $VARIANTS_TO_RUN
  noise levels:  $NOISE_LEVELS
  ensemble size: $ENSEMBLE_SIZE
  kernel:        $KERNEL_NAME
  force retrain: $FORCE_RETRAIN
  force reeval:  $FORCE_REEVAL
  executed nb:   $EXEC_NOTEBOOK
  log:           $LOG_FILE
EOF

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

cd "$REPO_DIR"

if [[ -n "$CONDA_ENV" ]]; then
  # Works on most conda installations; otherwise load/activate conda before running this script.
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "ERROR: conda not found, but --conda-env was set to '$CONDA_ENV'" >&2
    exit 2
  fi
fi

python - <<'PY'
import importlib.util
missing = [m for m in ['jupyter', 'nbconvert'] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('Missing Python packages: ' + ', '.join(missing) + '. Install jupyter/nbconvert in this environment.')
PY

# Execute all notebook cells. The notebook reads the exported CASCADE_* variables above.
jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output "$EXEC_NOTEBOOK" \
  --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
  --ExecutePreprocessor.timeout=-1 \
  2>&1 | tee "$LOG_FILE"

echo "Finished CASCADE training. Executed notebook: $EXEC_NOTEBOOK"
echo "Models:  $MODEL_OUTPUT_ROOT"
echo "Reports: $REPORT_OUTPUT_ROOT"
