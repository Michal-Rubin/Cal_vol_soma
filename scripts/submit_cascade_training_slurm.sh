#!/usr/bin/env bash
set -euo pipefail

# Submit cascade_training to a Slurm cluster.
# This wrapper requests cluster resources, then runs scripts/run_cascade_training_cluster.sh.
# Use from the cluster login node, not from Windows.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_cascade_training_slurm.sh [slurm options] -- [run_cascade_training_cluster.sh options]

Slurm resource options:
  --job-name NAME        Slurm job name. Default: cascade_training
  --partition NAME      Slurm partition/queue, e.g. gpu or long. Optional.
  --account NAME        Slurm account/project. Optional.
  --nodes N             Number of nodes. Default: 1
  --ntasks N            Number of tasks. Default: 1
  --cpus-per-task N     CPU cores for the notebook process. Default: 8
  --mem SIZE            Memory, e.g. 64G, 128G. Default: 64G
  --time HH:MM:SS       Walltime. Default: 48:00:00
  --gpu N               Request N GPUs using --gres=gpu:N. Default: 0
  --gpu-type TYPE       Request GPUs of type TYPE using --gres=gpu:TYPE:N. Optional.
  --constraint TEXT     Slurm node constraint. Optional.
  --qos NAME            Slurm QoS. Optional.
  --exclude NODES       Slurm exclude list. Optional.
  --dry-run             Print sbatch command without submitting.

Everything after -- is passed directly to run_cascade_training_cluster.sh.

Examples:
  # CPU job, 1 node, 16 CPUs, 128 GB RAM
  bash scripts/submit_cascade_training_slurm.sh \
    --nodes 1 --cpus-per-task 16 --mem 128G --time 72:00:00 \
    -- --repo /home/michal.rubin1/cal_vol_soma \
       --csv /data/Michal_Rubin/Dendrites/PyrLowFR.csv \
       --base-out /data/Michal_Rubin/cascade_training \
       --conda-env Cascade38

  # GPU job, 1 GPU
  bash scripts/submit_cascade_training_slurm.sh \
    --partition gpu --gpu 1 --cpus-per-task 12 --mem 96G --time 72:00:00 \
    -- --repo /home/michal.rubin1/cal_vol_soma \
       --csv /data/Michal_Rubin/Dendrites/PyrLowFR.csv \
       --base-out /data/Michal_Rubin/cascade_training \
       --conda-env Cascade38

  # GPU type if your cluster requires it, e.g. a6000, rtx6000, v100, a100
  bash scripts/submit_cascade_training_slurm.sh \
    --partition gpu --gpu 1 --gpu-type a100 --mem 128G --time 48:00:00 \
    -- --repo /home/michal.rubin1/cal_vol_soma \
       --csv /data/Michal_Rubin/Dendrites/PyrLowFR.csv \
       --base-out /data/Michal_Rubin/cascade_training \
       --conda-env Cascade38
EOF
}

JOB_NAME="cascade_training"
PARTITION=""
ACCOUNT=""
NODES="1"
NTASKS="1"
CPUS_PER_TASK="8"
MEM="64G"
TIME="48:00:00"
GPU_COUNT="0"
GPU_TYPE=""
CONSTRAINT=""
QOS=""
EXCLUDE=""
DRY_RUN=0

SBATCH_ARGS=()
RUN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --ntasks) NTASKS="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --gpu|--gpus) GPU_COUNT="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --exclude) EXCLUDE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; RUN_ARGS=("$@"); break ;;
    *) echo "Unknown Slurm option before --: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ${#RUN_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: pass run_cascade_training_cluster.sh options after --" >&2
  usage
  exit 2
fi

SBATCH_ARGS+=(--job-name="$JOB_NAME")
SBATCH_ARGS+=(--nodes="$NODES")
SBATCH_ARGS+=(--ntasks="$NTASKS")
SBATCH_ARGS+=(--cpus-per-task="$CPUS_PER_TASK")
SBATCH_ARGS+=(--mem="$MEM")
SBATCH_ARGS+=(--time="$TIME")
SBATCH_ARGS+=(--output="slurm-%x-%j.out")
SBATCH_ARGS+=(--error="slurm-%x-%j.err")

[[ -n "$PARTITION" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "$CONSTRAINT" ]] && SBATCH_ARGS+=(--constraint="$CONSTRAINT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos="$QOS")
[[ -n "$EXCLUDE" ]] && SBATCH_ARGS+=(--exclude="$EXCLUDE")

if [[ "$GPU_COUNT" != "0" ]]; then
  if [[ -n "$GPU_TYPE" ]]; then
    SBATCH_ARGS+=(--gres="gpu:${GPU_TYPE}:${GPU_COUNT}")
  else
    SBATCH_ARGS+=(--gres="gpu:${GPU_COUNT}")
  fi
fi

CMD=(sbatch "${SBATCH_ARGS[@]}" --wrap "bash scripts/run_cascade_training_cluster.sh ${RUN_ARGS[*]}")

printf 'Submitting command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. This wrapper assumes Slurm. Run 'which sbatch' on the cluster." >&2
  exit 2
fi

"${CMD[@]}"
