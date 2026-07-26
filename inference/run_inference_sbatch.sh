#!/bin/bash

#SBATCH --job-name=inference_pipeline
#SBATCH --output=logs/inference/%x_%j.log
#SBATCH --error=logs/inference/%x_%j.err
#SBATCH --partition=p_48G
#SBATCH --mem=48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/inference}"
mkdir -p "${LOG_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"

cd "${PROJECT_ROOT}"

if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda activate "${CONDA_ENV:-rag_env}"

TASK="${TASK:-generate}"

if [ "$TASK" = "classify" ]; then
    python -u inference/main.py classify "$@"
else
    python -u inference/main.py generate "$@"
fi

echo "Finished at: $(date)"
