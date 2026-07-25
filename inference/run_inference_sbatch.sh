#!/bin/bash

#SBATCH --job-name=inference_pipeline
#SBATCH --chdir=/nfs/home/jeanpetityvelosb/project/python/Dataset
#SBATCH --output=/nfs/home/jeanpetityvelosb/project/python/Dataset/logs/inference/%x_%j.log
#SBATCH --error=/nfs/home/jeanpetityvelosb/project/python/Dataset/logs/inference/%x_%j.err
#SBATCH --partition=p_48G
#SBATCH --mem=48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --ntasks=1

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"

source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

TASK="${TASK:-generate}"

if [ "$TASK" = "classify" ]; then
    python -u inference/main.py classify "$@"
else
    python -u inference/main.py generate "$@"
fi

echo "Finished at: $(date)"
