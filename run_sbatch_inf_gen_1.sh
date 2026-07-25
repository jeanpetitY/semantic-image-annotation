#!/bin/bash

#SBATCH --job-name=exp_dinov3_falcon
#SBATCH --chdir=/nfs/home/jeanpetityvelosb/project/python/Dataset
#SBATCH --output=/nfs/home/jeanpetityvelosb/project/python/Dataset/logs/inference/generation/dinov3/no_rag/v2/%x_%j.log
#SBATCH --error=/nfs/home/jeanpetityvelosb/project/python/Dataset/logs/inference/generation/dinov3/no_rag/v2/%x_%j.err
#SBATCH --partition=p_12G
#SBATCH --mem=48G
#SBATCH --gres=gpu:t2080ti:3
#SBATCH --ntasks=1

set -euo pipefail


echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python -u inference/nutrient_generator.py \
    --mode no_rag \
    --selective \
    --input-file dataset/multimodal/not_merged/test/food101_test.json \
    --output-file results/falcon/dinov3/ablation/20/food101.csv \
    --index-name icml-paper \
    --ablation \
    --rag-ratio 0.2

echo "Finished at: $(date)"
