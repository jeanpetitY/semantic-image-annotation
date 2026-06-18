#!/bin/zsh

#SBATCH --job-name=exp_clip_falcon
#SBATCH --output=logs/inference/generation/clip/no_rag/inf_clipFalcon_merged.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --ntasks=1

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python inference/nutrient_generator.py --mode no_rag --selective --input-file dataset/multimodal/merged/merged_test.json --output-file results/falcon/clip/no_rag/merged.csv --index-name icml-paper