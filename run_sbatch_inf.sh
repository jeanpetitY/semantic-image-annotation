#!/bin/zsh

#SBATCH --job-name=clip_rec_food
#SBATCH --output=inf_clipReco8_1.log
#SBATCH --partition=p_12G
#SBATCH --gres=gpu:t2080ti:1
#SBATCH --ntasks=1

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python inference/food_classifier.py