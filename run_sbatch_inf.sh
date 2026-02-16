#!/bin/zsh

#SBATCH --job-name=semantic_inf_food
#SBATCH --output=semantic.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1
#SBATCH --ntasks=1

# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

# python inference/food_classifier.py
python inference/nutrient_generator.py