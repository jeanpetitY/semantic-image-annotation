#!/bin/zsh

#SBATCH --job-name=import
#SBATCH --output=import.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1


# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python finetuning/recipe/train_classifier.py \
  --model_name microsoft/beit-base-patch16-384 \
  --data_dir dataset/image/merged \
  --output_dir ./model_saved/beit-food-389-v2 \
  --epochs 8 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
