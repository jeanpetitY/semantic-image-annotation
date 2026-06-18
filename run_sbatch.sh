#!/bin/zsh

#SBATCH --job-name=clip-finetune
#SBATCH --output=clip_16_epoch.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1


# Source Conda
source /nfs/home/jeanpetityvelosb/miniconda3/etc/profile.d/conda.sh
conda activate rag_env

python finetuning/train_classifier.py \
  --model_name "openai/clip-vit-base-patch32" \
  --data_dir dataset/image/merged \
  --output_dir ./model_saved/clip12/clip-food-389 \
  --epochs 12 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --cache_dir ./cached_dataset/clip-vit-base-patch32 \
  --fp16
