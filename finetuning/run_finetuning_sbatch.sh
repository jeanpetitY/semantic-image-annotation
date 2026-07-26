#!/bin/zsh

#SBATCH --job-name=exp_dinov3
#SBATCH --output=finetuning/logs/slurm/%x_%j.log
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:a3090:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

RUNNER="${RUNNER:-uv}"

if [[ "${RUNNER}" == "uv" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-finetuning/.uv-cache}"
  unset VIRTUAL_ENV
fi

if [[ "${RUNNER}" == "python" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate "${CONDA_ENV:-rag_env}"
fi

MODEL_KEY="${MODEL_KEY:-dinov3}"
MODEL_NAME="${MODEL_NAME:-org/example-dinov3-backbone}"
DATA_DIR="${DATA_DIR:-dataset/image/merged}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./model_saved/finetuning}"
CACHE_ROOT="${CACHE_ROOT:-./cached_dataset/finetuning}"
LOGS_ROOT="${LOGS_ROOT:-./finetuning/logs/runs}"

mkdir -p finetuning/logs/slurm

EXTRA_ARGS=()

if [[ "${FP16:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--fp16)
fi

if [[ "${FREEZE_BACKBONE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze_backbone)
fi

if [[ "${BACKBONE_CLASSIFIER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--backbone_classifier)
fi

if [[ "${OVERWRITE_CACHE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite_cache)
fi

COMMON_ARGS=(
  --model_name "${MODEL_NAME}"
  --data_dir "${DATA_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --cache_root "${CACHE_ROOT}"
  --logs_root "${LOGS_ROOT}"
  --run_name "${MODEL_KEY}_epochs_${EPOCHS}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${RUNNER}" == "uv" ]]; then
  uv --project finetuning run --python "${UV_PYTHON:-3.12}" food-finetune "${COMMON_ARGS[@]}"
elif [[ "${RUNNER}" == "python" ]]; then
  python finetuning/main.py "${COMMON_ARGS[@]}"
else
  echo "Unsupported RUNNER='${RUNNER}'. Use RUNNER=uv or RUNNER=python." >&2
  exit 1
fi
