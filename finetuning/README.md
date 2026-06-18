# Fine-Tuning Module

This folder contains the food image classification fine-tuning pipeline.

The module supports two training modes:

- Standard image classification models through `AutoModelForImageClassification`
  - BEiT
  - ViT
  - CLIP vision backbone when the checkpoint supports image classification
- Feature-extraction backbones through `AutoModel` + a custom linear head
  - DINOv3
  - Any other visual backbone with `pooler_output` or CLS-token features

## Files

- `train_classifier.py`: training implementation used as a library.
- `main.py`: only executable entry point for this module.
- `run_finetuning_sbatch.sh`: SLURM script dedicated to fine-tuning jobs.

## Output Organization

When `--output_dir` is not explicitly provided, outputs are organized by model
and epoch count:

```text
model_saved/finetuning/
  microsoft-beit-base-patch16-384/
    epochs_8/
    epochs_12/
    epochs_16/
  openai-clip-vit-base-patch32/
    epochs_8/
    epochs_12/
    epochs_16/
```

Logs follow the same structure:

```text
finetuning/logs/
  runs/
    microsoft-beit-base-patch16-384/
      epochs_8/
      epochs_12/
      epochs_16/
  slurm/
```

Caches are stored per model:

```text
cached_dataset/finetuning/
  microsoft-beit-base-patch16-384/
  openai-clip-vit-base-patch32/
```

The cache stores the raw dataset split, not pre-augmented tensors. Training
augmentations are applied dynamically during training.

## Saved Files For Each Run

Each output folder contains the model and files useful for analysis:

- `train_results.json`: training metrics saved by Hugging Face Trainer.
- `eval_results.json`: final evaluation metrics.
- `run_summary.json`: model name, epochs, paths, start/end time, and duration.
- `trainer_log_history.json`: per-epoch/per-step logs for plotting loss and metrics.
- `trainer_state.json`: Trainer state.

`run_summary.json` includes:

- `train_duration_seconds`
- `train_duration_minutes`
- `train_duration_hours`
- `run_duration_seconds`
- `run_duration_minutes`
- `run_duration_hours`

The `train_*` values measure fine-tuning time only. The `run_*` values include
training plus final evaluation and artifact saving.

## Local Commands

Go into the module folder:

```bash
cd finetuning
```

Install/synchronize the module environment:

```bash
uv sync
```

The module also contains `.python-version` set to `3.12`, so `uv` should select
a PyTorch-compatible Python version for this module.

Show CLI help:

```bash
uv run food-finetune --help
```

You can also run the module with Python from the repository root:

```bash
python finetuning/main.py --help
```

Run BEiT for 8 epochs:

```bash
uv run food-finetune \
  --model_name microsoft/beit-base-patch16-384 \
  --data_dir ../dataset/image/merged \
  --epochs 8 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
```

This keeps the same hyperparameters as the previous BEiT/CLIP runs:
`batch_size=32`, `lr=5e-5`, `weight_decay=0.01`, `warmup_ratio=0.05`.

Run CLIP for 12 epochs:

```bash
uv run food-finetune \
  --model_name openai/clip-vit-base-patch32 \
  --data_dir ../dataset/image/merged \
  --epochs 12 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
```

This is the updated equivalent of the old command:

```bash
python finetuning/train_classifier.py \
  --model_name openai/clip-vit-base-patch32 \
  --data_dir dataset/image/merged \
  --epochs 12 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
```

The new version no longer needs manual `--output_dir` or `--cache_dir` for
standard runs because outputs and caches are automatically separated by model
and epoch count.

Run ViT for 16 epochs:

```bash
uv run food-finetune \
  --model_name google/vit-base-patch16-224 \
  --data_dir ../dataset/image/merged \
  --epochs 16 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
```

Run DINOv3:

```bash
uv run food-finetune \
  --model_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --data_dir ../dataset/image/merged \
  --epochs 8 \
  --batch_size 16 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
```

For a cheaper first experiment with DINOv3, train only the classifier head:

```bash
uv run food-finetune \
  --model_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --data_dir ../dataset/image/merged \
  --epochs 8 \
  --batch_size 16 \
  --lr 1e-4 \
  --freeze_backbone \
  --fp16
```

DINOv3 is a gated Hugging Face model. Accept the model conditions on Hugging Face
and set `HUB_TOKEN` in your environment or `.env` before running it.

If you prefer running from the repository root, use:

```bash
uv --project finetuning run food-finetune --help
```

## SLURM Usage

Run SLURM commands from the repository root:

Submit one job:

```bash
sbatch finetuning/run_finetuning_sbatch.sh
```

By default, the SLURM script uses:

```bash
uv --project finetuning run food-finetune
```

The script sets `UV_CACHE_DIR=finetuning/.uv-cache` by default when `RUNNER=uv`.
It also uses Python `3.12` by default because PyTorch 2.7.1 does not provide
wheels for Python 3.14. Override this with `UV_PYTHON=3.11` or `UV_PYTHON=3.12`
if needed.

To use the previous Conda/Python mode instead:

```bash
sbatch --export=ALL,RUNNER=python finetuning/run_finetuning_sbatch.sh
```

Override model and epochs:

```bash
sbatch --export=ALL,MODEL_KEY=clip,MODEL_NAME=openai/clip-vit-base-patch32,EPOCHS=12,BATCH_SIZE=32,LR=5e-5,WEIGHT_DECAY=0.01,WARMUP_RATIO=0.05 \
  finetuning/run_finetuning_sbatch.sh
```

Submit 8, 12, and 16 epochs for BEiT:

```bash
for EPOCHS in 8 12 16; do
  sbatch --export=ALL,MODEL_KEY=beit,MODEL_NAME=microsoft/beit-base-patch16-384,EPOCHS=${EPOCHS},BATCH_SIZE=32,LR=5e-5,WEIGHT_DECAY=0.01,WARMUP_RATIO=0.05 \
    finetuning/run_finetuning_sbatch.sh
done
```

Submit 8, 12, and 16 epochs for CLIP:

```bash
for EPOCHS in 8 12 16; do
  sbatch --export=ALL,MODEL_KEY=clip,MODEL_NAME=openai/clip-vit-base-patch32,EPOCHS=${EPOCHS},BATCH_SIZE=32,LR=5e-5,WEIGHT_DECAY=0.01,WARMUP_RATIO=0.05 \
    finetuning/run_finetuning_sbatch.sh
done
```

Submit 8, 12, and 16 epochs for ViT:

```bash
for EPOCHS in 8 12 16; do
  sbatch --export=ALL,MODEL_KEY=vit,MODEL_NAME=google/vit-base-patch16-224,EPOCHS=${EPOCHS} \
    finetuning/run_finetuning_sbatch.sh
done
```

Submit DINOv3:

```bash
sbatch --export=ALL,MODEL_KEY=dinov3,MODEL_NAME=facebook/dinov3-vitl16-pretrain-lvd1689m,EPOCHS=8,BATCH_SIZE=16,LR=1e-5 \
  finetuning/run_finetuning_sbatch.sh
```

## Notes

- Use a different `MODEL_KEY` for each model family to make SLURM logs readable.
- Raw SLURM output files are written as `.log` files under
  `finetuning/logs/slurm/`.
- Trainer logs and metrics are organized under
  `finetuning/logs/runs/<model>/epochs_<n>/`.
- Use `--overwrite_cache` or `OVERWRITE_CACHE=1` only when you want to rebuild the cached dataset split.
- If you pass `--output_dir` manually, automatic `model/epochs` output nesting is disabled.
- For papers/reports, describe CLIP here as supervised fine-tuning of the CLIP visual backbone, not full multimodal CLIP fine-tuning.
