# Push To Hub Module

This module uploads a fine-tuned model directory to the Hugging Face Hub.

It is intended for the final publication step after training and local
validation have completed.

## Files

- `utils.py`: upload implementation
- `main.py`: executable entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12** — this module includes a `.python-version` file pinned to `3.12`
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Reproducibility At A Glance

1. Enter the module directory:

```bash
cd push_to_hub
```

2. Recreate the module environment:

```bash
uv sync
```

3. Run the documented CLI:

```bash
uv run food-push-to-hub --help
```

If exact dependency replay is required and a lockfile is present, prefer:

```bash
uv sync --frozen
```

## Environment Variables

Required:

- `HUB_TOKEN` or `HF_TOKEN`
- `REPO_NAME`

Optional:

- `MODEL_PATH`
- `HF_PRIVATE`

## Input Contract

The module automatically detects and uploads the following fine-tuned model
formats:

1. Standard Hugging Face image-classification checkpoint used by CLIP and BEiT:

```text
<model_dir>/
  config.json
  model.safetensors
  preprocessor_config.json
```

2. Custom DINOv3 checkpoint used in this repository:

```text
<model_dir>/
  backbone/
    config.json
    model.safetensors
  classifier.pt
  classifier_config.json
  preprocessor_config.json
```

If `MODEL_PATH` is not provided, the module falls back to the local DINOv3
reference checkpoint.

## Usage

From inside the module directory:

```bash
REPO_NAME=org/example-model \
MODEL_PATH=../model_saved/clip/run\ 1:\ 8\ epochs \
HUB_TOKEN=your_token \
uv run food-push-to-hub
```

From the repository root:

```bash
REPO_NAME=org/example-model \
MODEL_PATH=model_saved/beit/run\ 1:\ 8\ epochs \
HUB_TOKEN=your_token \
uv --project push_to_hub run food-push-to-hub
```

Custom DINOv3 example:

```bash
REPO_NAME=org/example-dinov3-model \
MODEL_PATH=model_saved/finetuning/facebook-dinov3-vitl16-pretrain-lvd1689m/run\ 1:\ ep_8_0 \
uv --project push_to_hub run food-push-to-hub
```

## Reproducibility Notes

- Publication is deterministic with respect to the local model directory being
  uploaded.
- Exact replay requires the same model files, the same repository name, and the
  same authentication scope on the Hugging Face Hub.
- The module ignores transient training artifacts such as optimizer states and
  checkpoint folders during upload.
- Supported publication targets include CLIP, BEiT, and DINOv3 checkpoints
  stored in the repository layouts described above.
