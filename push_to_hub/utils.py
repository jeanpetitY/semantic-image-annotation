"""Publish fine-tuned food models as reproducibility artifacts.

Paper reference:
- Experimental pipeline where CLIP, BEiT, and DINOv3 checkpoints are released
  alongside the code and semantified data.
"""

import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "model_saved/finetuning/example-dinov3-backbone/run_3_epoch_8"
)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def detect_model_format(model_dir: Path) -> str:
    # Paper reproducibility release: support both standard Hugging Face exports
    # and the custom DINOv3 backbone-plus-classifier serialization.
    standard_required_files = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ]
    dinov3_required_files = [
        "backbone/config.json",
        "backbone/model.safetensors",
        "classifier.pt",
        "classifier_config.json",
        "preprocessor_config.json",
    ]

    if all((model_dir / relative_path).is_file() for relative_path in standard_required_files):
        return "hf_standard"

    if all((model_dir / relative_path).is_file() for relative_path in dinov3_required_files):
        return "dinov3_custom"

    missing_standard = [
        relative_path
        for relative_path in standard_required_files
        if not (model_dir / relative_path).is_file()
    ]
    missing_dinov3 = [
        relative_path
        for relative_path in dinov3_required_files
        if not (model_dir / relative_path).is_file()
    ]
    missing_standard_text = "\n".join(f"- {path}" for path in missing_standard)
    missing_dinov3_text = "\n".join(f"- {path}" for path in missing_dinov3)
    raise FileNotFoundError(
        "Model directory does not match a supported upload format: "
        f"{model_dir}\n\n"
        "Expected one of:\n"
        "1. Standard Hugging Face image-classification checkpoint (used by CLIP and BEiT)\n"
        f"{missing_standard_text}\n\n"
        "2. Custom DINOv3 checkpoint\n"
        f"{missing_dinov3_text}"
    )


def validate_model_dir(model_dir: Path) -> str:
    return detect_model_format(model_dir)


def push_model_to_hub(
    model_path: str | None = None,
    repo_name: str | None = None,
    private: bool | None = None,
) -> None:
    # Paper supplementary material: publish only the stable model artifacts,
    # excluding transient optimizer/checkpoint files from training.
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Set HUB_TOKEN or HF_TOKEN in .env")

    repo_id = repo_name or require_env("REPO_NAME")
    model_dir = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_dir = model_dir.resolve()

    if private is None:
        private = os.getenv("HF_PRIVATE", "0") == "1"

    model_format = validate_model_dir(model_dir)

    login(token=token)
    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message=f"Upload {model_format} food classifier from {model_dir.name}",
        ignore_patterns=[
            "checkpoint-*",
            "*/optimizer.pt",
            "*/scheduler.pt",
            "*/scaler.pt",
            "*/rng_state.pth",
            "*/training_args.bin",
            "training_args.bin",
        ],
    )

    print(
        f"Detected format: {model_format}. "
        f"Uploaded {model_dir} to the configured Hugging Face repository '{repo_id}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a fine-tuned model directory to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model directory to upload. Falls back to MODEL_PATH or the default local path.",
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Target Hugging Face repository name. Falls back to REPO_NAME.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or use a private repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_model_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
    )


if __name__ == "__main__":
    main()
