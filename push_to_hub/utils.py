import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "model_saved/finetuning/facebook-dinov3-vitl16-pretrain-lvd1689m/epochs_8"
)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def validate_model_dir(model_dir: Path) -> None:
    required_files = [
        "backbone/config.json",
        "backbone/model.safetensors",
        "classifier.pt",
        "classifier_config.json",
        "preprocessor_config.json",
    ]

    missing_files = [
        relative_path
        for relative_path in required_files
        if not (model_dir / relative_path).is_file()
    ]

    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Model directory is incomplete: {model_dir}\nMissing files:\n{missing}"
        )


def push_model_to_hub() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Set HUB_TOKEN or HF_TOKEN in .env")

    repo_id = require_env("REPO_NAME")
    model_dir = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_dir = model_dir.resolve()

    private = os.getenv("HF_PRIVATE", "0") == "1"

    validate_model_dir(model_dir)

    login(token=token)
    create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        repo_type="model",
        commit_message=f"Upload DINOv3 food classifier from {model_dir.name}",
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

    print(f"Uploaded {model_dir} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    push_model_to_hub()
