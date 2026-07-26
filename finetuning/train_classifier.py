"""Fine-tune vision backbones for food recognition.

Paper reference:
- Experimental section reporting CLIP, BEiT, and DINOv3 fine-tuning on the
  semantified food-image collections.
"""

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from dotenv import load_dotenv
from huggingface_hub import login

from datasets import load_dataset, DatasetDict, load_from_disk

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    CenterCrop,
    ToTensor,
)

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import f1_score


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Ensure reproducibility by fixing random seeds.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str) -> str:
    """
    Convert a model name or run name into a filesystem-safe identifier.
    """

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")


# -------------------------------------------------------------------
# Main Trainer Class
# -------------------------------------------------------------------

class BackboneImageClassifier(nn.Module):
    """
    Generic image classifier for feature-extraction backbones such as DINOv3.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        hidden_size = self._resolve_hidden_size(backbone)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        if freeze_backbone:
            # Paper ablation: this supports the classifier-head-only setting
            # alongside the fully fine-tuned backbone configuration.
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _resolve_hidden_size(backbone: nn.Module) -> int:
        config = getattr(backbone, "config", None)

        for attr in ("hidden_size", "embed_dim", "projection_dim"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value

        hidden_sizes = getattr(config, "hidden_sizes", None)
        if hidden_sizes and isinstance(hidden_sizes[-1], int):
            return hidden_sizes[-1]

        raise ValueError(
            "Could not infer backbone hidden size. Add the model-specific "
            "hidden dimension to BackboneImageClassifier._resolve_hidden_size."
        )

    @staticmethod
    def _extract_features(outputs):
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state[:, 0]

        if isinstance(outputs, tuple):
            tensor_output = outputs[0]
            if tensor_output.ndim == 3:
                return tensor_output[:, 0]
            return tensor_output

        raise ValueError("Backbone output does not contain usable image features.")

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        features = self._extract_features(outputs)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }

    def save_pretrained(self, output_dir: str, **kwargs) -> None:
        os.makedirs(output_dir, exist_ok=True)

        backbone_dir = os.path.join(output_dir, "backbone")
        self.backbone.save_pretrained(backbone_dir)

        torch.save(
            self.classifier.state_dict(),
            os.path.join(output_dir, "classifier.pt"),
        )

        with open(
            os.path.join(output_dir, "classifier_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({
                "num_labels": self.num_labels,
                "id2label": self.id2label,
                "label2id": self.label2id,
                "backbone_dir": "backbone",
            }, f, indent=2)


class ImageClassificationTrainer:
    """
    A high-level trainer class for fine-tuning
    vision transformer models on image datasets.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the training pipeline.

        Args:
            args (argparse.Namespace): Parsed command line arguments.
        """

        self.args = args
        self._validate_args()
        self._setup_run_paths()

        self.dataset: DatasetDict = None
        self.model = None
        self.processor = None

        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

        self._setup_environment()
        set_seed(self.args.seed)

    def _validate_args(self) -> None:
        """
        Validate user-provided arguments before launching a long job.
        """

        if not 0 < self.args.val_ratio < 1:
            raise ValueError("--val_ratio must be between 0 and 1.")

        if self.args.batch_size <= 0:
            raise ValueError("--batch_size must be greater than 0.")

        if self.args.epochs <= 0:
            raise ValueError("--epochs must be greater than 0.")

    def _setup_run_paths(self) -> None:
        """
        Derive organized output, cache, and log paths for the run.
        """

        model_slug = slugify(self.args.model_name)
        epoch_slug = f"epochs_{self.args.epochs}"
        run_name = self.args.run_name or f"{model_slug}_{epoch_slug}"

        self.args.run_name = slugify(run_name)
        self.args.model_slug = model_slug

        if self.args.output_dir == "./model_saved":
            self.args.output_dir = os.path.join(
                self.args.output_root,
                model_slug,
                epoch_slug,
            )

        if self.args.cache_dir == "./cached_dataset":
            self.args.cache_dir = os.path.join(
                self.args.cache_root,
                model_slug,
            )

        if self.args.logging_dir is None:
            self.args.logging_dir = os.path.join(
                self.args.logs_root,
                model_slug,
                epoch_slug,
            )

        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(self.args.logging_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------

    def _setup_environment(self) -> None:
        """
        Load environment variables and authenticate with Hugging Face Hub
        when a token is available.
        """

        load_dotenv()

        token = os.getenv("HUB_TOKEN")

        if token:
            login(token=token)
        else:
            print("HUB_TOKEN not found; continuing without Hugging Face login.")

    # -------------------------------------------------------------------
    # Dataset Handling
    # -------------------------------------------------------------------

    def _infer_splits(self) -> Tuple[str, Optional[str]]:
        """
        Infer training and validation directories.

        Returns:
            Tuple[str, str]: Train and validation paths.
        """

        train_dir = os.path.join(self.args.data_dir, "train")
        val_dir = os.path.join(self.args.data_dir, "val")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        if os.path.isdir(val_dir):
            return train_dir, val_dir

        return train_dir, None

    def _cache_metadata_path(self) -> str:
        return os.path.join(self.args.cache_dir, "metadata.json")

    def _cache_metadata(self, train_dir: str, val_dir: str) -> Dict[str, object]:
        data_dir = os.path.abspath(self.args.data_dir)
        data_id = hashlib.sha1(data_dir.encode("utf-8")).hexdigest()[:12]

        return {
            "cache_version": 2,
            "cache_type": "raw_imagefolder_dataset",
            "data_id": data_id,
            "data_dir": data_dir,
            "train_dir": os.path.abspath(train_dir),
            "val_dir": os.path.abspath(val_dir) if val_dir else None,
            "val_ratio": None if val_dir else self.args.val_ratio,
            "seed": self.args.seed,
        }

    def _load_cached_dataset(
        self,
        expected_metadata: Dict[str, object]
    ) -> bool:
        if not os.path.exists(self.args.cache_dir):
            return False

        metadata_path = self._cache_metadata_path()

        if not os.path.isfile(metadata_path):
            if self.args.overwrite_cache:
                shutil.rmtree(self.args.cache_dir)
                return False

            raise ValueError(
                f"Cache exists without metadata: {self.args.cache_dir}. "
                "Use --overwrite_cache or choose another --cache_dir."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            cached_metadata = json.load(f)

        if cached_metadata != expected_metadata:
            if self.args.overwrite_cache:
                shutil.rmtree(self.args.cache_dir)
                return False

            raise ValueError(
                f"Cache metadata does not match current dataset config: "
                f"{self.args.cache_dir}. Use --overwrite_cache or choose "
                "another --cache_dir."
            )

        print("Loading cached raw dataset...")
        self.dataset = load_from_disk(self.args.cache_dir)
        return True

    def _save_dataset_cache(self, metadata: Dict[str, object]) -> None:
        if not self.args.cache_dir:
            return

        parent_dir = os.path.dirname(os.path.abspath(self.args.cache_dir))
        os.makedirs(parent_dir, exist_ok=True)

        print("Saving raw dataset cache...")
        self.dataset.save_to_disk(self.args.cache_dir)

        with open(self._cache_metadata_path(), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _validate_label_mapping(self) -> None:
        train_labels = self.dataset["train"].features["label"].names
        val_labels = self.dataset["validation"].features["label"].names

        if train_labels != val_labels:
            missing_in_val = sorted(set(train_labels) - set(val_labels))
            missing_in_train = sorted(set(val_labels) - set(train_labels))

            raise ValueError(
                "Train and validation label mappings differ. "
                f"Missing in validation: {missing_in_val[:10]}; "
                f"missing in train: {missing_in_train[:10]}"
            )

    def load_dataset(self) -> None:
        """
        Load dataset from disk and split if needed.
        """

        print("Loading dataset...")

        train_dir, val_dir = self._infer_splits()
        cache_metadata = self._cache_metadata(train_dir, val_dir)

        if self._load_cached_dataset(cache_metadata):
            self._validate_label_mapping()
            self._extract_labels()
            print(
                f"Dataset loaded: "
                f"{len(self.dataset['train'])} train / "
                f"{len(self.dataset['validation'])} validation"
            )
            return

        if val_dir:

            self.dataset = DatasetDict({
                "train": load_dataset(
                    "imagefolder",
                    data_dir=train_dir,
                    split="train"
                ),
                "validation": load_dataset(
                    "imagefolder",
                    data_dir=val_dir,
                    split="train"
                ),
            })

        else:

            full_dataset = load_dataset(
                "imagefolder",
                data_dir=train_dir,
                split="train"
            )

            try:
                split = full_dataset.train_test_split(
                    test_size=self.args.val_ratio,
                    seed=self.args.seed,
                    stratify_by_column="label",
                )
            except (TypeError, ValueError) as exc:
                print(
                    "Could not create a stratified validation split; "
                    f"falling back to random split. Reason: {exc}"
                )
                split = full_dataset.train_test_split(
                    test_size=self.args.val_ratio,
                    seed=self.args.seed
                )

            self.dataset = DatasetDict({
                "train": split["train"],
                "validation": split["test"]
            })

        self._validate_label_mapping()

        print(
            f"Dataset loaded: "
            f"{len(self.dataset['train'])} train / "
            f"{len(self.dataset['validation'])} validation"
        )

        self._extract_labels()
        self._save_dataset_cache(cache_metadata)

    def _extract_labels(self) -> None:
        """
        Extract label mappings from dataset.
        """

        labels = self.dataset["train"].features["label"].names

        self.id2label = dict(enumerate(labels))
        self.label2id = {v: k for k, v in self.id2label.items()}

    # -------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------

    def setup_transforms(self) -> None:
        """
        Configure image transformations.
        """

        self.processor = AutoImageProcessor.from_pretrained(
            self.args.model_name
        )

        image_size = self._resolve_image_size()

        normalize = Normalize(
            mean=self.processor.image_mean,
            std=self.processor.image_std
        )

        self.train_transforms = Compose([
            RandomResizedCrop(image_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])

        self.val_transforms = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            normalize,
        ])

    def _resolve_image_size(self) -> int:
        """
        Resolve the processor image size across model families.
        """

        for size_attr in ("crop_size", "size"):
            size = getattr(self.processor, size_attr, None)

            if isinstance(size, dict):
                for key in ("height", "shortest_edge", "width"):
                    value = size.get(key)
                    if isinstance(value, int):
                        return value

                for value in size.values():
                    if isinstance(value, int):
                        return value

            if isinstance(size, int):
                return size

        raise ValueError(
            f"Could not infer image size from processor for "
            f"{self.args.model_name}. Processor size={self.processor.size!r}"
        )

    def preprocess_datasets(self) -> None:
        """
        Apply preprocessing dynamically.
        """

        print("Registering dynamic image transformations...")

        def preprocess_train(batch):
            return {
                "pixel_values": [
                    self.train_transforms(img.convert("RGB"))
                    for img in batch["image"]
                ],
                "label": batch["label"],
            }

        def preprocess_val(batch):
            return {
                "pixel_values": [
                    self.val_transforms(img.convert("RGB"))
                    for img in batch["image"]
                ],
                "label": batch["label"],
            }

        self.dataset["train"].set_transform(preprocess_train)
        self.dataset["validation"].set_transform(preprocess_val)

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load pretrained model.
        """

        print(f"Loading model: {self.args.model_name}")

        if self._uses_backbone_classifier():
            backbone = AutoModel.from_pretrained(self.args.model_name)
            self.model = BackboneImageClassifier(
                backbone=backbone,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id,
                freeze_backbone=self.args.freeze_backbone,
            )
            return

        self.model = AutoModelForImageClassification.from_pretrained(
            self.args.model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

    def _uses_backbone_classifier(self) -> bool:
        if self.args.backbone_classifier:
            return True

        model_name = self.args.model_name.lower()
        return "dinov3" in model_name

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------

    @staticmethod
    def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred (tuple): Logits and labels.

        Returns:
            Dict[str, float]: Metrics dictionary.
        """

        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=1)

        accuracy = np.mean(predictions == labels).item()
        f1 = f1_score(labels, predictions, average="macro")
        top_k = min(5, logits.shape[1])
        top_k_predictions = np.argsort(logits, axis=1)[:, -top_k:]
        top5_accuracy = np.mean([
            label in top_k_predictions[index]
            for index, label in enumerate(labels)
        ]).item()

        return {
            "accuracy": accuracy,
            "f1_macro": f1,
            "top5_accuracy": top5_accuracy,
        }

    # -------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------

    def _collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Custom batch collation.

        Args:
            batch (list): Dataset batch.

        Returns:
            Dict[str, Tensor]: Collated batch.
        """

        pixel_values = torch.stack(
            [item["pixel_values"] for item in batch]
        )

        labels = torch.tensor(
            [item["label"] for item in batch]
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    def train(self) -> None:
        """
        Execute training and evaluation.
        """

        training_kwargs = {
            "output_dir": self.args.output_dir,
            "logging_dir": self.args.logging_dir,
            "remove_unused_columns": False,
            "save_strategy": "epoch",
            "learning_rate": self.args.lr,
            "per_device_train_batch_size": self.args.batch_size,
            "per_device_eval_batch_size": self.args.batch_size,
            "num_train_epochs": self.args.epochs,
            "warmup_ratio": self.args.warmup_ratio,
            "weight_decay": self.args.weight_decay,
            "logging_steps": 50,
            "fp16": self.args.fp16,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "greater_is_better": True,
            "report_to": "none",
            "run_name": self.args.run_name,
        }

        training_args_signature = inspect.signature(
            TrainingArguments.__init__
        ).parameters

        if "evaluation_strategy" in training_args_signature:
            training_kwargs["evaluation_strategy"] = "epoch"
        else:
            training_kwargs["eval_strategy"] = "epoch"

        training_args = TrainingArguments(**training_kwargs)

        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": self.dataset["train"],
            "eval_dataset": self.dataset["validation"],
            "compute_metrics": self.compute_metrics,
            "data_collator": self._collate_fn,
        }

        trainer_signature = inspect.signature(Trainer.__init__).parameters

        if "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = self.processor
        else:
            trainer_kwargs["tokenizer"] = self.processor

        trainer = Trainer(**trainer_kwargs)

        print("Starting training...")

        started_at = datetime.now(timezone.utc).isoformat()
        run_start_time = time.time()
        start_time = time.time()

        train_result = trainer.train()

        duration = time.time() - start_time
        train_finished_at = datetime.now(timezone.utc).isoformat()

        print(f"Training completed in {duration / 3600:.2f} hours")

        trainer.save_model()
        if isinstance(self.model, BackboneImageClassifier):
            self.model.save_pretrained(self.args.output_dir)
            self.processor.save_pretrained(self.args.output_dir)

        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)
        run_duration = time.time() - run_start_time
        finished_at = datetime.now(timezone.utc).isoformat()
        self._save_run_artifacts(
            trainer=trainer,
            train_metrics=train_result.metrics,
            eval_metrics=eval_metrics,
            duration_seconds=duration,
            started_at=started_at,
            train_finished_at=train_finished_at,
            finished_at=finished_at,
            run_duration_seconds=run_duration,
        )

        print("Final evaluation:", eval_metrics)

    def _save_run_artifacts(
        self,
        trainer: Trainer,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        duration_seconds: float,
        started_at: str,
        train_finished_at: str,
        finished_at: str,
        run_duration_seconds: float,
    ) -> None:
        """
        Save run metadata and Trainer log history for plotting curves.
        """

        train_metrics = dict(train_metrics)
        train_metrics["train_wall_time_seconds"] = duration_seconds
        train_metrics["train_wall_time_minutes"] = duration_seconds / 60
        train_metrics["train_wall_time_hours"] = duration_seconds / 3600
        train_metrics["run_wall_time_seconds"] = run_duration_seconds
        train_metrics["run_wall_time_minutes"] = run_duration_seconds / 60
        train_metrics["run_wall_time_hours"] = run_duration_seconds / 3600
        trainer.save_metrics("train", train_metrics)

        summary = {
            "run_name": self.args.run_name,
            "model_name": self.args.model_name,
            "model_slug": self.args.model_slug,
            "epochs": self.args.epochs,
            "output_dir": self.args.output_dir,
            "logging_dir": self.args.logging_dir,
            "cache_dir": self.args.cache_dir,
            "started_at": started_at,
            "train_finished_at": train_finished_at,
            "finished_at": finished_at,
            "train_duration_seconds": duration_seconds,
            "train_duration_minutes": duration_seconds / 60,
            "train_duration_hours": duration_seconds / 3600,
            "run_duration_seconds": run_duration_seconds,
            "run_duration_minutes": run_duration_seconds / 60,
            "run_duration_hours": run_duration_seconds / 3600,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }

        summary_path = os.path.join(self.args.output_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        log_history_path = os.path.join(
            self.args.output_dir,
            "trainer_log_history.json",
        )
        with open(log_history_path, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    # -------------------------------------------------------------------
    # Pipeline Runner
    # -------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full training pipeline.
        """

        self.load_dataset()
        self.setup_transforms()
        self.preprocess_datasets()
        self.load_model()
        self.train()


# -------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Image Classification Trainer"
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="microsoft/beit-base-patch16-384")
    parser.add_argument("--output_dir", type=str,
                        default="./model_saved")
    parser.add_argument("--output_root", type=str,
                        default="./model_saved/finetuning")
    parser.add_argument("--logs_root", type=str,
                        default="./logs/finetuning")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--backbone_classifier", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--cache_dir", type=str,
                        default="./cached_dataset")
    parser.add_argument("--cache_root", type=str,
                        default="./cached_dataset/finetuning")
    parser.add_argument("--overwrite_cache", action="store_true")

    return parser.parse_args()
