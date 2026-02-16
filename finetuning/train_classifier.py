import argparse
import os
import time
import random
from typing import Dict, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image

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
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

import evaluate
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


# -------------------------------------------------------------------
# Main Trainer Class
# -------------------------------------------------------------------

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

        self.dataset: DatasetDict = None
        self.model = None
        self.processor = None

        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

        self._setup_environment()
        set_seed(self.args.seed)

    # -------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------

    def _setup_environment(self) -> None:
        """
        Load environment variables and authenticate with Hugging Face Hub.
        """

        load_dotenv()

        token = os.getenv("HUB_TOKEN")

        if token is None:
            raise ValueError("HUB_TOKEN not found in environment variables.")

        login(token=token)

    # -------------------------------------------------------------------
    # Dataset Handling
    # -------------------------------------------------------------------

    def _infer_splits(self) -> Tuple[str, str]:
        """
        Infer training and validation directories.

        Returns:
            Tuple[str, str]: Train and validation paths.
        """

        train_dir = os.path.join(self.args.data_dir, "train")
        val_dir = os.path.join(self.args.data_dir, "val")

        if os.path.isdir(val_dir):
            return train_dir, val_dir

        return train_dir, None

    def load_dataset(self) -> None:
        """
        Load dataset from disk and split if needed.
        """

        print("Loading dataset...")

        train_dir, val_dir = self._infer_splits()

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

            split = full_dataset.train_test_split(
                test_size=self.args.val_ratio,
                seed=self.args.seed
            )

            self.dataset = DatasetDict({
                "train": split["train"],
                "validation": split["test"]
            })

        print(
            f"Dataset loaded: "
            f"{len(self.dataset['train'])} train / "
            f"{len(self.dataset['validation'])} validation"
        )

        self._extract_labels()

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

        if isinstance(self.processor.size, dict):
            image_size = self.processor.size["height"]
        else:
            image_size = self.processor.size

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

    def preprocess_datasets(self) -> None:
        """
        Apply preprocessing and cache results.
        """

        if os.path.exists(self.args.cache_dir):

            print("Loading cached dataset...")
            self.dataset = load_from_disk(self.args.cache_dir)
            return

        print("Applying image transformations...")

        def preprocess_train(batch):
            batch["pixel_values"] = [
                self.train_transforms(img.convert("RGB"))
                for img in batch["image"]
            ]
            return batch

        def preprocess_val(batch):
            batch["pixel_values"] = [
                self.val_transforms(img.convert("RGB"))
                for img in batch["image"]
            ]
            return batch

        self.dataset["train"] = self.dataset["train"].map(
            preprocess_train,
            batched=True,
            num_proc=4
        )

        self.dataset["validation"] = self.dataset["validation"].map(
            preprocess_val,
            batched=True,
            num_proc=4
        )

        self.dataset["train"].set_format(
            type="torch",
            columns=["pixel_values", "label"]
        )

        self.dataset["validation"].set_format(
            type="torch",
            columns=["pixel_values", "label"]
        )

        print("Saving processed dataset...")
        self.dataset.save_to_disk(self.args.cache_dir)

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load pretrained model.
        """

        print(f"Loading model: {self.args.model_name}")

        self.model = AutoModelForImageClassification.from_pretrained(
            self.args.model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

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

        accuracy = evaluate.load("accuracy").compute(
            predictions=predictions,
            references=labels
        )["accuracy"]

        f1 = f1_score(labels, predictions, average="macro")

        return {
            "accuracy": accuracy,
            "f1_macro": f1
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

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.args.lr,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epochs,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            logging_steps=50,
            fp16=self.args.fp16,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to=None,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.processor,
            compute_metrics=self.compute_metrics,
            data_collator=self._collate_fn,
        )

        print("Starting training...")

        start_time = time.time()

        trainer.train()

        duration = time.time() - start_time

        print(f"Training completed in {duration / 3600:.2f} hours")

        trainer.save_model()

        train_metrics = trainer.state.log_history[-1]
        trainer.save_metrics("train", train_metrics)

        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)

        print("Final evaluation:", eval_metrics)

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

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--cache_dir", type=str,
                        default="./cached_dataset")

    return parser.parse_args()


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

def main() -> None:
    """
    Application entry point.
    """

    args = parse_arguments()

    trainer = ImageClassificationTrainer(args)

    trainer.run()


if __name__ == "__main__":
    main()
