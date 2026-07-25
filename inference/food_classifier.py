import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.food_recognition import FoodClassifier


class FoodEvaluator:

    def __init__(self, classifier: FoodClassifier):
        self.classifier = classifier

    # ------------------ Strict evaluation ------------------

    def evaluate(self, test_dir: str, limit: int = None):
        from datasets import load_dataset
        from sklearn.metrics import accuracy_score, f1_score
        from tqdm import tqdm

        print("Loading dataset...")
        ds = load_dataset(
            "imagefolder",
            data_dir=test_dir,
            split="train"
        )

        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))

        labels = ds.features["label"].names

        y_true = []
        y_pred = []

        print(f"Evaluating {len(ds)} images...\n")

        for ex in tqdm(ds):

            image = ex["image"]
            true_label = labels[ex["label"]]

            pred = self.classifier.predict(image)

            y_true.append(true_label)
            y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print("\nStrict Results:")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-macro : {f1:.4f}")

        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4)
        }

    # ------------------ Semantic evaluation ------------------

    def evaluate_semantic(self, test_dir: str, limit: int = None):
        from datasets import load_dataset
        from sklearn.metrics import accuracy_score, f1_score
        from tqdm import tqdm

        print("Loading dataset...")
        ds = load_dataset(
            "imagefolder",
            data_dir=test_dir,
            split="train"
        )

        if limit:
            ds = ds.shuffle(seed=42).select(range(limit))

        labels = ds.features["label"].names

        correct = 0
        total = 0

        y_true = []
        y_pred = []

        print(f"Evaluating {len(ds)} images...\n")

        for ex in tqdm(ds):

            image = ex["image"]

            true_label = labels[ex["label"]]
            pred_label = self.classifier.predict(image)

            y_true.append(true_label)
            y_pred.append(pred_label)

            if self.classifier.labels_semantically_equal(
                true_label,
                pred_label
            ):
                correct += 1

            total += 1

        semantic_acc = correct / total if total else 0

        acc_strict = accuracy_score(y_true, y_pred)
        f1_strict = f1_score(y_true, y_pred, average="macro")

        print("\n Semantic Results:")
        print(f"Accuracy   : {acc_strict:.4f}")
        print(f"Semantic Accuracy : {semantic_acc:.4f}")
        print(f"F1-macro   : {f1_strict:.4f}")

        return {
            "accuracy": round(acc_strict, 4),
            "accuracy_semantic": round(semantic_acc, 4),
            "f1_macro": round(f1_strict, 4)
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned food classification model."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.getenv("MODEL_DIR", "yvelos/dinov3-food-389-v1"),
        help="Local path or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.getenv("TEST_DIR", "dataset/image/not_merged/AFD/test"),
        help="Directory containing the test split in imagefolder format.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "semantic"],
        default="semantic",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to evaluate.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional JSON path for metrics.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)

    classifier = FoodClassifier(args.model_dir)
    evaluator = FoodEvaluator(classifier)

    if args.mode == "strict":
        results = evaluator.evaluate(args.test_dir, limit=args.limit)
    else:
        results = evaluator.evaluate_semantic(args.test_dir, limit=args.limit)

    if args.output_file is None:
        save_dir = f"results/{args.model_dir.split('/')[-1]}"
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(
            save_dir,
            f"{Path(args.test_dir).parent.name}_metrics.json"
        )
    else:
        output_file = args.output_file
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_file}")
    return results


if __name__ == "__main__":
    main()
