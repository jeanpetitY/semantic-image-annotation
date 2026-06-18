import json
import os
import sys
from pathlib import Path

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




if __name__ == "__main__":

    model_dir = os.getenv("MODEL_DIR", "yvelos/dinov3-food-389-v1")
    test_dir = os.getenv("TEST_DIR", "dataset/image/not_merged/AFD/test")

    # Init classifier
    classifier = FoodClassifier(model_dir)

    # Init evaluator
    evaluator = FoodEvaluator(classifier)

    # Evaluate
    results = evaluator.evaluate_semantic(test_dir)

    # Save results
    save_dir = f"results/{model_dir.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    out_file = os.path.join(
        save_dir,
        f"{test_dir.split('/')[-2]}_metrics.json"
    )

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {out_file}")
