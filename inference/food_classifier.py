import torch
import re
from difflib import SequenceMatcher
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FoodClassifier:
    def __init__(self, model_dir: str):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model = AutoModelForImageClassification.from_pretrained(
            model_dir
        ).to(device)

        self.processor = AutoImageProcessor.from_pretrained(model_dir)

        self.id2label = self.model.config.id2label

    # ------------------ Utils ------------------

    def clean_label(self, name: str) -> str:
        name = name.lower().strip()
        name = name.replace("&", "and")
        name = re.sub(r"[ \-_/']", "_", name)
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name)

        return name.strip("_")

    def token_overlap(self, a: str, b: str) -> float:
        ta = a.split("_")
        tb = b.split("_")

        if len(ta) == 1 and len(tb) == 1:
            return SequenceMatcher(None, ta[0], tb[0]).ratio()

        if len(ta) != len(tb):
            return 0.0

        return 0.0

    def labels_semantically_equal(
        self,
        y_true: str,
        y_pred: str,
        threshold: float = 0.85,
    ) -> bool:

        a = self.clean_label(y_true)
        b = self.clean_label(y_pred)

        if a == b:
            return True

        ratio = SequenceMatcher(None, a, b).ratio()
        if ratio < threshold:
            return False

        overlap = self.token_overlap(a, b)
        if overlap < threshold:
            return False

        return True

    # ------------------ Prediction ------------------

    def _load_image(self, image_source):

        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        elif isinstance(image_source, str):
            return Image.open(image_source).convert("RGB")

        elif hasattr(image_source, "file"):
            return Image.open(
                BytesIO(image_source.file.read())
            ).convert("RGB")

        elif isinstance(image_source, BytesIO):
            return Image.open(image_source).convert("RGB")

        else:
            raise ValueError(f"Unsupported input type: {type(image_source)}")

    def predict(self, image_source) -> str:

        image = self._load_image(image_source)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = logits.argmax(-1).item()

        return self.id2label[pred_id]


class FoodEvaluator:

    def __init__(self, classifier: FoodClassifier):
        self.classifier = classifier

    # ------------------ Strict evaluation ------------------

    def evaluate_strict(self, test_dir: str, limit: int = None):

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
        print(f"Strict Accuracy   : {acc_strict:.4f}")
        print(f"Semantic Accuracy : {semantic_acc:.4f}")
        print(f"Strict F1-macro   : {f1_strict:.4f}")

        return {
            "accuracy_strict": round(acc_strict, 4),
            "accuracy_semantic": round(semantic_acc, 4),
            "f1_macro_strict": round(f1_strict, 4)
        }




if __name__ == "__main__":

    model_dir = "nateraw/food"
    test_dir = "dataset/image/merged/test"

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
