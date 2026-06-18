import json
import os
import re
from difflib import SequenceMatcher
from io import BytesIO

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BackboneImageClassifier(nn.Module):
    """Inference wrapper for a feature backbone plus a linear classifier."""

    def __init__(self, backbone, num_labels, id2label, label2id):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        hidden_size = self._resolve_hidden_size(backbone)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def _resolve_hidden_size(backbone) -> int:
        config = getattr(backbone, "config", None)

        for attr in ("hidden_size", "embed_dim", "projection_dim"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value

        hidden_sizes = getattr(config, "hidden_sizes", None)
        if hidden_sizes and isinstance(hidden_sizes[-1], int):
            return hidden_sizes[-1]

        raise ValueError("Could not infer backbone hidden size.")

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


class FoodClassifier:
    """Reusable food image classifier loader.

    Supports both standard Hugging Face image-classification models and the
    custom DINOv3 backbone-plus-classifier format used by DINOv3-food.
    """

    def __init__(self, model_dir: str):
        from transformers import AutoImageProcessor

        self.model_dir = model_dir
        self.resolved_model_dir = self._resolve_model_dir(model_dir)

        self.model = self._load_model(model_dir).to(DEVICE)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained(
            self.resolved_model_dir
        )
        self.id2label = self._load_id2label()

    def _resolve_model_dir(self, model_dir: str) -> str:
        classifier_config_path = os.path.join(
            model_dir,
            "classifier_config.json",
        )

        if os.path.isfile(classifier_config_path):
            return model_dir

        try:
            snapshot_dir = snapshot_download(
                repo_id=model_dir,
                allow_patterns=[
                    "backbone/*",
                    "classifier.pt",
                    "classifier_config.json",
                    "preprocessor_config.json",
                ],
            )
        except Exception as exc:
            if "dinov3-food" in model_dir:
                raise RuntimeError(
                    "Could not download the custom DINOv3 food model files "
                    f"from '{model_dir}'. Make sure the repository contains "
                    "classifier_config.json, classifier.pt, backbone/*, and "
                    "preprocessor_config.json, and make sure HUB_TOKEN is set "
                    "if the repository is private."
                ) from exc
            return model_dir

        downloaded_config = os.path.join(snapshot_dir, "classifier_config.json")
        if "dinov3-food" in model_dir and not os.path.isfile(downloaded_config):
            raise FileNotFoundError(
                f"Repository '{model_dir}' was downloaded, but "
                "classifier_config.json was not found. This model cannot be "
                "loaded with AutoModelForImageClassification because it uses "
                "a custom DINOv3 backbone-plus-classifier format."
            )

        return snapshot_dir

    def _load_model(self, model_dir: str):
        classifier_config_path = os.path.join(
            self.resolved_model_dir,
            "classifier_config.json",
        )

        if os.path.isfile(classifier_config_path):
            return self._load_backbone_classifier(classifier_config_path)

        from transformers import AutoModelForImageClassification

        return AutoModelForImageClassification.from_pretrained(model_dir)

    def _load_backbone_classifier(self, classifier_config_path: str):
        from transformers import AutoModel

        with open(classifier_config_path, "r", encoding="utf-8") as file:
            classifier_config = json.load(file)

        backbone_dir = os.path.join(
            self.resolved_model_dir,
            classifier_config["backbone_dir"],
        )

        id2label = {
            int(label_id): label
            for label_id, label in classifier_config["id2label"].items()
        }
        label2id = {
            label: int(label_id)
            for label, label_id in classifier_config["label2id"].items()
        }

        try:
            backbone = AutoModel.from_pretrained(backbone_dir)
        except ValueError as exc:
            if "dinov3_vit" in str(exc):
                raise RuntimeError(
                    "DINOv3 requires a recent transformers version. "
                    "Install/use transformers>=4.57.1 before loading "
                    "yvelos/dinov3-food-389-v1."
                ) from exc
            raise

        model = BackboneImageClassifier(
            backbone=backbone,
            num_labels=int(classifier_config["num_labels"]),
            id2label=id2label,
            label2id=label2id,
        )

        classifier_path = os.path.join(self.resolved_model_dir, "classifier.pt")
        classifier_state = torch.load(classifier_path, map_location="cpu")
        model.classifier.load_state_dict(classifier_state)

        return model

    def _load_id2label(self):
        classifier_config_path = os.path.join(
            self.resolved_model_dir,
            "classifier_config.json",
        )

        if os.path.isfile(classifier_config_path):
            with open(classifier_config_path, "r", encoding="utf-8") as file:
                classifier_config = json.load(file)

            return {
                int(label_id): label
                for label_id, label in classifier_config["id2label"].items()
            }

        return {
            int(label_id): label
            for label_id, label in self.model.config.id2label.items()
        }

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

    def _load_image(self, image_source):
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        if isinstance(image_source, str):
            return Image.open(image_source).convert("RGB")

        if hasattr(image_source, "file"):
            return Image.open(BytesIO(image_source.file.read())).convert("RGB")

        if isinstance(image_source, BytesIO):
            return Image.open(image_source).convert("RGB")

        raise ValueError(f"Unsupported input type: {type(image_source)}")

    def predict(self, image_source) -> str:
        image = self._load_image(image_source)

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = (
                outputs["logits"]
                if isinstance(outputs, dict)
                else outputs.logits
            )
            pred_id = logits.argmax(-1).item()

        return self.id2label[pred_id]
