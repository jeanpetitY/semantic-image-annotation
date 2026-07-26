import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path

import pandas as pd


HAS_IMAGE_KEY = "http://example.org/food#hasImage"
HAS_COMPONENTS_KEY = "http://example.org/food#hasComponent"
HAS_TEXT_DESCRIPTION_KEY = "http://example.org/food#hasTextDescription"
RDFS_LABEL_KEY = "http://www.w3.org/2000/01/rdf-schema#label"
HAS_UNIT_KEY = "http://example.org/food#hasUnit"
HAS_VALUE_KEY = "http://example.org/food#hasValue"


def _extract_first_value(value):
    if isinstance(value, list) and value:
        first_item = value[0]
        if isinstance(first_item, dict):
            return first_item.get("@value")
        return first_item
    return value


def _extract_label_from_jsonld_identifier(record: dict) -> str:
    identifier = str(record.get("@id", ""))
    type_values = record.get("@type", [])

    candidates = []
    if "#" in identifier:
        candidates.append(identifier.split("#", 1)[1])
    if isinstance(type_values, list) and type_values:
        first_type = str(type_values[0])
        if "#" in first_type:
            candidates.append(first_type.split("#", 1)[1])
        else:
            candidates.append(first_type)

    for candidate in candidates:
        if "_Img_" in candidate:
            prefix = candidate.split("_Img_", 1)[0]
        elif candidate.endswith("_Images"):
            prefix = candidate[: -len("_Images")]
        else:
            continue

        if "_" in prefix:
            return prefix.rsplit("_", 1)[0]
        return prefix

    label_value = _extract_first_value(record.get(RDFS_LABEL_KEY))
    if label_value is not None:
        return str(label_value)

    return identifier


def load_ground_truth_items(input_file: str) -> list[dict]:
    input_path = Path(input_file)

    if input_path.suffix.lower() == ".jsonl":
        data = pd.read_json(input_file, lines=True).to_dict(orient="records")
    else:
        with open(input_file, "r", encoding="utf-8") as file:
            data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Ground-truth input must contain a list of records.")

    normalized_items = []
    for item in data:
        if not isinstance(item, dict):
            continue

        if "label" in item and "image" in item:
            normalized_items.append(item)
            continue

        if "@id" in item and HAS_IMAGE_KEY in item:
            image_path = _extract_first_value(item.get(HAS_IMAGE_KEY))
            if not image_path:
                continue

            normalized_items.append(
                {
                    "label": _extract_label_from_jsonld_identifier(item),
                    "image": str(image_path),
                    "components": item.get(HAS_COMPONENTS_KEY, item.get("components", [])),
                    "text_description": _extract_first_value(
                        item.get(HAS_TEXT_DESCRIPTION_KEY)
                    ),
                }
            )
            continue

        raise ValueError(
            "Unsupported ground-truth format. Expected records with "
            "'label'/'image' or JSON-LD records with image metadata."
        )

    return normalized_items


class Evaluator:
    MASS_UNITS_TO_G = {
        "g": 1.0,
        "gram": 1.0,
        "grams": 1.0,
        "mg": 0.001,
        "milligram": 0.001,
        "milligrams": 0.001,
        "ug": 0.000001,
        "mcg": 0.000001,
        "microgram": 0.000001,
        "micrograms": 0.000001,
        "kg": 1000.0,
        "kilogram": 1000.0,
        "kilograms": 1000.0,
    }

    ENERGY_UNITS_TO_KCAL = {
        "kcal": 1.0,
        "calorie": 1.0,
        "calories": 1.0,
        "kj": 0.239005736,
        "kilojoule": 0.239005736,
        "kilojoules": 0.239005736,
    }

    UNIT_ALIASES = {
        "µg": "ug",
        "μg": "ug",
        "cal": "kcal",
    }

    def __init__(self, epsilon=0.1, epsilon_abs=0.1):
        self.epsilon = epsilon
        self.epsilon_abs = epsilon_abs

    # ------------------ Loaders ------------------

    def make_sample_id(self, item):
        return f"{item.get('label', '')} - {item.get('image', '')}"

    def load_ground_truth_records(self, json_path):
        data = load_ground_truth_items(json_path)

        records = []
        for item in data:
            components = item.get("components", [])
            records.append({
                "id": self.make_sample_id(item),
                "label": str(item.get("label", "")).strip(),
                "components": self.normalize_component_list(components),
            })

        return records

    def load_prediction_records(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"Prediction CSV is empty: {csv_path}")

            required = {"id", "predicted_name", "components"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Prediction CSV is missing columns {sorted(missing)}: {csv_path}"
                )

            records = []
            for row_number, row in enumerate(reader, start=2):
                raw_components = row.get("components", "").strip()
                records.append({
                    "id": row.get("id", "").strip(),
                    "predicted_name": row.get("predicted_name", "").strip(),
                    "components": self.parse_prediction_components(
                        raw_components,
                        row_number,
                    ),
                })

        return records

    def parse_prediction_components(self, raw_components, row_number):
        """
        Parse the components field from prediction CSVs.
        Accepts multiple literal formats produced by different models:
        - A list of strings like ["protein: 10 g", ...]
        - A list of dicts like [{"protein": "10 g"}, ...]
        - A list of 2-tuples/lists like [("protein", "10 g"), ...]
        - The exact string "I don't know" (case-insensitive) which is treated as abstention.
        """
        if raw_components.lower() == "i don't know":
            return ["i don't know"]

        try:
            components = ast.literal_eval(raw_components)
        except (ValueError, SyntaxError):
            return []

        # Normalize single dict into a list
        if isinstance(components, dict):
            components = [components]

        if not isinstance(components, list):
            return []

        normalized_items = []
        for item in components:
            # dicts -> key: value
            if isinstance(item, dict):
                for k, v in item.items():
                    normalized_items.append(f"{k}: {v}")
            # tuples/lists of length 2 -> name, value
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized_items.append(f"{item[0]}: {item[1]}")
            else:
                normalized_items.append(str(item))

        return self.normalize_component_list(normalized_items)

    def normalize_component_list(self, components):
        normalized_components = []
        for component in components:
            if isinstance(component, dict):
                label = _extract_first_value(component.get(RDFS_LABEL_KEY))
                unit = _extract_first_value(component.get(HAS_UNIT_KEY))
                value = _extract_first_value(component.get(HAS_VALUE_KEY))

                if isinstance(value, dict):
                    value = value.get("@value")

                if label is not None and value is not None:
                    normalized_components.append(
                        f"{str(label).strip()}: {str(value).strip()} {str(unit).strip()}".strip().lower()
                    )
                    continue

            component_text = str(component).strip().strip('"').strip("'").lower()
            if component_text:
                normalized_components.append(component_text)

        return normalized_components

    def align_records(self, gt_records, pred_records):
        gt_by_id = {record["id"]: record for record in gt_records}
        aligned = []
        unmatched_predictions = 0

        for pred in pred_records:
            gt = gt_by_id.get(pred["id"])
            if gt is None:
                unmatched_predictions += 1
                continue
            aligned.append((gt, pred))

        return aligned, unmatched_predictions

    # ------------------ Parsing ------------------

    def parse_component(self, component_str):
        match = re.fullmatch(
            r"\s*(.*?):\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"\s*([^\d\s]+)\s*",
            component_str,
        )
        if not match:
            return None, None, None

        name = match.group(1).strip().lower()
        value = float(match.group(2))
        unit = self.normalize_unit(match.group(3))
        return name, value, unit

    def normalize_unit(self, unit):
        unit = unit.strip().lower()
        return self.UNIT_ALIASES.get(unit, unit)

    def normalize_value(self, value, unit):
        unit = self.normalize_unit(unit)

        if unit in self.MASS_UNITS_TO_G:
            return value * self.MASS_UNITS_TO_G[unit], "mass:g"

        if unit in self.ENERGY_UNITS_TO_KCAL:
            return value * self.ENERGY_UNITS_TO_KCAL[unit], "energy:kcal"

        if unit in {"%", "percent", "percentage"}:
            return value, "percent"

        return value, f"unit:{unit}"

    def parse_components_by_name(self, components):
        parsed = {}
        for component in components:
            name, value, unit = self.parse_component(component)
            if name is None:
                continue

            norm_value, unit_family = self.normalize_value(value, unit)
            parsed[name] = {
                "value": norm_value,
                "unit_family": unit_family,
                "raw_value": value,
                "raw_unit": unit,
            }

        return parsed

    # ------------------ Metrics ------------------
    
    def component_confusion(self, gt_sets, pred_sets):
        tp = fp = fn = 0

        for gt_set, pred_set in zip(gt_sets, pred_sets):
            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)

        return tp, fp, fn

    def set_prf(self, gt_sets, pred_sets):
        tp, fp, fn =  self.component_confusion(gt_sets, pred_sets)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return precision, recall, f1

    def mean_jaccard(self, gt_sets, pred_sets):
        scores = []
        for gt_set, pred_set in zip(gt_sets, pred_sets):
            union = gt_set | pred_set
            scores.append(len(gt_set & pred_set) / len(union) if union else 0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def value_metrics(self, aligned):
        errors = []
        correct = 0
        total = 0
        unit_mismatches = 0

        for gt, pred in aligned:
            gt_parsed = self.parse_components_by_name(gt["components"])
            pred_parsed = self.parse_components_by_name(pred["components"])

            for nutrient, gt_item in gt_parsed.items():
                pred_item = pred_parsed.get(nutrient)
                if pred_item is None:
                    continue

                total += 1

                if gt_item["unit_family"] != pred_item["unit_family"]:
                    unit_mismatches += 1
                    continue

                gt_val = gt_item["value"]
                pred_val = pred_item["value"]
                errors.append(abs(gt_val - pred_val))

                if gt_val == 0:
                    if abs(pred_val) <= self.epsilon_abs:
                        correct += 1
                elif abs(pred_val - gt_val) / abs(gt_val) <= self.epsilon:
                    correct += 1

        mae = sum(errors) / len(errors) if errors else 0.0
        accuracy = correct / total if total else 0.0
        return mae, accuracy, unit_mismatches, total

    def abstention_rate(self, aligned):
        if not aligned:
            return 0.0
        abstained = sum(
            1 for _, pred in aligned
            if pred["components"] == ["i don't know"]
        )
        return abstained / len(aligned)

    def food_accuracy(self, aligned):
        if not aligned:
            return 0.0
        correct = sum(
            1 for gt, pred in aligned
            if gt["label"].lower() == pred["predicted_name"].lower()
        )
        return correct / len(aligned)
    
    def build_component_sets(self, aligned, is_ablation=False):
        """
        Build component sets depending on evaluation mode.

        Standard mode:
            Compare only nutrient/component names.
            Example: "protein: 10 g" -> "protein"

        Ablation mode:
            Compare full component strings.
            Example: "protein: 10 g" stays "protein: 10 g"
        """

        if is_ablation:
            gt_component_sets = [
                set(gt["components"])
                for gt, _ in aligned
            ]

            pred_component_sets = [
                set(pred["components"])
                for _, pred in aligned
            ]

        else:
            gt_component_sets = [
                set(self.parse_components_by_name(gt["components"]))
                for gt, _ in aligned
            ]

            pred_component_sets = [
                set(self.parse_components_by_name(pred["components"]))
                for _, pred in aligned
            ]

        return gt_component_sets, pred_component_sets

    # ------------------ Evaluation ------------------

    def evaluate(self, ground_json, prediction_csv, output_file=None, is_ablation=False):
        gt_records = self.load_ground_truth_records(ground_json)
        pred_records = self.load_prediction_records(prediction_csv)
        aligned, unmatched_predictions = self.align_records(gt_records, pred_records)

        gt_component_sets, pred_component_sets = self.build_component_sets(
            aligned,
            is_ablation=is_ablation,
        )

        precision, recall, f1 = self.set_prf(
            gt_component_sets,
            pred_component_sets,
        )
        mae, value_acc, unit_mismatches, value_pairs = self.value_metrics(aligned)

        results = {
            "Evaluation mode": "ablation" if is_ablation else "standard",
            "Samples Evaluated": len(aligned),
            "Unmatched Predictions": unmatched_predictions,
            "Samples to be Evaluated": len(gt_records),
            "Predictions available": len(pred_records),
            "Prediction coverage": round(len(aligned) / len(gt_records), 4) if gt_records else 0.0,
            "Food accuracy": round(self.food_accuracy(aligned), 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "Jaccard": round(
                self.mean_jaccard(gt_component_sets, pred_component_sets),
                4,
            ),
            "MAE": round(mae, 4),
            "Accuracy@10%": round(value_acc, 4),
            "Abstention rate": round(self.abstention_rate(aligned), 4),
            "Answer rate": round(1 - self.abstention_rate(aligned), 4),
        }

        results["Output file"] = self.save_results(
            results,
            prediction_csv,
            output_file,
        )
        self.save_results(results, prediction_csv, results["Output file"])

        return results

    def save_results(self, results, prediction_csv, output_file=None):
        if output_file is None:
            pred_dir = os.path.dirname(prediction_csv)
            pred_name = os.path.splitext(os.path.basename(prediction_csv))[0]
            output_file = os.path.join(
                pred_dir,
                f"{pred_name}_evaluation_metrics.json",
            )

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        return output_file


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate nutrient predictions.")
    parser.add_argument(
        "--ground-json",
        default="dataset/multimodal/not_merged/test/example_test.jsonl",
    )
    parser.add_argument(
        "--prediction-csv",
        default="results/falcon/dinov3/rag/afd.csv",
    )
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--is-ablation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = Evaluator()
    print(evaluator.evaluate(
        ground_json=args.ground_json,
        prediction_csv=args.prediction_csv,
        output_file=args.output_file,
        is_ablation=args.is_ablation,
    ))


if __name__ == "__main__":
    main()
