import ast
import csv
import json
import re


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

    DIMENSIONLESS_UNITS = {"none", "(none)", "unit", "units"}

    UNIT_ALIASES = {
        "µg": "ug",
        "μg": "ug",
        "cal": "kcal",
    }

    def __init__(self, epsilon=0.1, epsilon_abs=0.1):
        """
        Args:
            epsilon (float): Relative tolerance (default 10%).
            epsilon_abs (float): Absolute tolerance for zero values.
        """
        self.epsilon = epsilon
        self.epsilon_abs = epsilon_abs

    # ----------------------------------
    # Loaders
    # ----------------------------------

    def load_ground_truth(self, json_path):
        """Load ground truth components."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        norm_components = []

        for item in data:
            components = item.get("components", [])
            comp_norm = [str(c).strip().lower() for c in components]
            norm_components.append(comp_norm)

        return norm_components

    def load_predictions(self, csv_path):
        """Load predicted components from CSV."""
        preds = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Prediction CSV is empty: {csv_path}")

            try:
                components_idx = header.index("components")
            except ValueError as exc:
                raise ValueError(
                    f"Prediction CSV must contain a 'components' column: {csv_path}"
                ) from exc

            for row_number, row in enumerate(reader, start=2):
                if len(row) <= components_idx:
                    raise ValueError(
                        f"Row {row_number} has no components column in {csv_path}"
                    )

                raw_components = row[components_idx].strip()

                if raw_components.lower() == "i don't know":
                    preds.append(["i don't know"])
                    continue

                try:
                    components = ast.literal_eval(raw_components)
                except (ValueError, SyntaxError):
                    preds.append([])
                    continue

                if isinstance(components, list):
                    preds.append([
                        str(c).strip('"').strip().lower()
                        for c in components
                    ])
                else:
                    preds.append([])

        return preds

    # ----------------------------------
    # Parsing
    # ----------------------------------

    def parse_component(self, component_str):
        """
        Parse: 'protein: 10 g' -> ('protein', 10.0, 'g').
        """
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

        if unit in self.DIMENSIONLESS_UNITS:
            return value, "dimensionless"

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

    # ----------------------------------
    # Metrics
    # ----------------------------------

    def validate_lengths(self, gt_all, pred_all):
        if len(gt_all) != len(pred_all):
            raise ValueError(
                "Ground truth and prediction counts differ: "
                f"{len(gt_all)} ground-truth samples vs {len(pred_all)} predictions"
            )

    def jaccard_similarity(self, list1, list2):
        s1 = {x.lower().strip() for x in list1}
        s2 = {x.lower().strip() for x in list2}

        union = s1.union(s2)

        return len(s1.intersection(s2)) / len(union) if union else 0.0

    def precision_recall_f1(self, gt_all, pred_all):
        self.validate_lengths(gt_all, pred_all)

        tp = fp = fn = 0

        for gt, pred in zip(gt_all, pred_all):
            gt_names = set(self.parse_components_by_name(gt))
            pred_names = set(self.parse_components_by_name(pred))

            tp += len(gt_names & pred_names)
            fp += len(pred_names - gt_names)
            fn += len(gt_names - pred_names)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        return precision, recall, f1

    def compute_mae_and_accuracy(self, gt_all, pred_all):
        self.validate_lengths(gt_all, pred_all)

        errors = []
        correct = 0
        total = 0
        unit_mismatches = 0

        for gt, pred in zip(gt_all, pred_all):
            gt_parsed = self.parse_components_by_name(gt)
            pred_parsed = self.parse_components_by_name(pred)

            for nutrient, gt_item in gt_parsed.items():
                if nutrient not in pred_parsed:
                    continue

                pred_item = pred_parsed[nutrient]

                if gt_item["unit_family"] != pred_item["unit_family"]:
                    unit_mismatches += 1
                    total += 1
                    continue

                gt_val = gt_item["value"]
                pred_val = pred_item["value"]

                errors.append(abs(gt_val - pred_val))
                total += 1

                if gt_val == 0:
                    if abs(pred_val) <= self.epsilon_abs:
                        correct += 1
                elif abs(pred_val - gt_val) / abs(gt_val) <= self.epsilon:
                    correct += 1

        mae = sum(errors) / len(errors) if errors else 0.0
        acc = correct / total if total else 0.0

        return mae, acc, unit_mismatches

    def compute_abstention_rate(self, preds):
        if not preds:
            return 0.0

        abstained = sum(
            1 for p in preds
            if len(p) == 1 and p[0].lower() == "i don't know"
        )

        return abstained / len(preds)

    def compute_jaccard(self, gt_all, pred_all):
        self.validate_lengths(gt_all, pred_all)

        scores = [
            self.jaccard_similarity(gt, pred)
            for gt, pred in zip(gt_all, pred_all)
        ]

        return sum(scores) / len(scores) if scores else 0.0

    # ----------------------------------
    # Main Evaluation
    # ----------------------------------

    def evaluate(self, ground_json, prediction_csv):
        gt = self.load_ground_truth(ground_json)
        preds = self.load_predictions(prediction_csv)

        self.validate_lengths(gt, preds)
        n = len(gt)

        precision, recall, f1 = self.precision_recall_f1(gt, preds)
        mae, acc, unit_mismatches = self.compute_mae_and_accuracy(gt, preds)
        abstention = self.compute_abstention_rate(preds)
        jaccard = self.compute_jaccard(gt, preds)

        return {
            "Samples Evaluated": n,
            "Samples to be Evaluated": len(gt),
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1-score": round(f1, 2),
            "MAE": round(mae, 2),
            "Accuracy@10%": round(acc, 2),
            "Unit mismatches": unit_mismatches,
            "Abstention rate": round(abstention, 2),
            "Jaccard": round(jaccard, 2),
        }


if __name__ == "__main__":
    evaluator = Evaluator()

    print(evaluator.evaluate(
        ground_json="dataset/multimodal/merged/merged_test.json",
        prediction_csv="results/falcon/clip/rag/merged.csv"
    ))
