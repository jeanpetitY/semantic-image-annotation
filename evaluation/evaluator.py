import json
import csv
import re
import ast

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class Evaluator:

    def __init__(self, epsilon=0.1, epsilon_abs=0.1):
        """
        Args:
            epsilon (float): Relative tolerance (default 10%)
            epsilon_abs (float): Absolute tolerance for zero values
        """
        self.epsilon = epsilon
        self.epsilon_abs = epsilon_abs


    # ----------------------------------
    # Loaders
    # ----------------------------------

    def load_ground_truth(self, json_path):
        """Load ground truth components"""

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        norm_components = []

        for item in data:
            components = item.get("components", [])
            comp_norm = [c.strip().lower() for c in components]
            norm_components.append(comp_norm)

        return norm_components


    def load_predictions(self, csv_path):
        """Load predicted components from CSV"""

        preds = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:

                raw_components = row[2].strip()

                # Case: abstention
                if raw_components.lower() == "i don't know":
                    preds.append(["i don't know"])
                    continue

                try:
                    components = ast.literal_eval(raw_components)

                    if isinstance(components, list):
                        preds.append([
                            c.strip('"').strip().lower()
                            for c in components
                        ])
                    else:
                        preds.append([])

                except (ValueError, SyntaxError):
                    preds.append([])

        return preds


    # ----------------------------------
    # Parsing
    # ----------------------------------

    def parse_component(self, component_str):
        """
        Parse: 'protein: 10 g' -> ('protein', 10.0, 'g')
        """

        match = re.match(r"(.*?):\s*([\d\.]+)\s*(\w+)", component_str)

        if match:
            name = match.group(1).strip().lower()
            value = float(match.group(2))
            unit = match.group(3).lower()

            return name, value, unit

        return None, None, None


    # ----------------------------------
    # Metrics
    # ----------------------------------

    def jaccard_similarity(self, list1, list2):

        s1 = {x.lower().strip() for x in list1}
        s2 = {x.lower().strip() for x in list2}

        union = s1.union(s2)

        return len(s1.intersection(s2)) / len(union) if union else 0.0


    def precision_recall_f1(self, gt_all, pred_all):

        tp = fp = fn = 0

        for gt, pred in zip(gt_all, pred_all):

            gt_set = set(gt)
            pred_set = set(pred)

            tp += len(gt_set & pred_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        return precision, recall, f1


    def precision_recall_f1_names(self, gt_all, pred_all):

        tp = fp = fn = 0

        for gt, pred in zip(gt_all, pred_all):

            gt_names = {
                self.parse_component(c)[0]
                for c in gt
                if self.parse_component(c)[0] is not None
            }

            pred_names = {
                self.parse_component(c)[0]
                for c in pred
                if self.parse_component(c)[0] is not None
            }

            tp += len(gt_names & pred_names)
            fp += len(pred_names - gt_names)
            fn += len(gt_names - pred_names)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        return precision, recall, f1


    def compute_mae_and_accuracy(self, gt_all, pred_all):

        errors = []
        correct = 0
        total = 0

        for gt, pred in zip(gt_all, pred_all):

            gt_parsed = {
                name: val
                for name, val, _ in
                (self.parse_component(c) for c in gt)
                if name is not None
            }

            pred_parsed = {
                name: val
                for name, val, _ in
                (self.parse_component(c) for c in pred)
                if name is not None
            }

            for nutrient in gt_parsed:

                if nutrient in pred_parsed:

                    gt_val = gt_parsed[nutrient]
                    pred_val = pred_parsed[nutrient]

                    errors.append(abs(gt_val - pred_val))
                    total += 1

                    if gt_val == 0:
                        if abs(pred_val) <= self.epsilon_abs:
                            correct += 1
                    else:
                        if abs(pred_val - gt_val) / gt_val <= self.epsilon:
                            correct += 1

        mae = sum(errors) / len(errors) if errors else 0.0
        acc = correct / total if total else 0.0

        return mae, acc


    def compute_abstention_rate(self, preds):

        abstained = sum(
            1 for p in preds
            if len(p) == 1 and p[0].lower() == "i don't know"
        )

        return abstained / len(preds)


    def compute_jaccard(self, gt_all, pred_all):

        scores = [
            self.jaccard_similarity(gt, pred)
            for gt, pred in zip(gt_all, pred_all)
        ]

        return sum(scores) / len(scores)


    # ----------------------------------
    # Main Evaluation
    # ----------------------------------

    def evaluate(self, ground_json, prediction_csv):

        gt = self.load_ground_truth(ground_json)
        preds = self.load_predictions(prediction_csv)

        n = min(len(gt), len(preds))

        gt = gt[:n]
        preds = preds[:n]

        precision, recall, f1 = self.precision_recall_f1(gt, preds)

        mae, acc = self.compute_mae_and_accuracy(gt, preds)

        abstention = self.compute_abstention_rate(preds)

        jaccard = self.compute_jaccard(gt, preds)

        results = {
            "Samples Evaluated": n,
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1-score": round(f1, 2),
            "MAE": round(mae, 2),
            "Accuracy@10%": round(acc, 2),
            "Abstention rate": round(abstention, 2),
            "Jaccard": round(jaccard, 2)
        }

        return results
