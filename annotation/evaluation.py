import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


FDC_ID_PATTERN = re.compile(r"/food-details/(\d+)/nutrients")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_fdc_id(annotation: Dict) -> Optional[int]:
    if annotation.get("fdc_id") is not None:
        return int(annotation["fdc_id"])

    source = annotation.get("usda_source", "")
    match = FDC_ID_PATTERN.search(source)
    if not match:
        return None

    return int(match.group(1))


def load_gold(path: Path) -> Dict[str, int]:
    gold_items = load_json(path)
    gold = {}

    for item in gold_items:
        food_class = item["food_class"]
        expected_fdc_id = int(item["expected_fdc_id"])
        gold[food_class] = expected_fdc_id

    return gold


def load_annotations(path: Path) -> Dict[str, Dict]:
    annotations = load_json(path)
    result = {}

    for item in annotations:
        food_class = item["food_class"]
        result[food_class] = item

    return result


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate(annotations: Dict[str, Dict], gold: Dict[str, int]) -> Dict:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    correct = []
    incorrect = []
    missing = []

    for food_class, expected_fdc_id in gold.items():
        annotation = annotations.get(food_class)

        if annotation is None:
            false_negative += 1
            missing.append(
                {
                    "food_class": food_class,
                    "expected_fdc_id": expected_fdc_id,
                }
            )
            continue

        predicted_fdc_id = extract_fdc_id(annotation)

        if predicted_fdc_id == expected_fdc_id:
            true_positive += 1
            correct.append(
                {
                    "food_class": food_class,
                    "fdc_id": predicted_fdc_id,
                    "usda_name": annotation.get("usda_name"),
                }
            )
        else:
            false_positive += 1
            false_negative += 1
            incorrect.append(
                {
                    "food_class": food_class,
                    "expected_fdc_id": expected_fdc_id,
                    "predicted_fdc_id": predicted_fdc_id,
                    "usda_name": annotation.get("usda_name"),
                    "usda_source": annotation.get("usda_source"),
                }
            )

    unexpected = []

    for food_class, annotation in annotations.items():
        if food_class not in gold:
            false_positive += 1
            unexpected.append(
                {
                    "food_class": food_class,
                    "predicted_fdc_id": extract_fdc_id(annotation),
                    "usda_name": annotation.get("usda_name"),
                }
            )

    expected_total = len(gold)
    produced_total = len(annotations)

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "metrics": {
            "expected_total": expected_total,
            "produced_total": produced_total,
            "correct": true_positive,
            "incorrect": len(incorrect),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "accuracy": safe_divide(true_positive, expected_total),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        },
        "incorrect_annotations": incorrect,
        "missing_annotations": missing,
        "unexpected_annotations": unexpected,
        "correct_annotations": correct,
    }


def print_summary(report: Dict) -> None:
    metrics = report["metrics"]

    print("Annotation evaluation")
    print("---------------------")
    print(f"Expected annotations: {metrics['expected_total']}")
    print(f"Produced annotations: {metrics['produced_total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Missing: {metrics['missing']}")
    print(f"Unexpected: {metrics['unexpected']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")

    if report["incorrect_annotations"]:
        print("\nIncorrect annotations:")
        for item in report["incorrect_annotations"]:
            print(
                "- "
                f"{item['food_class']}: expected {item['expected_fdc_id']}, "
                f"predicted {item['predicted_fdc_id']} "
                f"({item.get('usda_name')})"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate USDA annotation quality against a strict gold file."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to the generated annotation JSON file."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to the strict gold JSON file."
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path where the full evaluation report is saved."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    annotations = load_annotations(args.annotations)
    gold = load_gold(args.gold)
    report = evaluate(annotations, gold)

    print_summary(report)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
