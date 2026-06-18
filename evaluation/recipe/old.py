from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:
    SmoothingFunction = None
    sentence_bleu = None


COMPONENT_PATTERN = re.compile(r"^(.*?):\s*([-+]?\d*\.?\d+)\s*([a-zA-Z%µ]+)$")


def normalize_text(value: str) -> str:
    return value.strip().lower()


def ensure_parent_dir(file_path: str) -> None:
    parent = Path(file_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def parse_component(component_str: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Parse 'protein: 10 g' -> ('protein', 10.0, 'g')."""
    text = normalize_text(component_str)
    match = COMPONENT_PATTERN.match(text)
    if not match:
        return None, None, None

    name = match.group(1).strip()
    value = float(match.group(2))
    unit = match.group(3).strip()
    return name, value, unit


def _safe_parse_list(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.lower() == "i don't know":
        return ["i don't know"]

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(parsed, list):
        return []

    out: List[str] = []
    for item in parsed:
        if isinstance(item, str):
            out.append(normalize_text(item))
    return out


def load_ground_truth(json_path: str) -> Dict[str, List[str]]:
    """Load ground truth as {id: [components]} where id = '<label> - <image>'."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Ground-truth JSON must be a list of objects.")

    gt_dict: Dict[str, List[str]] = {}
    for item in data:
        label = str(item.get("label", "")).strip()
        image = str(item.get("image", "")).strip()
        food_id = f"{label} - {image}"

        components_raw = item.get("components", [])
        if isinstance(components_raw, list):
            components = [normalize_text(str(c)) for c in components_raw]
        else:
            components = []

        gt_dict[food_id] = components

    return gt_dict


def load_predictions(csv_path: str) -> Dict[str, List[str]]:
    """Load predictions as {id: [predicted_components]} from CSV column index 0 and 2."""
    preds_dict: Dict[str, List[str]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header

        for row in reader:
            if len(row) < 3:
                continue
            food_id = str(row[0]).strip()
            raw_components = str(row[2])
            preds_dict[food_id] = _safe_parse_list(raw_components)

    return preds_dict


def align_samples(
    gt_dict: Dict[str, List[str]],
    pred_dict: Dict[str, List[str]],
) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """Align GT and predictions on common IDs (sorted for determinism)."""
    common_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    gt_all = [gt_dict[idx] for idx in common_ids]
    pred_all = [pred_dict[idx] for idx in common_ids]
    return gt_all, pred_all, common_ids


def jaccard_similarity(list1: Sequence[str], list2: Sequence[str]) -> float:
    s1 = {normalize_text(x) for x in list1}
    s2 = {normalize_text(x) for x in list2}
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def precision_recall_f1(gt_all: Sequence[Sequence[str]], pred_all: Sequence[Sequence[str]]) -> Tuple[float, float, float]:
    tp = fp = fn = 0

    for gt, pred in zip(gt_all, pred_all):
        gt_set = set(gt)
        pred_set = set(pred)
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_mae_and_accuracy(
    gt_all: Sequence[Sequence[str]],
    pred_all: Sequence[Sequence[str]],
    epsilon: float = 0.1,
    epsilon_abs: float = 0.1,
) -> Tuple[float, float]:
    """Compute MAE and accuracy within epsilon tolerance on matched nutrient names."""
    errors: List[float] = []
    correct = 0
    total = 0

    for gt, pred in zip(gt_all, pred_all):
        gt_parsed = {name: val for name, val, _ in (parse_component(c) for c in gt) if name is not None}
        pred_parsed = {name: val for name, val, _ in (parse_component(c) for c in pred) if name is not None}

        for nutrient, gt_val in gt_parsed.items():
            if nutrient not in pred_parsed:
                continue

            pred_val = pred_parsed[nutrient]
            errors.append(abs(gt_val - pred_val))
            total += 1

            if gt_val == 0:
                if abs(pred_val - gt_val) <= epsilon_abs:
                    correct += 1
            else:
                if abs(pred_val - gt_val) / abs(gt_val) <= epsilon:
                    correct += 1

    mae = sum(errors) / len(errors) if errors else 0.0
    acc_eps = correct / total if total > 0 else 0.0
    return mae, acc_eps


def compute_abstention_rate(pred_all: Sequence[Sequence[str]]) -> float:
    if not pred_all:
        return 0.0
    abstained = sum(1 for p in pred_all if len(p) == 1 and p[0] == "i don't know")
    return abstained / len(pred_all)


def compute_bleu(gt_all: Sequence[Sequence[str]], pred_all: Sequence[Sequence[str]]) -> Optional[float]:
    if SmoothingFunction is None or sentence_bleu is None:
        return None

    smoothie = SmoothingFunction().method4
    scores: List[float] = []

    for gt, pred in zip(gt_all, pred_all):
        if not pred:
            scores.append(0.0)
        else:
            scores.append(sentence_bleu([list(gt)], list(pred), smoothing_function=smoothie))

    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge_l(gt_all: Sequence[Sequence[str]], pred_all: Sequence[Sequence[str]]) -> Optional[float]:
    if rouge_scorer is None:
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores: List[float] = []

    for gt, pred in zip(gt_all, pred_all):
        if not pred:
            scores.append(0.0)
            continue
        score = scorer.score(" ".join(gt), " ".join(pred))["rougeL"].fmeasure
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def compute_jaccard(gt_all: Sequence[Sequence[str]], pred_all: Sequence[Sequence[str]]) -> float:
    if not gt_all:
        return 0.0
    scores = [jaccard_similarity(gt, pred) for gt, pred in zip(gt_all, pred_all)]
    return sum(scores) / len(scores)


def evaluate(
    ground_json: str,
    prediction_csv: str,
    epsilon: float = 0.1,
    epsilon_abs: float = 0.1,
) -> Dict[str, Any]:
    """Main evaluation: exact match + value metrics + abstention + jaccard."""
    gt_dict = load_ground_truth(ground_json)
    pred_dict = load_predictions(prediction_csv)
    gt_all, pred_all, common_ids = align_samples(gt_dict, pred_dict)

    precision, recall, f1 = precision_recall_f1(gt_all, pred_all)
    mae, acc_eps = compute_mae_and_accuracy(gt_all, pred_all, epsilon=epsilon, epsilon_abs=epsilon_abs)
    abstention = compute_abstention_rate(pred_all)
    jaccard = compute_jaccard(gt_all, pred_all)

    return {
        "samples_evaluated": len(common_ids),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mae": round(mae, 4),
        f"accuracy@{int(epsilon * 100)}%": round(acc_eps, 4),
        "abstention_rate": round(abstention, 4),
        "jaccard": round(jaccard, 4),
    }


def evaluate1(ground_json: str, prediction_csv: str) -> Dict[str, Any]:
    """Lexical evaluation profile kept for backward compatibility."""
    gt_dict = load_ground_truth(ground_json)
    pred_dict = load_predictions(prediction_csv)
    gt_all, pred_all, common_ids = align_samples(gt_dict, pred_dict)

    bleu = compute_bleu(gt_all, pred_all)
    rouge_l = compute_rouge_l(gt_all, pred_all)
    jaccard = compute_jaccard(gt_all, pred_all)

    results: Dict[str, Any] = {
        "samples_evaluated": len(common_ids),
        "jaccard": round(jaccard, 5),
    }

    if bleu is None:
        results["bleu"] = None
        results["warning_bleu"] = "nltk is not installed; BLEU not computed."
    else:
        results["bleu"] = round(bleu, 5)

    if rouge_l is None:
        results["rouge_l"] = None
        results["warning"] = "rouge-score is not installed; ROUGE-L not computed."
    else:
        results["rouge_l"] = round(rouge_l, 5)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nutrient generation predictions.")

    parser.add_argument(
        "--ground-json",
        type=str,
        required=True,
        help="Path to ground-truth JSON file.",
    )
    parser.add_argument(
        "--prediction-csv",
        type=str,
        required=True,
        help="Path to predictions CSV file.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["main", "lexical", "both"],
        default="main",
        help="Evaluation profile: main metrics, lexical metrics, or both.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Relative tolerance for +-10%.",
    )
    parser.add_argument(
        "--epsilon-abs",
        type=float,
        default=0.1,
        help="Absolute tolerance for zero-value nutrients.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Optional path to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation for printed/saved results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.profile == "main":
        results: Dict[str, Any] = evaluate(
            ground_json=args.ground_json,
            prediction_csv=args.prediction_csv,
            epsilon=args.epsilon,
            epsilon_abs=args.epsilon_abs,
        )
    elif args.profile == "lexical":
        results = evaluate1(
            ground_json=args.ground_json,
            prediction_csv=args.prediction_csv,
        )
    else:
        results = {
            "main": evaluate(
                ground_json=args.ground_json,
                prediction_csv=args.prediction_csv,
                epsilon=args.epsilon,
                epsilon_abs=args.epsilon_abs,
            ),
            "lexical": evaluate1(
                ground_json=args.ground_json,
                prediction_csv=args.prediction_csv,
            ),
        }

    output = json.dumps(results, ensure_ascii=False, indent=args.indent)
    print(output)

    if args.output_file:
        ensure_parent_dir(args.output_file)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output)


if __name__ == "__main__":
    main()
