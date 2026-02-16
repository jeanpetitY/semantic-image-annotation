import json
import csv
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import ast


def load_ground_truth(json_path):
    """Load ground truth and extract only the component list in order."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract components list only
    norm_components = []
    component_list = [item["components"] for item in data]
    for components in component_list:
        component = [c.strip().lower() for c in components]
        norm_components.append(component)
    return norm_components


def load_predictions(csv_path):
    """
    Load CSV predictions.

    Parse the 'components' column, which contains a string
    representation of a Python list.

    Returns:
        List[List[str]]: list of predicted component lists
    """
    preds = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            raw_components = row[2].strip()

            # Case: model does not know
            if raw_components.lower() == "i don't know":
                preds.append(["i don't know"])
                continue

            try:
                # Safely parse string into Python list
                components = ast.literal_eval(raw_components)

                if isinstance(components, list):
                    preds.append([c.strip('"').strip().lower() for c in components])
                else:
                    preds.append([])

            except (ValueError, SyntaxError):
                preds.append([])

    return preds

def parse_component(component_str):
    """
    Parse: 'protein: 10 g' -> ('protein', 10.0, 'g')
    """
    if match := re.match(r"(.*?):\s*([\d\.]+)\s*(\w+)", component_str):
        name = match.group(1).strip().lower()
        value = float(match.group(2))
        unit = match.group(3).lower()
        return name, value, unit
    return None, None, None


# ------------------------------
# Metrics
# ------------------------------
def jaccard_similarity(list1, list2):
    s1 = {x.lower().strip() for x in list1}
    s2 = {x.lower().strip() for x in list2}
    union = s1.union(s2)
    return len(s1.intersection(s2)) / len(union) if union else 0.0

def precision_recall_f1(gt_all, pred_all):
    tp = fp = fn = 0

    for gt, pred in zip(gt_all, pred_all):
        gt_set = set(gt)
        pred_set = set(pred)

        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, f1

def precision_recall_f1_2(gt_all, pred_all):
    tp = fp = fn = 0

    for gt, pred in zip(gt_all, pred_all):

        # Extraire uniquement les noms
        gt_names = {
            parse_component(c)[0]
            for c in gt
            if parse_component(c)[0] is not None
        }

        pred_names = {
            parse_component(c)[0]
            for c in pred
            if parse_component(c)[0] is not None
        }

        tp += len(gt_names & pred_names)
        fp += len(pred_names - gt_names)
        fn += len(gt_names - pred_names)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return precision, recall, f1


def compute_mae_and_accuracy(gt_all, pred_all, epsilon=0.1, epsilon_abs=0.1):
    """
    Compute Mean Absolute Error (MAE) and Accuracy within a tolerance for nutrient values.

    Parameters:
        gt_all (List[List[str]]): List of ground-truth nutrient lists per sample.
        pred_all (List[List[str]]): List of predicted nutrient lists per sample.
        epsilon (float): Relative tolerance for Accuracy@ε (default 0.1 = 10%).
        epsilon_abs (float): Absolute tolerance for nutrients with 0 value (default 0.1 g).

    Returns:
        mae (float): Mean Absolute Error across all matched nutrients.
        acc_eps (float): Accuracy within tolerance epsilon.
    """
    errors = []
    correct = 0
    total = 0

    for gt, pred in zip(gt_all, pred_all):
        # Parse components to dict {name: value}
        gt_parsed = {name: val for name, val, _ in (parse_component(c) for c in gt) if name is not None}
        pred_parsed = {name: val for name, val, _ in (parse_component(c) for c in pred) if name is not None}

        for nutrient in gt_parsed:
            if nutrient in pred_parsed:
                gt_val = gt_parsed[nutrient]
                pred_val = pred_parsed[nutrient]

                errors.append(abs(gt_val - pred_val))
                total += 1

                if gt_val == 0:
                    # Use absolute tolerance for zero nutrients
                    if abs(pred_val - gt_val) <= epsilon_abs:
                        correct += 1
                else:
                    # Use relative tolerance for non-zero nutrients
                    if abs(pred_val - gt_val) / gt_val <= epsilon:
                        correct += 1

    mae = sum(errors) / len(errors) if errors else 0.0
    acc_eps = correct / total if total > 0 else 0.0

    return mae, acc_eps



def compute_abstention_rate(preds):
    abstained = sum(1 for p in preds if len(p) == 1 and p[0].lower() == "i don't know")
    return abstained / len(preds)


def compute_bleu(gt_all, pred_all):
    smoothie = SmoothingFunction().method4
    scores = []

    for gt, pred in zip(gt_all, pred_all):
        if not pred:
            scores.append(0)
        else:
            scores.append(sentence_bleu([gt], pred, smoothing_function=smoothie))

    return sum(scores) / len(scores)


def compute_rouge(gt_all, pred_all):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for gt, pred in zip(gt_all, pred_all):
        if not pred:
            scores.append(0)
            continue
        score = scorer.score(" ".join(gt), " ".join(pred))["rougeL"].fmeasure
        scores.append(score)

    return sum(scores) / len(scores)


def compute_jaccard(gt_all, pred_all):
    scores = [jaccard_similarity(gt, pred) for gt, pred in zip(gt_all, pred_all)]
    return sum(scores) / len(scores)


# --------------------------------------------
# MAIN EVALUATION FUNCTION
# --------------------------------------------
def evaluate1(ground_json, prediction_csv):
    gt = load_ground_truth(ground_json)
    preds = load_predictions(prediction_csv)

    n = min(len(gt), len(preds))

    gt = gt[:n]
    preds = preds[:n]

    results = {
        "Samples Evaluated": n,
        "BLEU": round(compute_bleu(gt, preds), 5),
        "ROUGE-L": round(compute_rouge(gt, preds), 5),
        "Jaccard": round(compute_jaccard(gt, preds), 4)
    }

    return results


def evaluate(ground_json, prediction_csv):
    gt = load_ground_truth(ground_json)
    preds = load_predictions(prediction_csv)

    n = min(len(gt), len(preds))
    gt = gt[:n]
    preds = preds[:n]

    precision, recall, f1 = precision_recall_f1(gt, preds)
    mae, acc_eps = compute_mae_and_accuracy(gt, preds, epsilon=0.1)
    abstention = compute_abstention_rate(preds)

    results = {
        "Samples Evaluated": n,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-score": round(f1, 2),
        "MAE": round(mae, 2),
        "Accuracy@10%": round(acc_eps, 2),
        "Abstention rate": round(abstention, 2),
        "Jaccard": round(compute_jaccard(gt, preds), 2)
    }

    return results



# -----------------------
# Run evaluation
# -----------------------
# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/AFD_test.json",
#     prediction_csv="results/falcon/beit/not_rag/afd.csv"
# )

results = evaluate(
    ground_json="dataset/multimodal/not_merged/test/AFD_test.json",
    prediction_csv="results/falcon/beit/80%/afd.csv"
)

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/AFD_test.json",
#     prediction_csv="results/falcon/beit/rag/afd.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/AFD_test.json",
#     prediction_csv="results/falcon/beit/semantic_search/afd.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/fruitveg81_test.json",
#     prediction_csv="results/falcon/beit/rag/fruitveg81.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/fruitveg81_test.json",
#     prediction_csv="results/falcon/beit/not_rag/fruitveg81.csv"
# )

# results = evaluate(
#    ground_json="dataset/multimodal/not_merged/test/fruitveg81_test.json",
#    prediction_csv="results/falcon/beit/semantic_search/fruitveg81.csv"
# )


# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/food101_test.json",
#     prediction_csv="results/falcon/beit/rag/food101.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/food101_test.json",
#     prediction_csv="results/falcon/beit/not_rag/food101.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/food101_test.json",
#     prediction_csv="results/falcon/beit/semantic_search/food101.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/uecfood256_test.json",
#     prediction_csv="results/falcon/beit/rag/uecfood256.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/uecfood256_test.json",
#     prediction_csv="results/falcon/beit/not_rag/uecfood256.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/not_merged/test/uecfood256_test.json",
#     prediction_csv="results/falcon/beit/semantic_search/uecfood256.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/merged/merged_test.json",
#     prediction_csv="results/falcon/beit/rag/merged.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/merged/merged_test.json",
#     prediction_csv="results/falcon/beit/not_rag/merged.csv"
# )

# results = evaluate(
#     ground_json="dataset/multimodal/merged/merged_test.json",
#     prediction_csv="results/falcon/beit/semantic_search/merged.csv"
# )

print(results)
