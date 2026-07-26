import argparse
import json
import os
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_JSON_PATH = "outputs/uecfood256/uecfood256_usda_raw.json"
DEFAULT_DATASET_PATH = "../dataset/image/not_merged/uecfood256/test"
DEFAULT_LABEL_KEY = "food_class"
DEFAULT_USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def open_json_file(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_labels(path, label_key=DEFAULT_LABEL_KEY):
    data = open_json_file(path)

    if isinstance(data, dict):
        return len(data)

    if isinstance(data, list):
        labels = [
            item[label_key]
            for item in data
            if isinstance(item, dict) and item.get(label_key)
        ]
        return len(set(labels)) if labels else len(data)

    raise TypeError(f"Unsupported JSON root type: {type(data).__name__}")


def get_annotated_labels(json_path, label_key=DEFAULT_LABEL_KEY):
    data = open_json_file(json_path)

    if not isinstance(data, list):
        raise TypeError("The JSON file must contain a list of dictionaries.")

    annotated_labels = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item at index {index} is not a dictionary.")
        if label_key not in item:
            raise KeyError(f"Missing key '{label_key}' in item at index {index}.")
        annotated_labels.append(item[label_key])

    return annotated_labels


def get_initial_labels(dataset_path):
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Dataset directory not found: {dataset_path}")
    return os.listdir(dataset_path)


def get_annotation_stats(json_path, dataset_path, label_key=DEFAULT_LABEL_KEY):
    annotated_labels = get_annotated_labels(json_path, label_key)
    initial_labels = get_initial_labels(dataset_path)

    annotated_label_set = set(annotated_labels)
    not_annotated_labels = [
        label for label in initial_labels if label not in annotated_label_set
    ]

    return annotated_labels, initial_labels, not_annotated_labels


def format_usda_query(label):
    return label.replace("_", " ").title()


def get_usda_api_key():
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("USDA_KEY")
    if api_key:
        return api_key

    raise EnvironmentError("USDA_KEY is missing in environment variables.")


def search_usda_label(label, api_key=None, base_url=DEFAULT_USDA_BASE_URL):
    if api_key is None:
        api_key = get_usda_api_key()

    endpoint = f"{base_url}/foods/search"
    params = {
        "api_key": api_key,
        "query": format_usda_query(label),
        "pageSize": 1,
    }

    response = requests.get(endpoint, params=params, timeout=30)
    response.raise_for_status()

    foods = response.json().get("foods", [])
    return foods[0].get("description") if foods else None


def get_usda_pairs_for_not_annotated(
    json_path,
    dataset_path,
    label_key=DEFAULT_LABEL_KEY,
    sleep_seconds=0.5,
):
    _, _, not_annotated_labels = get_annotation_stats(
        json_path,
        dataset_path,
        label_key,
    )
    api_key = get_usda_api_key()

    usda_label_pairs = []
    for label in not_annotated_labels:
        try:
            usda_label = search_usda_label(label, api_key=api_key)
        except requests.RequestException as error:
            print(f"[WARNING] USDA search failed for '{label}': {error}")
            usda_label = None

        usda_label_pairs.append((label, usda_label))
        time.sleep(sleep_seconds)

    return usda_label_pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare annotated food labels against an image-dataset label directory."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help=f"Path to the annotation JSON file. Default: {DEFAULT_JSON_PATH}",
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the dataset label directory. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--label-key",
        default=DEFAULT_LABEL_KEY,
        help=f"Key containing the label name. Default: {DEFAULT_LABEL_KEY}",
    )
    parser.add_argument(
        "--search-usda",
        action="store_true",
        help="Search USDA suggestions for not-annotated labels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    annotated_labels, initial_labels, not_annotated_labels = get_annotation_stats(
        args.json_path,
        args.dataset_path,
        args.label_key,
    )

    print(f"Annotated labels ({len(annotated_labels)}):")
    print(annotated_labels)
    print(f"\nInitial labels ({len(initial_labels)}):")
    print(initial_labels)
    print(f"\nNot annotated labels ({len(not_annotated_labels)}):")
    print(not_annotated_labels)

    if args.search_usda:
        usda_label_pairs = get_usda_pairs_for_not_annotated(
            args.json_path,
            args.dataset_path,
            args.label_key,
        )
        print(f"\nUSDA label pairs ({len(usda_label_pairs)}):")
        print(usda_label_pairs)


if __name__ == "__main__":
    main()
