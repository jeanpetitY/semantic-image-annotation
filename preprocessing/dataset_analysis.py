"""Lightweight distribution analysis for image datasets.

Paper reference:
- Supplementary reproducibility material for the merged food-image dataset and
  the released excerpt.
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_dataset_root(root_dir: str) -> Path:
    candidate = Path(root_dir).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    return project_candidate


def _summarize_class_counts(class_counts: Dict[str, int]):
    if not class_counts:
        print("No classes found.")
        return None

    counts = list(class_counts.values())
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    mean_images = np.mean(counts)
    std_images = np.std(counts)

    print("===== Dataset Statistics =====")
    print(f"Total classes: {len(class_counts)}\n")
    print(f"Class with MIN images: {min_class} ({class_counts[min_class]} images)")
    print(f"Class with MAX images: {max_class} ({class_counts[max_class]} images)\n")
    print(f"Average images per class: {mean_images:.2f}")
    print(f"Standard deviation: {std_images:.2f}")

    return {
        "min_class": (min_class, class_counts[min_class]),
        "max_class": (max_class, class_counts[max_class]),
        "mean": float(mean_images),
        "std": float(std_images),
    }


def analyze_flat_dataset(root_dir: str):
    """
    Analyze a folder-per-class dataset without train/test subdirectories.
    """
    root_dir = str(_resolve_dataset_root(root_dir))
    class_counts = {}

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_files = [
            file_name
            for file_name in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, file_name))
            and file_name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        class_counts[class_name] = len(image_files)

    return _summarize_class_counts(class_counts)


def analyze_split_dataset(root_dir: str):
    """
    Analyze a dataset structured with `train/` and `test/` subdirectories.
    """
    root_dir = str(_resolve_dataset_root(root_dir))
    class_counts = {}

    for split in ["train", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            image_files = [
                file_name
                for file_name in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, file_name))
                and file_name.lower().endswith(IMAGE_EXTENSIONS)
            ]

            class_counts[class_name] = class_counts.get(class_name, 0) + len(image_files)

    return _summarize_class_counts(class_counts)


def analyze_dataset_layout(root_dir: str):
    """
    Automatically analyze either a flat dataset layout or a train/test layout.
    """
    # Paper reproducibility: support both the full merged dataset layout and
    # the compact excerpt layout with the same analysis command.
    root_path = _resolve_dataset_root(root_dir)
    if (root_path / "train").exists() or (root_path / "test").exists():
        return analyze_split_dataset(str(root_path))
    return analyze_flat_dataset(str(root_path))
