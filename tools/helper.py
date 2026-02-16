import csv
import random
import re
from flask import json
import os
import shutil
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Set
from dataset_splitter import DatasetSplitter
from dataset_balancer import DatasetBalancer
from PIL import Image



fruitveg81_labels = os.listdir("dataset/image/not_merged/fruitveg81/train")
food101_labels = os.listdir("dataset/image/not_merged/food101/train")
uecfood256_labels = os.listdir("dataset/image/not_merged/uecfood256/train")
afd_labels = os.listdir("dataset/image/not_merged/AFD/train")
merged_labels = os.listdir("dataset/image/merged/train")


class Helper:

    def clean_label(self, name: str) -> str:
        """
        Normalize a label into a clean identifier suitable for filenames, URLs, and IDs.
        """
        name = name.lower().strip()

        # Replace semantic symbols
        name = name.replace("&", "and")

        # Replace separators by underscore
        name = re.sub(r"[ \-_/']", "_", name)

        # Remove remaining non-alphanumeric characters
        name = re.sub(r"[^a-z0-9_]", "", name)

        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)

        # Strip leading/trailing underscores
        return name.strip("_")

    def check_and_clean_labels(self, labels: list):
        """
        Check which labels are already clean and apply cleaning if needed.

        Args:
            labels (list): List of string labels.

        Returns:
            tuple:
                - cleaned_labels (list): Final list of labels (cleaned or unchanged)
                - modified_count (int): Number of labels that were modified
                - modified_labels (list): Labels that were changed
        """
        cleaned_labels = []
        modified_labels = []

        for label in labels:
            cleaned = self.clean_label(label)
            if label != cleaned:
                modified_labels.append(label)
            cleaned_labels.append(cleaned)

        return cleaned_labels, len(modified_labels), modified_labels

    def check_all_datasets(self):
        """
        Check and clean labels for all datasets and print a summary.
        """
        datasets = {
            "UECFood256": uecfood256_labels,
            "AFD": afd_labels,
            "Food101": food101_labels,
            "FruitVeg81": fruitveg81_labels,
            "Merged": merged_labels,
        }

        summary = {}

        for name, labels in datasets.items():
            _, modified_count, modified_labels = self.check_and_clean_labels(labels)
            summary[name] = {
                "total": len(labels),
                "modified": modified_count,
                "unchanged": len(labels) - modified_count,
                "modified_labels": modified_labels,
            }

        return summary
    
    def flatten_image_folders(self, extensions=(".jpg", ".jpeg", ".png")):
        """
        Flatten all nested subfolders inside each class folder (e.g. apples, bananas)(fruitveg81 dataset)
        so that all images end up directly in their class folder.

        Args:
            root_dir (str): Path to the dataset root (e.g. 'fruitveg81/')
            extensions (tuple): Allowed image extensions
        """
        root = Path(self)
        class_dirs = [d for d in root.iterdir() if d.is_dir()]

        for class_dir in class_dirs:
            print(f"Processing class: {class_dir.name}")

            for subpath in class_dir.rglob("*"):
                if subpath.is_file() and subpath.suffix.lower() in extensions:
                    # New unique name to avoid overwriting
                    new_name = f"{subpath.stem}_{hash(subpath)}{subpath.suffix}"
                    dest_path = class_dir / new_name

                    try:
                        shutil.move(str(subpath), str(dest_path))
                    except Exception as e:
                        print(f"Could not move {subpath}: {e}")

            # Remove old empty subfolders
            for subfolder in class_dir.rglob("*"):
                if subfolder.is_dir() and not any(subfolder.iterdir()):
                    subfolder.rmdir()

            print(f"Flattened: {class_dir.name}\n")

        print("All folders have been flattened successfully!")
        
    
    """ Originally this method is used to construct the set of images we want to 
    link in the ORKG KG for uecfood256 dataset.  It extracts one image per class, 
    renames it with a cleaned label and copies it into a flat destination folder.
    """    
    def extract_uecfood256_images(
        self,
        category_file: str,
        source_dir: str,
        dest_dir: str,
        extensions=(".jpg", ".jpeg", ".png"),
    ):
        """
        Extract one image per UECFood256 class, rename it using a cleaned label,
        and copy it into a flat destination directory.

        Args:
            category_file (str): Path to category.txt file (id, name).
            source_dir (str): Root directory containing UECFood256 class folders.
            dest_dir (str): Output directory for renamed images.
            extensions (tuple): Allowed image extensions.
        """
        os.makedirs(dest_dir, exist_ok=True)

        # Detect separator automatically
        with open(category_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sep = "\t" if "\t" in first_line else ","

        df = pd.read_csv(category_file, sep=sep)

        for _, row in df.iterrows():
            food_id = str(row["id"])
            raw_name = str(row["name"])
            cleaned_name = self.clean_label(raw_name)

            folder_path = os.path.join(source_dir, food_id)
            if not os.path.exists(folder_path):
                print(f"Folder not found for ID {food_id}")
                continue

            images = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(extensions)
            ]

            if not images:
                print(f"No image found in folder {folder_path}")
                continue

            selected_image = images[0]
            src_path = os.path.join(folder_path, selected_image)
            dest_path = os.path.join(dest_dir, f"{cleaned_name}.jpg")

            try:
                shutil.copy(src_path, dest_path)
                print(f"Copied {selected_image} to {cleaned_name}.jpg")
            except Exception as e:
                print(f"Failed to copy image for {raw_name}: {e}")
                
                
    def add_image_urls_to_usda_json(
        self,
        input_json: str,
        output_json: str,
        base_url: str,
        class_key: str = "class",
        image_extension: str = ".jpg",
    ):
        """
        Add an image URL to each USDA food entry based on its class label.

        Args:
            input_json (str): Path to the input USDA JSON file.
            output_json (str): Path to the output enriched JSON file.
            base_url (str): Base URL where images are hosted.
            class_key (str): JSON key containing the class label.
            image_extension (str): Image file extension.
        """
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if class_key in item and isinstance(item[class_key], str):
                cleaned_label = self.clean_label(item[class_key])
                item["image"] = f"{base_url}{cleaned_label}{image_extension}"
            else:
                item["image"] = None

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    """
    Fix known structural issues in the UECFood256 dataset.
    """

    def merge_duplicate_classes(
        self,
        metadata_path: str,
        dataset_root: str,
        duplicate_ids=(40, 67),
        output_metadata_path="uecfood256_fixed.txt",
    ):
        """
        Merge duplicate classes and renumber dataset folders.

        Args:
            metadata_path (str): Path to category metadata file.
            dataset_root (str): Root folder of dataset.
            duplicate_ids (tuple): IDs of duplicate classes.
            output_metadata_path (str): Output corrected metadata file.
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = [row for row in reader if len(row) >= 2]

        df = pd.DataFrame(rows[1:], columns=rows[0])
        df["id"] = df["id"].astype(int)

        id1, id2 = duplicate_ids
        name1 = df.loc[df["id"] == id1, "name"].values[0]
        name2 = df.loc[df["id"] == id2, "name"].values[0]

        if name1.lower().strip() != name2.lower().strip():
            raise ValueError("Duplicate class names do not match.")

        folder1 = os.path.join(dataset_root, str(id1))
        folder2 = os.path.join(dataset_root, str(id2))

        for root, _, files in os.walk(folder2):
            for file in files:
                shutil.move(
                    os.path.join(root, file),
                    os.path.join(folder1, file),
                )

        shutil.rmtree(folder2)

        df = df[df["id"] != id2].reset_index(drop=True)
        df["id"] = range(1, len(df) + 1)
        df.to_csv(output_metadata_path, sep="\t", index=False)

        all_folders = sorted([int(f) for f in os.listdir(dataset_root) if f.isdigit()])
        for expected_id, folder_id in enumerate(all_folders, start=1):
            old_path = os.path.join(dataset_root, str(folder_id))
            new_path = os.path.join(dataset_root, str(expected_id))
            if folder_id != expected_id:
                shutil.move(old_path, new_path)
                
    def prepare_food101_dataset(
        self,
        output_dir: str = "food101",
        train_per_class: int = 750,
        test_per_class: int = 250,
    ):
        """
        Download the Food101 dataset from Hugging Face and store it locally
        using a folder-per-class structure.

        Structure:
            output_dir/
            ├── train/
            │   ├── apple_pie/
            │   ├── ...
            └── test/
                ├── apple_pie/
                ├── ...

        Args:
            output_dir (str): Root directory where the dataset will be saved.
            train_per_class (int): Maximum number of training images per class.
            test_per_class (int): Maximum number of testing images per class.
        """
        print("Loading Food101 dataset from Hugging Face...")
        dataset = load_dataset("ethz/food101")

        output_dir = Path(output_dir)
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        def save_split(split_name, split_data, max_per_class):
            print(f"Processing split: {split_name}")
            class_counts = {}

            for idx, example in enumerate(split_data):
                label_id = example["label"]
                label_name = split_data.features["label"].int2str(label_id)

                # Normalize label for folder name
                label_name = self.clean_label(label_name)

                if label_name not in class_counts:
                    class_counts[label_name] = 0

                if class_counts[label_name] >= max_per_class:
                    continue

                class_dir = output_dir / split_name / label_name
                class_dir.mkdir(parents=True, exist_ok=True)

                img_path = class_dir / f"{label_name}_{class_counts[label_name]:04d}.jpg"
                example["image"].save(img_path)

                class_counts[label_name] += 1

                if idx % 1000 == 0 and idx > 0:
                    print(f"  {idx} images processed")

            print(
                f"Finished {split_name}: {sum(class_counts.values())} images saved"
            )

        # Save splits
        save_split("train", dataset["train"], train_per_class)
        save_split("test", dataset["validation"], test_per_class)

        print(f"Dataset successfully prepared in {output_dir}")
        
    def rename_uecfood256_folders(
        self,
        dataset_dir: str,
        txt_file: str,
        splits=("train", "test"),
        delimiter="\t",
    ):
        """
        Rename UECFood256 numeric class folders (e.g., 1/, 2/, ...)
        into semantic class names (e.g., apple_pie/, bibimbap/)
        using the metadata file.

        The class names are normalized using self.clean_label().

        Args:
            dataset_dir (str): Root dataset directory containing train/ and test/.
            txt_file (str): Metadata file with columns [id, name].
            splits (tuple): Dataset splits to process (default: ("train", "test")).
            delimiter (str): Column delimiter in the metadata file.
        """

        # --------------------------------------------------
        # Step 1: Build ID -> cleaned label mapping
        # --------------------------------------------------
        id_to_label = {}

        with open(txt_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = [row for row in reader if len(row) >= 2]

        for row in rows[1:]:  # skip header
            class_id = row[0].strip()
            class_name = row[1].strip()
            id_to_label[class_id] = self.clean_label(class_name)

        print(f"Loaded {len(id_to_label)} class mappings.")

        # --------------------------------------------------
        # Step 2: Rename folders in each split
        # --------------------------------------------------
        dataset_root = Path(dataset_dir)

        for split in splits:
            split_path = dataset_root / split
            if not split_path.exists():
                print(f"Split not found, skipping: {split_path}")
                continue

            print(f"Renaming folders in: {split_path}")

            for folder in split_path.iterdir():
                if not folder.is_dir():
                    continue

                old_name = folder.name  # numeric ID
                new_name = id_to_label.get(old_name)

                if not new_name:
                    print(f"No mapping found for folder: {old_name}")
                    continue

                new_path = folder.parent / new_name

                if new_path.exists():
                    print(f"Target folder already exists: {new_name}")
                    continue

                try:
                    os.rename(folder, new_path)
                    print(f"{old_name} -> {new_name}")
                except Exception as e:
                    print(f"Failed to rename {old_name}: {e}")

        print("UECFood256 folder renaming completed.")
        
    def token_overlap(self, a: str, b: str) -> float:
        ta = a.split("_")
        tb = b.split("_")

        # Single-token labels (e.g. churro / churros)
        if len(ta) == 1 and len(tb) == 1:
            return SequenceMatcher(None, ta[0], tb[0]).ratio()

        # Multi-token labels must have same length
        if len(ta) != len(tb):
            return 0.0

        # scores = []
        # for x, y in zip(ta, tb):
        #     scores.append(SequenceMatcher(None, x, y).ratio())

        return 0.0
        
    def compare_food_labels(
        self,
        food101_labels: List[str],
        uecfood256_labels: List[str],
        threshold: float = 0.8,
    ) -> Dict[str, List]:
        """
        Compare two food label lists using clean_label() to compute:
          - exact matches (after normalization)
          - fuzzy matches based on SequenceMatcher similarity

        Args:
            food101_labels (List[str]): Labels from Food101 dataset.
            uecfood256_labels (List[str]): Labels from UECFood256 dataset.
            threshold (float): Similarity threshold for fuzzy matching.

        Returns:
            dict: {
                "exact_matches": List[str],
                "fuzzy_matches": List[Tuple[str, str, float]]
            }
        """
        normalized_food101 = [self.clean_label(n) for n in food101_labels]
        normalized_uec = [self.clean_label(n) for n in uecfood256_labels]

        exact_matches = sorted(set(normalized_food101) & set(normalized_uec))

        fuzzy_matches: List[Tuple[str, str, float]] = []

        for f in normalized_food101:
            if f in exact_matches:
                continue

            for u in normalized_uec:
                ratio = SequenceMatcher(None, f, u).ratio()
                score = round(ratio, 2)

                if score > threshold:
                    overlap = self.token_overlap(f, u)

                    # Require at least one strong shared token
                    if overlap < 0.85:
                        continue

                    fuzzy_matches.append((f, u, score))

        print(f"Exact matches: {len(exact_matches)}")
        print(f"Fuzzy matches (>{threshold}): {len(fuzzy_matches)}")

        return {
            "exact_matches": exact_matches,
            "fuzzy_matches": fuzzy_matches,
        }
        
    def _list_labels(self, dataset_root: Path, split: str) -> List[str]:
        """
        List class labels (folder names) for a given dataset split.
        """
        split_dir = dataset_root / split
        if not split_dir.exists():
            return []

        return [
            d.name for d in split_dir.iterdir()
            if d.is_dir()
        ]
        
    def get_dataset_labels(self, dataset_root: str):
        train = set(os.listdir(os.path.join(dataset_root, "train")))
        test = set(os.listdir(os.path.join(dataset_root, "test")))
        return sorted(train | test)
    
    def _copy_split(
        self,
        src_root: Path,
        split: str,
        dst_root: Path,
        skip_labels: Set[str] = None,
    ):
        """
        Copy class folders from src_root/split to dst_root/split,
        optionally skipping some class labels.
        """
        skip_labels = skip_labels or set()

        src_split = src_root / split
        dst_split = dst_root / split
        dst_split.mkdir(parents=True, exist_ok=True)

        for class_dir in src_split.iterdir():
            if not class_dir.is_dir():
                continue

            label = class_dir.name
            if label in skip_labels:
                continue

            dst_class_dir = dst_split / label
            if dst_class_dir.exists():
                continue

            shutil.copytree(class_dir, dst_class_dir)
            
    def merge_food_datasets(
        self,
        dataset_dirs: Dict[str, str],
        output_dir: str,
        threshold: float = 0.85,
    ):
        """
        Merge multiple food datasets into a unified dataset.

        Dataset structure (for all datasets):
            dataset/
            train/<class_label>/
            test/<class_label>/

        Rules:
        - Food101 is the reference dataset
        - Exact + fuzzy matches (via compare_food_labels) are considered equivalent
        - For equivalent labels, keep Food101 and skip others
        - Non-equivalent labels are merged

        Args:
            dataset_dirs (dict): {
                "food101": "/path/to/food101",
                "uecfood256": "/path/to/uecfood256",
                "fruitveg81": "/path/to/fruitveg81",
                "afd": "/path/to/afd",
            }
            output_dir (str): Destination directory
            threshold (float): Similarity threshold for fuzzy matching
        """

        out_root = Path(output_dir)
        (out_root / "train").mkdir(parents=True, exist_ok=True)
        (out_root / "test").mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # Step 1: Collect Food101 labels (reference)
        # --------------------------------------------------
        food101_root = Path(dataset_dirs["food101"])

        f101_labels = sorted(
            set(
                self._list_labels(food101_root, "train")
                + self._list_labels(food101_root, "test")
            )
        )

        print(f"Food101 labels: {len(f101_labels)}")

        # --------------------------------------------------
        # Step 2: Copy Food101 entirely
        # --------------------------------------------------
        print("Copying Food101...")
        self._copy_split(food101_root, "train", out_root)
        self._copy_split(food101_root, "test", out_root)

        # --------------------------------------------------
        # Step 3: Merge remaining datasets
        # --------------------------------------------------
        for name, path in dataset_dirs.items():
            if name == "food101":
                continue

            print(f"\nProcessing dataset: {name}")
            ds_root = Path(path)

            ds_labels = sorted(
                set(
                    self._list_labels(ds_root, "train")
                    + self._list_labels(ds_root, "test")
                )
            )

            print(f"{name} labels: {len(ds_labels)}")

            # Semantic comparison against Food101
            comp = self.compare_food_labels(
                food101_labels=f101_labels,
                uecfood256_labels=ds_labels,
                threshold=threshold,
            )

            exact = set(comp["exact_matches"])
            fuzzy = {u for _, u, _ in comp["fuzzy_matches"]}
            skip_labels = exact | fuzzy

            print(f"{name} skipped labels (duplicates): {len(skip_labels)}")

            # Copy dataset, skipping duplicates
            self._copy_split(ds_root, "train", out_root, skip_labels)
            self._copy_split(ds_root, "test", out_root, skip_labels)

        print(f"\nMerged dataset created at: {out_root.resolve()}")
        
    def build_multimodal_dataset(
        self,
        json_source: str,
        train_dir: str,
        output_file: str,
    ):
        """
        Build a multimodal JSON dataset linking image paths to nutrient data.

        Args:
            json_source (str): Path to USDA-enriched JSON file.
            train_dir (str): Path to training dataset (with class subfolders).
            output_file (str): Path to save the final multimodal JSON.
        """

        # --------------------------------------------------
        # Load nutrient data
        # --------------------------------------------------
        with open(json_source, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []

        # Fixed instruction (can be externalized later)
        instruction = (
            "Given the food image, identify its nutritional components "
            "and their corresponding values."
        )

        train_dir = Path(train_dir)

        for item in data:
            raw_class = item.get("food_class", "")
            class_name = self.clean_label(raw_class)

            if not class_name:
                continue

            class_dir = train_dir / class_name
            if not class_dir.is_dir():
                print(f"Skipping missing class directory: {class_dir}")
                continue

            # Build components list
            components = []
            for nutrient in item.get("components", []):
                name = nutrient.get("name")
                value = nutrient.get("value")
                unit = nutrient.get("unit")
                if name and value is not None:
                    components.append(f"{name}: {value} {unit}".strip())

            # Link each image to the same nutritional profile
            for img in class_dir.iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                record = {
                    "image": str(img),
                    "label": class_name,
                    # "instruction": instruction,
                    "components": components,
                    "ingredients": item.get("ingredients", ""),
                    "orkg_link": item.get("id"),
                    "usda_link": item.get("usda_link", ""),
                }

                # Remove empty fields (except 0)
                record = {
                    k: v for k, v in record.items()
                    if v is not None
                    and v != ""
                    and v != []
                    and v != {}
                }

                records.append(record)
        # --------------------------------------------------
        # Save output
        # --------------------------------------------------
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"Multimodal dataset saved: {len(records)} samples -> {output_file}")
        
    def build_same_as_map(
        self,
        reference_labels: List[str],
        other_labels: List[str],
        threshold: float = 0.85,
    ) -> Dict[str, str]:
        """
        Build a mapping {other_label -> reference_label}
        using exact and fuzzy semantic matching.
        """

        comp = self.compare_food_labels(
            food101_labels=reference_labels,
            uecfood256_labels=other_labels,
            threshold=threshold,
        )

        same_as = {}

        # Exact matches
        for label in comp["exact_matches"]:
            same_as[label] = label

        # Fuzzy matches
        for ref, other, _ in comp["fuzzy_matches"]:
            same_as[other] = ref

        return same_as

        
    def merge_food_datasets_with_same_as(
        self,
        dataset_dirs: Dict[str, str],
        output_dir: str,
        threshold: float = 0.85,
    ):
        """
        Merge datasets into a unified structure while preserving
        semantic equivalences via a `same_as` key.

        Food101 is used as the semantic reference.
        """

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # Step 1: Load Food101 labels (reference)
        # --------------------------------------------------
        food101_root = Path(dataset_dirs["food101"])
        food101_labels = self.get_dataset_labels(str(food101_root))
        food101_labels = [self.clean_label(l) for l in food101_labels]

        merged_index = {
            label: {
                "label": label,
                "source": "food101",
                "same_as": None,
            }
            for label in food101_labels
        }
        # --------------------------------------------------
        # Step 3: Process other datasets
        # --------------------------------------------------
        for name, path in dataset_dirs.items():
            if name == "food101":
                continue

            ds_root = Path(path)
            ds_labels = self.get_dataset_labels(str(ds_root))
            ds_labels = [self.clean_label(l) for l in ds_labels]

            same_as_map = self.build_same_as_map(
                reference_labels=food101_labels,
                other_labels=ds_labels,
                threshold=threshold,
            )

            for label in ds_labels:
                entry = {
                    "label": label,
                    "source": name,
                    "same_as": same_as_map.get(label),
                }

                merged_index[f"{name}:{label}"] = entry

        # --------------------------------------------------
        # Step 4: Save merged index
        # --------------------------------------------------
        output_file = out_root / "merged_labels_with_same_as.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(list(merged_index.values()), f, indent=2, ensure_ascii=False)

        print(f"Merged dataset index saved to {output_file}")
    
    def analyze_image_dataset(
        self,
        dataset_root: str,
        splits=("train", "test"),
        extensions=(".jpg", ".jpeg", ".png"),
        portion: float = 0.05,
    ) -> Dict:  # sourcery skip: low-code-quality
        """
        Analyze a food image dataset with a folder-per-class structure.

        Notes:
        - total_images is computed on the full dataset
        - image-level statistics are computed on a sampled subset if portion < 1.0
        """

        if not (0 < portion <= 1.0):
            raise ValueError("portion must be in the range (0, 1].")

        dataset_root = Path(dataset_root)

        stats = {
            "dataset_root": str(dataset_root),
            "splits": {},
            "total_classes": 0,
            "total_images": 0,  # REAL total images (train + test)
            "images_per_class": {},
            "extensions": set(),
            "image_stats": {
                "avg_width": 0,
                "avg_height": 0,
                "avg_resolution": 0,
                "avg_file_size_kb": 0,
                "dataset_size_gb": 0,
                "sampled_portion": portion,
            },
        }

        all_classes = set()

        # Image-level (sampled) accumulators
        total_width = 0
        total_height = 0
        total_pixels = 0
        total_file_size = 0
        image_count_sampled = 0

        total_images_real = 0

        for split in splits:
            split_path = dataset_root / split
            if not split_path.exists():
                continue

            split_images = 0
            split_classes = 0

            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                all_classes.add(class_name)

                images = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in extensions
                ]

                num_images_class = len(images)
                split_images += num_images_class
                total_images_real += num_images_class
                split_classes += 1

                stats["images_per_class"].setdefault(class_name, 0)
                stats["images_per_class"][class_name] += num_images_class

                # --------------------------------------------------
                # Sampling ONLY for image-level statistics
                # --------------------------------------------------
                sampled_images = images
                if portion < 1.0 and images:
                    sample_size = max(1, int(len(images) * portion))
                    sampled_images = random.sample(images, sample_size)

                for img_path in sampled_images:
                    stats["extensions"].add(img_path.suffix.lower())

                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            total_width += w
                            total_height += h
                            total_pixels += w * h

                        total_file_size += img_path.stat().st_size
                        image_count_sampled += 1

                    except Exception:
                        continue

            stats["splits"][split] = {
                "num_classes": split_classes,
                "num_images": split_images,
            }

        stats["total_classes"] = len(all_classes)
        stats["total_images"] = total_images_real

        # --------------------------------------------------
        # Distribution statistics (exact)
        # --------------------------------------------------
        counts = list(stats["images_per_class"].values())
        if counts:
            stats["distribution"] = {
                "min_images_per_class": min(counts),
                "max_images_per_class": max(counts),
                "avg_images_per_class": round(sum(counts) / len(counts), 2),
            }
        else:
            stats["distribution"] = {
                "min_images_per_class": 0,
                "max_images_per_class": 0,
                "avg_images_per_class": 0,
            }

        # --------------------------------------------------
        # Image-level statistics (sampled)
        # --------------------------------------------------
        if image_count_sampled > 0:
            stats["image_stats"].update({
                "avg_width": round(total_width / image_count_sampled, 2),
                "avg_height": round(total_height / image_count_sampled, 2),
                "avg_resolution": round(total_pixels / image_count_sampled, 2),
                "avg_file_size_kb": round((total_file_size / image_count_sampled) / 1024, 2),
                "dataset_size_gb": round(total_file_size / (1024 ** 3), 3),
            })

        stats["extensions"] = sorted(stats["extensions"])

        return stats

def main():
    helper = Helper()
    splitter = DatasetSplitter()
    balancer = DatasetBalancer()

    while True:
        print("\n=== Dataset Utility CLI ===")
        print("1 - Check & clean dataset labels")
        print("2 - Prepare Food101 dataset (from HuggingFace)")
        print("3 - Rename UECFood256 folders (ID -> label)")
        print("4 - Add image URLs to USDA JSON")
        print("5 - Merge food datasets")
        print("6 - Build multimodal dataset (image + nutrients)")
        print("7 - Create train/test split")
        print("8 - Balance dataset (class imbalance)")
        print("9 - Analyze image dataset statistics")
        print("10 - Exit")

        choice = input("\nSelect an option (1-10): ").strip()

        # --------------------------------------------------
        # 1) Check & clean labels
        # --------------------------------------------------
        if choice == "1":
            results = helper.check_all_datasets()
            for dataset, info in results.items():
                print(f"\nDataset: {dataset}")
                print(f"  Total labels: {info['total']}")
                print(f"  Modified labels: {info['modified']}")
                print(f"  Unchanged labels: {info['unchanged']}")

        # --------------------------------------------------
        # 2) Prepare Food101 dataset
        # --------------------------------------------------
        elif choice == "2":
            output_dir = input("Output directory [food101]: ").strip() or "food101"
            helper.prepare_food101_dataset(output_dir=output_dir)

        # --------------------------------------------------
        # 3) Rename UECFood256 folders
        # --------------------------------------------------
        elif choice == "3":
            dataset_dir = input("UECFood256 dataset directory: ").strip()
            metadata_file = input("UECFood256 category.txt path: ").strip()

            helper.rename_uecfood256_folders(
                dataset_dir=dataset_dir,
                txt_file=metadata_file,
            )

        # --------------------------------------------------
        # 4) Add image URLs to USDA JSON
        # --------------------------------------------------
        elif choice == "4":
            input_json = input("Input USDA JSON file: ").strip()
            output_json = input("Output JSON file: ").strip()
            base_url = input("Base image URL: ").strip()

            helper.add_image_urls_to_usda_json(
                input_json=input_json,
                output_json=output_json,
                base_url=base_url,
            )

        # --------------------------------------------------
        # 5) Merge datasets
        # --------------------------------------------------
        elif choice == "5":
            dataset_dirs = {
                "food101": input("Food101 root directory: ").strip(),
                "uecfood256": input("UECFood256 root directory: ").strip(),
                "fruitveg81": input("FruitVeg81 root directory: ").strip(),
                "afd": input("AFD root directory: ").strip(),
            }

            output_dir = input("Output merged dataset directory: ").strip()

            helper.merge_food_datasets(
                dataset_dirs=dataset_dirs,
                output_dir=output_dir,
            )

        # --------------------------------------------------
        # 6) Build multimodal dataset
        # --------------------------------------------------
        elif choice == "6":
            json_source = input("USDA enriched JSON file: ").strip()
            train_dir = input("Train directory (merged dataset): ").strip()
            output_file = input("Output multimodal JSON file: ").strip()

            helper.build_multimodal_dataset(
                json_source=json_source,
                train_dir=train_dir,
                output_file=output_file,
            )

        # --------------------------------------------------
        # 7) Create train/test split
        # --------------------------------------------------
        elif choice == "7":
            dataset_path = input("Dataset root (class folders): ").strip()
            train_path = input("Output train directory: ").strip()
            test_path = input("Output test directory: ").strip()
            test_split = input("Test split ratio [0.25]: ").strip()

            test_split = float(test_split) if test_split else 0.25

            splitter.create_train_test_split(
                dataset_path=dataset_path,
                train_path=train_path,
                test_path=test_path,
                test_split=test_split,
            )

        # --------------------------------------------------
        # 8) Balance dataset
        # --------------------------------------------------
        elif choice == "8":
            source_path = input("Source dataset directory: ").strip()
            output_path = input("Balanced output directory: ").strip()
            target = input("Target images per class [1000]: ").strip()

            target = int(target) if target else 1000

            balancer.balance_dataset(
                source_path=source_path,
                output_path=output_path,
                target=target,
            )

        # --------------------------------------------------
        # 9) Analyze image dataset statistics
        # --------------------------------------------------
        elif choice == "9":
            dataset_path = input("Dataset directory to analyze: ").strip()
            portion = float(input("Portion of images to sample for stats [0.05]: ")) or 0.05

            stats = helper.analyze_image_dataset(dataset_path, portion=portion)

            print("\nDataset statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
        

        # --------------------------------------------------
        # 10) Exit
        # --------------------------------------------------
        elif choice == "10":
            print("Exiting...")
            break

        # --------------------------------------------------
        # Invalid option
        # --------------------------------------------------
        else:
            print("Invalid choice. Please select a number between 1 and 10.")



# if __name__ == "__main__":
#     main()
    

helper = Helper()

# AFD dataset
helper.build_multimodal_dataset(
    json_source="json/new/data_retrieve_from_orkg.json",
    train_dir="dataset/image/not_merged/AFD/test",
    output_file="dataset/multimodal/not_merged/test/AFD_test.json",
)
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/AFD/train",
#     output_file="dataset/multimodal/not_merged/train/AFD_train.json",
# )

# FruitVeg81 dataset
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/fruitveg81/test",
#     output_file="dataset/multimodal/not_merged/test/fruitveg81_test.json",
# )
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/fruitveg81/train",
#     output_file="dataset/multimodal/not_merged/train/fruitveg81_train.json",
# )

# UECFood256 dataset
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/uecfood256/test",
#     output_file="dataset/multimodal/not_merged/test/uecfood256_test.json",
# )
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/uecfood256/train",
#     output_file="dataset/multimodal/not_merged/train/uecfood256_train.json",
# )

# Food101 dataset
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/food101/test",
#     output_file="dataset/multimodal/not_merged/test/food101_test.json",
# )

# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/not_merged/food101/train",
#     output_file="dataset/multimodal/not_merged/train/food101_train.json",
# )

# Merged dataset
# helper.build_multimodal_dataset(
#     json_source="json/new/data_retrieve_from_orkg.json",
#     train_dir="dataset/image/merged/test",
#     output_file="dataset/multimodal/merged/merged_test.json",
# )
helper.build_multimodal_dataset(
    json_source="json/new/data_retrieve_from_orkg.json",
    train_dir="dataset/image/merged/train",
    output_file="dataset/multimodal/merged/merged_train.json",
)