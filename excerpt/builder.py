"""Build the supplementary excerpt of the semantified dataset.

Paper reference:
- Introduction statement describing an excerpt of semantically annotated images
  distributed as supplementary material.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_IMAGE_KEY = "http://example.org/food#hasImage"


def resolve_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return project_candidate


class DatasetExcerptBuilder:
    DEFAULT_MAX_SIZE_MB = 45
    DISK_OVERHEAD_RESERVE_MB = 2

    @staticmethod
    def clean_label(name: str) -> str:
        name = name.lower().strip()
        name = name.replace("&", "and")
        name = re.sub(r"[ \-_/']", "_", name)
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    def load_annotation_records(self, annotation_source: Path) -> list[dict]:
        with open(annotation_source, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Annotation source must contain a JSON array of records.")

        return data

    def extract_image_path(self, record: dict) -> str | None:
        if "image" in record and record["image"]:
            return str(record["image"])
        values = record.get(HAS_IMAGE_KEY, [])
        if not isinstance(values, list) or not values:
            return None
        first_item = values[0]
        if isinstance(first_item, dict):
            return first_item.get("@value")
        return str(first_item)

    def extract_class_name(self, image_path: str) -> str:
        image_path_obj = Path(image_path)
        if len(image_path_obj.parts) < 2:
            raise ValueError(f"Unsupported image path in annotation record: {image_path}")
        return self.clean_label(image_path_obj.parent.name)

    def extract_record_class_name(self, record: dict) -> str | None:
        if "label" in record and record["label"]:
            return self.clean_label(str(record["label"]))

        image_path = self.extract_image_path(record)
        if image_path:
            return self.extract_class_name(image_path)

        return None

    def build_class_source_index(self, not_merged_root: Path) -> dict[str, list[str]]:
        # Paper dataset-construction step: the merged dataset comes from several
        # source collections, so we preserve that provenance for the excerpt.
        class_sources: dict[str, list[str]] = {}
        if not not_merged_root.is_dir():
            return class_sources

        for dataset_dir in sorted(not_merged_root.iterdir()):
            test_dir = dataset_dir / "test"
            if not test_dir.is_dir():
                continue

            dataset_name = self.clean_label(dataset_dir.name)
            for class_dir in sorted(test_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = self.clean_label(class_dir.name)
                class_sources.setdefault(class_name, []).append(dataset_name)

        return class_sources

    def select_classes(
        self,
        records: list[dict],
        num_classes: int,
        class_sources: dict[str, list[str]],
    ) -> list[str]:
        # Paper supplementary excerpt: class selection is deterministic and
        # diversified across the original source datasets.
        if num_classes <= 0:
            raise ValueError("--num-classes must be greater than 0.")

        classes = []
        seen = set()
        for record in records:
            class_name = self.extract_record_class_name(record)
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            classes.append(class_name)

        dataset_buckets: dict[str, list[str]] = defaultdict(list)
        dataset_order: list[str] = []
        fallback_classes: list[str] = []

        for class_name in classes:
            sources = class_sources.get(class_name, [])
            if not sources:
                fallback_classes.append(class_name)
                continue

            dataset_name = sources[0]
            if dataset_name not in dataset_buckets:
                dataset_order.append(dataset_name)
            dataset_buckets[dataset_name].append(class_name)

        selected: list[str] = []
        indices = {dataset_name: 0 for dataset_name in dataset_order}

        progress = True
        while len(selected) < num_classes and progress:
            progress = False
            for dataset_name in dataset_order:
                bucket = dataset_buckets[dataset_name]
                index = indices[dataset_name]
                if index >= len(bucket):
                    continue
                selected.append(bucket[index])
                indices[dataset_name] += 1
                progress = True
                if len(selected) >= num_classes:
                    break

        for class_name in fallback_classes:
            if len(selected) >= num_classes:
                break
            selected.append(class_name)

        if not selected:
            raise ValueError("No class could be extracted from the annotation source.")

        return selected

    def resolve_source_image(
        self,
        image_root: Path,
        annotation_source: Path,
        image_path: str,
    ) -> Path:
        normalized = image_path.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        while normalized.startswith("../"):
            normalized = normalized[3:]

        path_from_repo = resolve_path(normalized)
        if path_from_repo.is_file():
            return path_from_repo

        class_name = Path(normalized).parent.name
        file_name = Path(normalized).name
        fallback = image_root / class_name / file_name
        if fallback.is_file():
            return fallback.resolve()

        candidate_from_annotation = (annotation_source.parent / image_path).resolve()
        if candidate_from_annotation.is_file():
            return candidate_from_annotation

        raise FileNotFoundError(f"Source image not found for annotation path: {image_path}")

    def rewrite_record_image_path(self, record: dict, class_name: str, file_name: str) -> dict:
        updated = json.loads(json.dumps(record))
        rewritten_path = f"images/{class_name}/{file_name}"
        if "image" in updated:
            updated["image"] = rewritten_path
        else:
            updated[HAS_IMAGE_KEY] = [{"@value": rewritten_path}]
        return updated

    def is_augmented_image(self, image_path: Path) -> bool:
        return image_path.name.lower().startswith("aug_")

    def allocated_size(self, path: Path) -> int:
        stat = path.stat()
        return getattr(stat, "st_blocks", 0) * 512 or stat.st_size

    def build_excerpt(
        self,
        image_root: str,
        annotation_source: str,
        output_dir: str,
        num_classes: int,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
    ) -> dict:
        # Paper supplementary material: keep a representative multimodal subset
        # under a strict size budget while preserving source-dataset diversity.
        image_root_path = resolve_path(image_root)
        annotation_source_path = resolve_path(annotation_source)
        output_dir_path = resolve_path(output_dir)
        output_images_dir = output_dir_path / "images"
        output_annotation_path = output_dir_path / "annotation.jsonld"
        not_merged_root = PROJECT_ROOT / "dataset/image/not_merged"
        max_size_bytes = max_size_mb * 1024 * 1024
        safety_reserve_bytes = self.DISK_OVERHEAD_RESERVE_MB * 1024 * 1024
        source_records = self.load_annotation_records(annotation_source_path)
        class_sources = self.build_class_source_index(not_merged_root)
        selected_classes = self.select_classes(source_records, num_classes, class_sources)
        selected_class_sources = {
            class_name: class_sources.get(class_name, [])
            for class_name in selected_classes
        }

        if not image_root_path.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {image_root_path}")
        if not annotation_source_path.is_file():
            raise FileNotFoundError(
                f"Annotation source does not exist: {annotation_source_path}"
            )

        if output_images_dir.exists():
            shutil.rmtree(output_images_dir)
        if output_annotation_path.exists():
            output_annotation_path.unlink()
        output_images_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        written_records = 0
        copied_bytes = 0
        estimated_annotation_bytes = 2
        seen_classes = set()
        missing_classes = set(selected_classes)
        skipped_size_limit = 0
        total_source_records_selected = 0
        total_source_images_selected = 0

        grouped_records = defaultdict(list)
        for record in source_records:
            image_path = self.extract_image_path(record)
            if not image_path:
                continue

            class_name = self.extract_class_name(image_path)
            if class_name not in selected_classes:
                continue

            total_source_records_selected += 1
            source_image = self.resolve_source_image(
                image_root=image_root_path,
                annotation_source=annotation_source_path,
                image_path=image_path,
            )
            total_source_images_selected += 1
            if self.is_augmented_image(source_image):
                continue

            grouped_records[class_name].append((record, source_image))

        excerpt_records = []
        record_index = {class_name: 0 for class_name in selected_classes}
        progress = True
        while progress:
            progress = False
            for class_name in selected_classes:
                items = grouped_records.get(class_name, [])
                index = record_index[class_name]
                if index >= len(items):
                    continue

                progress = True
                record, source_image = items[index]
                record_index[class_name] += 1

                updated_record = self.rewrite_record_image_path(
                    record=record,
                    class_name=class_name,
                    file_name=source_image.name,
                )
                record_json = json.dumps(updated_record, ensure_ascii=False)
                record_size = len(record_json.encode("utf-8"))
                separator_overhead = 1 if excerpt_records else 0

                destination_dir = output_images_dir / class_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination_image = destination_dir / source_image.name

                shutil.copy2(source_image, destination_image)
                image_size = self.allocated_size(destination_image)
                projected_size = (
                    copied_bytes
                    + estimated_annotation_bytes
                    + separator_overhead
                    + record_size
                    + image_size
                    + safety_reserve_bytes
                )
                if projected_size > max_size_bytes:
                    destination_image.unlink()
                    if not any(destination_dir.iterdir()):
                        destination_dir.rmdir()
                    skipped_size_limit += 1
                    continue

                excerpt_records.append(updated_record)

                copied_images += 1
                copied_bytes += image_size
                estimated_annotation_bytes += separator_overhead + record_size
                written_records += 1
                seen_classes.add(class_name)
                missing_classes.discard(class_name)

        with open(output_annotation_path, "w", encoding="utf-8") as output_file:
            json.dump(excerpt_records, output_file, ensure_ascii=False)

        annotation_size = (
            self.allocated_size(output_annotation_path)
            if output_annotation_path.exists()
            else 0
        )
        total_size_bytes = copied_bytes + annotation_size

        return {
            "output_dir": str(output_dir_path),
            "annotation_file": str(output_annotation_path),
            "selected_classes": selected_classes,
            "selected_class_sources": selected_class_sources,
            "classes_written": sorted(seen_classes),
            "missing_classes": sorted(missing_classes),
            "total_selected_classes": len(selected_classes),
            "total_source_records_selected": total_source_records_selected,
            "total_source_images_selected": total_source_images_selected,
            "records_written": written_records,
            "images_copied": copied_images,
            "skipped_size_limit": skipped_size_limit,
            "max_size_mb": max_size_mb,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        }
