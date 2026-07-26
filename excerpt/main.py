"""CLI entry point for building the released excerpt dataset.

Paper reference:
- Supplementary material describing the compact excerpt released for
  inspection and visualization in tools such as Protégé.
"""

import argparse

from excerpt.builder import DatasetExcerptBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an excerpt dataset from merged images and merged JSON-LD annotations."
    )
    parser.add_argument(
        "--image-root",
        default="dataset/image/merged/test",
        help="Root directory containing merged test images grouped by class.",
    )
    parser.add_argument(
        "--annotation-source",
        default="dataset/multimodal/merged/merged_test.jsonld",
        help="Source JSON-LD file containing merged linkage annotations.",
    )
    parser.add_argument(
        "--output-dir",
        default="excerpt_dataset",
        help="Output directory where images/ and annotation.jsonld will be written.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes to include in the excerpt, selected deterministically from the source file.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=45,
        help="Maximum total size of the generated excerpt directory in megabytes.",
    )
    return parser


def main() -> None:
    # The excerpt is derived from the merged test split plus the linked
    # JSON-LD annotations so the released sample stays consistent.
    args = build_parser().parse_args()
    builder = DatasetExcerptBuilder()
    summary = builder.build_excerpt(
        image_root=args.image_root,
        annotation_source=args.annotation_source,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        max_size_mb=args.max_size_mb,
    )
    print(summary)


if __name__ == "__main__":
    main()
