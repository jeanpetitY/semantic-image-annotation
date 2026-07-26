"""CLI entry point for the image-to-KG linkage stage.

Paper reference:
- "An Approach for Image Annotation via Knowledge Graph Linkage", where each
  image is aligned with a semantically structured JSON-LD description.
"""

import argparse

from linkage.linker import ImageTextLinker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Link each image to its textual food description and export JSON-LD."
    )
    parser.add_argument(
        "--json-source",
        required=True,
        help="Path to the source JSON file containing food metadata and components.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Root directory containing one subdirectory per food class.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output JSON-LD file.",
    )
    return parser


def main():
    # Keep the entry point thin so the linkage logic remains centralized in the
    # implementation file that mirrors the paper methodology.
    args = build_parser().parse_args()
    linker = ImageTextLinker()
    count = linker.build_jsonld(
        json_source=args.json_source,
        image_dir=args.image_dir,
        output_file=args.output_file,
    )
    print(f"Linked {count} images into {args.output_file}")


if __name__ == "__main__":
    main()
