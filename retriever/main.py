"""CLI entry point for data retrieval from USDA and ORKG.

Paper reference:
- The enrichment phase combines USDA-derived nutrient data and ORKG exports
  before constructing the semantified multimodal dataset.
"""

import argparse
import sys

from retriever import get_from_orkg, get_from_usda


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retriever module for USDA enrichment and ORKG export."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    subparsers.add_parser("usda", help="Run USDA enrichment.")
    subparsers.add_parser("orkg", help="Export data from ORKG.")

    return parser


def main():
    parser = build_parser()
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.task == "usda":
        # USDA retrieval supports the nutrient-enrichment branch of the paper.
        return get_from_usda.main()

    # ORKG export supports the structured-knowledge branch used before linkage.
    return get_from_orkg.main()


if __name__ == "__main__":
    main()
