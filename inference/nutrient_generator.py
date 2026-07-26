"""Nutrient generation for the plain and KG-grounded settings.

Paper reference:
- Evaluation section comparing no-RAG generation against KG-grounded RAG.
"""

import argparse
from inference.generation_core import build_assistant, build_shared_parser


def build_parser() -> argparse.ArgumentParser:
    parser = build_shared_parser("Food nutrient generation for no-RAG and KG-grounded RAG.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "no_rag"],
        default="no_rag",
        help="Execution mode",
    )

    parser.add_argument(
        "--selective",
        action="store_true",
        help="Enable selective prompting",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    assistant = build_assistant(args.index_name)
    is_rag = args.mode == "rag"
    assistant.process_json_and_predict(
        input_file=args.input_file,
        output_file=args.output_file,
        is_rag=is_rag,
        is_selective=args.selective,
    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
