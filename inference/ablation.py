"""Progressive RAG ablation for nutrient generation.

Paper reference:
- Ablation setting where only a controlled fraction of samples uses KG-based
  retrieval, enabling cumulative comparison against the no-RAG baseline.
"""

import argparse
from inference.generation_core import build_assistant, build_shared_parser


def build_parser() -> argparse.ArgumentParser:
    parser = build_shared_parser("Progressive RAG ablation for nutrient generation.")
    parser.add_argument(
        "--rag-ratio",
        type=float,
        default=0.2,
        help="RAG ratio for ablation (0.2, 0.4, 0.6, 0.8, 1.0)",
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
    assistant.process_json_and_predict_ablation(
        input_file=args.input_file,
        output_file=args.output_file,
        is_selective=args.selective,
        rag_ratio=args.rag_ratio,
    )
    print("\nAblation finished successfully.")


if __name__ == "__main__":
    main()
