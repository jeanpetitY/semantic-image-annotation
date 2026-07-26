"""Semantic-retrieval baseline for nutrient prediction.

Paper reference:
- Evaluation setting where the image is matched to the nearest semantic record
  through CLIP retrieval, without invoking the generative RAG prompt.
"""

from inference.generation_core import build_assistant, build_shared_parser


def build_parser():
    return build_shared_parser("Semantic retrieval baseline for nutrient inference.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    assistant = build_assistant(args.index_name)
    assistant.predict_with_semantic_search(
        input_file=args.input_file,
        output_file=args.output_file,
    )
    print("\nSemantic retrieval finished successfully.")


if __name__ == "__main__":
    main()
