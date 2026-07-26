import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import ablation, food_classifier, nutrient_generator, semantic_retrieval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified inference entrypoint for classification and nutrient generation."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Run food classification evaluation.",
    )
    for action in food_classifier.build_parser()._actions:
        if action.dest != "help":
            classify_parser._add_action(action)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run nutrient generation inference.",
    )
    for action in nutrient_generator.build_parser()._actions:
        if action.dest != "help":
            generate_parser._add_action(action)

    semantic_parser = subparsers.add_parser(
        "semantic-retrieval",
        help="Run the semantic-retrieval baseline.",
    )
    for action in semantic_retrieval.build_parser()._actions:
        if action.dest != "help":
            semantic_parser._add_action(action)

    ablation_parser = subparsers.add_parser(
        "ablation",
        help="Run the progressive RAG ablation study.",
    )
    for action in ablation.build_parser()._actions:
        if action.dest != "help":
            ablation_parser._add_action(action)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args_dict = vars(args).copy()
    task = args_dict.pop("task")

    forwarded_args = []
    for key, value in args_dict.items():
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                forwarded_args.append(option)
        elif value is not None:
            forwarded_args.extend([option, str(value)])

    if task == "classify":
        return food_classifier.main(forwarded_args)
    if task == "semantic-retrieval":
        return semantic_retrieval.main(forwarded_args)
    if task == "ablation":
        return ablation.main(forwarded_args)

    return nutrient_generator.main(forwarded_args)


if __name__ == "__main__":
    main()
