import argparse

from preprocessing.dataset_analysis import analyze_dataset_layout
from preprocessing.dataset_balancer import DatasetBalancer
from preprocessing.dataset_splitter import DatasetSplitter
from preprocessing.helper import Helper, helper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-processing utilities for dataset preparation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("interactive", help="Launch the interactive preprocessing menu.")
    subparsers.add_parser("check-labels", help="Check and summarize label normalization across datasets.")

    build_multimodal = subparsers.add_parser(
        "build-multimodal",
        help="Build a multimodal dataset by matching images with nutrient annotations.",
    )
    build_multimodal.add_argument("--json-source", required=True)
    build_multimodal.add_argument("--train-dir", required=True)
    build_multimodal.add_argument("--output-file", required=True)

    split_dataset = subparsers.add_parser(
        "split-dataset",
        help="Create a train/test split from a folder-per-class dataset.",
    )
    split_dataset.add_argument("--dataset-path", required=True)
    split_dataset.add_argument("--train-path", required=True)
    split_dataset.add_argument("--test-path", required=True)
    split_dataset.add_argument("--test-split", type=float, default=0.25)

    balance_dataset = subparsers.add_parser(
        "balance-dataset",
        help="Balance a folder-per-class dataset by downsampling or augmentation.",
    )
    balance_dataset.add_argument("--source-path", required=True)
    balance_dataset.add_argument("--output-path", required=True)
    balance_dataset.add_argument("--target", type=int, default=1000)

    analyze_dataset = subparsers.add_parser(
        "analyze-dataset",
        help="Compute dataset statistics for a folder-per-class dataset.",
    )
    analyze_dataset.add_argument("--dataset-path", required=True)
    analyze_dataset.add_argument("--portion", type=float, default=0.05)

    analyze_layout = subparsers.add_parser(
        "analyze-layout",
        help="Analyze class distribution for either a flat dataset or a train/test layout.",
    )
    analyze_layout.add_argument("--dataset-path", required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "interactive":
        return helper()

    helper_instance = Helper()

    if args.command == "check-labels":
        print(helper_instance.check_all_datasets())
        return

    if args.command == "build-multimodal":
        helper_instance.build_multimodal_dataset(
            json_source=args.json_source,
            train_dir=args.train_dir,
            output_file=args.output_file,
        )
        return

    if args.command == "split-dataset":
        DatasetSplitter().create_train_test_split(
            dataset_path=args.dataset_path,
            train_path=args.train_path,
            test_path=args.test_path,
            test_split=args.test_split,
        )
        return

    if args.command == "balance-dataset":
        DatasetBalancer().balance_dataset(
            source_path=args.source_path,
            output_path=args.output_path,
            target=args.target,
        )
        return

    if args.command == "analyze-dataset":
        print(
            helper_instance.analyze_image_dataset(
                dataset_path=args.dataset_path,
                portion=args.portion,
            )
        )
        return

    if args.command == "analyze-layout":
        print(analyze_dataset_layout(args.dataset_path))


if __name__ == "__main__":
    main()
