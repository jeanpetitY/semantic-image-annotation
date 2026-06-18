try:
    from .train_classifier import ImageClassificationTrainer, parse_arguments
except ImportError:
    from train_classifier import ImageClassificationTrainer, parse_arguments


def main() -> None:
    args = parse_arguments()
    trainer = ImageClassificationTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
