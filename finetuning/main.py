"""CLI entry point for fine-tuning the vision backbones.

Paper reference:
- Experimental setup used to train CLIP, BEiT, and DINOv3 food classifiers.
"""

try:
    from .train_classifier import ImageClassificationTrainer, parse_arguments
except ImportError:
    from train_classifier import ImageClassificationTrainer, parse_arguments


def main() -> None:
    # Training orchestration is kept here, while model-specific details remain
    # in the trainer implementation documented against the paper settings.
    args = parse_arguments()
    trainer = ImageClassificationTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
