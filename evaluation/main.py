"""CLI entry point for nutrient-generation evaluation.

Paper reference:
- Experimental section reporting abstention-aware and component-level
  evaluation metrics on the semantified benchmark.
"""

from evaluation.evaluator import main


if __name__ == "__main__":
    main()
