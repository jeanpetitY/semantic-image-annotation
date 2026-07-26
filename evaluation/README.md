# Evaluation Module

This module evaluates nutrient-generation predictions against a ground-truth
reference file.

It reports:

- food-name accuracy
- component precision, recall, F1, and Jaccard
- value MAE
- `Accuracy@10%` for matched nutrient values
- abstention rate and answer rate

## Files

- `evaluator.py`: evaluation implementation and CLI logic
- `main.py`: executable entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12** — this module includes a `.python-version` file pinned to `3.12`
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Setup With uv

Go into the module folder:

```bash
cd evaluation
```

Install or synchronize the module environment:

```bash
uv sync
```

Show CLI help:

```bash
uv run food-evaluation --help
```

If you prefer running from the repository root, use:

```bash
uv --project evaluation run food-evaluation --help
```

## Input Contract

The evaluator expects:

- a ground-truth `.jsonld` file containing records with `label`, `image`, and `components`
- a prediction `.csv` file containing the columns `id`, `predicted_name`, and `components`

The `id` field is aligned with the convention:

```text
<label> - <image>
```

During evaluation, image paths inside `id` are normalized before matching.
For example, `../dataset/...` and `dataset/...` are treated as the same image.

The prediction `components` field may contain:

- a list of strings such as `["protein: 10 g", "iron: 1.2 mg"]`
- a list of dicts
- a list of 2-tuples
- the exact string `I don't know`

For ground-truth files, `components` may also be stored in ontology-style JSON
objects with `rdfs:label`, `hasUnit`, and `hasValue`. The evaluator normalizes
that structure before computing metrics.

## Usage

Standard evaluation:

```bash
uv run food-evaluation \
  --ground-json dataset/multimodal/not_merged/test/example_test.jsonld \
  --prediction-csv results/text-model/vision-model/example.csv
```

Evaluation on the released excerpt dataset:

```bash
uv run food-evaluation \
  --ground-json excerpt_dataset/annotation.jsonld \
  --prediction-csv results/text-model/vision-model/excerpt_rag.csv
```

Ablation evaluation:

```bash
uv run food-evaluation \
  --ground-json dataset/multimodal/not_merged/test/example_test.jsonld \
  --prediction-csv results/text-model/vision-model/example.csv \
  --is-ablation
```

Excerpt ablation evaluation:

```bash
uv run food-evaluation \
  --ground-json excerpt_dataset/annotation.jsonld \
  --prediction-csv results/text-model/vision-model/excerpt_ablation.csv \
  --is-ablation
```

Optional output path:

```bash
uv run food-evaluation \
  --ground-json dataset/multimodal/not_merged/test/example_test.jsonld \
  --prediction-csv results/text-model/vision-model/example.csv \
  --output-file results/text-model/vision-model/example_evaluation_metrics.json
```

## Output

If `--output-file` is omitted, the evaluator writes:

```text
<prediction_csv_dir>/<prediction_csv_stem>_evaluation_metrics.json
```

When every aligned prediction is an abstention (`I don't know`), the evaluator
reports:

- `Abstention rate = 1.0`
- `Answer rate = 0.0`
- `Precision`, `Recall`, `F1-score`, `Jaccard`, `MAE`, and `Accuracy@10%` as
  `null` in the JSON output

This behavior avoids reporting misleading zeros when no effective nutrient
comparison can be performed.

## Reproducibility Notes

- The evaluator is deterministic for fixed input files.
- Standard mode compares nutrient names after parsing and unit normalization.
- Ablation mode compares full component strings directly.
- Value comparison normalizes mass units to grams and energy units to kilocalories.
