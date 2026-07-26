# Annotation Module

This module enriches food labels with USDA FoodData Central nutritional data.

It can:

- read food class labels from an image-folder dataset directory;
- search matching foods in the USDA API;
- validate label equivalence with an LLM;
- write an enriched JSON file containing nutrients, ingredients, USDA source, and image URL.
- write labels that could not be validated to a `not_annotated` JSON file.
- evaluate annotation quality against a strict gold file.

## Files

- `get_from_usda.py`: USDA enrichment implementation.
- `apply_corrections.py`: applies human-in-the-loop annotation corrections.
- `annotation_stats.py`: compares annotated labels with dataset labels and reports missing coverage.
- `evaluation.py`: strict annotation evaluation implementation.
- `main.py`: executable entry point for this module.
- `pyproject.toml`: local `uv` project configuration for annotation only.
- `.python-version`: Python version used by this module.

## Output Organization

Generated files stay inside the module:

```text
annotation/
  evaluation/
    fruitveg81_gold.json
    fruitveg81_human_corrections.json
  outputs/
    fruitveg81/
      fruitveg81_usda.json
      fruitveg81_usda_raw.json
      fruitveg81_usda_corrected.json
    food101/
      food101_usda.json
      food101_usda_raw.json
      food101_not_annotated.json
    uecfood256/
```

## Environment Variables

Required:

```bash
USDA_KEY=your_usda_api_key
```

Optional:

```bash
USDA_BASE_URL=https://example.org/usda-api
BASE_IMAGE_URL=https://your-image-base-url
OPENAI_API_KEY=your_openai_key
```

`OPENAI_API_KEY` is only required if you use an external validator model. The
default validator remains configurable through `--model-name`.

## Setup With uv

Go into the module folder:

```bash
cd annotation
```

Install/synchronize the module environment:

```bash
uv sync
```

Show CLI help:

```bash
uv run image-annotation --help
```

Show annotation-coverage stats help:

```bash
uv run image-annotation-stats --help
```

## Usage

Fruit/vegetable mode:

```bash
uv run image-annotation \
  --food-labels ../dataset/image/not_merged/fruitveg81/test \
  --output-file outputs/fruitveg81/fruitveg81_usda.json \
  --not-annotated-file outputs/fruitveg81/fruitveg81_not_annotated.json \
  --fruit-veg \
  --image-base-url https://example.org/images/fruitveg81/
```

`--fruit-veg` keeps the FruitVeg81 workflow simple: the script appends `raw`
to the USDA query and keeps the first USDA result. It does not use the LLM and
does not search by data type priority.

General food mode with the default validator:

```bash
uv run image-annotation \
  --food-labels ../dataset/image/not_merged/food101/test \
  --output-file outputs/food101/food101_usda_raw.json \
  --not-annotated-file outputs/food101/food101_not_annotated.json \
  --image-base-url https://example.org/images/food101/
```

Use an alternative validator instead:

```bash
uv run image-annotation \
  --food-labels ../dataset/image/not_merged/food101/test \
  --output-file outputs/food101/food101_usda_raw.json \
  --not-annotated-file outputs/food101/food101_not_annotated.json \
  --image-base-url https://example.org/images/food101/ \
  --model-name example-validator
```

UECFood256 uses the same flow with its own output folder:

```bash
uv run image-annotation \
  --food-labels ../dataset/image/not_merged/uecfood256/test \
  --output-file outputs/uecfood256/uecfood256_usda_raw.json \
  --not-annotated-file outputs/uecfood256/uecfood256_not_annotated.json \
  --image-base-url https://example.org/images/uecfood256/
```

If you prefer running from the repository root, use:

```bash
uv --project annotation run image-annotation --help
```

## Evaluation

Each dataset can have a strict gold file where each `food_class` maps to one
expected USDA `fdc_id`.

```json
[
  {
    "food_class": "damsons",
    "expected_fdc_id": 2119551
  }
]
```

For reproducible experiments, keep the raw annotations and human corrections
separate:

```text
outputs/fruitveg81/fruitveg81_usda_raw.json
evaluation/fruitveg81_human_corrections.json
outputs/fruitveg81/fruitveg81_usda_corrected.json
```

Evaluate raw `fruitveg81` annotations before human correction:

```bash
uv run image-annotation-eval \
  --annotations outputs/fruitveg81/fruitveg81_usda_raw.json \
  --gold evaluation/fruitveg81_gold.json
```

Apply human-in-the-loop corrections:

```bash
uv run image-annotation-apply-corrections \
  --annotations outputs/fruitveg81/fruitveg81_usda_raw.json \
  --corrections evaluation/fruitveg81_human_corrections.json \
  --output outputs/fruitveg81/fruitveg81_usda_corrected.json \
  --fetch-usda-details
```

Inspect annotation coverage against a dataset label directory:

```bash
uv run image-annotation-stats \
  outputs/uecfood256/uecfood256_usda_raw.json \
  ../dataset/image/not_merged/uecfood256/test
```

Evaluate corrected annotations:

```bash
uv run image-annotation-eval \
  --annotations outputs/fruitveg81/fruitveg81_usda_corrected.json \
  --gold evaluation/fruitveg81_gold.json
```

Use `--output-report path/to/report.json` only when you want to persist a
generated evaluation report.

The evaluator reports:

- `accuracy`: correct annotations divided by expected annotations.
- `precision`: correct annotations divided by produced annotations, with wrong IDs counted as false positives.
- `recall`: correct annotations divided by annotations that should be correct, with wrong or missing IDs counted as false negatives.
- `f1_score`: harmonic mean of precision and recall.

## Output

The output JSON contains entries shaped like:

```json
{
  "food_class": "apple",
  "portion": "100 g",
  "usda_name": "APPLE, RAW",
  "usda_data_type": "Foundation",
  "usda_food_category": "Fruits and Fruit Juices",
  "description": "APPLE, RAW",
  "ingredients": [],
  "nutrients": [
    {
      "name": "Energy",
      "value": 52,
      "unit": "KCAL"
    }
  ],
  "usda_source": "https://fdc.nal.usda.gov/food-details/<fdc_id>/nutrients",
  "image": "https://my-bucket/fruitveg81/apple.jpg"
}
```

## Notes

- USDA requests use a timeout; override it with `--timeout`.
- If `--image-base-url` and `BASE_IMAGE_URL` are both missing, the `image` field is empty.
- For dish datasets such as Food101 and UECFood256, USDA data types are searched in this order: `Survey (FNDDS)`, `Foundation`, `SR Legacy`, `Experimental`, `Branded`.
- For each data type, the LLM chooses one semantically equivalent label among the top candidates, or returns `none`.
- If no candidate is selected after all data types are checked, the label is written to `not_annotated`.
- In `--fruit-veg` mode, the script appends `raw` to USDA queries and keeps the first USDA result without LLM validation.
