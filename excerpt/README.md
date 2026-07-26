# Excerpt Module

This module creates a compact excerpt of the merged test dataset.

It writes:

- an `images/` directory containing one subdirectory per selected class
- an `annotation.jsonld` file containing the filtered linkage annotations

By design, the excerpt:
- keeps the output directory at the repository root
- enforces a maximum total size of `45` MB by default

The excerpt is built from:

- `dataset/image/merged/test`
- `dataset/multimodal/merged/merged_test.jsonld`

## Files

- `builder.py`: excerpt construction logic
- `main.py`: executable entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Reproducibility At A Glance

1. Enter the module directory:

```bash
cd excerpt
```

2. Recreate the module environment:

```bash
uv sync
```

3. Show the CLI help:

```bash
uv run food-excerpt --help
```

## Input Contract

The module expects:

- an image root directory with one subdirectory per class
- a JSON-LD annotation file containing ontology-style linkage records
- a target number of classes given with `--num-classes`
- a size limit in megabytes given with `--max-size-mb` when the default value `45` must be changed

Each selected annotation record must contain:

- `http://example.org/food#hasImage`

The class is inferred from the parent directory of the image path.
The selected classes are chosen deterministically while preserving diversity
across the original source datasets represented in
`dataset/image/not_merged/*/test`.

## Output Contract

The module creates:

```text
<output_dir>/
  images/
    <class_1>/
    <class_2>/
    ...
  annotation.jsonld
```

For every retained record, `http://example.org/food#hasImage` is rewritten to:

```text
images/<class>/<filename>
```

All other JSON-LD fields are preserved.

The total excerpt size is capped with `--max-size-mb` and defaults to `50`.
The excerpt is then filled progressively and may keep only part of the images
from the selected classes in order to stay within the size budget.

## Usage

From inside the module directory:

```bash
uv run food-excerpt \
  --output-dir ../excerpt_dataset \
  --num-classes 20 \
  --max-size-mb 50
```

From the repository root:

```bash
uv --project excerpt run food-excerpt \
  --image-root dataset/image/merged/test \
  --annotation-source dataset/multimodal/merged/merged_test.jsonld \
  --output-dir excerpt_dataset \
  --num-classes 20 \
  --max-size-mb 45
```

Once the excerpt has been created, its outputs can be reused directly by the
other modules:

- `excerpt_dataset/images` for classification and dataset analysis
- `excerpt_dataset/annotation.jsonld` for nutrient inference and evaluation

Typical follow-up commands from the repository root:

```bash
uv --project preprocessing run food-preprocessing analyze-layout \
  --dataset-path excerpt_dataset/images
```

```bash
uv --project inference run food-inference generate \
  --index-name example-index \
  --input-file excerpt_dataset/annotation.jsonld \
  --output-file results/text-model/vision-model/excerpt_rag.csv \
  --mode rag \
  --selective
```

```bash
uv --project evaluation run food-evaluation \
  --ground-json excerpt_dataset/annotation.jsonld \
  --prediction-csv results/text-model/vision-model/excerpt_rag.csv
```

## Reproducibility Notes

- The excerpt is deterministic for fixed inputs and a fixed class count.
- Class selection is diversified across the original merged datasets instead of
  taking all classes from a single source first.
- Annotation records are written in the same order as they appear in the source JSON-LD file.
- Image metadata is preserved, except for the rewritten relative `hasImage` path.
- The module copies only the non-augmented images referenced by the retained JSON-LD records.
- The module stops adding new records when the configured size budget is reached.
- The excerpt may therefore contain only a subset of the available images for the selected classes.
- The command summary reports both the total available images for the selected classes and the number actually copied into the excerpt.

## Protégé Note

`annotation.jsonld` is written as a JSON array of JSON-LD-style records so that
it is easier to inspect and exchange than JSON Lines.
