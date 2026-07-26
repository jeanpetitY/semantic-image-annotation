# Pre-processing Module

This module contains the dataset-preparation utilities used before annotation,
fine-tuning, inference, and evaluation.

It reorganizes the former `tools/` code as a standalone module without changing
the intended preprocessing workflows.

## Files

- `helper.py`: high-level preprocessing utilities
- `dataset_splitter.py`: train/test split creation
- `dataset_balancer.py`: class balancing through downsampling or augmentation
- `dataset_analysis.py`: lightweight dataset-distribution analysis
- `main.py`: executable entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12** — this module includes a `.python-version` file pinned to `3.12`
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Reproducibility At A Glance

For the strongest reproduction path, use the commands and conventions below:

1. Enter the module directory:

```bash
cd preprocessing
```

2. Recreate the module environment:

```bash
uv sync
```

3. Run the documented CLI from the locked environment:

```bash
uv run food-preprocessing --help
```

If exact dependency replay is required and a lockfile is present, prefer:

```bash
uv sync --frozen
```

## Input Dataset Contracts

This module works with two common image-dataset layouts.

Flat folder-per-class layout:

```text
<dataset_root>/
  <class_a>/
  <class_b>/
  ...
```

Split layout:

```text
<dataset_root>/
  train/
    <class_a>/
    <class_b>/
    ...
  test/
    <class_a>/
    <class_b>/
    ...
```

For multimodal construction, the module also expects a nutrient source file in
JSON format containing records with a food class and its associated components.

## Main Operations

The module provides reproducible utilities for:

- checking and normalizing dataset labels
- renaming dataset folders
- flattening nested image folders when required by a dataset source
- preparing train/test splits
- balancing classes toward a target number of images per class
- merging image datasets
- computing dataset statistics

## Fuzzy Label Matching

Semantic duplicate detection is implemented in
`Helper.fuzzy_match_food_labels(...)`.

Inputs:

- `reference_labels`
- `candidate_labels`
- `threshold`

Outputs:

- `exact_matches`
- `fuzzy_matches`

Behavior:

- labels are normalized with `clean_label()`
- string similarity is computed with `SequenceMatcher`
- token overlap is used as an additional safeguard before accepting a fuzzy match

This function is used by the dataset-merging workflow to avoid copying labels
that are already represented by an equivalent class in the reference dataset.

## Commands

Show CLI help:

```bash
uv run food-preprocessing --help
```

Launch the interactive menu:

```bash
uv run food-preprocessing interactive
```

Check label normalization across datasets:

```bash
uv run food-preprocessing check-labels
```

Create a train/test split:

```bash
uv run food-preprocessing split-dataset \
  --dataset-path dataset/image/source \
  --train-path dataset/image/train \
  --test-path dataset/image/test \
  --test-split 0.25
```

Balance a dataset:

```bash
uv run food-preprocessing balance-dataset \
  --source-path dataset/image/train \
  --output-path dataset/image/balanced \
  --target 1000
```

Analyze a dataset with image-level statistics:

```bash
uv run food-preprocessing analyze-dataset \
  --dataset-path dataset/image/merged \
  --portion 0.05
```

Analyze a dataset layout with the lightweight class-distribution analyzer:

```bash
uv run food-preprocessing analyze-layout \
  --dataset-path dataset/image/merged
```

If you prefer running from the repository root, use:

```bash
uv --project preprocessing run food-preprocessing --help
```

## Outputs

Depending on the command, this module may produce:

- split datasets under user-specified `train/` and `test/` folders
- balanced datasets under a user-specified output directory
- merged image datasets
- printed dataset statistics and label-matching summaries

The module does not impose a single output root; outputs are controlled by the
explicit paths passed to each command.

## Reproducibility Notes

- Preprocessing behavior is deterministic when the underlying helper uses fixed
  ordering and no random augmentation path is involved.
- Dataset balancing and sampling steps may rely on randomized behavior.
  Reproducing those outputs exactly therefore requires keeping the same code,
  inputs, and Python random-state behavior.
- The module removes the previous import-time side effects from `tools/`, so the
  documented CLI is now the authoritative execution path.
- The previous standalone `utils.py` and `utils1.py` analyzers have been
  consolidated into `dataset_analysis.py`.
