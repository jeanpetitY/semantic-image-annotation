# Retriever Module

This module contains the data-retrieval utilities used upstream of inference and
evaluation.

It supports two workflows:

- USDA enrichment from dataset labels
- ORKG export from existing graph resources

## Files

- `get_from_usda.py`: retrieve USDA nutrient data for dataset labels
- `get_from_orkg.py`: export food resources from ORKG
- `main.py`: unified entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Environment Variables

Depending on the workflow, the module may require:

- `USDA_KEY`
- `USDA_BASE_URL`
- `BASE_IMAGE_URL`
- `OPENAI_API_KEY`
- `EMAIL`
- `PASSWORD`
- `ORKG_HOST`
- `ORKG_FOOD_CLASS_ID`
- `ORKG_COMPONENT_CLASS_ID`

## Setup With uv

```bash
cd retriever
uv sync
uv run food-retriever --help
```

If exact dependency replay is required and a lockfile is present, prefer:

```bash
uv sync --frozen
```

From the repository root:

```bash
uv --project retriever run food-retriever --help
```

## Usage

USDA enrichment:

```bash
uv run food-retriever-usda \
  --dataset-dir dataset/image/not_merged/example/train \
  --output-file json/old/example_usda_enriched.json
```

Inspect retrieval coverage against a dataset label directory:

```bash
uv run food-retriever-stats \
  json/old/example_usda_enriched.json \
  dataset/image/not_merged/example/test
```

ORKG export from explicit entity IDs:

```bash
uv run food-retriever-orkg \
  --mode entity-ids \
  --entity-ids ENTITY_001 ENTITY_002 \
  --output-file exported_foods.json
```

ORKG export by food-class scan:

```bash
uv run food-retriever-orkg \
  --mode class-scan \
  --output-file exported_foods.json
```

## Input Contracts

USDA enrichment expects:

- a folder-per-class image dataset directory passed through `--dataset-dir`
- valid API credentials and optional image-base URL metadata

ORKG export expects:

- valid ORKG credentials
- either an explicit list of entity IDs or a class-scan configuration
- deployment-specific class identifiers supplied through environment variables

## Outputs

The retriever produces:

- USDA-enriched JSON files from dataset labels
- ORKG-exported JSON files from existing graph resources
- retrieval-coverage summaries printed to the console through `food-retriever-stats`

Output locations are fully controlled by the CLI arguments.

## Reproducibility Notes

- USDA enrichment depends on the upstream USDA API, optional LLM-based semantic
  validation, and the exact dataset label directory provided as input.
- ORKG export depends on the current state of the ORKG instance and the
  entity-selection mode used for the export.
- Exact replay therefore requires the same credentials, the same external
  services, the same input dataset labels, and the same command arguments.
