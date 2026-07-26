# Importer Module

This module imports USDA-enriched food data into ORKG.

## Files

- `import_data.py`: import implementation
- `main.py`: executable entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Environment Variables

- `EMAIL`
- `PASSWORD`
- `ORKG_HOST`
- `ORKG_FOOD_CLASS_PRIMARY`
- `ORKG_FOOD_CLASS_SECONDARY`
- `ORKG_COMPONENT_CLASS`
- `ORKG_DATASET_CLASS`
- `ORKG_PROP_HAS_COMPONENT`
- `ORKG_PROP_COMPONENT_NAME`
- `ORKG_PROP_COMPONENT_VALUE`
- `ORKG_PROP_NAME`
- `ORKG_PROP_SOURCE`
- `ORKG_PROP_IMAGE`
- `ORKG_PROP_PORTION`
- `ORKG_DATASET_RESOURCE`

## Setup With uv

```bash
cd importer
uv sync
uv run food-importer --help
```

If exact dependency replay is required and a lockfile is present, prefer:

```bash
uv sync --frozen
```

From the repository root:

```bash
uv --project importer run food-importer --help
```

## Usage

```bash
uv run food-importer \
  --input-file json/old/example_usda_enriched.json \
  --dataset-name example_dataset \
  --start 0 \
  --end 100
```

If you prefer running from the repository root, use:

```bash
uv --project importer run food-importer --help
```

## Input Contract

The importer expects a USDA-enriched JSON file containing one food record per
entry, with nutrient and metadata fields already prepared upstream.

The key parameters controlling a reproducible import are:

- `--input-file`
- `--dataset-name`
- `--start`
- `--end`
- `--host`

## Reproducibility Notes

- Import behavior depends on the exact content of the input JSON file and the
  target ORKG instance and the deployment-specific class/predicate identifiers
  supplied through the environment.
- Exact replay therefore requires the same input file, the same ORKG host, and
  credentials with comparable permissions.
- The importer writes to an external knowledge-graph service, so reproducibility
  is operational rather than bitwise.
