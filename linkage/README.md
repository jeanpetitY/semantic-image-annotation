# Linkage Module

This module links each food image to its associated textual food description and
exports a JSON-LD file containing ontology-style records.

It is the dedicated module for image-text linkage.

## Files

- `linker.py`: linkage implementation
- `main.py`: executable entry point for the module
- `pyproject.toml`: local `uv` project configuration
- `.python-version`: Python version used by this module

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment

## Reproducibility At A Glance

1. Enter the module directory:

```bash
cd linkage
```

2. Recreate the module environment:

```bash
uv sync
```

3. Run the documented CLI:

```bash
uv run food-linkage --help
```

## Input Contract

The module expects:

- a source JSON file with one food record per entry, including at least:
  - `food_class`
  - `components`
  - optionally `food_name`, `description`, `ingredients`, `id`, and `usda_link`
- an image root directory containing one subdirectory per normalized food class

The class directory name is matched using the same label normalization strategy
as the existing preprocessing code.

In the intended pipeline, the source JSON file is the output produced by
`retriever/get_from_orkg.py`.

## Output Format

The module writes a JSON-LD file as a single JSON array of records.

Each linked image record includes:

- `@id`
- `@type`
- `http://www.w3.org/2000/01/rdf-schema#label`
- `http://example.org/food#hasImage`
- `http://example.org/food#hasTextDescription`
- `http://example.org/food#hasComponent`

Optional fields may include:

- `http://example.org/food#hasIngredient`
- `http://example.org/food#hasORKGLink`
- `http://example.org/food#hasUSDALink`

Each linked component record includes:

- `@id`
- `@type`
- `http://www.w3.org/2000/01/rdf-schema#label`
- `http://example.org/food#hasUnit`
- `http://example.org/food#hasValue`

## Usage

```bash
uv run food-linkage \
  --json-source ../json/new/data_retrieve_from_orkg.json \
  --image-dir ../dataset/image/not_merged/AFD/test \
  --output-file ../dataset/multimodal/not_merged/test/AFD_test.jsonld
```
From the repository root:

```bash
uv --project linkage run food-linkage \
  --json-source json/new/data_retrieve_from_orkg.json \
  --image-dir dataset/image/not_merged/AFD/test \
  --output-file dataset/multimodal/not_merged/test/AFD_test.jsonld
```

## Textual Description Construction

For each linked image, the textual description stored in
`http://example.org/food#hasTextDescription` is built deterministically from
the available metadata:

- food name or description
- ingredient list, when available
- nutrient-component strings derived from the ontology-style component records

## Reproducibility Notes

- The linkage is deterministic for fixed source JSON content and fixed image
  directories.
- Images are processed in sorted filename order inside each class directory.
- The output is written as a single JSON-LD array, which is easier to inspect in tools such as Protégé.
