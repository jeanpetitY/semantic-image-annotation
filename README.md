# Semantic Food Image Annotation and Nutrient Inference

This repository contains the codebase used to build, enrich, link, model, and evaluate a semantified food-image dataset for nutrient-aware inference.

The project combines:

- food-image datasets such as `Food-101`, `FruitVeg81`, `AFD`, and `UECFood256`
- nutritional enrichment from external sources
- knowledge-graph structuring and image-to-description linkage
- vision-model fine-tuning for food recognition
- nutrient generation with `no_rag`, `rag`, semantic retrieval, and ablation settings

## Pipeline Overview

The full workflow is organized into dedicated modules:

1. `preprocessing/`: prepare image datasets, normalize labels, create splits, balance classes, and analyze dataset layouts
2. `retriever/`: retrieve nutritional information from USDA and export structured resources from ORKG
3. `importer/`: import semantified food descriptions into ORKG
4. `linkage/`: link each image to its semantic food description and export JSON-LD annotations
5. `excerpt/`: create a compact `excerpt_dataset/` for inspection, inference, and evaluation
6. `finetuning/`: fine-tune food-classification backbones such as CLIP, BEiT, and DINOv3
7. `inference/`: run food classification and nutrient inference
8. `evaluation/`: evaluate nutrient predictions against JSON-LD ground truth
9. `push_to_hub/`: publish trained model artifacts

## Repository Layout

- [`preprocessing`](preprocessing/README.md): dataset preparation and analysis
- [`retriever`](retriever/README.md): USDA enrichment and ORKG export
- [`importer`](importer/README.md): ORKG import pipeline
- [`linkage`](linkage/README.md): image-to-description JSON-LD linkage
- [`excerpt`](excerpt/README.md): excerpt-dataset construction
- [`finetuning`](finetuning/README.md): food-recognition training workflows
- [`inference`](inference/README.md): classification, nutrient generation, semantic retrieval, and ablation
- [`evaluation`](evaluation/README.md): prediction evaluation metrics and CLI
- [`push_to_hub`](push_to_hub/README.md): model publication utilities

## Excerpt Dataset

When generated, `excerpt_dataset/` provides a compact end-to-end example shared across modules:

- `excerpt_dataset/images`: image classification and dataset analysis
- `excerpt_dataset/annotation.jsonld`: nutrient inference and evaluation

The exact commands for creating and using this excerpt are documented in the module READMEs, especially:

- [`excerpt/README.md`](excerpt/README.md)
- [`inference/README.md`](inference/README.md)
- [`evaluation/README.md`](evaluation/README.md)
- [`preprocessing/README.md`](preprocessing/README.md)

## Reproducibility

Each module is documented and executable independently. For environment setup, commands, inputs, and outputs, use the README inside the relevant module rather than the repository root.
