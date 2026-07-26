# Inference Reproducibility

This directory contains the inference code used for two downstream tasks:

- `food classification`, implemented in [`food_classifier.py`](food_classifier.py)
- `nutrient generation (no_rag / rag)`, implemented in [`nutrient_generator.py`](nutrient_generator.py)
- `semantic retrieval baseline`, implemented in [`semantic_retrieval.py`](semantic_retrieval.py)
- `progressive RAG ablation`, implemented in [`ablation.py`](ablation.py)

Shared nutrient-generation utilities are centralized in [`generation_core.py`](generation_core.py).

Both tasks can be launched from the unified entrypoint [`main.py`](main.py).

## Prerequisites

- **Python 3.12** — this module includes a `.python-version` file pinned to `3.12`
- **[uv](https://docs.astral.sh/uv/)** — used to create and run the module environment
- **Conda environment `inference_env`** — used for the cluster jobs
- **GPU access** — the recommended Slurm configuration uses one 24 GB GPU with `48G` host memory

## Required Environment Variables

The nutrient-generation pipeline depends on authenticated services. Set the following variables before running:

- `HUB_TOKEN` — required when using gated or private Hugging Face models; optional otherwise
- `PINECONE_API_KEY`

The module environment also includes `accelerate`, which is required by the
Falcon3 generation pipeline when `transformers` loads the model with
`device_map="auto"`.

If a `.env` file is used, it should define at least these variables.

## Setup With uv

Go into the module folder:

```bash
cd inference
```

Install or synchronize the module environment:

```bash
uv sync
```

For exact dependency reproduction, prefer:

```bash
uv sync --frozen
```

Show CLI help:

```bash
uv run food-inference --help
```

## Unified Entrypoint

### Food Classification

Semantic evaluation:

```bash
uv run food-inference classify \
  --model-dir anonymous-eval/food-recognition \
  --test-dir dataset/image/not_merged/AFD/test \
  --mode semantic
```

Strict evaluation:

```bash
uv run food-inference classify \
  --model-dir anonymous-eval/food-recognition \
  --test-dir dataset/image/not_merged/AFD/test \
  --mode strict
```

Classification on the released excerpt dataset:

```bash
uv run food-inference classify \
  --model-dir anonymous-eval/food-recognition \
  --test-dir excerpt_dataset/images \
  --mode semantic
```

Optional arguments:

- `--limit`
- `--output-file`

Default output:

- `results/<model_name>/<split_name>_metrics`

### Nutrient Generation

Example with RAG and selective prompting:

```bash
uv run food-inference generate \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonld \
  --output-file results/text-model/vision-model/example.csv \
  --mode rag \
  --selective
```

Example on the released excerpt dataset:

```bash
uv run food-inference generate \
  --index-name icml-paper \
  --input-file excerpt_dataset/annotation.jsonld \
  --output-file results/falcon/dinov3/excerpt_rag.csv \
  --mode rag \
  --selective
```

Example on the released excerpt dataset without retrieval:

```bash
uv run food-inference generate \
  --index-name example-index \
  --input-file excerpt_dataset/annotation.jsonld \
  --output-file results/text-model/vision-model/excerpt_no_rag.csv \
  --mode no_rag \
  --selective
```

Available modes:

- `rag`
- `no_rag`

The `generate` command is restricted to the two main settings reported for nutrient generation:

- `no_rag`
- `rag`

### Semantic Retrieval Baseline

This command retrieves the closest linked semantic record from the image embedding space and writes the matched food label and nutrient list directly.

```bash
uv run food-inference semantic-retrieval \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonld \
  --output-file results/text-model/vision-model/example_semantic.csv
```

Excerpt example:

```bash
uv run food-inference semantic-retrieval \
  --index-name icml-paper-image \
  --input-file excerpt_dataset/annotation.jsonld \
  --output-file results/falcon/dinov3/excerpt_rag.csv
```

### Progressive RAG Ablation

This command mixes `no_rag` and `rag` predictions in a cumulative evaluation protocol controlled by `--rag-ratio`.

```bash
uv run food-inference ablation \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonld \
  --output-file results/text-model/vision-model/example_ablation.csv \
  --rag-ratio 0.2 \
  --selective
```

Excerpt example:

```bash
uv run food-inference ablation \
  --index-name icml-paper \
  --input-file excerpt_dataset/annotation.jsonld \
  --output-file results/falcon/dinov3/excerpt_rag.csv \
  --rag-ratio 0.2 \
  --selective
```

## Single Slurm Script

Use the unified Slurm launcher [`run_inference_sbatch.sh`](run_inference_sbatch.sh), which follows the documented single-GPU cluster configuration.

Submit nutrient generation:

```bash
sbatch inference/run_inference_sbatch.sh \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonld \
  --output-file results/text-model/vision-model/example.csv \
  --mode rag \
  --selective
```

Submit semantic retrieval:

```bash
TASK=semantic-retrieval sbatch inference/run_inference_sbatch.sh \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonld \
  --output-file results/text-model/vision-model/example_semantic.csv
```

Submit the progressive ablation study:

```bash
TASK=ablation sbatch inference/run_inference_sbatch.sh \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonld \
  --output-file results/text-model/vision-model/example_ablation.csv \
  --rag-ratio 0.2 \
  --selective
```

Submit food classification:

```bash
TASK=classify sbatch inference/run_inference_sbatch.sh \
  --model-dir org/dinov3-food-model \
  --test-dir dataset/image/not_merged/AFD/test \
  --mode semantic
```

## Reproducibility Notes

- The same entrypoint is used for local and Slurm execution.
- The unified Slurm script standardizes the hardware profile to `1 x 24 GB GPU`, `48G` RAM, and the `inference_env` environment.
- Classification outputs are saved as JSON files.
- Nutrient-generation, semantic-retrieval, and ablation outputs are saved as CSV files.
- Input annotation files are expected in JSON-LD format.
- Relative input paths are resolved against the repository root.
- Relative output paths are also resolved against the repository root, and CSV rows are flushed as they are written.
- Exact reruns require the same model identifiers, input files, environment variables, and resource configuration.

If you prefer running from the repository root, use:

```bash
uv --project inference run food-inference --help
```
