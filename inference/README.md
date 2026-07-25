# Inference Reproducibility

This directory contains the inference code used for two downstream tasks:

- `food classification`, implemented in [`food_classifier.py`](food_classifier.py)
- `nutrient generation`, implemented in [`nutrient_generator.py`](nutrient_generator.py)

Both tasks can be launched from the unified entrypoint [`main.py`](main.py).

## Prerequisites

- **Python 3.14+** — the project includes a `.python-version` file for tool compatibility
- **[uv](https://docs.astral.sh/uv/)** — used as the package manager, build system, and script runner
- **Conda environment `inference_env`** — used for the cluster jobs
- **GPU access** — the recommended Slurm configuration uses one 24 GB GPU with `48G` host memory

## Required Environment Variables

The nutrient-generation pipeline depends on authenticated services. Set the following variables before running:

- `HUB_TOKEN`
- `PINECONE_API_KEY`

If a `.env` file is used, it should define at least these variables.

## Unified Entrypoint

### Food Classification

Semantic evaluation:

```bash
python inference/main.py classify \
  --model-dir org/dinov3-food-model \
  --test-dir dataset/image/not_merged/AFD/test \
  --mode semantic
```

Strict evaluation:

```bash
python inference/main.py classify \
  --model-dir org/dinov3-food-model \
  --test-dir dataset/image/not_merged/AFD/test \
  --mode strict
```

Optional arguments:

- `--limit`
- `--output-file`

Default output:

- `results/<model_name>/<split_name>_metrics`

### Nutrient Generation

Example with RAG and selective prompting:

```bash
python inference/main.py generate \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonl \
  --output-file results/text-model/vision-model/example.csv \
  --mode rag \
  --selective
```

Available modes:

- `rag`
- `no_rag`
- `semantic_search`

Optional arguments:

- `--ablation`
- `--rag-ratio`

## Single Slurm Script

Use the unified Slurm launcher [`run_inference_sbatch.sh`](run_inference_sbatch.sh), which follows the documented single-GPU cluster configuration.

Submit nutrient generation:

```bash
sbatch inference/run_inference_sbatch.sh \
  --index-name example-index \
  --input-file dataset/multimodal/not_merged/test/example_test.jsonl \
  --output-file results/text-model/vision-model/example.csv \
  --mode rag \
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
- Nutrient-generation outputs are saved as CSV files.
- Exact reruns require the same model identifiers, input files, environment variables, and resource configuration.
