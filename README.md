# 🍽️ Experimentation for Semantic Image Annotation

This project focuses on **automatic semantic annotation of food images** and **nutritional knowledge augmentation** using multiple food datasets:

- **Food101**
- **FruitVeg81**
- **AFD**
- **UECFOOD256**

The pipeline supports:

- Food image classification  
- Nutrient prediction  
- Knowledge-augmented inference (RAG)  
- USDA-based nutritional enrichment  

---

# ⚙️ Requirements

To reproduce the experiments, we recommend the following hardware:

- **GPU**: NVIDIA RTX-class GPU (24GB VRAM)
- **RAM**: 48 GB
- **CPU**: 12 cores

### Software

- **Python ≥ 3.10**

If Python is not installed:

👉 https://www.python.org/downloads/

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure
.
├── evaluation        # Evaluation scripts and metrics
├── finetuning        # Model training scripts
├── inference         # Classification & nutrient prediction
├── push_to_hub       # HuggingFace model publishing tools
├── annotation        # USDA nutritional annotation/enrichment pipeline
├── retrieve          # Other retrieval utilities
├── tools             # Utility functions


## Fine-Tuning
The finetuning folder contains the code used to fine-tune the food classification model.

You can launch the training process with the following command:

you can run the code by run this command in your terminal
```bash
python finetuning/train_classifier.py \
  --model_name microsoft/beit-base-patch16-384 \
  --data_dir dataset/image/merged \
  --output_dir ./model_saved/food-model \
  --epochs 8 \
  --batch_size 32 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --fp16
```

## Inference
The inference folder contain to file: one for classify food image and the other to predict nutrient from food image.

### Inference for classify food image
To classify and evaluate our food images test dataset, you can run the following command

```bash
python inference/food_classifier.py \
  --model-dir ./path_to_model_name \
  --data-dir path_to_test_data \
  --result-dir results \
  --mode semantic
```
or 
```bash
python inference/food_classifier.py \
  --model-dir ./path_to_model_name \
  --data-dir dpath_to_test_data\
  --result-dir results \
  --limit 500
```
the mode argument can only take two values semantic(in case your dataset is UECFOOD) and strict for the other dataset.

### Inference for nutrient generation 
  - Without Knowledge augmentation
      ```bash
      python inference/nutrient_generator.py \
    --index-name example-index \
    --input-file dataset/multimodal/not_merged/test/example_test.jsonl \
    --output-file results/text-model/vision-model/no_rag/example.csv \
    --mode no_rag \
    --selective
    ```
  - With Knowledge Augmentation
    ```bash
    python inference/nutrient_generator.py \
    --index-name example-index \
    --input-file dataset/multimodal/not_merged/test/example_test.jsonl \
    --output-file results/text-model/vision-model/rag/example.csv \
    --mode rag \
    --selective
    ```
  - Case: Ablation Study
    ```bash
    python inference/nutrient_generator.py \
    --index-name example-index \
    --input-file dataset/multimodal/not_merged/test/example_test.jsonl \
    --output-file results/text-model/vision-model/ablation/example.csv \
    --mode no_rag \
    --selective \
    --ablation \
    --rag-ratio 0.4
    ```

  - case: Baseline with clip-based approach(semantic searc)
    ```bash
    python inference/nutrient_generator.py \
    --index-name example-index \
    --input-file dataset/multimodal/not_merged/test/example_test.jsonl \
    --output-file results/text-model/vision-model/semantic_search/example.csv \
    --mode semantic_search
    ```

## Retrieval
This folder contains one folder one fold to get nutrient data from USDA 

you can run this command to if you want to perform this task
```bash
cd annotation
uv run image-annotation \
  --food-labels ../dataset/image/not_merged/fruitveg81/test \
  --output-file outputs/fruitveg81/fruitveg81_usda.json \
  --not-annotated-file outputs/fruitveg81/fruitveg81_not_annotated.json \
  --fruit-veg \
  --image-base-url https://my-bucket/fruitveg81/
```
