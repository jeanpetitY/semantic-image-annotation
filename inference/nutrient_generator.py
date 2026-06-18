from __future__ import annotations

import argparse
import ast
import csv
import gc
import os
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.food_recognition import FoodClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
CLASSIFIER_MODEL_NAME = "yvelos/dinov3-food-389-v1"
TEXT_EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_parent_dir(file_path: str) -> None:
    """Create the parent directory of a file path if it does not exist."""
    parent = Path(file_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def safe_literal_eval_list(value: str) -> List[Any]:
    """Safely parse a Python-list-like string; return [] on parse failure."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


@dataclass
class GenerationConfig:
    """Text generation configuration for the LLM pipeline."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True


class RAGRecipe:
    """RAG helper for image/text embedding and Pinecone retrieval."""

    def __init__(
        self,
        index_name: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
    ) -> None:
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.metric = metric

        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise EnvironmentError("PINECONE_API_KEY is missing in environment variables.")

        self.pc = Pinecone(api_key=pinecone_key)
        self.text_model: Optional[SentenceTransformer] = None
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None

    # ------------------ Setup ------------------

    def set_text_model(self, model_name: str = TEXT_EMBED_MODEL_NAME) -> None:
        self.text_model = SentenceTransformer(model_name)

    def load_clip(self, model_name: str = CLIP_MODEL_NAME) -> None:
        if self.clip_model is None or self.clip_processor is None:
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)

    def _ensure_index(self, dimension: int = 512):
        existing_indexes = set(self.pc.list_indexes().names())
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        return self.pc.Index(self.index_name)

    def use_index(self, name: Optional[str] = None):
        return self.pc.Index(name or self.index_name)

    # ------------------ Embeddings ------------------

    def embed_text(self, text: str) -> np.ndarray:
        if self.text_model is None:
            raise RuntimeError("Text model is not initialized. Call set_text_model() first.")
        return self.text_model.encode(text, convert_to_numpy=True)

    def embed_image(self, image_source: Union[str, BytesIO]) -> np.ndarray:
        if self.clip_model is None or self.clip_processor is None:
            raise RuntimeError("CLIP model is not initialized. Call load_clip() first.")

        if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            response = requests.get(image_source, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        elif isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, BytesIO):
            image = Image.open(image_source).convert("RGB")
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return features[0].cpu().numpy()

    # ------------------ Search ------------------

    def _search(self, embedding: np.ndarray, top_k: int, namespace: str) -> List[Dict[str, Any]]:
        index = self.use_index()
        results = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [match.get("metadata", {}) for match in results.get("matches", [])]

    def search_by_text(self, query: str, top_k: int = 5, namespace: str = "text") -> List[Dict[str, Any]]:
        emb = self.embed_text(query)
        return self._search(emb, top_k, namespace)

    def search_by_image(
        self,
        image_source: Union[str, BytesIO],
        top_k: int = 5,
        namespace: str = "image",
    ) -> List[Dict[str, Any]]:
        emb = self.embed_image(image_source)
        return self._search(emb, top_k, namespace)

    def search_topk_by_image(
        self,
        image_source: Union[str, BytesIO],
        top_k: int = 1,
        namespace: str = "image",
    ) -> Optional[Dict[str, Any]]:
        results = self.search_by_image(image_source, top_k=top_k, namespace=namespace)
        if top_k == 1:
            return results[0] if results else None
        elif top_k == 2:
            return results[1] if results else None
        elif top_k == 3:
            return results[2] if results else None
        
        elif top_k == 4:
            return results[3] if results else None
        
        elif top_k >= 5:
            return results[4] if results else None 
        return results[0] if results else None


class FoodAssistant:
    """Food prediction + nutrient generation pipeline."""

    def __init__(
        self,
        rag: RAGRecipe,
        text_model_name: str = TEXT_MODEL_NAME,
        classifier_model_name: str = CLASSIFIER_MODEL_NAME,
    ) -> None:
        self.rag = rag
        self.text_model_name = text_model_name

        self.food_classifier = FoodClassifier(classifier_model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self._text_pipe = None
        self._generation_config = GenerationConfig()

    def _get_text_pipe(self):
        if self._text_pipe is None:
            self._text_pipe = pipeline(
                "text-generation",
                model=self.text_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=self._generation_config.max_new_tokens,
                temperature=self._generation_config.temperature,
                top_p=self._generation_config.top_p,
                top_k=self._generation_config.top_k,
                do_sample=self._generation_config.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
                use_cache=True,
            )
        return self._text_pipe

    # ------------------ Prompting ------------------

    @staticmethod
    def _format_rag_context(docs: Sequence[Dict[str, Any]], query: str) -> str:
        if not docs:
            context = "- Information not found."
        else:
            context = "\n".join(f"- {doc}" for doc in docs)
        return (
            f"Here is information found on ORKG about {query}:\n"
            f"{context}\n\n"
            f"Please provide all nutritional components for {query}."
        )

    @staticmethod
    def _messages_no_rag(prompt: str, selective: bool) -> List[Dict[str, str]]:
        if not selective:
            system = (
                "You are a specialized assistant in food information engineering.\n\n\
                Your task is to provide the nutritional components of a food item.\n\n\
                STRICT RULES (MUST BE FOLLOWED EXACTLY):\n\
                1. you should use your own knowledge.\n\
                2. The output must be a python list of strings, and only that.\n\
                3. Each element must follow this exact format:\n\
                \"nutrient: value unit\"\n\
                4. Example of a valid output:\n\
                ['iron: 1.42 mg', 'zinc: 1.11 mg']\n\
                5. Do not add explanations, comments, reasoning, headers, or any extra text.\n\
                6. Do not include nutrient names without values or units.\n\
                7. Return only the final python list - nothing else."
            )
        else:
            system = (
                "You are a specialized assistant in food information engineering.\n\n\
                Your task is to provide the nutritional components of a food item.\n\n\
                STRICT RULES (MUST BE FOLLOWED EXACTLY):\n\
                1. you should use your own knowledge and do not invent nutritional component if you don't know.\n\
                2. The output must be a python list of strings, and only that.\n\
                3. Each element must follow this exact format:\n\
                \"nutrient: value unit\"\n\
                4. Example of a valid output:\n\
                ['iron: 1.42 mg', 'zinc: 1.11 mg',.... 'compN: valueN unitN']\n\
                5. Do not add explanations, comments, reasoning, headers, or any extra text.\n\
                6. Do not include nutrient names without values or units.\n\
                7. Return only the final python list OR \"I don't know\" - nothing else."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    @staticmethod
    def _messages_rag(prompt: str, selective: bool) -> List[Dict[str, str]]:
        if selective:
            system = (
                "You are a specialized assistant in food information engineering.\n\n\
                Your task is to extract and return the nutritional components of a food item.\n\n\
                STRICT RULES (MUST BE FOLLOWED EXACTLY):\n\
                1. Use only the information explicitly present in the user-provided context.\n\
                2. Do NOT invent, infer, estimate, or guess any nutrient, value, or unit.\n\
                3. If the requested food does NOT exist in the context, reply EXACTLY:\n\
                \"I don't know\"\n\
                4. You should extract all the nutritional components corresponding to the user query.\n\
                5. The output must be a python list of strings, and only that.\n\
                6. Each element must follow this exact format:\n\
                nutrient: value unit\n\
                7. Example of a VALID output:\n\
                ['iron: 1.42 mg', 'zinc: 1.11 mg',.... 'compN: valueN unitN']\n\
                8. Do not add explanations, comments, reasoning, headers, or any extra text.\n\
                9. Do not include nutrient names without values or units.\n\
                10. Do not reference ingredients, sources, or descriptions.\n\
                11. Return only the final python list OR \"I don't know\" - nothing else."
            )
        else:
            system = (
                "You are a specialized assistant in food information engineering.\n\n\
                Your task is to extract and return the nutritional components of a food item.\n\n\
                STRICT RULES (MUST BE FOLLOWED EXACTLY):\n\
                1. Use only the information explicitly present in the user-provided context.\n\
                2. Do NOT invent, infer, estimate, or guess any nutrient, value, or unit.\n\
                4. The output must be a python list of strings, and only that.\n\
                5. Each element must follow this exact format:\n\
                \"nutrient: value unit\"\n\
                6. Example of a valid output:\n\
                ['iron: 1.42 mg', 'zinc: 1.11 mg']\n\
                7. Do not add explanations, comments, reasoning, headers, or any extra text.\n\
                8. Do not include nutrient names without values or units.\n\
                9. Do not reference ingredients, sources, or descriptions.\n\
                10. Return only the final python list - nothing else."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    def _generate_components(self, prompt: str, is_rag: bool, selective: bool) -> str:
        messages = (
            self._messages_rag(prompt, selective)
            if is_rag
            else self._messages_no_rag(prompt, selective)
        )
        response = self._get_text_pipe()(messages)
        try:
            return response[0]["generated_text"][2]["content"].strip().strip(".")
        except (KeyError, IndexError, TypeError):
            return "I don't know"

    # ------------------ Classification ------------------

    def predict_food_name(self, image_source: Union[str, BytesIO]) -> str:
        return self.food_classifier.predict(image_source)

    # ------------------ End-to-end pipelines ------------------

    def process_json_and_predict(
        self,
        input_file: str,
        output_file: str,
        is_rag: bool = False,
        is_selective: bool = False,
        is_ablation_study: bool = False,
        rag_ratio: float = 0.2,
    ) -> None:
        df = pd.read_json(input_file)
        n_total = len(df)

        if is_ablation_study:
            rag_ratio = max(0.0, min(1.0, rag_ratio))
            n_rag = int(n_total * rag_ratio)
            print(f"[Ablation] Progressive RAG: {n_rag}/{n_total} ({rag_ratio * 100:.0f}%)")
        else:
            n_rag = 0

        ensure_parent_dir(output_file)

        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components", "used_rag"])

            for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=n_total, desc="Processing rows", unit="row")):
                food_id = f"{row['label']} - {row['image']}"

                try:
                    predicted_name = self.predict_food_name(row["image"])
                except Exception as exc:  # noqa: BLE001
                    writer.writerow([food_id, "I don't know", f"Error: {exc}", False])
                    continue

                # if predicted_name.lower() != str(row["label"]).lower():
                #     writer.writerow([food_id, predicted_name, "I don't know", False])
                #     continue

                if is_ablation_study:
                    use_rag_here = idx < n_rag
                else:
                    use_rag_here = is_rag

                label = f"Food name: {predicted_name}"

                if use_rag_here:
                    docs = self.rag.search_by_text(predicted_name, top_k=4, namespace="text")
                    prompt = self._format_rag_context(docs, label)
                else:
                    prompt = (
                        "Given this food name identify its main nutrients and their corresponding "
                        "nutritional values in grams, milligrams, kcal etc.\n"
                        f"{label}?"
                    )

                try:
                    components = self._generate_components(prompt, is_rag=use_rag_here, selective=is_selective)
                except Exception as exc:  # noqa: BLE001
                    components = f"Error: {exc}"

                writer.writerow([food_id, predicted_name, components, use_rag_here])

    def predict_with_semantic_search(self, input_file: str, output_file: str, top_k: int =1) -> None:
        df = pd.read_json(input_file)

        ensure_parent_dir(output_file)

        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components"])

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
                food_id = f"{row['label']} - {row['image']}"

                try:
                    match = self.rag.search_topk_by_image(row["image"], top_k=top_k, namespace="image")
                    if not match:
                        writer.writerow([food_id, "I don't know", []])
                        continue

                    predicted_name = match.get("food_class", "I don't know")
                    components_raw = str(match.get("components", "[]"))
                    parsed = safe_literal_eval_list(components_raw)

                    formatted_components: List[str] = []
                    for item in parsed:
                        if isinstance(item, dict):
                            name = item.get("name", "unknown")
                            value = item.get("value", "")
                            unit = item.get("unit", "")
                            formatted_components.append(f"{name}: {value} {unit}".strip())

                    writer.writerow([food_id, predicted_name, formatted_components])

                except Exception as exc:  # noqa: BLE001
                    writer.writerow([food_id, "I don't know", [f"Error searching image: {exc}"]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food Multimodal RAG Pipeline")

    parser.add_argument("--index-name", default="icml-paper", type=str, required=True, help="Pinecone index name")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output CSV file")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "no_rag", "semantic_search"],
        default="no_rag",
        help="Execution mode",
    )
    parser.add_argument("--selective", action="store_true", help="Enable selective prompting")
    parser.add_argument("--ablation", action="store_true", help="Enable progressive RAG ablation study")
    parser.add_argument("--rag-ratio", type=float, default=0.2, help="RAG ratio for ablation (0.0 to 1.0)")

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    set_seed(42)

    hub_token = os.getenv("HUB_TOKEN")
    if hub_token:
        login(token=hub_token)

    args = parse_args()

    rag = RAGRecipe(index_name=args.index_name)
    rag.load_clip()
    rag.set_text_model()

    assistant = FoodAssistant(rag=rag)

    ensure_parent_dir(args.output_file)

    if args.mode == "semantic_search":
        assistant.predict_with_semantic_search(input_file=args.input_file, output_file=args.output_file)
    else:
        assistant.process_json_and_predict(
            input_file=args.input_file,
            output_file=args.output_file,
            is_rag=args.mode == "rag",
            is_selective=args.selective,
            is_ablation_study=args.ablation,
            rag_ratio=args.rag_ratio,
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
