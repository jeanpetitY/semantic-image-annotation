"""Shared inference utilities for nutrient generation variants.

Paper reference:
- Evaluation section comparing plain Falcon3 generation, CLIP-based semantic
  retrieval, and KG-grounded generation over the semantified dataset.
"""

from io import BytesIO
from pathlib import Path
from typing import Union

import ast
import csv
import os
import random
import sys

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv  # type: ignore
from fastapi import UploadFile
from huggingface_hub import login
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.food_recognition import FoodClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


load_dotenv()

model_name = "tiiuae/Falcon3-7B-Instruct"
classifier_model_name = "anonymous-eval/food-recognition"
tokenizer = None
pipe = None
model_img_emb = None
processor_img_emb = None

HAS_IMAGE_KEY = "http://example.org/food#hasImage"
RDFS_LABEL_KEY = "http://www.w3.org/2000/01/rdf-schema#label"


def _extract_first_value(value):
    if isinstance(value, list) and value:
        first_item = value[0]
        if isinstance(first_item, dict):
            return first_item.get("@value")
        return first_item
    return value


def _extract_label_from_jsonld_identifier(record: dict) -> str:
    identifier = str(record.get("@id", ""))
    type_values = record.get("@type", [])

    candidates = []
    if "#" in identifier:
        candidates.append(identifier.split("#", 1)[1])
    if isinstance(type_values, list) and type_values:
        first_type = str(type_values[0])
        if "#" in first_type:
            candidates.append(first_type.split("#", 1)[1])
        else:
            candidates.append(first_type)

    for candidate in candidates:
        if "_Img_" in candidate:
            prefix = candidate.split("_Img_", 1)[0]
        elif candidate.endswith("_Images"):
            prefix = candidate[: -len("_Images")]
        else:
            continue

        if "_" in prefix:
            return prefix.rsplit("_", 1)[0]
        return prefix

    label_value = _extract_first_value(record.get(RDFS_LABEL_KEY))
    if label_value is not None:
        return str(label_value)

    return identifier


def _resolve_record_image_path(image_path: str, base_dir: Path) -> str:
    candidate = Path(str(image_path)).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    if candidate.exists():
        return str(candidate.resolve())

    base_candidate = (base_dir / candidate).resolve()
    if base_candidate.exists():
        return str(base_candidate)

    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return str(project_candidate)

    return str(candidate)


def resolve_output_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _normalize_input_dataframe(df: pd.DataFrame, base_dir: Path | None = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = PROJECT_ROOT

    if {"label", "image"}.issubset(df.columns):
        normalized_df = df.copy()
        normalized_df["image"] = normalized_df["image"].map(
            lambda path: _resolve_record_image_path(path, base_dir)
        )
        return normalized_df

    if "@id" in df.columns and HAS_IMAGE_KEY in df.columns:
        normalized_rows = []
        for record in df.to_dict(orient="records"):
            image_path = _extract_first_value(record.get(HAS_IMAGE_KEY))
            if not image_path:
                continue

            normalized_rows.append(
                {
                    "label": _extract_label_from_jsonld_identifier(record),
                    "image": _resolve_record_image_path(str(image_path), base_dir),
                }
            )

        return pd.DataFrame(normalized_rows)

    raise ValueError(
        "Unsupported input format. Expected columns {'label', 'image'} "
        "or JSON-LD records with '@id' and image metadata."
    )


def load_input_dataframe(input_file: str) -> pd.DataFrame:
    # Paper downstream task: nutrient generation consumes the semantified
    # multimodal dataset, not only raw imagefolder labels.
    input_path = Path(input_file).expanduser()
    if not input_path.is_absolute():
        if input_path.exists():
            input_path = input_path.resolve()
        else:
            project_candidate = (PROJECT_ROOT / input_path).resolve()
            if project_candidate.exists():
                input_path = project_candidate

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)

    return _normalize_input_dataframe(df, base_dir=input_path.parent)


class RAGRecipe:
    def __init__(
        self,
        index_name,
        cloud="aws",
        region="us-east-1",
        metric="cosine",
    ):
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.metric = metric
        self.text_model = None
        self.model_emb = None
        self.processor = None
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def create_index(self, dimension=512):
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        return self.pc.Index(self.index_name)

    def use_index(self, name):
        self.index_name = name
        return self.pc.Index(self.index_name)

    def set_text_model(self):
        self.text_model = SentenceTransformer("intfloat/multilingual-e5-large")

    def load_clip(self):
        global model_img_emb, processor_img_emb
        if self.model_emb is None:
            if model_img_emb is None or processor_img_emb is None:
                model_img_emb = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor_img_emb = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model_emb = model_img_emb
            self.processor = processor_img_emb

    def embed_image(self, image_source: Union[str, UploadFile, BytesIO]):
        if isinstance(image_source, str) and (
            image_source.startswith("http://") or image_source.startswith("https://")
        ):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        elif isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, UploadFile):
            image_bytes = image_source.file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_source, BytesIO):
            image = Image.open(image_source).convert("RGB")
        else:
            raise ValueError("Unsupported image source type.")

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model_emb.get_image_features(**inputs)
        return features[0].cpu().numpy()

    def embed_text(self, text: str):
        return self.text_model.encode(text, convert_to_numpy=True)

    def store_embedding(
        self,
        is_new_index: bool,
        id: str,
        embedding,
        metadata: dict,
        namespace="ns1",
    ):
        if namespace == "text":
            index = self.use_index(name="tsotsatext")
        else:
            index = self.create_index() if is_new_index else self.use_index()

        index.upsert(
            [{"id": id, "values": embedding, "metadata": metadata}],
            namespace=namespace,
            batch_size=12,
        )

    def store_text_embedding(self, id: str, text: str, metadata: dict, namespace="text"):
        emb = self.embed_text(text)
        self.store_embedding(id, emb, metadata, namespace)

    def store_image_embedding(self, id: str, image_path: any, metadata: dict, namespace="image"):
        emb = self.embed_image(image_path)
        self.store_embedding(id, emb, metadata, namespace)

    def search_by_text(self, query: str, top_k=5, namespace="text"):
        emb = self.embed_text(query)
        return self._search(emb, top_k, namespace)

    def search_by_image(self, image_path: Union[str, UploadFile, BytesIO], top_k=5, namespace="image"):
        emb = self.embed_image(image_path)
        return self._search(emb, top_k, namespace)

    def search_first_by_image(self, image_path: Union[str, UploadFile, BytesIO], namespace="image"):
        results = self.search_by_image(image_path, top_k=1, namespace=namespace)
        return results[0] if results else None

    def _search(self, embedding, top_k, namespace):
        index = self.use_index(self.index_name)
        results = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [match["metadata"] for match in results["matches"]]


class FoodAssistant:
    def __init__(self, rag: RAGRecipe):
        self.rag = rag
        self.food_classifier = FoodClassifier(classifier_model_name)

    def _format_context(self, docs, query, fallback="Information not found."):
        if not docs:
            docs = [fallback]
        context = "\n".join([f"- {doc}" for doc in docs])
        return (
            "Here is the information found on ORKG(Open Research Knowledge "
            f"Graph) about {query} food: {context}\n\nPlease provide all the "
            f"food components of: {query} food?"
        )

    def choose_message(
        self,
        prompt: Union[dict, str],
        is_rag: bool = False,
        is_selective: bool = False,
    ):
        if not is_rag:
            if not is_selective:
                return [
                    {
                        "role": "system",
                        "content": "You are a specialized assistant in food information engineering.\n\n\
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
                        7. Return only the final python list - nothing else.",
                    },
                    {"role": "user", "content": f"{prompt}"},
                ]
            return [
                {
                    "role": "system",
                    "content": "You are a specialized assistant in food information engineering.\n\n\
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
                    7. Return only the final python list OR \"I don't know\" - nothing else.",
                },
                {"role": "user", "content": f"{prompt}"},
            ]

        if not is_selective:
            return [
                {
                    "role": "system",
                    "content": "You are a specialized assistant in food information engineering.\n\n\
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
                    10. Return only the final python list - nothing else.",
                },
                {"role": "user", "content": f"{prompt}"},
            ]
        return [
            {
                "role": "system",
                "content": "You are a specialized assistant in food information engineering.\n\n\
                Your task is to extract and return the nutritional components of a food item.\n\n\
                STRICT RULES (MUST BE FOLLOWED EXACTLY):\n\
                1. Use only the information explicitly present in the user-provided context.\n\
                2. Do NOT invent, infer, estimate, or guess any nutrient, value, or unit.\n\
                3. If the requested food does NOT exist in the context, reply EXACTLY:\n\
                \"I don't know\"\n\
                4. You must extract EVERY nutritional component corresponding to the user query from the context.\n\
                Keep in mind that that some components can have 0 as their value, but they must still be included.\n\
                The components should be extracted in the order they appear in the context.\n\
                5. Do NOT skip, summarize, merge, filter, rank, or select only the most important components.\n\
                6. If a component in the context has a nutrient name, value, and unit, it MUST appear in the output.\n\
                7. The output must be a python list of strings, and only that.\n\
                8. Each element must follow this exact format:\n\
                nutrient: value unit\n\
                9. Example of a VALID output:\n\
                ['iron: 1.42 mg', 'zinc: 1.11 mg',.... 'compN: valueN unitN']\n\
                10. Do not add explanations, comments, reasoning, headers, or any extra text.\n\
                11. Do not include nutrient names without values or units.\n\
                12. Do not reference ingredients, sources, or descriptions.\n\
                13. Return only the final python list OR \"I don't know\" - nothing else.",
            },
            {"role": "user", "content": f"{prompt}"},
        ]

    def predict_food_name(self, image_source: Union[str, UploadFile, BytesIO]) -> str:
        return self.food_classifier.predict(image_source)

    def use_model(self, query, is_rag: bool = False, is_selective: bool = False):
        global pipe, tokenizer
        if pipe is None:
            hub_token = os.getenv("HUB_TOKEN")
            if hub_token:
                login(token=hub_token)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                use_cache=True,
            )
        messages = self.choose_message(
            query,
            is_rag=is_rag,
            is_selective=is_selective,
        )
        response = pipe(messages)
        return response[0]["generated_text"][2]["content"].strip().strip(".")

    def process_json_and_predict(
        self,
        input_file: str,
        output_file: str,
        is_rag: bool = False,
        is_selective: bool = False,
    ):
        df = load_input_dataframe(input_file)
        output_path = resolve_output_path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components"])
            outfile.flush()

            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Processing rows",
                unit="row",
            ):
                food_id = f"{row['label']} - {row['image']}"
                food_name_predicted = self.predict_food_name(row["image"])
                label = f"Food name: {food_name_predicted}"

                if is_rag:
                    docs = self.rag.search_by_text(
                        food_name_predicted,
                        top_k=4,
                        namespace="text",
                    )
                    query = self._format_context(
                        docs,
                        label,
                        fallback="Information not found.",
                    )
                else:
                    query = (
                        "Given this food name identify its main nutrients "
                        "and their corresponding nutritional values "
                        "in grams, milligrams, kcal etc.\n"
                        f"{label}?"
                    )

                try:
                    components = self.use_model(
                        query,
                        is_rag=is_rag,
                        is_selective=is_selective,
                    )
                except Exception as error:
                    components = f"Error: {error}"

                writer.writerow([food_id, food_name_predicted, components])
                outfile.flush()

    def predict_with_semantic_search(self, input_file: str, output_file: str):
        df = load_input_dataframe(input_file)
        output_path = resolve_output_path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components"])
            outfile.flush()

            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Processing rows",
                unit="row",
            ):
                food_id = f"{row['label']} - {row['image']}"

                try:
                    match = self.rag.search_first_by_image(
                        row["image"],
                        namespace="image",
                    )
                    if match is None:
                        food_name_predicted = "I don't know"
                        components_list = []
                    else:
                        food_name_predicted = match.get("food_class", "I don't know")
                        components_raw = match.get("components", "[]")
                        try:
                            components_data = ast.literal_eval(components_raw)
                            components_list = [
                                f"{item['name']}: {item['value']} {item['unit']}"
                                for item in components_data
                            ]
                        except Exception as error:
                            components_list = [f"Error parsing components: {error}"]
                except Exception as error:
                    food_name_predicted = "I don't know"
                    components_list = [f"Error searching image: {error}"]

                writer.writerow([food_id, food_name_predicted, components_list])
                outfile.flush()

    def process_json_and_predict_ablation(
        self,
        input_file: str,
        output_file: str,
        is_selective: bool = False,
        rag_ratio: float = 0.2,
    ):
        # Paper ablation: progressively increase the fraction of test samples
        # that use KG-grounded retrieval while keeping the rest ungrounded.
        df = load_input_dataframe(input_file)
        n_total = len(df)
        n_rag = int(n_total * rag_ratio)

        print(f"[Ablation] Progressive RAG: {n_rag}/{n_total} ({rag_ratio*100:.0f}%)")

        output_path = resolve_output_path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components", "used_rag"])
            outfile.flush()

            for idx, (_, row) in enumerate(
                tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row")
            ):
                food_id = f"{row['label']} - {row['image']}"
                food_name_predicted = self.predict_food_name(row["image"])
                use_rag_here = idx < n_rag
                label = f"Food name: {food_name_predicted}"

                if use_rag_here:
                    docs = self.rag.search_by_text(
                        food_name_predicted,
                        top_k=4,
                        namespace="text",
                    )
                    query = self._format_context(
                        docs,
                        label,
                        fallback="Information not found.",
                    )
                else:
                    query = (
                        "Given this food name identify its main nutrients "
                        "and their corresponding nutritional values "
                        "in grams, milligrams,  kcal etc.\n"
                        f"{label}?"
                    )

                try:
                    components = self.use_model(
                        query,
                        is_rag=use_rag_here,
                        is_selective=is_selective,
                    )
                except Exception as error:
                    components = f"Error: {error}"

                writer.writerow([food_id, food_name_predicted, components, use_rag_here])
                outfile.flush()


def build_shared_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--index-name", type=str, required=True, help="Pinecone index name")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSON-LD file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output CSV file")
    return parser


def build_assistant(index_name: str) -> FoodAssistant:
    rag = RAGRecipe(index_name=index_name)
    rag.load_clip()
    rag.set_text_model()
    return FoodAssistant(rag=rag)
