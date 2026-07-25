from io import BytesIO
from pathlib import Path
from typing import Union

import argparse
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

    # for a deterministic behavior (may impact performance):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


load_dotenv()

login(token=os.getenv("HUB_TOKEN"))


model_name = "tiiuae/Falcon3-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier_model_name = "yvelos/dinov3-food-389-v1"


model_img_emb = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_img_emb = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    # repetition_penalty=1.2,
    # num_return_sequences=1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    use_cache=True,
)

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


def _normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if {"label", "image"}.issubset(df.columns):
        return df.copy()

    if "@id" in df.columns and HAS_IMAGE_KEY in df.columns:
        normalized_rows = []
        for record in df.to_dict(orient="records"):
            image_path = _extract_first_value(record.get(HAS_IMAGE_KEY))
            if not image_path:
                continue

            normalized_rows.append(
                {
                    "label": _extract_label_from_jsonld_identifier(record),
                    "image": str(image_path),
                }
            )

        return pd.DataFrame(normalized_rows)

    raise ValueError(
        "Unsupported input format. Expected columns {'label', 'image'} "
        "or JSON-LD records with '@id' and image metadata."
    )


def load_input_dataframe(input_file: str) -> pd.DataFrame:
    input_path = Path(input_file)

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_file)
    elif input_path.suffix.lower() == ".jsonl":
        df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_json(input_file)

    return _normalize_input_dataframe(df)


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

    # ------------------ Pinecone setup ------------------

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

    def set_text_model(self):  # e.g. SentenceTransformer(...)
        self.text_model = SentenceTransformer("intfloat/multilingual-e5-large")

    def load_clip(self):
        if self.model_emb is None:
            self.model_emb = model_img_emb
            self.processor = processor_img_emb

    # ------------------ Image embedding ------------------
    def embed_image(self, image_source: Union[str, UploadFile, BytesIO]):
        # Case 1 : Image URL (str)
        if isinstance(image_source, str) and (
            image_source.startswith("http://")
            or image_source.startswith("https://")
        ):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        # Case 2 : Local path (str)
        elif isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")

        # Cas3 3 : Uploaded File (UploadFile ou BytesIO)
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

    # ------------------ Text embedding ------------------
    def embed_text(self, text: str):
        return self.text_model.encode(text, convert_to_numpy=True)

    # ------------------ Store embeddings ------------------
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

    def store_text_embedding(
        self,
        id: str,
        text: str,
        metadata: dict,
        namespace="text",
    ):
        emb = self.embed_text(text)
        self.store_embedding(id, emb, metadata, namespace)

    def store_image_embedding(
        self,
        id: str,
        image_path: any,
        metadata: dict,
        namespace="image",
    ):
        emb = self.embed_image(image_path)
        self.store_embedding(id, emb, metadata, namespace)

    # ------------------ Semantic search ------------------

    def search_by_text(self, query: str, top_k=5, namespace="text"):
        emb = self.embed_text(query)
        return self._search(emb, top_k, namespace)

    def search_by_image(
        self,
        image_path: Union[str, UploadFile, BytesIO],
        top_k=5,
        namespace="image",
    ):
        emb = self.embed_image(image_path)
        return self._search(emb, top_k, namespace)

    def search_first_by_image(
        self,
        image_path: Union[str, UploadFile, BytesIO],
        namespace="image",
    ):
        results = self.search_by_image(image_path, top_k=1, namespace=namespace)
        return results[0] if results else None

    def _search(self, embedding, top_k, namespace):
        # if namespace == "text":
        #     index = self.use_index("tsotsatext")
        # else:
        index = self.use_index(self.index_name)
        results = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        # print(results)
        return [match["metadata"] for match in results["matches"]]

    # ------------------ Unified query ------------------

    def smart_search(self, input_data, top_k=5):
        """
        input_data: str (text) or str (image_path ending with .jpg/.png/.jpeg)
        """
        if isinstance(input_data, str) and (
            ".jpg" or ".png" or ".jpeg" in input_data.lower()
        ):
            return self.search_by_image(input_data, top_k=top_k, namespace="image")
        else:
            return self.search_by_text(input_data, top_k=top_k, namespace="text")


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
        """
        Chooses a message based on the value of `is_rag`.
        """
    
        if not is_rag:
            if not is_selective:
                # Message for without knowledge graph
                # This message is used when the model does not need to access a knowledge graph.
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
            else:
                # Message for without knowledge graph but selective
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
        else:
            if not is_selective:
                # Message for with knowledge graph
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
            else:
                # Message for with knowledge graph but selective
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
        """
        Predict the name of a food item from an image input.

        Args:
            image_source (Union[str, UploadFile, BytesIO]):
                - str: Path to a local image file
                - UploadFile: Uploaded image file (FastAPI)
                - BytesIO: In-memory image bytes

        Returns:
            str: Predicted food class name (e.g., "pizza", "sushi", etc.)
        """

        return self.food_classifier.predict(image_source)

    def use_model(self, query, is_rag: bool = False, is_selective: bool = False):
        """
        Args:
            query (str): The input query to the model.

            message_type (str): The type of message to choose.
            is_selective (bool): Whether the response should be selective.
        Returns:
            str: The generated response from the model.
        """

        prompt = query

        # If RAG is enabled, we need to search for relevant documents
        messages = self.choose_message(
            prompt,
            is_rag=is_rag,
            is_selective=is_selective,
        )

        # Generate a response
        response = pipe(messages)

        return response[0]["generated_text"][2]["content"].strip().strip(".")

    def predict_with_image(
        self,
        input_file: str,
        output_file: str,
        is_rag: bool = False,
        is_vllm: bool = True,
    ):
        df = load_input_dataframe(input_file)
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Food Name", "image", "components"])

            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Processing rows",
                unit="row",
            ):
                name = str(row["name"]).strip()
                image_path = str(row["image"]).strip()
                print(image_path)

                if is_rag:
                    # If RAG is enabled, we need to search for relevant documents
                    docs = self.rag.search_by_image(
                        image_path,
                        top_k=4,
                        namespace="image",
                    )
                    query = self._format_context(
                        docs,
                        name,
                        fallback="Information not found.",
                    )
                else:
                    query = image_path

                prompt = {
                    "query": query,
                    "image": image_path,
                }

                try:
                    response_str = self.use_vllm_model(
                        prompt,
                        is_rag=is_rag,
                        is_vllm=is_vllm,
                    )
                    if response_str.strip().startswith("[") and response_str.strip().endswith(
                        "]"
                    ):

                        components = response_str
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    components = []

                writer.writerow([name, image_path, components])

    def process_json_and_predict_old(
        self,
        input_file: str,
        output_file: str,
        is_rag: bool = False,
        is_selective: bool = False,
    ):
        """
        :param input_file: path of the input file
        :param output_file: path of the output file
        """
        df = load_input_dataframe(input_file)
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components"])

            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Processing rows",
                unit="row",
            ):
                food_id = f"{row['label']} - {row['image']}"
                food_name_predicted = self.predict_food_name(row["image"])
                if food_name_predicted.lower() != row["label"].lower():
                    writer.writerow([food_id, food_name_predicted, "I don't know"])
                else:
                    print(
                        f"predicted_food={food_name_predicted}, "
                        f"target_food={row['label']}"
                    )
                    label = f"Food name: {food_name_predicted}"

                    if is_rag:
                        # If RAG is enabled, we need to search for relevant documents
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
                            "and their corresponding nutritional values in "
                            "grams, milligrams, kcal etc. \n"
                            f"{label}?"
                        )

                    try:
                        response_str = self.use_model(
                            query,
                            is_rag=is_rag,
                            is_selective=is_selective,
                        )
                        # response_dict = json.loads(response_str.response)
                        # print(response_str)

                        components = response_str
                    except Exception as e:
                        ingredients = f"Error: {e}"
                        components = f"Error: {e}"

                    writer.writerow([food_id, food_name_predicted, components])

    def process_json_and_predict(
        self,
        input_file: str,
        output_file: str,
        is_rag: bool = False,
        is_selective: bool = False,
        is_ablation_study: bool = False,
        rag_ratio: float = 0.2,
    ):
        """
        Progressive ablation study (cumulative RAG)

        rag_ratio: 0.2, 0.4, 0.6, 0.8, 1.0
        """

        df = load_input_dataframe(input_file)

        n_total = len(df)

        # ----------------------------
        # Number of samples using RAG
        # ----------------------------
        n_rag = 0

        if is_ablation_study:
            n_rag = int(n_total * rag_ratio)

            print(
                f"[Ablation] Progressive RAG: "
                f"{n_rag}/{n_total} ({rag_ratio*100:.0f}%)"
            )

        # ----------------------------
        # Write output
        # ----------------------------
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:

            writer = csv.writer(outfile)
            writer.writerow([
                "id",
                "predicted_name",
                "components",
                "used_rag",
            ])

            for idx, (_, row) in enumerate(
                tqdm(
                    df.iloc[0:].iterrows(),
                    total=len(df.iloc[0:]),
                    desc="Processing rows",
                    unit="row",
                )
            ):

                food_id = f"{row['label']} - {row['image']}"

                food_name_predicted = self.predict_food_name(
                    row["image"]
                )

                # Progressive RAG decision
                if is_ablation_study:
                    use_rag_here = idx < n_rag
                elif is_rag:
                    use_rag_here = True
                else:
                    use_rag_here = False

                # Wrong prediction
                # if food_name_predicted.lower() != row["label"].lower():

                #     writer.writerow([
                #         food_id,
                #         food_name_predicted,
                #         "I don't know",
                #         use_rag_here,
                #     ])
                #     continue

                label = f"Food name: {food_name_predicted}"

                # Build query
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

                    response = self.use_model(
                        query,
                        is_rag=use_rag_here,
                        is_selective=is_selective,
                    )

                    components = response

                except Exception as e:

                    components = f"Error: {e}"

                writer.writerow([
                    food_id,
                    food_name_predicted,
                    components,
                    use_rag_here,
                ])

    def predict_with_semantic_search(self, input_file: str, output_file: str):
        """
        Process JSON input using search_first_by_image, extract predicted food_class and components,
        and write results to CSV.

        :param input_file: path of the input JSON file
        :param output_file: path of the output CSV file
        """
        df = load_input_dataframe(input_file)

        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "predicted_name", "components"])

            for _, row in tqdm(
                df.iterrows(),
                total=len(df),
                desc="Processing rows",
                unit="row",
            ):
                food_id = f"{row['label']} - {row['image']}"

                # Search for the first match by image
                try:
                    match = self.rag.search_first_by_image(
                        row["image"],
                        namespace="image",
                    )
                    if match is None:
                        food_name_predicted = "I don't know"
                        components_list = []
                    else:
                        # retrieve the label predicted and components from metadata
                        food_name_predicted = match.get("food_class", "I don't know")

                        # Extract and format components
                        components_raw = match.get("components", "[]")
                        try:
                            # Safely evaluate the string representation of the list
                            components_data = ast.literal_eval(components_raw)
                            components_list = [
                                f"{item['name']}: {item['value']} {item['unit']}"
                                for item in components_data
                            ]
                        except Exception as e:
                            components_list = [f"Error parsing components: {e}"]
                except Exception as e:
                    food_name_predicted = "I don't know"
                    components_list = [f"Error searching image: {e}"]

                writer.writerow([food_id, food_name_predicted, components_list])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Food Multimodal RAG Pipeline"
    )

    # ------------------ Core ------------------
    parser.add_argument(
        "--index-name",
        type=str,
        required=True,
        help="Pinecone index name",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input JSON file",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output CSV file",
    )

    # ------------------ Modes ------------------
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "no_rag", "semantic_search"],
        default="no_rag",
        help="Execution mode",
    )

    parser.add_argument(
        "--selective",
        action="store_true",
        help="Enable selective prompting",
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Enable progressive RAG ablation study",
    )

    parser.add_argument(
        "--rag-ratio",
        type=float,
        default=0.2,
        help="RAG ratio for ablation (0.2, 0.4, 0.6, 0.8, 1.0)",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # ------------------ Init RAG ------------------
    rag = RAGRecipe(index_name=args.index_name)
    rag.load_clip()
    rag.set_text_model()

    assistant = FoodAssistant(rag=rag)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ------------------ Execution ------------------
    if args.mode == "semantic_search":

        assistant.predict_with_semantic_search(
            input_file=args.input_file,
            output_file=args.output_file,
        )

    else:

        is_rag = args.mode == "rag"

        assistant.process_json_and_predict(
            input_file=args.input_file,
            output_file=args.output_file,
            is_rag=is_rag,
            is_selective=args.selective,
            is_ablation_study=args.ablation,
            rag_ratio=args.rag_ratio,
        )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
