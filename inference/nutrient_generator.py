from io import BytesIO
from fastapi import UploadFile
from dotenv import load_dotenv # type: ignore
import requests
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, pipeline, AutoImageProcessor, AutoModelForImageClassification
from sentence_transformers import SentenceTransformer
import csv, os, pandas as pd, torch, json, gc
from PIL import Image
from typing import Literal, Union
from tqdm import tqdm
from huggingface_hub import login
from pinecone import Pinecone, ServerlessSpec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import random
import numpy as np, ast

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Pour un comportement complètement déterministe :
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

load_dotenv()

login(token=os.getenv("HUB_TOKEN")) 

MODEL_TYPE = Literal["mistral", "falcon", "llama"]

# model_image_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# processor = AutoProcessor.from_pretrained(model_image_id, trust_remote_code=True)
# model_vllm = AutoModelForVision2Seq.from_pretrained(model_image_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")


model_name = "tiiuae/Falcon3-7B-Instruct"
# model_name = "mistralai/Ministral-8B-Instruct-2410"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_emb = SentenceTransformer("intfloat/multilingual-e5-large")
vllm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
vllm_model_name = "llava-hf/llava-1.5-7b-hf"
# vllm_model_name = "OpenGVLab/InternVL2-4B"



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
    use_cache=True
)

class RAGRecipe:
    def __init__(self, index_name, cloud='aws', region="us-east-1", metric='cosine'):
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
            spec=ServerlessSpec(cloud=self.cloud, region=self.region)
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
        if isinstance(image_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
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
    def store_embedding(self, is_new_index: bool, id: str, embedding, metadata: dict, namespace="ns1"):
        if namespace == "text":
            index = self.use_index(name="tsotsatext")
        else:
            index = self.create_index() if is_new_index else self.use_index()

        index.upsert([{"id": id, 'values': embedding, "metadata": metadata}], namespace=namespace, batch_size=12)

    def store_text_embedding(self, id: str, text: str, metadata: dict, namespace='text'):
        emb = self.embed_text(text)
        self.store_embedding(id, emb, metadata, namespace)

    def store_image_embedding(self, id: str, image_path: any, metadata: dict, namespace='image'):
        emb = self.embed_image(image_path)
        self.store_embedding(id, emb, metadata, namespace)

    # ------------------ Semantic search ------------------

    def search_by_text(self, query: str, top_k=5, namespace='text'):
        emb = self.embed_text(query)
        return self._search(emb, top_k, namespace)

    def search_by_image(self, image_path: Union[str, UploadFile, BytesIO], top_k=5, namespace='image'):
        emb = self.embed_image(image_path)
        return self._search(emb, top_k, namespace)
    
    def search_first_by_image(self, image_path: Union[str, UploadFile, BytesIO], namespace='image'):
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
            include_metadata=True
        )
        # print(results)
        return [match['metadata'] for match in results['matches']]

    # ------------------ Unified query ------------------

    def smart_search(self, input_data, top_k=5):
        """
        input_data: str (text) or str (image_path ending with .jpg/.png/.jpeg)
        """
        if isinstance(input_data, str) and ('.jpg' or'.png' or'.jpeg' in input_data.lower()):
            return self.search_by_image(input_data, top_k=top_k, namespace='image')
        else:
            return self.search_by_text(input_data, top_k=top_k, namespace='text')
        
class FoodAssistant:
    def __init__(self, rag: RAGRecipe):
        self.rag = rag
        self.model_cv = AutoModelForImageClassification.from_pretrained("yvelos/beit-food-389")
        self.processor = AutoImageProcessor.from_pretrained("yvelos/beit-food-389")
        
    def _format_context(self, docs, query, fallback="Information not found."):
        if not docs:
            docs = [fallback]
        context = "\n".join([f"- {doc}" for doc in docs])
        return f"Here is the information found on ORKG(Open Resource Knowledge Graph) about {query} food: {context}\n\nPlease provide all the food components of: {query} food?"

    def choose_message(self, prompt: Union[dict, str], is_rag: bool = False, is_selective: bool = False, is_vllm=False):
        # sourcery skip: inline-immediately-returned-variable, switch
        """
        Chooses a message based on the value of `is_rag`.
        """
        if is_vllm:
            #  Message for VLLM
            if is_rag:
                # Message for VLLM with RAG
                return [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "url": prompt['image']},  # Assuming prompt is an image URL or path
                            {
                                "type": "text", 
                                "text": "You are a helpful assistant in the food engineering domain."
                                "don't include explanation or any other text, Note that the answer should be only in the three case above."
                                "your response should look like this: ['comp1 (value unit)', 'comp2 (value unit)', ..., 'compN'] example: ['Cholesterol (0 mg)','Folate, total (133.0 µg)','Fiber, total dietary (0 g)','Niacin (3.556 mg)', etc]"
                                "Please extract all the food nutritional components in the context provided"
                                f"{prompt['query']}"
                            }
        
                        ]
                    }
                ]
            return [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "url": prompt.image},  # Assuming prompt is an image URL or path
                        {
                            "type": "text", 
                            "text": "You are a helpful assistant in the food engineering domain."
                            "Given this food image, identify its main nutrients and their corresponding nutritional values in grams, milligrams, or kcal."
                        }
                    ]
                }
            ]
        elif not is_rag:
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
                        7. Return only the final python list - nothing else."
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
                        7. Return only the final python list OR \"I don't know\" - nothing else."
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
                        10. Return only the final python list - nothing else."
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
        set_seed(42)
        # --- Load image depending on input type ---
        if (
            isinstance(image_source, str)
            or not isinstance(image_source, UploadFile)
            and isinstance(image_source, BytesIO)
        ):
            # Case 1: local file path
            image = Image.open(image_source).convert("RGB")

        elif isinstance(image_source, UploadFile):
            # Case 2: uploaded file from FastAPI
            image = Image.open(BytesIO(image_source.file.read())).convert("RGB")

        else:
            raise ValueError("Unsupported input type. Use str, UploadFile, or BytesIO.")
        self.model_cv.to(device)
        id2label = self.model_cv.config.id2label

        # --- Preprocess the image ---
        inputs = self.processor(images=image, return_tensors="pt").to(device)

        # --- Run model inference ---
        with torch.no_grad():
            logits = self.model_cv(**inputs).logits
            pred_id = logits.argmax(-1).item()

        return id2label[pred_id]
    
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
        messages = self.choose_message(prompt, is_rag=is_rag, is_selective=is_selective)

        # Generate a response
        response = pipe(messages)
        
        return response[0]['generated_text'][2]['content'].strip().strip('.')
    
    def use_vllm_model(self, prompt: Union[dict, str], is_rag: bool = False, is_vllm: bool = False):
        """
        Args:
            prompt (str): The input prompt to the model.
            is_rag (bool): Whether to use RAG for the query.
            is_selective (bool): Whether the response should be selective.
        Returns:
            str: The generated response from the model.
        """

        # If RAG is enabled, we need to search for relevant documents        
        messages = self.choose_message(prompt, is_rag=is_rag, is_vllm=is_vllm)

        # Generate a response
        response = pipe_vllm(
            messages,
            use_fast=False,
            max_new_tokens=512,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        output = response[0]['generated_text'][1]['content'].strip().strip('.')
        del response
        torch.cuda.empty_cache()
        gc.collect()
        
        return output

    def predict_with_image(self, input_file: str, output_file: str, is_rag: bool = False, is_vllm: bool = True):
        
        df = pd.read_csv(input_file)
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Food Name', 'image', 'components'])
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
                name = str(row['name']).strip()
                image_path = str(row['image']).strip()
                print(image_path)
                
                if is_rag:
                    # If RAG is enabled, we need to search for relevant documents
                    docs = self.rag.search_by_image(image_path, top_k=4, namespace="image")
                    query = self._format_context(docs, name, fallback="Information not found.")
                else:
                    query = image_path
                    
                prompt = {
                    'query': query,
                    'image': image_path,
                }

                try:
                    response_str = self.use_vllm_model(prompt, is_rag=is_rag, is_vllm=is_vllm)
                    if response_str.strip().startswith("[") and response_str.strip().endswith("]"):

                        components = response_str
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    components = []

                writer.writerow([name, image_path, components])

    def process_json_and_predict_old(self, input_file: str, output_file: str, is_rag: bool = False, is_selective: bool = False):
        """
        :param input_file: path of the input file
        :param output_file: path of the output file
        """
        df = pd.read_json(input_file)
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['id', 'predicted_name', 'components'])

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
                food_id = f"{row['label']} - {row['image']}"
                food_name_predicted = self.predict_food_name(row["image"])
                if food_name_predicted.lower() != row["label"].lower():
                    writer.writerow([food_id, food_name_predicted, "I don't know"])
                else:   
                    print(f"predicted_food={food_name_predicted}, target_food={row['label']}")
                    label = f"Food name: {food_name_predicted}"
                    
                    if is_rag:
                        # If RAG is enabled, we need to search for relevant documents
                        docs = self.rag.search_by_text(food_name_predicted, top_k=4, namespace="text")
                        query = self._format_context(docs, label, fallback="Information not found.")
                    else:
                        query = f"Given this food name identify its main nutrients and their corresponding nutritional values in grams, milligrams, kcal etc. \n{label}?"

                    try:
                        response_str = self.use_model(query, is_rag=is_rag, is_selective=is_selective)
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
        rag_ratio: float = 0.2
    ):
        """
        Progressive ablation study (cumulative RAG)

        rag_ratio: 0.2, 0.4, 0.6, 0.8, 1.0
        """

        df = pd.read_json(input_file)

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
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:

            writer = csv.writer(outfile)
            writer.writerow([
                'id',
                'predicted_name',
                'components',
                'used_rag'
            ])

            for idx, (_, row) in enumerate(
                tqdm(df.iterrows(), total=n_total,
                     desc="Processing rows", unit="row")
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
                if food_name_predicted.lower() != row["label"].lower():

                    writer.writerow([
                        food_id,
                        food_name_predicted,
                        "I don't know",
                        use_rag_here
                    ])
                    continue

                label = f"Food name: {food_name_predicted}"

                # Build query
                if use_rag_here:

                    docs = self.rag.search_by_text(
                        food_name_predicted,
                        top_k=4,
                        namespace="text"
                    )

                    query = self._format_context(
                        docs,
                        label,
                        fallback="Information not found."
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
                        is_selective=is_selective
                    )

                    components = response

                except Exception as e:

                    components = f"Error: {e}"

                writer.writerow([
                    food_id,
                    food_name_predicted,
                    components,
                    use_rag_here
                ])

    
    def predict_with_semantic_search(self, input_file: str, output_file: str):
        """
        Process JSON input using search_first_by_image, extract predicted food_class and components,
        and write results to CSV.

        :param input_file: path of the input JSON file
        :param output_file: path of the output CSV file
        """
        df = pd.read_json(input_file)

        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['id', 'predicted_name', 'components'])

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
                food_id = f"{row['label']} - {row['image']}"
                
                # Search for the first match by image
                try:
                    match = self.rag.search_first_by_image(row['image'], namespace='image')
                    if match is None:
                        food_name_predicted = "I don't know"
                        components_list = []
                    else:
                        # retrieve the label predicted and components from metadata
                        food_name_predicted = match.get('food_class', "I don't know")
                        
                        # Extract and format components
                        components_raw = match.get('components', "[]")
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



rag = RAGRecipe(index_name="icml-paper")
rag.load_clip()
rag.set_text_model()
# output_file = "results/falcon/beit/not_rag/merged.csv"
# output_file = "results/falcon/beit/not_rag/food101.csv"
# output_file = "results/falcon/beit/not_rag/uecfood256.csv"
output_file = "results/falcon/beit/not_rag/fruitveg81.csv"

# output_file = "results/falcon/beit/rag/fruitveg81.csv"
# output_file = "results/falcon/beit/rag/food101.csv"
# output_file = "results/falcon/beit/rag/merged.csv"
# output_file = "results/falcon/beit/rag/uecfood256.csv"

# output_file = "results/falcon/vit/merged.csv"
# output_file = "results/falcon/vit/food101.csv"
# output_file = "results/falcon/vit/uecfood256.csv"
# output_file = "results/falcon/vit/fruitveg81.csv"



assistant = FoodAssistant(rag=rag)
def predict_beit_with_falcon():
    """ AFD output without rag"""
    # print("======================AFD without rag================================")
    # input_file = "dataset/multimodal/not_merged/test/AFD_test.json"
    # output_file = "results/falcon/beit/not_rag/afd.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=False
    # )
    # ===================
    """ AFD output with rag"""
    # print("======================AFD with rag================================")
    # input_file = "dataset/multimodal/not_merged/test/AFD_test.json"
    # output_file = "results/falcon/beit/rag/afd.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=True
    # )
    
    """ AFD semantic search output"""
    # input_file = "dataset/multimodal/not_merged/test/AFD_test.json"
    # output_file = "results/falcon/beit/semantic_search/afd.csv"
    # assistant.predict_with_semantic_search(
    #     input_file=input_file,
    #     output_file=output_file
    # )
    
    # """ AFD semantic search output"""
    # input_file = "dataset/multimodal/not_merged/test/AFD_test.json"
    # output_file = "results/falcon/beit/60%/afd.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=False,
    #     is_ablation_study=True,
    #     rag_ratio=0.6
    # )
    
    """ FruitVege81 output without rag"""
    # print("======================FruitVeg81 without rag================================")
    # input_file = "dataset/multimodal/not_merged/test/fruitveg81_test.json"
    # output_file = "results/falcon/beit/not_rag/fruitveg81.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=False
    # )
    # ===================
    """ FruitVege81 output with rag"""
    # print("======================FruitVeg81 with rag================================")
    # input_file = "dataset/multimodal/not_merged/test/fruitveg81_test.json"
    # output_file = "results/falcon/beit/rag/fruitveg81.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=True
    # )
    
    """ FruitVege81 semantic search output"""
    # input_file = "dataset/multimodal/not_merged/test/fruitveg81_test.json"
    # output_file = "results/falcon/beit/semantic_search/fruitveg81.csv"
    # assistant.predict_with_semantic_search(
    #     input_file=input_file,
    #     output_file=output_file
    # )
    
    """ AFD semantic search output"""
    input_file = "dataset/multimodal/not_merged/test/fruitveg81_test.json"
    output_file = "results/falcon/beit/40%/fruitveg81.csv"
    assistant.process_json_and_predict(
        input_file=input_file,
        output_file=output_file,
        is_selective=True,
        is_rag=False,
        is_ablation_study=True,
        rag_ratio=0.4
    )
    
    """ Food101 output with rag"""
    # print("======================Food101 with rag================================")
    # input_file = "dataset/multimodal/not_merged/test/food101_test.json"
    # output_file = "results/falcon/beit/rag/food101.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=True
    # )
    # # =====================
    """ Food101 output without rag"""
    # print("======================Food101 without rag================================")
    # input_file = "dataset/multimodal/not_merged/test/food101_test.json"
    # output_file = "results/falcon/beit/not_rag/food101.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=False
    # )
    
    """ Food101 semantic search output"""
    # input_file = "dataset/multimodal/not_merged/test/food101_test.json"
    # output_file = "results/falcon/beit/semantic_search/food101.csv"
    # assistant.predict_with_semantic_search(
    #     input_file=input_file,
    #     output_file=output_file
    # )
    
    # ======================
    """ UECFOOD256 output with rag"""
    # print("======================uecfood256 without rag================================")
    # input_file = "dataset/multimodal/not_merged/test/uecfood256_test.json"
    # output_file = "results/falcon/beit/rag/uecfood256_1.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=True
    # )
    
    
    # # =====================
    """ UECFOOD256 output without rag"""
    # print("======================uecfood256 without rag================================")
    # input_file = "dataset/multimodal/not_merged/test/uecfood256_test.json"
    # output_file = "results/falcon/beit/not_rag/uecfood256.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=False
    # )
    
    """ UECFOOD256 semantic search output"""
    # input_file = "dataset/multimodal/not_merged/test/uecfood256_test.json"
    # output_file = "results/falcon/beit/semantic_search/uecfood256.csv"
    # assistant.predict_with_semantic_search(
    #     input_file=input_file,
    #     output_file=output_file
    # )
    
    
    
    # =====================
    # """ Merged output without rag"""
    # input_file = "dataset/multimodal/merged/merged_test.json"
    # output_file = "results/falcon/beit/not_rag/merged.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=False
    # )
    
    # # =====================
    # """ Merged output with rag"""
    # input_file = "dataset/multimodal/merged/merged_test.json"
    # output_file = "results/falcon/beit/rag/merged.csv"
    # assistant.process_json_and_predict(
    #     input_file=input_file,
    #     output_file=output_file,
    #     is_selective=True,
    #     is_rag=True
    # )
    
    """ Merged semantic search output"""
    # input_file = "dataset/multimodal/merged/merged_test.json"
    # output_file = "results/falcon/beit/semantic_search/merged.csv"
    # assistant.predict_with_semantic_search(
    #     input_file=input_file,
    #     output_file=output_file
    # )
    

if __name__ == "__main__":
    predict_beit_with_falcon()