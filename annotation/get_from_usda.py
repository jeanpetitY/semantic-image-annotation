import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


DEFAULT_USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEFAULT_MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
DEFAULT_TIMEOUT = 30
DEFAULT_SEARCH_PAGE_SIZE = 5
DEFAULT_OUTPUT_FILE = "outputs/default/usda_enriched.json"
DEFAULT_NOT_ANNOTATED_FILE = "outputs/default/not_annotated.json"
DISH_DATA_TYPE_PRIORITY = [
    "Survey (FNDDS)",
    "Foundation",
    "SR Legacy",
    "Experimental",
    "Branded",
]
NUTRIENT_GROUP_NAMES = {
    "Proximates",
    "Carbohydrates",
    "Minerals",
    "Vitamins and Other Components",
    "Vitamins",
    "Lipids",
    "Amino acids",
}
TOKEN_ALIASES = {
    "macaroni": {"macaroni", "mac"},
    "fries": {"fries", "fry"},
    "mangos": {"mango", "mangos"},
    "tomatos": {"tomato", "tomatos", "tomatoes"},
}
IGNORED_MATCH_TOKENS = {
    "and",
    "with",
    "the",
    "food",
    "foods",
}


class Response(BaseModel):
    response: str


def load_client(model_name: str):
    import torch
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=96,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )


def load_json_list(file_path: str) -> List[Dict]:
    if not file_path or not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_list(data: List[Dict], file_path: str) -> None:
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        file.write("\n")


def upsert_by_food_class(records: List[Dict], record: Dict) -> List[Dict]:
    food_class = record.get("food_class")

    return [
        item
        for item in records
        if item.get("food_class") != food_class
    ] + [record]


def extract_food_category(data: Dict) -> Optional[str]:
    for key in ("foodCategory", "wweiaFoodCategory", "brandedFoodCategory"):
        value = data.get(key)

        if isinstance(value, dict):
            description = (
                value.get("description")
                or value.get("name")
                or value.get("wweiaFoodCategoryDescription")
            )
            if description:
                return description

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def normalize_match_tokens(value: str) -> set:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    tokens = set()

    for token in value.split():
        if len(token) <= 2 or token in IGNORED_MATCH_TOKENS:
            continue

        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]

        tokens.add(token)

    return tokens


def token_matches(label_token: str, candidate_tokens: set) -> bool:
    accepted = TOKEN_ALIASES.get(label_token, {label_token})

    return bool(accepted & candidate_tokens)


def is_candidate_label_eligible(label: str, candidate: str) -> bool:
    label_tokens = normalize_match_tokens(label)
    candidate_tokens = normalize_match_tokens(candidate)

    if not label_tokens or not candidate_tokens:
        return False

    if len(label_tokens) == 1:
        return token_matches(next(iter(label_tokens)), candidate_tokens)

    return all(
        token_matches(token, candidate_tokens)
        for token in label_tokens
    )


def parse_model_response(value: str) -> str:
    value = value.strip().strip(".")

    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return str(parsed.get("response", "")).strip()
    except json.JSONDecodeError:
        pass

    match = re.search(r'"response"\s*:\s*"([^"]+)"', value)
    if match:
        return match.group(1).strip()

    return value.strip('"').strip("'").strip()


class AskFoodSearch:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, openai_api_key=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.client = None
        self.pipe = None

    @staticmethod
    def is_gpt_model(model_name: str) -> bool:
        return model_name in ["gpt-4o-mini", "gpt-3.5-turbo"]

    def get_openai_client(self):
        if self.client is None:
            if not self.openai_api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is required when using an OpenAI validator."
                )
            self.client = OpenAI(api_key=self.openai_api_key)

        return self.client

    def get_pipe(self):
        if self.pipe is None:
            self.pipe = load_client(self.model_name)

        return self.pipe

    @staticmethod
    def choose_candidate_messages(label: str, candidates: List[str]):
        candidate_lines = "\n".join(
            f"{index}. {candidate}"
            for index, candidate in enumerate(candidates, start=1)
        )

        prompt = (
            "Dataset food label:\n"
            f"{label}\n\n"
            "USDA candidate labels:\n"
            f"{candidate_lines}\n\n"
            "Choose the single USDA candidate label that is semantically "
            "equivalent to the dataset food label. Return exactly that USDA "
            "candidate label. If none are equivalent, return none."
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are a food annotation validator. "
                    "Respond only as JSON with the key 'response'. "
                    "The value must be exactly one candidate label or 'none'."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def normalize_choice(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    def choose_equivalent_candidate(
        self,
        label: str,
        candidates: List[str],
    ) -> Optional[str]:
        if not candidates:
            return None

        messages = self.choose_candidate_messages(label, candidates)

        if self.is_gpt_model(self.model_name):
            completion = self.get_openai_client().beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=Response,
            )
            raw_choice = completion.choices[0].message.parsed.response
        else:
            response = self.get_pipe()(messages)
            raw_choice = response[0]["generated_text"][2]["content"]

        choice = parse_model_response(raw_choice)

        if self.normalize_choice(choice) == "none":
            return None

        normalized_choice = self.normalize_choice(choice)

        for candidate in candidates:
            if self.normalize_choice(candidate) == normalized_choice:
                return candidate

        for candidate in candidates:
            if normalized_choice in self.normalize_choice(candidate):
                return candidate

        return None

    def ask_if_food_is_fruit_or_vegetable(self, label):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a food science assistant. "
                    "Your task is to determine whether a given food label "
                    "refers to a fruit or a vegetable. "
                    "Respond only as JSON with the key 'response'. "
                    "The value must be either 'fruit' or 'vegetable'."
                ),
            },
            {"role": "user", "content": f"Is this food a fruit or vegetable? '{label}'"},
        ]

        if self.is_gpt_model(self.model_name):
            completion = self.get_openai_client().beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=Response,
            )
            return completion.choices[0].message.parsed.response

        response = self.get_pipe()(messages)
        return parse_model_response(response[0]["generated_text"][2]["content"])

    def update_food_file_with_food_type(
        self,
        input_file_path: str,
        output_file_path: str,
        label_field: str = "food_class",
        output_key: str = "food_type",
    ):
        with open(input_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of food objects")

        with open(output_file_path, "w", encoding="utf-8") as output:
            output.write("[\n")
            first_item = True

            for item in data:
                label = item.get(label_field)
                if label:
                    try:
                        result = self.ask_if_food_is_fruit_or_vegetable(label)
                        print(f"[INFO] Classified '{label}' as '{result}'")
                        item[output_key] = result
                    except Exception as exc:
                        print(f"[WARNING] Failed to classify '{label}': {exc}")
                        item[output_key] = None
                else:
                    item[output_key] = None

                if not first_item:
                    output.write(",\n")
                else:
                    first_item = False

                json.dump(item, output, ensure_ascii=False, indent=2)
                output.flush()

            output.write("\n]")

        print(f"File updated incrementally: {output_file_path}")


class USDAFoodEnrichmentService:
    def __init__(
        self,
        asker: AskFoodSearch,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_USDA_BASE_URL,
        base_image_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        search_page_size: int = DEFAULT_SEARCH_PAGE_SIZE,
    ) -> None:
        self.asker = asker
        if not api_key:
            raise EnvironmentError("USDA_KEY is missing in environment variables.")

        self.api_key = api_key
        self.base_url = base_url
        self.base_image_url = base_image_url
        self.timeout = timeout
        self.search_page_size = search_page_size
        self.details_by_id: Dict[int, Dict] = {}
        self.search_calls = 0
        self.detail_calls = 0

    @staticmethod
    def normalize_label(label: str) -> str:
        label = label.lower().strip()
        label = label.replace("&", "and")
        label = re.sub(r"[ \-_/']", "_", label)
        label = re.sub(r"[^a-z0-9_]", "", label)
        label = re.sub(r"_+", "_", label)

        return label.strip("_")

    @staticmethod
    def format_query(label: str) -> str:
        return label.replace("_", " ").title()

    def build_image_url(
        self,
        label: str,
        base_url: Optional[str] = None,
    ) -> str:
        if base_url is None:
            if self.base_image_url is None:
                return ""

            base_url = f"{self.base_image_url.rstrip('/')}/"

        normalized = self.normalize_label(label)
        base_url = f"{base_url.rstrip('/')}/"

        return f"{base_url}{normalized}.jpg"

    def search_usda_candidates(
        self,
        query: str,
        data_type: str,
    ) -> List[Dict]:
        endpoint = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": self.search_page_size,
            "dataType": data_type,
        }

        self.search_calls += 1

        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            print(
                f"[WARNING] USDA search failed for '{query}' "
                f"in {data_type}: {exc}"
            )
            return []

        if response.status_code != 200:
            if data_type == "Survey (FNDDS)" and self.search_page_size > 1:
                fallback_params = dict(params)
                fallback_params["pageSize"] = 1
                fallback_response = requests.get(
                    endpoint,
                    params=fallback_params,
                    timeout=self.timeout,
                )

                if fallback_response.status_code == 200:
                    print(
                        "[WARNING] USDA Survey search only accepted "
                        f"pageSize=1 for '{query}'."
                    )
                    return fallback_response.json().get("foods", [])

            print(
                f"[WARNING] USDA search returned {response.status_code} "
                f"for '{query}' in {data_type}"
            )
            return []

        return response.json().get("foods", [])

    def search_first_food(self, query: str) -> Dict:
        endpoint = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": 1,
        }

        self.search_calls += 1

        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            print(f"[WARNING] USDA search failed for '{query}': {exc}")
            return {
                "fdc_id": None,
                "name": None,
                "data_type": None,
                "attempts": [],
                "reason": "usda_search_failed",
            }

        if response.status_code != 200:
            return {
                "fdc_id": None,
                "name": None,
                "data_type": None,
                "attempts": [],
                "reason": "usda_search_bad_status",
            }

        foods = response.json().get("foods", [])

        if not foods:
            return {
                "fdc_id": None,
                "name": None,
                "data_type": None,
                "attempts": [],
                "reason": "no_usda_results",
            }

        food = foods[0]

        return {
            "fdc_id": food.get("fdcId"),
            "name": food.get("description"),
            "data_type": food.get("dataType"),
            "attempts": [
                {
                    "data_type": food.get("dataType"),
                    "candidates": [food.get("description")],
                    "selected": food.get("description"),
                }
            ],
            "reason": "first_usda_result",
        }

    def choose_food(
        self,
        label: str,
        query: str,
        data_type_priority: List[str],
    ) -> Dict:
        attempts = []
        saw_candidates = False

        for data_type in data_type_priority:
            candidates = self.search_usda_candidates(query, data_type)
            raw_candidate_labels = [
                food.get("description")
                for food in candidates
                if food.get("fdcId") and food.get("description")
            ]
            eligible_candidates = [
                food
                for food in candidates
                if (
                    food.get("fdcId")
                    and food.get("description")
                    and is_candidate_label_eligible(label, food["description"])
                )
            ]
            candidate_labels = [
                food["description"]
                for food in eligible_candidates
            ]

            if not raw_candidate_labels:
                attempts.append(
                    {
                        "data_type": data_type,
                        "candidates": [],
                        "eligible_candidates": [],
                        "selected": None,
                    }
                )
                continue

            saw_candidates = True

            if not candidate_labels:
                attempts.append(
                    {
                        "data_type": data_type,
                        "candidates": raw_candidate_labels,
                        "eligible_candidates": [],
                        "selected": None,
                        "reason": "no_candidate_passed_token_filter",
                    }
                )
                continue

            selected_label = self.asker.choose_equivalent_candidate(
                label=label,
                candidates=candidate_labels,
            )
            selected_food = None

            if selected_label:
                for food in eligible_candidates:
                    if food.get("description") == selected_label:
                        selected_food = food
                        break

            attempts.append(
                {
                    "data_type": data_type,
                    "candidates": raw_candidate_labels,
                    "eligible_candidates": candidate_labels,
                    "selected": selected_label,
                }
            )

            if selected_food:
                return {
                    "fdc_id": selected_food["fdcId"],
                    "name": selected_food["description"],
                    "data_type": data_type,
                    "attempts": attempts,
                    "reason": "llm_equivalence",
                }

        return {
            "fdc_id": None,
            "name": None,
            "data_type": None,
            "attempts": attempts,
            "reason": (
                "needs_human_verification"
                if saw_candidates
                else "no_usda_results"
            ),
        }

    def get_food_details(self, fdc_id: int) -> Optional[Dict]:
        if fdc_id in self.details_by_id:
            return self.details_by_id[fdc_id]

        endpoint = f"{self.base_url}/food/{fdc_id}"
        params = {"api_key": self.api_key}
        self.detail_calls += 1

        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            print(f"[WARNING] USDA details failed for '{fdc_id}': {exc}")
            return None

        if response.status_code != 200:
            return None

        data = response.json()

        nutrients = []

        for nutrient in data.get("foodNutrients", []):
            nutrient_info = nutrient.get("nutrient")
            value = nutrient.get("amount")

            if not nutrient_info or value is None:
                continue

            name = nutrient_info.get("name")
            if name in NUTRIENT_GROUP_NAMES:
                continue

            nutrients.append(
                {
                    "name": name,
                    "value": value,
                    "unit": nutrient_info.get("unitName"),
                }
            )

        if data.get("ingredients"):
            ingredients = [
                item.strip()
                for item in data["ingredients"].split(",")
                if item.strip()
            ]
        else:
            ingredients = [
                food.get("ingredientDescription", "").strip()
                for food in data.get("inputFoods", [])
                if food.get("ingredientDescription", "").strip()
            ]

        details = {
            "fdc_id": fdc_id,
            "portion": "100 g",
            "description": data.get("description"),
            "data_type": data.get("dataType"),
            "food_category": extract_food_category(data),
            "ingredients": ingredients,
            "nutrients": nutrients,
            "source_url": (
                f"https://fdc.nal.usda.gov/food-details/"
                f"{fdc_id}/nutrients"
            ),
        }
        self.details_by_id[fdc_id] = details

        return details

    def build_food_entry(
        self,
        label: str,
        details: Dict,
        image_base_url: Optional[str],
    ) -> Dict:
        entry = {
            "food_class": label,
            "portion": details["portion"],
            "usda_name": details["description"],
            "description": details["description"],
            "ingredients": details["ingredients"],
            "nutrients": details["nutrients"],
            "usda_source": details["source_url"],
            "image": self.build_image_url(label, image_base_url),
        }

        if details.get("data_type"):
            entry["usda_data_type"] = details["data_type"]

        if details.get("food_category"):
            entry["usda_food_category"] = details["food_category"]

        return entry

    def retrieve_food(
        self,
        label: str,
        is_fruit_veg: bool,
    ) -> Dict:
        query = self.format_query(label)
        if is_fruit_veg:
            query = f"{query}, raw"
            choice = self.search_first_food(query)

            if not choice.get("fdc_id"):
                return {
                    "details": None,
                    "choice": choice,
                    "query": query,
                }

            return {
                "details": self.get_food_details(choice["fdc_id"]),
                "choice": choice,
                "query": query,
            }

        choice = self.choose_food(
            label=label,
            query=query,
            data_type_priority=DISH_DATA_TYPE_PRIORITY,
        )

        if not choice.get("fdc_id"):
            return {
                "details": None,
                "choice": choice,
                "query": query,
            }

        return {
            "details": self.get_food_details(choice["fdc_id"]),
            "choice": choice,
            "query": query,
        }

    def process_dataset(
        self,
        labels: List,
        output_file: str,
        not_annotated_file: str,
        is_fruit_veg: bool = False,
        image_base_url: Optional[str] = None,
    ) -> None:
        enriched = load_json_list(output_file)
        not_annotated = load_json_list(not_annotated_file)
        processed = {
            item.get("food_class")
            for item in enriched
            if item.get("food_class")
        }

        total = len(labels)
        print(f"Processing {total} food items...")

        for index, label in enumerate(labels, start=1):
            if label in processed:
                continue

            print(f"[{index}/{total}] {self.format_query(label)}")

            result = self.retrieve_food(
                label=label,
                is_fruit_veg=is_fruit_veg,
            )
            details = result["details"]
            choice = result["choice"]

            if details is None:
                not_annotated = upsert_by_food_class(
                    not_annotated,
                    {
                        "food_class": label,
                        "query": result["query"],
                        "reason": choice.get("reason", "details_unavailable"),
                        "attempts": choice.get("attempts", []),
                    },
                )
                save_json_list(not_annotated, not_annotated_file)
                continue

            entry = self.build_food_entry(
                label=label,
                details=details,
                image_base_url=image_base_url,
            )

            enriched.append(entry)
            save_json_list(enriched, output_file)
            time.sleep(0.5)

        save_json_list(enriched, output_file)
        save_json_list(not_annotated, not_annotated_file)

        print(f"Saved {len(enriched)} entries to {output_file}")
        print(f"Saved {len(not_annotated)} entries to {not_annotated_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="USDA Food Dataset Enrichment Service"
    )

    parser.add_argument(
        "--food-labels",
        type=str,
        default="../dataset/image/not_merged/food101/test",
        help="Path of the image dataset folder containing food class folders.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output enriched JSON file.",
    )
    parser.add_argument(
        "--not-annotated-file",
        type=str,
        default=DEFAULT_NOT_ANNOTATED_FILE,
        help="Path where not annotated labels are saved.",
    )
    parser.add_argument(
        "--image-base-url",
        type=str,
        default=None,
        help="Base URL for images. Example: https://example.org/dataset/food101/",
    )
    parser.add_argument(
        "--fruit-veg",
        action="store_true",
        help="Append raw to USDA queries for fruit and vegetable datasets.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="LLM validator model name. Defaults to Falcon3.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="USDA API request timeout in seconds.",
    )
    parser.add_argument(
        "--search-page-size",
        type=int,
        default=DEFAULT_SEARCH_PAGE_SIZE,
        help="Number of USDA candidates inspected per data type.",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_arguments()

    if args.search_page_size <= 0:
        raise ValueError("--search-page-size must be greater than 0")

    labels = sorted(os.listdir(args.food_labels))
    asker = AskFoodSearch(
        model_name=args.model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    service = USDAFoodEnrichmentService(
        asker=asker,
        api_key=os.getenv("USDA_KEY"),
        base_url=os.getenv("USDA_BASE_URL", DEFAULT_USDA_BASE_URL),
        base_image_url=os.getenv("BASE_IMAGE_URL"),
        timeout=args.timeout,
        search_page_size=args.search_page_size,
    )

    service.process_dataset(
        labels=labels,
        output_file=args.output_file,
        not_annotated_file=args.not_annotated_file,
        is_fruit_veg=args.fruit_veg,
        image_base_url=args.image_base_url,
    )

    print("\nEnrichment pipeline completed successfully.")


if __name__ == "__main__":
    main()
