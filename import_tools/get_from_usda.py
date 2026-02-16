"""
USDA Food Data Enrichment Service

This module provides a complete pipeline for enriching
food classification datasets with nutritional information
retrieved from the USDA FoodData Central API.

The system integrates:
- Local JSON caching
- LLM-based semantic validation
- Automatic dataset enrichment
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from huggingface_hub import login

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from import_tools.utils import (
    uecfood256_labels,
    food101_labels,
    afd_labels,
    fruitveg81_labels,
)

from ask.chat import AskFoodSearch


# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------

load_dotenv()

USDA_API_KEY = os.getenv("USDA_KEY")
USDA_BASE_URL = os.getenv("USDA_BASE_URL", "https://api.nal.usda.gov/fdc/v1")
BASE_IMAGE_URL = os.getenv("BASE_IMAGE_URL")


if USDA_API_KEY is None:
    raise EnvironmentError("USDA_KEY is missing in environment variables.")


# ---------------------------------------------------------------------
# Main Service Class
# ---------------------------------------------------------------------

class USDAFoodEnrichmentService:
    """
    Service for enriching food labels with USDA nutritional data.

    Features:
    - Intelligent search with LLM validation
    - Local JSON cache
    - Automatic progress persistence
    - Image URL generation
    """

    def __init__(
        self,
        asker: AskFoodSearch,
        cache_file: str = "usda_cache.json"
    ) -> None:
        """
        Initialize the USDA enrichment service.

        Args:
            asker (AskFoodSearch): LLM validation client.
            cache_file (str): Path to local cache file.
        """

        self.asker = asker
        self.api_key = USDA_API_KEY
        self.base_url = USDA_BASE_URL

        self.cache_file = cache_file
        self.cache: Dict[str, dict] = self._load_cache()

        self.search_calls = 0
        self.detail_calls = 0

    # -----------------------------------------------------------------
    # Label Processing
    # -----------------------------------------------------------------

    @staticmethod
    def normalize_label(label: str) -> str:
        """
        Normalize a label for filenames and identifiers.

        Args:
            label (str): Raw food label.

        Returns:
            str: Normalized label.
        """

        label = label.lower().strip()
        label = label.replace("&", "and")

        label = re.sub(r"[ \-_/']", "_", label)
        label = re.sub(r"[^a-z0-9_]", "", label)
        label = re.sub(r"_+", "_", label)

        return label.strip("_")

    @staticmethod
    def format_query(label: str) -> str:
        """
        Convert label to USDA search format.

        Args:
            label (str): Dataset label.

        Returns:
            str: Formatted query.
        """

        return label.replace("_", " ").title()

    # -----------------------------------------------------------------
    # Image Handling
    # -----------------------------------------------------------------

    def build_image_url(
        self,
        label: str,
        base_url: Optional[str] = None
    ) -> str:
        """
        Build standardized image URL.

        Args:
            label (str): Food label.
            base_url (str): Image base path.

        Returns:
            str: Full image URL.
        """

        if base_url is None:
            base_url = f"{BASE_IMAGE_URL}/fruitveg81/"

        normalized = self.normalize_label(label)

        return f"{base_url}{normalized}.jpg"

    # -----------------------------------------------------------------
    # Cache Management
    # -----------------------------------------------------------------

    def _load_cache(self) -> Dict[str, dict]:
        """
        Load local cache.

        Returns:
            Dict[str, dict]: Cache dictionary.
        """

        if not os.path.exists(self.cache_file):
            return {}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    def _save_cache(self) -> None:
        """
        Persist cache to disk.
        """

        with open(self.cache_file, "w", encoding="utf-8") as file:
            json.dump(
                self.cache,
                file,
                indent=2,
                ensure_ascii=False
            )

    # -----------------------------------------------------------------
    # USDA API Interaction
    # -----------------------------------------------------------------

    def search_food(
        self,
        query: str,
        is_fruit_veg: bool = False
    ) -> Optional[int]:
        """
        Search food using intelligent validation.

        Args:
            query (str): Search query.
            is_fruit_veg (bool): Enable raw food mode.

        Returns:
            Optional[int]: FDC ID if found.
        """

        if is_fruit_veg:
            query = f"{query}, raw"

        if query in self.cache:
            return self.cache[query].get("fdc_id")

        endpoint = f"{self.base_url}/foods/search"

        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": 1,
        }

        response = requests.get(endpoint, params=params)

        self.search_calls += 1

        if response.status_code != 200:
            self.cache[query] = {"fdc_id": None}
            self._save_cache()
            return None

        foods = response.json().get("foods", [])

        if not foods:
            self.cache[query] = {"fdc_id": None}
            self._save_cache()
            return None

        top_food = foods[0]
        fdc_id = top_food["fdcId"]
        description = top_food.get("description")

        if is_fruit_veg:
            self.cache[query] = {
                "fdc_id": fdc_id,
                "name": description
            }
            self._save_cache()

            return fdc_id

        verdict = self.asker.use_gpt4(query, description)

        if verdict.lower() == "yes":

            self.cache[query] = {
                "fdc_id": fdc_id,
                "name": description
            }

            self._save_cache()

            return fdc_id

        self.cache[query] = {"fdc_id": None}
        self._save_cache()

        return None

    def get_food_details(self, fdc_id: int) -> Optional[Dict]:
        """
        Retrieve food nutritional details.

        Args:
            fdc_id (int): FoodData Central ID.

        Returns:
            Optional[Dict]: Food details.
        """

        endpoint = f"{self.base_url}/food/{fdc_id}"

        params = {"api_key": self.api_key}

        response = requests.get(endpoint, params=params)

        self.detail_calls += 1

        if response.status_code != 200:
            return None

        data = response.json()

        nutrients = [
            {
                "name": n["nutrient"].get("name"),
                "value": n.get("amount"),
                "unit": n["nutrient"].get("unitName"),
            }
            for n in data.get("foodNutrients", [])
            if "nutrient" in n
        ]

        if data.get("ingredients"):

            ingredients = [
                item.strip()
                for item in data["ingredients"].split(",")
            ]

        else:

            ingredients = [
                item.strip()
                for f in data.get("inputFoods", [])
                for item in f.get(
                    "ingredientDescription", ""
                ).split(",")
                if item.strip()
            ]

        return {
            "fdc_id": fdc_id,
            "portion": "100 g",
            "description": data.get("description"),
            "ingredients": ingredients,
            "nutrients": nutrients,
            "source_url": (
                f"https://fdc.nal.usda.gov/food-details/"
                f"{fdc_id}/nutrients"
            ),
        }

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    @staticmethod
    def load_output(file_path: str) -> List[Dict]:
        """
        Load existing output file.

        Args:
            file_path (str): JSON path.

        Returns:
            List[Dict]: Loaded data.
        """

        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def save_output(data: List[Dict], file_path: str) -> None:
        """
        Save enriched dataset.

        Args:
            data (List[Dict]): Dataset.
            file_path (str): Output path.
        """

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

    # -----------------------------------------------------------------
    # Data Construction
    # -----------------------------------------------------------------

    def build_food_entry(
        self,
        label: str,
        details: Dict,
        image_base_url: Optional[str]
    ) -> Dict:
        """
        Build standardized food entry.

        Args:
            label (str): Dataset label.
            details (Dict): USDA details.
            image_base_url (str): Image base URL.

        Returns:
            Dict: Food entry.
        """

        image_url = self.build_image_url(
            label,
            image_base_url
        )

        return {
            "food_class": label,
            "portion": details["portion"],
            "usda_name": details["description"],
            "description": details["description"],
            "ingredients": details["ingredients"],
            "nutrients": details["nutrients"],
            "usda_source": details["source_url"],
            "image": image_url,
        }

    # -----------------------------------------------------------------
    # Retrieval Pipeline
    # -----------------------------------------------------------------

    def retrieve_food(
        self,
        label: str,
        is_fruit_veg: bool
    ) -> Optional[Dict]:
        """
        Retrieve full food information.

        Args:
            label (str): Food label.
            is_fruit_veg (bool): Raw food mode.

        Returns:
            Optional[Dict]: Food details.
        """

        query = self.format_query(label)

        fdc_id = self.search_food(query, is_fruit_veg)

        if not fdc_id:
            return None

        return self.get_food_details(fdc_id)

    def process_dataset(
        self,
        labels: List,
        output_file: str,
        is_fruit_veg: bool = False,
        image_base_url: Optional[str] = None,
    ) -> None:
        """
        Process a dataset and enrich labels.

        Args:
            labels (List): Dataset labels.
            output_file (str): Output JSON file.
            is_fruit_veg (bool): Raw food mode.
            image_base_url (str): Image base path.
        """

        enriched = self.load_output(output_file)

        processed = {
            item["food_class"]
            for item in enriched
        }

        total = len(labels)

        print(f"Processing {total} food items...")

        for index, label in enumerate(labels, start=1):

            if label in processed:
                continue

            query = self.format_query(label)

            print(f"[{index}/{total}] {query}")

            details = self.retrieve_food(
                label,
                is_fruit_veg
            )

            if details is None:
                continue

            entry = self.build_food_entry(
                label,
                details,
                image_base_url
            )

            enriched.append(entry)

            self.save_output(enriched, output_file)

            time.sleep(0.5)

        print(f"Saved {len(enriched)} entries to {output_file}")


# ---------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Execute dataset enrichment.
    """

    asker = AskFoodSearch()

    service = USDAFoodEnrichmentService(asker)

    output_uec = "json/old/uecfood256_usda_enriched.json"

    service.process_dataset(
        labels=uecfood256_labels,
        output_file=output_uec,
        is_fruit_veg=False,
        image_base_url=f"{BASE_IMAGE_URL}/uecfood256/",
    )


if __name__ == "__main__":
    main()
