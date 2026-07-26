import json
import os
import re
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple

from difflib import SequenceMatcher
from dotenv import load_dotenv
from orkg import ORKG


# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

if not EMAIL or not PASSWORD:
    raise EnvironmentError("Missing ORKG credentials")


# ---------------------------------------------------------------------
# ORKG Client Wrapper
# ---------------------------------------------------------------------

class ORKGClient:
    """
    High-level ORKG API client with safety mechanisms.
    """

    def __init__(self, host: str, email: str, password: str) -> None:

        self.client = ORKG(
            host=host,
            creds=(email, password)
        )

    # -------------------------------------------------------------
    # Resource Handling
    # -------------------------------------------------------------

    def find_resource(
        self,
        label: str,
        exact: bool = True,
        classes: Optional[str] = None
    ) -> Optional[str]:

        response = self.client.resources.get_unpaginated(
            q=label,
            exact=exact,
            size=30,
            include=classes or ""
        )

        if not response.succeeded:
            return None

        if not response.content:
            return None

        return response.content[0]["id"]

    def create_resource(
        self,
        label: str,
        classes: Optional[List[str]] = None
    ) -> Optional[str]:

        response = self.client.resources.add(
            label=label,
            classes=classes or []
        )

        if response.succeeded:
            return response.content["id"]

        return None

    def get_or_create_resource(
        self,
        label: str,
        classes: Optional[List[str]] = None
    ) -> Optional[str]:

        rid = self.find_resource(label, classes=classes[0] if classes else None)

        if rid:
            return rid

        return self.create_resource(label, classes)

    # -------------------------------------------------------------
    # Predicate Handling
    # -------------------------------------------------------------

    def get_or_create_predicate(self, label: str) -> Optional[str]:

        response = self.client.predicates.get_unpaginated(
            q=label,
            exact=True,
            size=10
        )

        if response.succeeded and response.content:
            return response.content[0]["id"]

        created = self.client.predicates.add(label=label)

        if created.succeeded:
            return created.content["id"]

        return None

    # -------------------------------------------------------------
    # Literal Handling
    # -------------------------------------------------------------

    def create_literal(self, value: str, datatype: str) -> Optional[str]:

        response = self.client.literals.add(
            label=value,
            datatype=datatype
        )

        if response.succeeded:
            return response.content["id"]

        return None

    # -------------------------------------------------------------
    # Statement Handling
    # -------------------------------------------------------------

    def statement_exists(
        self,
        subject_id: str,
        predicate_id: str,
        object_id: str
    ) -> bool:

        response = self.client.statements.get_by_subject(
            subject_id=subject_id,
            size=200
        )

        if not response.succeeded:
            return False

        for stmt in response.content:
            if (
                stmt["predicate"]["id"] == predicate_id and
                stmt["object"]["id"] == object_id
            ):
                return True

        return False

    def safe_add_statement(
        self,
        subject_id: str,
        predicate_id: str,
        object_id: str
    ) -> Optional[str]:

        if self.statement_exists(subject_id, predicate_id, object_id):
            return None

        response = self.client.statements.add(
            subject_id=subject_id,
            predicate_id=predicate_id,
            object_id=object_id
        )

        if response.succeeded:
            return response.content["id"]

        return None


# ---------------------------------------------------------------------
# USDA  --> ORKG Importer
# ---------------------------------------------------------------------

class USDAORKGImporter:
    """
    Imports USDA enriched dataset into ORKG.
    """

    # ORKG CLASS IDS (Centralized)
    FOOD_CLASSES = ["C123036", "C77009"]
    COMPONENT_CLASS = ["C34009"]
    DATASET_CLASS = ["C14025"]

    # PROPERTY IDS
    PROP_HAS_COMPONENT = "P62073"
    PROP_COMPONENT_NAME = "P62093"
    PROP_COMPONENT_VALUE = "P5086"

    PROP_NAME = "P183114"
    PROP_SOURCE = "P183112"
    PROP_IMAGE = "P142026"
    PROP_PORTION = "P20080"

    DATASET_RESOURCE = "wikidata:Q7866384"

    def __init__(self, orkg: ORKGClient) -> None:

        self.orkg = orkg
        self.dataset: List[Dict] = []

    # -------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------

    @staticmethod
    def normalize(label: str) -> str:

        label = label.lower().strip()
        label = label.replace("&", "and")

        label = re.sub(r"[ \-_/']", "_", label)
        label = re.sub(r"[^a-z0-9_]", "", label)
        label = re.sub(r"_+", "_", label)

        return label.strip("_")

    @staticmethod
    def similarity(a: str, b: str) -> float:

        return SequenceMatcher(None, a, b).ratio()

    # -------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------

    def load_dataset(self, path: str) -> List[Dict]:

        with open(path, "r", encoding="utf-8") as file:
            self.dataset = json.load(file)

        return self.dataset

    # -------------------------------------------------------------
    # Core Import Logic
    # -------------------------------------------------------------

    def import_dataset(
        self,
        dataset_name: str,
        start: int = 0,
        end: Optional[int] = None
    ) -> None:

        subset = self.dataset[start:end]

        print(f"Importing {len(subset)} food items...")

        for index, item in enumerate(subset, start=1):

            label = self.normalize(item["food_class"])

            resource_label = f"{label}_{dataset_name.lower()}"

            print(f"\n[{index}] {resource_label}")

            food_id = self.orkg.get_or_create_resource(
                resource_label,
                self.FOOD_CLASSES
            )

            if not food_id:
                continue

            self._add_food_metadata(food_id, item)

            self._add_nutrients(food_id, resource_label, item)

            self._link_to_dataset(resource_label)

            time.sleep(0.5)

        print("\nImport completed.")

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    def _add_food_metadata(self, food_id: str, item: Dict) -> None:

        self._add_literal(
            food_id,
            self.PROP_NAME,
            item["usda_name"]
        )

        if item.get("portion"):
            self._add_literal(
                food_id,
                self.PROP_PORTION,
                item["portion"]
            )

        if item.get("usda_source"):
            self._add_literal(
                food_id,
                self.PROP_SOURCE,
                item["usda_source"],
                "xsd:uri"
            )

        if item.get("image"):
            self._add_literal(
                food_id,
                self.PROP_IMAGE,
                item["image"],
                "xsd:uri"
            )

    def _add_literal(
        self,
        subject_id: str,
        predicate_id: str,
        value: str,
        datatype: str = "xsd:string"
    ) -> None:

        literal_id = self.orkg.create_literal(value, datatype)

        if not literal_id:
            return

        self.orkg.safe_add_statement(
            subject_id,
            predicate_id,
            literal_id
        )

    # -------------------------------------------------------------
    # Nutrients
    # -------------------------------------------------------------

    def _add_nutrients(
        self,
        food_id: str,
        parent_label: str,
        item: Dict
    ) -> None:

        if not item.get("nutrients"):
            return

        for nutrient in item["nutrients"]:

            name = nutrient["name"]
            value = nutrient["value"] or 0
            unit = nutrient["unit"]

            full_value = f"{value} {unit}"

            comp_label = (
                f"{self.normalize(name)}_{parent_label}"
            )

            comp_id = self.orkg.get_or_create_resource(
                comp_label,
                self.COMPONENT_CLASS
            )

            if not comp_id:
                continue

            # Link to food
            self.orkg.safe_add_statement(
                food_id,
                self.PROP_HAS_COMPONENT,
                comp_id
            )

            # Name
            self._add_literal(
                comp_id,
                self.PROP_COMPONENT_NAME,
                name
            )

            # Value
            self._add_literal(
                comp_id,
                self.PROP_COMPONENT_VALUE,
                full_value
            )

    # -------------------------------------------------------------
    # Dataset Linking
    # -------------------------------------------------------------

    def _link_to_dataset(self, food_label: str) -> None:

        food_id = self.orkg.find_resource(food_label)

        if not food_id:
            return

        dataset_id = self.orkg.get_or_create_resource(
            self.DATASET_RESOURCE,
            self.DATASET_CLASS
        )

        if not dataset_id:
            return

        pred = self.orkg.get_or_create_predicate(
            "has contribution"
        )

        if not pred:
            return

        self.orkg.safe_add_statement(
            dataset_id,
            pred,
            food_id
        )


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a USDA-enriched dataset into ORKG."
    )
    parser.add_argument(
        "--host",
        default="https://orkg.org",
        help="ORKG host URL.",
    )
    parser.add_argument(
        "--input-file",
        default="json/old/uecfood256_usda_enriched.json",
        help="Path to the USDA-enriched JSON file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="uecfood256",
        help="Dataset name suffix used for generated ORKG resources.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=75,
        help="Start index within the loaded dataset.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=90,
        help="End index within the loaded dataset.",
    )
    args = parser.parse_args()

    orkg_client = ORKGClient(
        host=args.host,
        email=EMAIL,
        password=PASSWORD
    )

    importer = USDAORKGImporter(orkg_client)

    importer.load_dataset(args.input_file)

    importer.import_dataset(
        dataset_name=args.dataset_name,
        start=args.start,
        end=args.end
    )


if __name__ == "__main__":
    main()
