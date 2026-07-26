"""Export food resources and component statements from ORKG.

Paper reference:
- The semantically structured description layer stored in the Open Research
  Knowledge Graph before linkage to the image side of the dataset.
"""

import argparse
import json
import os
import hashlib

from dotenv import load_dotenv
from orkg import ORKG

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

ORKG_HOST = os.getenv("ORKG_HOST", "https://example.org/kg")

# Graph class identifiers are injected through environment variables so no
# deployment-specific IDs are hard-coded in the public code.
FOOD_CLASS = os.getenv("ORKG_FOOD_CLASS_ID", "CLASS_FOOD_PLACEHOLDER")
COMPONENT_CLASS = os.getenv("ORKG_COMPONENT_CLASS_ID", "CLASS_COMPONENT_PLACEHOLDER")

# Output file
OUTPUT_FILE = "exported_foods.json"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export food resources from ORKG."
    )
    parser.add_argument(
        "--mode",
        choices=["entity-ids", "resource-ids", "class-scan"],
        default="entity-ids",
        help="Export from explicit graph entity IDs or by scanning the configured food class.",
    )
    parser.add_argument(
        "--output-file",
        default="exported_foods.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--entity-ids",
        "--resource-ids",
        nargs="+",
        default=[],
        help="Graph entity IDs used in explicit-ID export mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of exported foods.",
    )
    parser.add_argument(
        "--host",
        default=ORKG_HOST,
        help="Knowledge-graph host URL.",
    )
    return parser.parse_args()


orkg = None



def extract_literal_value(value):
    """
    Extract a readable value from an ORKG object.

    If the value is a dictionary, return its label or ID.
    Otherwise, return the value as a string.
    """
    if isinstance(value, dict):
        return value.get("label") or value.get("id") or ""
    return str(value)


def parse_component_statements(component_id):
    """Retrieve and parse statements describing a food component.

    Paper KG-construction step: component entities provide the nutrient side of
    the multimodal representation later linked to images.
    """
    statements = orkg.statements.get_by_subject(component_id)

    component = {
        "name": "",
        "value": None,
        "unit": None,
    }

    for statement in statements.content:
        predicate = statement["predicate"]["label"]
        obj = statement["object"]

        if predicate == "food component name":
            component["name"] = extract_literal_value(obj)

        elif predicate == "value":
            raw_value = extract_literal_value(obj)

            try:
                parts = raw_value.split()
                if len(parts) == 2:
                    component["value"] = float(parts[0])
                    component["unit"] = parts[1]
                else:
                    component["value"] = raw_value
            except ValueError:
                component["value"] = raw_value

    return component

def parse_component_statements_1(component_id):
    """
    Retrieve and parse a food component (nutrient) using ORKG statements.

    The 'value' predicate points to a Quantity Value resource,
    which contains 'numeric value' and 'unit'.
    """
    statements = orkg.statements.get_by_subject(component_id)

    component = {
        "name": "",
        "value": None,
        "unit": None,
    }

    value_resource_id = None

    # First hop: component-level statements
    for s in statements.content:
        predicate = s["predicate"]["label"]
        obj = s["object"]

        # Component name (e.g., iron)
        if predicate == "food component name":
            component["name"] = extract_literal_value(obj)

        elif predicate == "value":
            value_resource_id = obj.get("id")

    # Second hop: Quantity Value resource
    if value_resource_id:
        value_statements = orkg.statements.get_by_subject(value_resource_id)

        for s in value_statements.content:
            predicate = s["predicate"]["label"]
            obj = s["object"]

            if predicate == "numericValue":
                try:
                    component["value"] = float(extract_literal_value(obj))
                except ValueError:
                    component["value"] = extract_literal_value(obj)

            elif predicate == "unit":
                component["unit"] = extract_literal_value(obj)

    return component

def parse_ingredient_statements(ingredient_id):
    """
    Retrieve and parse an ingredient using ORKG statements.

    The ingredient name is extracted from the 'common name' property.
    If not found, the resource label is used as a fallback.

    Args:
        ingredient_id (str): Graph entity ID of the ingredient.

    Returns:
        str: Ingredient label.
    """
    statements = orkg.statements.get_by_subject(ingredient_id)

    ingredient_name = None

    for s in statements.content:
        predicate = s["predicate"]["label"]
        if predicate == "common name":
            obj = s["object"]

            ingredient_name = extract_literal_value(obj)
            break

    # Fallback: use resource label if common name not found
    if not ingredient_name:
        try:
            resource = orkg.resources.get(ingredient_id)
            ingredient_name = resource.get("label", "")
        except Exception:
            ingredient_name = ""

    return ingredient_name




def parse_food_resource(resource_id):
    """
    Reconstruct all information related to a single food resource.

    Args:
        resource_id (str): Graph entity ID of the food.

    Returns:
        dict: Structured food data.
    """
    statements = orkg.statements.get_by_subject(resource_id)

    food_data = {
        "source_record_id": f"record_{hashlib.sha1(resource_id.encode('utf-8')).hexdigest()[:12]}",
        "food": [],
        "components": [],
        "ingredients": [],
        "image": [],
        "areas": [],
        "also_known_as": [],
    }
    # print(statements.content)
    for statement in statements.content:
        predicate = statement["predicate"]["label"]
        obj = statement["object"]
        value = extract_literal_value(obj)

        if predicate in ["usda food name", "usda food class", "common name", "food name"]:
            food_data["food"].append(value)
            food_data["food_name"] = value

        elif predicate == "local name":
            food_data["also_known_as"].append(value)

        elif predicate == "food class":
            food_data["food_class"] = value

        elif predicate == "description":
            food_data["description"] = value

        elif predicate in ["usda food ingredient", "ingredient", "has ingredient", "food ingredient"]:
            if ingredient_id := obj.get("id"):
                ingredient_name = parse_ingredient_statements(ingredient_id)
                if ingredient_name and ingredient_name not in food_data["ingredients"]:
                    food_data["ingredients"].append(ingredient_name)

        elif predicate in ["usda link", "source", "link"]:
            food_data["usda_link"] = value

        elif predicate in ["food image", "image", "photo"]:
            food_data["image"].append(value)

        elif predicate in ["area", "origin", "region"]:
            food_data["areas"].append(value)

        elif predicate == "food component":
            if component_id := obj.get("id"):
                component = parse_component_statements_1(component_id)
                if component["name"]:
                    food_data["components"].append(component)

    return food_data

def fetch_foods_by_resource_ids(resource_ids, limit=None):
    """
    Fetch food resources from ORKG using an explicit list of entity IDs
    and export them in the same JSON format as class-based export.

    Args:
        resource_ids (List[str]): List of graph entity IDs.
        limit (int, optional): Maximum number of resources to export.

    Returns:
        None
    """
    count = 0
    results = []

    # Load existing output file if it exists (resume support)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            try:
                results = json.load(file)
            except json.JSONDecodeError:
                results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("[")

        for resource_id in resource_ids:
            try:
                food_data = parse_food_resource(resource_id)

                # Try to infer dataset source from resource label
                try:
                    resource = orkg.resources.get(resource_id)
                    label = resource.get("label", "")
                    if "_" in label:
                        dataset = label.split("_")[-1]
                        food_data["extract_from"] = f"{dataset} dataset"
                except Exception:
                    pass

                results.append(food_data)

                if count > 0:
                    file.write(",\n")

                file.write(json.dumps(food_data, ensure_ascii=False, indent=2))
                file.flush()

                count += 1
                print(f"Saved entity [{count}]")

                if limit and count >= limit:
                    break

            except Exception as error:
                print(f"Error processing one explicit entity: {error}")

        file.write("]")

    print(f"Export finished. {count} resources written to {OUTPUT_FILE}")



def fetch_and_stream_foods(limit=None):
    """
    Fetch USDA food resources from ORKG and write them incrementally to a JSON file.

    Args:
        limit (int, optional): Maximum number of foods to export. Defaults to None.
    """
    foods = orkg.resources.get_unpaginated(size=200, include=FOOD_CLASS)
    count = 0
    results = []

    # Load existing file if present (to resume export)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            try:
                results = json.load(file)
            except json.JSONDecodeError:
                results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("[")

        for resource in foods.content:
            if FOOD_CLASS not in resource.get("classes", []):
                continue

            resource_id = resource["id"]

            try:
                food_data = parse_food_resource(resource_id)

                label = resource.get("label", "")
                if "_" in label:
                    dataset = label.split("_")[-1]
                    food_data["extract_from"] = f"{dataset} dataset"

                results.append(food_data)

                if count > 0:
                    file.write(",\n")

                file.write(json.dumps(food_data, ensure_ascii=False, indent=2))
                file.flush()

                count += 1
                print(f"Saved {resource.get('label', 'unknown')} [{count}]")

                if limit and count >= limit:
                    break

            except Exception as error:
                print(f"Error processing {resource.get('label', 'unknown')}: {error}")

        file.write("]")

    print(f"Export finished. {count} food items written to {OUTPUT_FILE}")


def main():
    global orkg, OUTPUT_FILE

    args = parse_args()
    OUTPUT_FILE = args.output_file
    orkg = ORKG(host=args.host, creds=(EMAIL, PASSWORD))

    print("Fetching food data from ORKG...")
    if args.mode in {"entity-ids", "resource-ids"}:
        fetch_foods_by_resource_ids(args.entity_ids, limit=args.limit)
    else:
        fetch_and_stream_foods(limit=args.limit)


if __name__ == "__main__":
    main()
