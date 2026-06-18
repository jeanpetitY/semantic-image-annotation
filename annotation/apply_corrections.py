import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


DEFAULT_USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEFAULT_TIMEOUT = 30
NUTRIENT_GROUP_NAMES = {
    "Proximates",
    "Carbohydrates",
    "Minerals",
    "Vitamins and Other Components",
    "Vitamins",
    "Lipids",
    "Amino acids",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        file.write("\n")


def extract_ingredients(data: Dict) -> List[str]:
    if data.get("ingredients"):
        return [
            item.strip()
            for item in data["ingredients"].split(",")
            if item.strip()
        ]

    return [
        food.get("ingredientDescription", "").strip()
        for food in data.get("inputFoods", [])
        if food.get("ingredientDescription", "").strip()
    ]


def extract_nutrients(data: Dict) -> List[Dict]:
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

    return nutrients


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


def fetch_usda_details(
    fdc_id: int,
    api_key: str,
    base_url: str,
    timeout: int,
) -> Dict:
    response = requests.get(
        f"{base_url}/food/{fdc_id}",
        params={"api_key": api_key},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    details = {
        "portion": "100 g",
        "usda_name": data.get("description"),
        "description": data.get("description"),
        "ingredients": extract_ingredients(data),
        "nutrients": extract_nutrients(data),
        "usda_source": (
            f"https://fdc.nal.usda.gov/food-details/{fdc_id}/nutrients"
        ),
    }

    if data.get("dataType"):
        details["usda_data_type"] = data["dataType"]

    food_category = extract_food_category(data)
    if food_category:
        details["usda_food_category"] = food_category

    return details


def apply_correction(
    annotation: Dict,
    correction: Dict,
    fetched_details: Optional[Dict],
) -> Dict:
    updated = dict(annotation)

    if fetched_details:
        updated.update(fetched_details)
    else:
        corrected_fdc_id = int(correction["corrected_fdc_id"])
        updated["usda_name"] = correction["corrected_usda_name"]
        updated["description"] = correction["corrected_usda_name"]
        updated["usda_source"] = (
            f"https://fdc.nal.usda.gov/food-details/"
            f"{corrected_fdc_id}/nutrients"
        )

    return updated


def apply_corrections(
    annotations: List[Dict],
    corrections: List[Dict],
    fetch_usda_details_enabled: bool,
    api_key: Optional[str],
    base_url: str,
    timeout: int,
) -> List[Dict]:
    corrections_by_class = {
        item["food_class"]: item
        for item in corrections
    }
    corrected = []

    if fetch_usda_details_enabled and not api_key:
        raise EnvironmentError(
            "USDA_KEY is required when --fetch-usda-details is enabled."
        )

    details_by_id = {}

    for annotation in annotations:
        food_class = annotation["food_class"]
        correction = corrections_by_class.get(food_class)

        if not correction:
            corrected.append(annotation)
            continue

        fetched_details = None

        if fetch_usda_details_enabled:
            corrected_fdc_id = int(correction["corrected_fdc_id"])

            if corrected_fdc_id not in details_by_id:
                details_by_id[corrected_fdc_id] = fetch_usda_details(
                    fdc_id=corrected_fdc_id,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )

            fetched_details = details_by_id[corrected_fdc_id]

        corrected.append(
            apply_correction(
                annotation=annotation,
                correction=correction,
                fetched_details=fetched_details,
            )
        )

    return corrected


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply human-in-the-loop USDA annotation corrections."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to raw annotation JSON."
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        required=True,
        help="Path to human correction JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to corrected annotation JSON."
    )
    parser.add_argument(
        "--fetch-usda-details",
        action="store_true",
        help="Fetch corrected USDA nutrients and ingredients by fdc_id."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="USDA request timeout in seconds."
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_arguments()

    annotations = load_json(args.annotations)
    corrections = load_json(args.corrections)

    corrected = apply_corrections(
        annotations=annotations,
        corrections=corrections,
        fetch_usda_details_enabled=args.fetch_usda_details,
        api_key=os.getenv("USDA_KEY"),
        base_url=os.getenv("USDA_BASE_URL", DEFAULT_USDA_BASE_URL),
        timeout=args.timeout,
    )

    save_json(corrected, args.output)
    print(
        f"Applied {len(corrections)} human corrections. "
        f"Saved corrected annotations to {args.output}."
    )


if __name__ == "__main__":
    main()
