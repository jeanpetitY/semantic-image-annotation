import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


class ImageTextLinker:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    FOOD_NS = "http://example.org/food#"
    RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
    XSD_FLOAT = "http://www.w3.org/2001/XMLSchema#float"
    XSD_URI = "http://www.w3.org/2001/XMLSchema#anyURI"
    HAS_IMAGE = f"{FOOD_NS}hasImage"
    HAS_COMPONENT = f"{FOOD_NS}hasComponent"
    HAS_TEXT_DESCRIPTION = f"{FOOD_NS}hasTextDescription"
    HAS_INGREDIENT = f"{FOOD_NS}hasIngredient"
    HAS_ORKG_LINK = f"{FOOD_NS}hasORKGLink"
    HAS_USDA_LINK = f"{FOOD_NS}hasUSDALink"

    @staticmethod
    def clean_label(name: str) -> str:
        name = name.lower().strip()
        name = name.replace("&", "and")
        name = re.sub(r"[ \-_/']", "_", name)
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    def load_source_records(self, json_source: str) -> List[Dict]:
        with open(json_source, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("The linkage source file must contain a list of records.")

        return data

    def build_component_records(self, item: Dict, class_name: str) -> List[Dict]:
        components = []
        source_name = item.get("food_name") or item.get("food_class") or class_name
        for nutrient in item.get("components", []):
            name = nutrient.get("name")
            value = nutrient.get("value")
            unit = nutrient.get("unit")
            if name and value is not None:
                component_label = self.clean_label(str(name))
                component_id = (
                    f"{self.FOOD_NS}{self.clean_label(str(source_name))}"
                    f"_Comp_{component_label}"
                )
                components.append(
                    {
                        "@id": component_id,
                        "@type": [f"{self.FOOD_NS}FoodComponent"],
                        self.RDFS_LABEL: [{"@value": str(name)}],
                        f"{self.FOOD_NS}hasUnit": [{"@value": str(unit)}],
                        f"{self.FOOD_NS}hasValue": [
                            {
                                "@type": self.XSD_FLOAT,
                                "@value": str(float(value)),
                            }
                        ],
                    }
                )
        return components

    def build_component_strings(self, component_records: List[Dict]) -> List[str]:
        component_strings = []
        for component in component_records:
            label_values = component.get(self.RDFS_LABEL, [])
            unit_values = component.get(f"{self.FOOD_NS}hasUnit", [])
            value_values = component.get(f"{self.FOOD_NS}hasValue", [])

            name = label_values[0].get("@value") if label_values else None
            unit = unit_values[0].get("@value") if unit_values else None
            value = value_values[0].get("@value") if value_values else None

            if name is not None and value is not None:
                component_strings.append(f"{name}: {value} {unit}".strip())

        return component_strings

    def build_text_description(self, item: Dict, component_strings: List[str]) -> str:
        food_name = item.get("food_name") or item.get("description") or item.get("food_class", "")
        description_parts = [str(food_name).strip().rstrip(".")]

        ingredients = item.get("ingredients", [])
        if isinstance(ingredients, list) and ingredients:
            description_parts.append(
                "Ingredients: " + ", ".join(str(ingredient).strip() for ingredient in ingredients if str(ingredient).strip())
            )

        if component_strings:
            description_parts.append("Nutritional components: " + "; ".join(component_strings))

        return ". ".join(part for part in description_parts if part).strip()

    def build_image_record(
        self,
        item: Dict,
        class_name: str,
        image_path: Path,
        image_index: int,
        component_records: List[Dict],
        text_description: str,
        dataset_name: str,
    ) -> Dict:
        record_id = f"{self.FOOD_NS}{class_name}_{dataset_name}_Img_{image_index}"
        record_type = f"{self.FOOD_NS}{class_name}_{dataset_name}_Images"

        record = {
            "@id": record_id,
            "@type": [record_type],
            self.RDFS_LABEL: [{"@value": f"Image_{image_index}"}],
            self.HAS_IMAGE: [{"@value": str(image_path)}],
            self.HAS_TEXT_DESCRIPTION: [{"@value": text_description}],
            self.HAS_COMPONENT: component_records,
        }

        ingredients = item.get("ingredients", [])
        if isinstance(ingredients, list) and ingredients:
            record[self.HAS_INGREDIENT] = [
                {"@value": str(ingredient).strip()}
                for ingredient in ingredients
                if str(ingredient).strip()
            ]

        if item.get("id"):
            record[self.HAS_ORKG_LINK] = [
                {"@type": self.XSD_URI, "@value": str(item["id"])}
            ]

        if item.get("usda_link"):
            record[self.HAS_USDA_LINK] = [
                {"@type": self.XSD_URI, "@value": str(item["usda_link"])}
            ]

        return record

    def iter_linked_records(self, json_source: str, image_dir: str) -> Iterable[Dict]:
        source_records = self.load_source_records(json_source)
        image_root = Path(image_dir)
        dataset_name = self.clean_label(image_root.parent.name or "dataset")

        for item in source_records:
            raw_class = item.get("food_class", "")
            class_name = self.clean_label(raw_class)
            if not class_name:
                continue

            class_dir = image_root / class_name
            if not class_dir.is_dir():
                print(f"Skipping missing class directory: {class_dir}")
                continue

            component_records = self.build_component_records(item, class_name)
            component_strings = self.build_component_strings(component_records)
            text_description = self.build_text_description(item, component_strings)

            image_index = 0
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue
                image_index += 1
                yield self.build_image_record(
                    item=item,
                    class_name=class_name,
                    image_path=image_path,
                    image_index=image_index,
                    component_records=component_records,
                    text_description=text_description,
                    dataset_name=dataset_name,
                )

    def build_jsonl(self, json_source: str, image_dir: str, output_file: str) -> int:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as file:
            for record in self.iter_linked_records(json_source, image_dir):
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return count
