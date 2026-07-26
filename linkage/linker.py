"""Link food images to KG-style semantic descriptions.

Paper reference:
- "An Approach for Image Annotation via Knowledge Graph Linkage"
- The code below materializes the paper's image-to-KG linkage by converting
  nutrient features into graph entities and attaching each image to its
  semantic description.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class ImageTextLinker:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    FOOD_NS = "http://example.org/food#"
    RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
    RDFS_CLASS = "http://www.w3.org/2000/01/rdf-schema#Class"
    RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    XSD_FLOAT = "http://www.w3.org/2001/XMLSchema#float"
    XSD_URI = "http://www.w3.org/2001/XMLSchema#anyURI"
    HAS_IMAGE = f"{FOOD_NS}hasImage"
    HAS_COMPONENT = f"{FOOD_NS}hasComponent"
    HAS_TEXT_DESCRIPTION = f"{FOOD_NS}hasTextDescription"
    HAS_INGREDIENT = f"{FOOD_NS}hasIngredient"
    HAS_ORKG_LINK = f"{FOOD_NS}hasORKGLink"
    HAS_USDA_LINK = f"{FOOD_NS}hasUSDALink"
    FOOD_COMPONENT_NAME = f"{FOOD_NS}foodComponentName"

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

    def build_component_records(self, item: Dict, class_name: str, dataset_name: str) -> List[Dict]:
        # Paper Eq. (1), semantic mapping Ψ: nutrient features are converted
        # into explicit component entities before they are linked to an image.
        components = []
        source_name = item.get("food_name") or item.get("food_class") or class_name
        for nutrient in item.get("components", []):
            name = nutrient.get("name")
            value = nutrient.get("value")
            unit = nutrient.get("unit")
            if name and value is not None:
                component_label = self.clean_label(str(name))
                component_id = (
                    f"{self.FOOD_NS}{class_name}_{dataset_name}"
                    f"_Comp_{component_label}"
                )
                components.append(
                    {
                        "@id": component_id,
                        "@type": [f"{self.FOOD_NS}{class_name}_{dataset_name}_Components"],
                        self.FOOD_COMPONENT_NAME: [{"@value": str(name)}],
                        f"{self.FOOD_NS}hasUnit": [{"@value": str(unit)}],
                        f"{self.FOOD_NS}hasValue": [
                            {
                                "@type": self.XSD_FLOAT,
                                "@value": str(float(value)),
                            }
                        ],
                        self.RDFS_LABEL: [{"@value": f"{component_label}_{self.clean_label(str(source_name))}"}],
                    }
                )
        return components

    def build_component_strings(self, component_records: List[Dict]) -> List[str]:
        component_strings = []
        for component in component_records:
            label_values = component.get(self.FOOD_COMPONENT_NAME, component.get(self.RDFS_LABEL, []))
            unit_values = component.get(f"{self.FOOD_NS}hasUnit", [])
            value_values = component.get(f"{self.FOOD_NS}hasValue", [])

            name = label_values[0].get("@value") if label_values else None
            unit = unit_values[0].get("@value") if unit_values else None
            value = value_values[0].get("@value") if value_values else None

            if name is not None and value is not None:
                component_strings.append(f"{name}: {value} {unit}".strip())

        return component_strings

    def build_text_description(self, item: Dict, component_strings: List[str]) -> str:
        # Paper motivation: the image is linked to richer semantic context than
        # a coarse label, so we preserve name, ingredients, and nutrients here.
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
        class_name: str,
        image_path: Path,
        image_index: int,
        component_records: List[Dict],
        text_description: str,
        dataset_name: str,
        ingredients: List[str],
        orkg_link: Optional[str],
        usda_link: Optional[str],
    ) -> Dict:
        # Paper Eq. (1): this is the concrete multimodal record linking an
        # image instance in I to the semantic entities extracted for that food.
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

        if ingredients:
            record[self.HAS_INGREDIENT] = [
                {"@value": str(ingredient).strip()}
                for ingredient in ingredients
            ]

        if orkg_link:
            record[self.HAS_ORKG_LINK] = [
                {"@type": self.XSD_URI, "@value": str(orkg_link)}
            ]

        if usda_link:
            record[self.HAS_USDA_LINK] = [
                {"@type": self.XSD_URI, "@value": str(usda_link)}
            ]

        return record

    def build_class_records(
        self,
        class_name: str,
        dataset_name: str,
        has_ingredients: bool,
    ) -> List[Dict]:
        records = [
            {"@id": f"{self.FOOD_NS}{dataset_name}", "@type": [self.RDFS_CLASS]},
            {
                "@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Food",
                "@type": [self.RDFS_CLASS],
                self.RDFS_LABEL: [{"@value": class_name}],
                self.RDFS_SUBCLASS_OF: [{"@id": f"{self.FOOD_NS}{dataset_name}"}],
            },
            {
                "@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Images",
                "@type": [self.RDFS_CLASS],
                self.RDFS_LABEL: [{"@value": "Images"}],
                self.RDFS_SUBCLASS_OF: [{"@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Food"}],
            },
            {
                "@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Components",
                "@type": [self.RDFS_CLASS],
                self.RDFS_LABEL: [{"@value": "Components"}],
                self.RDFS_SUBCLASS_OF: [{"@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Food"}],
            },
        ]
        if has_ingredients:
            records.append(
                {
                    "@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Ingredients",
                    "@type": [self.RDFS_CLASS],
                    self.RDFS_LABEL: [{"@value": "Ingredients"}],
                    self.RDFS_SUBCLASS_OF: [{"@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Food"}],
                }
            )
        return records

    def build_ingredient_records(
        self,
        class_name: str,
        dataset_name: str,
        ingredients: List[str],
    ) -> List[Dict]:
        records = []
        for ingredient in ingredients:
            ingredient_slug = self.clean_label(ingredient)
            records.append(
                {
                    "@id": f"{self.FOOD_NS}{class_name}_{dataset_name}_Ing_{ingredient_slug}",
                    "@type": [f"{self.FOOD_NS}{class_name}_{dataset_name}_Ingredients"],
                    self.RDFS_LABEL: [{"@value": ingredient}],
                }
            )
        return records

    def build_food_record(
        self,
        item: Dict,
        class_name: str,
        dataset_name: str,
        image_records: List[Dict],
        component_records: List[Dict],
        ingredient_records: List[Dict],
        text_description: str,
    ) -> Dict:
        record = {
            "@id": f"{self.FOOD_NS}{class_name}_{dataset_name}",
            "@type": [f"{self.FOOD_NS}{class_name}_{dataset_name}_Food"],
            self.RDFS_LABEL: [{"@value": class_name}],
            self.HAS_IMAGE: [{"@id": image_record["@id"]} for image_record in image_records],
            self.HAS_COMPONENT: [{"@id": component_record["@id"]} for component_record in component_records],
            self.HAS_TEXT_DESCRIPTION: [{"@value": text_description}],
        }

        if ingredient_records:
            record[self.HAS_INGREDIENT] = [{"@id": ingredient_record["@id"]} for ingredient_record in ingredient_records]

        if item.get("id"):
            record[self.HAS_ORKG_LINK] = [{"@type": self.XSD_URI, "@value": str(item["id"])}]

        if item.get("usda_link"):
            record[self.HAS_USDA_LINK] = [{"@type": self.XSD_URI, "@value": str(item["usda_link"])}]

        return record

    def iter_linked_records(self, json_source: str, image_dir: str) -> Iterable[Dict]:
        # Paper pipeline: iterate jointly over semantic food metadata and image
        # folders, then emit ontology classes plus image-linked instances.
        source_records = self.load_source_records(json_source)
        image_root = Path(image_dir)
        dataset_name = self.clean_label(image_root.parent.name or "dataset")
        emitted_dataset_classes = set()

        for item in source_records:
            raw_class = item.get("food_class", "")
            class_name = self.clean_label(raw_class)
            if not class_name:
                continue

            class_dir = image_root / class_name
            if not class_dir.is_dir():
                print(f"Skipping missing class directory: {class_dir}")
                continue

            component_records = self.build_component_records(item, class_name, dataset_name)
            component_strings = self.build_component_strings(component_records)
            text_description = self.build_text_description(item, component_strings)
            ingredients = [
                str(ingredient).strip()
                for ingredient in item.get("ingredients", [])
                if str(ingredient).strip()
            ]

            if dataset_name not in emitted_dataset_classes:
                yield {"@id": f"{self.FOOD_NS}{dataset_name}", "@type": [self.RDFS_CLASS]}
                emitted_dataset_classes.add(dataset_name)

            for class_record in self.build_class_records(
                class_name=class_name,
                dataset_name=dataset_name,
                has_ingredients=bool(ingredients),
            )[1:]:
                yield class_record

            image_records = []

            image_index = 0
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue
                image_index += 1
                image_record = self.build_image_record(
                    class_name=class_name,
                    image_path=image_path,
                    image_index=image_index,
                    component_records=component_records,
                    text_description=text_description,
                    dataset_name=dataset_name,
                    ingredients=ingredients,
                    orkg_link=item.get("id"),
                    usda_link=item.get("usda_link"),
                )
                image_records.append(image_record)
                yield image_record

            ingredient_records = self.build_ingredient_records(
                class_name=class_name,
                dataset_name=dataset_name,
                ingredients=ingredients,
            )
            for ingredient_record in ingredient_records:
                yield ingredient_record

            for component_record in component_records:
                component_record["@type"] = [f"{self.FOOD_NS}{class_name}_{dataset_name}_Components"]
                yield component_record

            yield self.build_food_record(
                item=item,
                class_name=class_name,
                dataset_name=dataset_name,
                image_records=image_records,
                component_records=component_records,
                ingredient_records=ingredient_records,
                text_description=text_description,
            )

    def build_jsonld(self, json_source: str, image_dir: str, output_file: str) -> int:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = list(self.iter_linked_records(json_source, image_dir))
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

        image_record_count = sum(
            1
            for record in records
            if isinstance(record, dict)
            and self.HAS_IMAGE in record
            and any(str(record_type).endswith("_Images") for record_type in record.get("@type", []))
        )
        return image_record_count
