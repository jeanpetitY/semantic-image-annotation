import json
from typing import Dict, List
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv # type: ignore
from load_model import load_client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Response(BaseModel):
    response: str
    

class AskFoodSearch:
    def __init__(self, model_name="tiiuae/Falcon3-7B-Instruct"):
        self.client =  OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
    
    def choose_message(self, prompt, is_gpt=True):
        return (
            [
                {
                    "role": "system",
                    "content": "You are a food science assistant."
                    "Your task is to check whether two food labels refer to the same food item"
                    "Respond ONLY with a JSON object containing the key 'response'."
                    "The value MUST be either 'yes' or 'no'."
                    "Do not include explanations."
                },
                {"role": "user", "content": prompt},
            ]
            if is_gpt
            else [
                {
                    "role": "system",
                    "content": "You are a food science assistant."
                    "Your task is to check whether two food labels refer to the same food item"
                    # "Respond ONLY with a JSON object containing the key 'response'."
                    "The value MUST be either 'yes' or 'no'."
                    "Do not include explanations."
                },
                {"role": "user", "content": prompt},
            ]
        )
        
    def ask_if_food_is_fruit_or_vegetable(self, label):
        messages = [
            {
                "role": "system",
                "content": "You are a food science assistant."
                "Your task is to determine whether a given food label refers to a fruit or a vegetable."
                "Respond ONLY with a JSON object containing the key 'response'."
                "The value MUST be either 'fruit' or 'vegetable'."
                "Do not include explanations."
            },
            {"role": "user", "content": f"Is this food a fruit or vegetable? '{label}'"},
        ]
        if self.model_name in ["gpt-4o-mini", "gpt-3.5-turbo"]:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=Response
            )
            return completion.choices[0].message.parsed.response
        else:
            pipe = load_client(self.model_name)
            response = pipe(messages)
            return response[0]['generated_text'][2]['content'].strip().strip('.')
    
    def use_gpt4(self, label_1: str, label_2: str):

        prompt = f"Are these two food labels semantically equivalent? '{label_1}' and '{label_2}'"
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.choose_message(prompt),
            response_format=Response
        )
        return completion.choices[0].message.parsed.response
    
    def use_free_model(self, label_1: str, label_2: str):
        pipe = load_client(self.model_name)
        prompt = f"Are these two food labels semantically equivalent? '{label_1}' and '{label_2}'"
        messages = self.choose_message(prompt, is_gpt=False)
        response = pipe(messages)
        return response[0]['generated_text'][2]['content'].strip().strip('.')
    

    def update_food_file_with_food_type(
        self,
        input_file_path: str,
        output_file_path: str,
        label_field: str = "food_class",
        output_key: str = "food_type"
    ):
        """
        Stream-process a JSON file containing a list of food items,
        classify each food as fruit or vegetable using the LLM,
        and write each updated item immediately to the output file.
        """

        # 1. Load input file
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of food objects")

        # 2. Open output file and start JSON array
        with open(output_file_path, "w", encoding="utf-8") as out:
            out.write("[\n")

            first_item = True

            for item in data:
                if label := item.get(label_field):
                    try:
                        result = self.ask_if_food_is_fruit_or_vegetable(label)
                        print(f"[INFO] Classified '{label}' as '{result}'")
                        item[output_key] = result
                    except Exception as e:
                        print(f"[WARNING] Failed to classify '{label}': {e}")
                        item[output_key] = None
                else:
                    item[output_key] = None

                # Write comma if not the first element
                if not first_item:
                    out.write(",\n")
                else:
                    first_item = False

                # Write the item immediately
                json.dump(item, out, ensure_ascii=False, indent=2)

                # Force flush to disk (important for long runs)
                out.flush()

            # 3. Close JSON array
            out.write("\n]")

        print(f"✅ File updated incrementally: {output_file_path}")

        
askfood = AskFoodSearch(model_name="gpt-4o-mini")

askfood.update_food_file_with_food_type(
    input_file_path="json/old/fruitveg81_usda_enriched.json",
    output_file_path="json/old/fruitveg81_usda_enriched_updated.json"
)