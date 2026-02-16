from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import AutoModelForImageClassification, AutoImageProcessor

load_dotenv()

token = os.getenv("HUB_TOKEN")
repo_name = os.getenv("REPO_NAME")

model_path = "model_saved/beit-food-389-v1"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

login(token)


model.push_to_hub(repo_name)
processor.push_to_hub(repo_name)