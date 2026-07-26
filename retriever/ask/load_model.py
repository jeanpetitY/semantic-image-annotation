from transformers import  AutoTokenizer, pipeline
import torch

# model_name = "tiiuae/Falcon3-7B-Instruct"
model_name = "mistralai/Ministral-8B-Instruct-2410"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

def load_client(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        # repetition_penalty=1.2,
        # num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )