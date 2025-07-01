import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

kagglehub.login()

model_path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

prompt = "how many digits does googol have"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(
    max_new_tokens=150, do_sample=True, temperature=0.7
)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
