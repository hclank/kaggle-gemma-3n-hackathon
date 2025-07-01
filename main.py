import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

kagglehub.login()

model_path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
