import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    kagglehub.login()

    model_path = kagglehub.model_download(
        "google/gemma-3n/transformers/gemma-3n-e2b-it"
    )

    # Check MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:
        device = "cpu"
        print("MPS not available, using CPU")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load with reduced precision and memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use half precision
        low_cpu_mem_usage=True,
        device_map=None,  # We'll manually move to MPS
    )

    # Move model to MPS device
    model = model.to(device)
    print(f"Model loaded successfully on {device}")

    prompt = "how do i make pizza the italian way"
    inputs = tokenizer(prompt, return_tensors="pt").to(
        device
    )  # Move inputs to same device

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,  # Reduced for memory (max_new_tokens is basically used to get the number of words)
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:")
    print(result)

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
