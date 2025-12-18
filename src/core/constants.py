import os

# Set HuggingFace cache to net directory for all models
# All models will be downloaded/cached to this centralized location
HF_HOME = "/net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME

# Model paths - All using HuggingFace Hub IDs
# Models will be downloaded to: /net/tokyo100-10g/data/str01_01/y-guo/huggingface_cache/hub/
MODEL_PATHs = {
    # LLaMA models - HuggingFace Hub IDs
    "llama3.2_3b_it": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2_1b_it": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.1_8b_it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B",
    "llama3.1_8b": "meta-llama/Llama-3.1-8B",
    
    # Qwen models - HuggingFace Hub IDs
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_7b_it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}
