import os

# Set HuggingFace cache to net directory for all models
# All models will be downloaded/cached to this centralized location
HF_HOME = "/net/tokyo100-10g/data/str01_01/y-guo"
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME
import os

# Dynamic path configuration based on current user

_model_base = "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf"

# Model paths - All using HuggingFace Hub IDs
MODEL_PATHs = {
    # LLaMA models - local paths or HuggingFace Hub IDs
    "llama3.2_3b_it": f"{_model_base}/llama3.2_3b_it",
    "llama3.2_1b_it": f"{_model_base}/llama3.2_1b_it",
    "llama3.1_8b_it": f"{_model_base}/llama3.1_8b_it",
    "llama3.2_3b": f"{_model_base}/llama3.2_3b",
    "llama3.2_1b": f"{_model_base}/llama3.2_1b",
    "llama3.1_8b": f"{_model_base}/llama3.1_8b",
    "llama3.1_70b": "meta-llama/Llama-3.1-70B",
    # DeepSeek models - HuggingFace Hub IDs
    "deepseek_r1_distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek_r1_distill_qwen_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_r1_distill_qwen_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # Qwen models - HuggingFace Hub IDs
    "qwen2.5_3b": "Qwen/Qwen2.5-3B",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_14b": "Qwen/Qwen2.5-14B",
    "qwen2.5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_7b_it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_14b_it": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3_0.6b": "Qwen/Qwen3-0.6B",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B",
    "qwen3_235b": "Qwen/Qwen3-235B-A22B",
    # Pythia Model
    "pythia_2.8b": "EleutherAI/pythia-2.8b"
}
_current_user = os.environ.get('USER', 'unknown')
if _current_user == 'y-guo':
    _model_base = "/home/y-guo/models/llama_hf"
    MODEL_PATHs = {
    # LLaMA models - HuggingFace Hub IDs
    "llama3.2_3b_it": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2_1b_it": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.1_8b_it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B",
    "llama3.1_8b": "meta-llama/Llama-3.1-8B",
    "llama3.1_70b": "meta-llama/Llama-3.1-70B",
    "llama3.2_8b": "meta-llama/Llama-3.2-8B",
    # DeepSeek models - HuggingFace Hub IDs
    "deepseek_r1_distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek_r1_distill_qwen_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_r1_distill_qwen_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # Qwen models - HuggingFace Hub IDs
    "qwen2.5_3b": "Qwen/Qwen2.5-3B",
    "qwen2.5_3b_it": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B",
    "qwen2.5_7b_it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_14b": "Qwen/Qwen2.5-14B",
    "qwen2.5_14b_it": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}