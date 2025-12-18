"""
MyriadLama Custom Attention Generation.

This script implements a custom attention-based ensemble generation specifically
designed for the MyriadLAMA dataset.

Key features:
- Uses HuggingFace's native attention mask mechanism
- Patches LLaMA's _update_causal_mask to inject custom structure masks
- Implements segment-based masking where:
  * Causal mask is always applied
  * Shared part (instruction + few-shot) uses normal causal attention
  * Each para part can attend to itself and the shared part
  * Para parts are isolated from each other (cannot attend to other paras)
- Provides placeholder interface for Qwen model support

Attention Pattern (visualized):
    shared_part | para_1 | para_2 | para_3 | ...
    ------------+--------+--------+--------+----
    normal      | see    | see    | see    | ...   <- shared tokens
    causal      | shared | shared | shared | ...
                | only   | only   | only   |
    ------------+--------+--------+--------+----
    can see     | normal | X      | X      | ...   <- para_1 tokens
    shared      | causal |        |        |
    ------------+--------+--------+--------+----
    can see     | X      | normal | X      | ...   <- para_2 tokens
    shared      |        | causal |        |
    ------------+--------+--------+--------+----
    ...
"""

import os
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel

from constants import MODEL_PATHs
from mask_visualization import visualize_mask_heatmap
from utils import DATASET_ROOT

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

# ==============================================================================
# Lemmatization functions
# ==============================================================================

nlp = None
num_parts = 8


def init_spacy():
    """Reused from generate.py"""
    global nlp
    nlp = spacy.load("en_core_web_lg")


def lemmaize_predicts(predict):
    """Reused from generate.py"""
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]


def lemmaize_chunk(chunk):
    """Reused from generate.py"""
    predict_lemmas = []
    answer_lemmas = []
    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
    return predict_lemmas, answer_lemmas


def append_lemmas(df, results):
    """Reused from generate.py"""
    all_predict_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df


# ==============================================================================
# Prompt construction
# ==============================================================================

def get_few_shot_examples_with_paraphrases(dataset, k=5, num_fs_paraphrases=3, seed=42):
    """
    Get few-shot examples formatted with multiple paraphrase questions + one answer.
    
    Each few-shot example has:
    - Multiple paraphrase questions (same count as main question paraphrases)
    - One answer (shared by all paraphrases)
    
    Args:
        dataset: MyriadLamaDataset instance
        k: Number of few-shot examples
        num_fs_paraphrases: Number of paraphrase questions per few-shot 
                            (should match main question paraphrase count)
        seed: Random seed
        
    Returns:
        List of few-shot examples, each as:
        {'paraphrases': [q1, q2, ...], 'answer': ans}
    """
    import random
    from datasets import load_from_disk
    
    if not os.path.exists(dataset.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset.dataset_path}")
    
    train_ds = load_from_disk(dataset.dataset_path)['train']
    random.seed(seed)
    indices = random.sample(range(len(train_ds)), k)
    
    few_shot_examples = []
    for idx in indices:
        example = train_ds[idx]
        # Get available paraphrases (manual_paraphrases + auto_paraphrases)
        all_paras = example["manual_paraphrases"] + example.get("auto_paraphrases", [])
        # Select first num_fs_paraphrases
        selected_paras = all_paras[:min(num_fs_paraphrases, len(all_paras))]
        answer = example["answers"][0]
        
        few_shot_examples.append({
            'paraphrases': selected_paras,
            'answer': answer
        })
    
    return few_shot_examples


def construct_prompt_new_format(instruction, few_shot_examples, question_paraphrases):
    """
    Construct prompt parts: one shared part + multiple paraphrase parts.
    
    Returns a list where:
    - First element: shared part (instruction + few-shot examples)
    - Remaining elements: individual paraphrase parts (one per paraphrase)
    
    Shared part format:
    {instruction}
    
    Q: {fs1_para1} A: {fs1_answer}
    Q: {fs2_para1} A: {fs2_answer}
    Q: {fs3_para1} A: {fs3_answer}
    
    Each paraphrase part format:
    Q: {main_question_paraphrase} A:
    
    Args:
        instruction: Instruction string
        few_shot_examples: List of {'paraphrases': [...], 'answer': ...}
        question_paraphrases: List of all main question paraphrase strings
        
    Returns:
        List of strings: [shared_part, para_part_1, para_part_2, ...]
    """
    # Build shared part: instruction + few-shot examples
    shared_parts = []
    
    # Add instruction with trailing newline
    shared_parts.append(instruction + "\n")
    
    # Add few-shot examples (each on one line: Q: ... A: ...)
    fs_lines = []
    for fs_example in few_shot_examples:
        # Use only the FIRST paraphrase from each few-shot example
        para = fs_example['paraphrases'][0]
        answer = fs_example['answer']
        fs_lines.append(f"Q: {para} A: {answer}")
    
    # Join few-shot lines with newlines, add trailing newline
    shared_parts.append("\n".join(fs_lines) + "\n")
    
    # Combine shared parts
    shared_part = "\n".join(shared_parts)
    
    # Build individual paraphrase parts
    # SWAP: Put para2 first (will be group 1), para1 second (will be group 2)
    para_parts = []
    for para in question_paraphrases:
        para_part = f"Q: {para} A:"
        para_parts.append(para_part)
    
    # Swap positions if we have at least 2 paraphrases
    if len(para_parts) >= 2:
        para_parts[0], para_parts[1] = para_parts[1], para_parts[0]
    
    # Return list: [shared_part, para_2 (swapped to position 1), para_1 (swapped to position 2), ...]
    return [shared_part] + para_parts


def tokenize_prompt_parts(prompt_parts, tokenizer, add_special_tokens=True):
    """
    Tokenize prompt parts and track positions for each segment.
    
    Args:
        prompt_parts: List of strings [shared_part, para_part_1, para_part_2, ...]
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether to add BOS/EOS tokens (default: True)
        
    Returns:
        input_ids: Tensor of token IDs [1, seq_len]
        segment_positions: List of (start, end) tuples for each segment (after special tokens)
        question_group_ids: Tensor of group IDs for each token position
        total_length: Total number of tokens (including special tokens)
    """
    # Tokenize each part individually
    tokenized_parts = []
    separator = "\n"  # Single newline separator
    sep_tokens = tokenizer.encode(separator, add_special_tokens=False)
    
    for part in prompt_parts:
        tokens = tokenizer.encode(part, add_special_tokens=False)
        tokenized_parts.append(tokens)
    
    # Build full token sequence and track positions
    full_tokens = []
    segment_positions = []
    current_pos = 0
    
    # Add BOS token if needed
    if add_special_tokens and tokenizer.bos_token_id is not None:
        full_tokens.append(tokenizer.bos_token_id)
        current_pos = 1
    
    for i, tokens in enumerate(tokenized_parts):
        if i > 0:
            # Add separator
            full_tokens.extend(sep_tokens)
            current_pos += len(sep_tokens)
        
        start_pos = current_pos
        full_tokens.extend(tokens)
        current_pos += len(tokens)
        end_pos = current_pos
        
        segment_positions.append((start_pos, end_pos))
    
    # Build question_group_ids tensor
    # Group 0 = shared part (instruction + few-shot)
    # Group 1 = para_part_1
    # Group 2 = para_part_2
    # ...
    total_length = len(full_tokens)
    question_group_ids = torch.zeros(total_length, dtype=torch.long)
    
    # BOS token (if exists) belongs to group 0
    for group_idx, (start, end) in enumerate(segment_positions):
        question_group_ids[start:end] = group_idx
    
    # Convert to tensor
    input_ids = torch.tensor([full_tokens], dtype=torch.long)
    
    return input_ids, segment_positions, question_group_ids, total_length


# ==============================================================================
# NEW - Custom attention mask building (per reference document)
# ==============================================================================

from pdb import set_trace
def build_question_struct_mask(question_group_ids, Q, K, dtype, device):
    """
    Build the per-query structure mask [1, 1, Q, K].
    
    NEW STRATEGY: para2 (group 1) is completely isolated to eliminate its influence.
    
    Rule (token-level):
    - Let g_q = question_group_ids[q], g_k = question_group_ids[k].
    - If g_q == 0 (shared part): can see all EXCEPT group 1 (para2)
    - If g_q == 1 (para2): COMPLETELY ISOLATED - sees nothing, no one can see it
    - If g_q >= 2 (para1 and others): can see shared (group 0) + themselves only
    
    Args:
        question_group_ids: [S] tensor with values in {0, 1, 2, ...}
                           0 = shared part, 1 = para2 (isolated), 2+ = para1 and others
        Q: Query sequence length
        K: Key sequence length
        dtype: Data type for mask
        device: Device for mask
        
    Returns:
        [1, 1, Q, K] additive mask (0 or -inf)
    """

    neg_inf = torch.finfo(dtype).min
    
    # Ensure question_group_ids is on the correct device
    q_group_ids = question_group_ids[:Q].to(device)
    k_group_ids = question_group_ids[:K].to(device)
    
    # [Q, 1] and [1, K]
    g_q = q_group_ids.view(Q, 1)  # [Q, 1]
    g_k = k_group_ids.view(1, K)  # [1, K]
    
    # NEW MASK STRATEGY:
    # - Group 0 (shared): normal, can see all EXCEPT para2 (group 1)
    # - Group 1 (para2, now in first position): COMPLETELY ISOLATED
    #   * Cannot be attended to by anyone (including itself)
    #   * Cannot attend to anything (including itself)
    # - Group 2+ (para1 and others): can see shared + themselves only (NOT para2)
    
    is_shared = (g_q == 0)  # Query is in shared part
    is_para2 = (g_q == 1)   # Query is in para2 (the isolated one)
    is_other_para = (g_q >= 2)  # Query is in para1 or other paras
    
    # For shared part queries: allow all keys EXCEPT para2 (group 1)
    allowed_shared = is_shared & (g_k != 1)  # Shared cannot see para2
    
    # For para2 queries: block everything (completely isolated)
    allowed_para2 = torch.zeros_like(allowed_shared, dtype=torch.bool)  # Para2 sees nothing
    
    # For other para queries (para1, etc.): allow shared (except para2) OR same group
    is_k_shared = (g_k == 0)  # [1, K]
    same_group = (g_q == g_k)  # [Q, K]
    allowed_other_para = is_other_para & ((is_k_shared) | same_group)
    
    # Combine: allow if any condition is true
    allowed = allowed_shared | allowed_para2 | allowed_other_para  # [Q, K]
    
    # Additionally, NO ONE can attend to para2 (group 1)
    allowed = allowed & (g_k != 1)  # Block all attention to para2
    
    # Create mask: 0 for allowed, -inf for blocked
    mask = torch.zeros((1, 1, Q, K), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, neg_inf)

    fig = visualize_mask_heatmap(mask[0, 0], title="Question Struct Mask", show=False, save=False)
    return mask

def build_test_mask(question_group_ids, Q, K, dtype, device):
    """
    Build the per-query structure mask [1, 1, Q, K].
    
    Rule (token-level):
    - Let g_q = question_group_ids[q], g_k = question_group_ids[k].
    - If g_q == 0 (shared part): allow everything (structure doesn't constrain it)
    - If g_q >= 1 (para part N): allow g_k == 0 OR g_k == g_q (can see shared + self)
    
    Args:
        question_group_ids: [S] tensor with values in {0, 1, 2, ...}
                           0 = shared part, 1+ = para parts
        Q: Query sequence length
        K: Key sequence length
        dtype: Data type for mask
        device: Device for mask
        
    Returns:
        [1, 1, Q, K] additive mask (0 or -inf)
    """

    neg_inf = torch.finfo(dtype).min
    
    # Ensure question_group_ids is on the correct device
    q_group_ids = question_group_ids[:Q].to(device)
    k_group_ids = question_group_ids[:K].to(device)
    
    # [Q, 1] and [1, K]
    g_q = q_group_ids.view(Q, 1)  # [Q, 1]
    g_k = k_group_ids.view(1, K)  # [1, K]
    
    # Boolean conditions
    is_shared = (g_q == 0)  # Query is in shared part
    is_para = (g_q >= 1)    # Query is in some para part
    
    # For shared part queries: allow all keys (normal causal will be applied separately)
    allowed_shared = is_shared  # [Q, 1] broadcast to [Q, K]
    
    # For para part queries: allow keys in shared (g_k == 0) OR same group (g_q == g_k)
    is_k_shared = (g_k == 0)  # [1, K]
    same_group = (g_q == g_k)  # [Q, K]
    allowed_para = is_para & (is_k_shared | same_group)  # [Q, K]
    
    # Combine: allow if either condition is true
    allowed = allowed_shared | allowed_para  # [Q, K]
    
    # Create mask: 0 for allowed, -inf for blocked
    mask = torch.zeros((1, 1, Q, K), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, neg_inf)
    print()
    return mask

# ==============================================================================
# LLaMA model patching (for transformers >= 4.55)
# ==============================================================================

def install_llama_struct_mask(model):
    """
    Install custom structure mask for LLaMA model.
    
    Patches each LlamaDecoderLayer.forward to inject custom structure mask
    on top of the attention_mask.
    
    Args:
        model: HuggingFace LlamaForCausalLM model
        
    Returns:
        dict with patching info for restoration
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    base: LlamaModel = model.model
    patch_info = {'method': 'decoder_layer', 'originals': {}}
    
    # Initialize cache for struct masks
    model._struct_mask_cache = {}
    
    for layer_idx, layer in enumerate(base.layers):
        if not isinstance(layer, LlamaDecoderLayer):
            continue
            
        old_forward = layer.forward
        patch_info['originals'][layer_idx] = old_forward
        
        def make_patched_forward(old_fwd, idx):
            def patched_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs
            ):
                # Modify attention_mask if we have custom structure
                if attention_mask is not None and attention_mask.dim() == 4:
                    if hasattr(model, 'question_group_ids') and model.question_group_ids is not None:
                        B, H1, Q, K = attention_mask.shape
                        if B == 1:
                            # Use cache to avoid recomputing mask
                            cache_key = (Q, K, attention_mask.dtype, attention_mask.device)
                            
                            if cache_key not in model._struct_mask_cache:
                                # First time: compute and cache
                                struct_mask = build_question_struct_mask(
                                    model.question_group_ids, Q, K,
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device,
                                )
                                model._struct_mask_cache[cache_key] = struct_mask
                            else:
                                # Use cached mask
                                struct_mask = model._struct_mask_cache[cache_key]
                            
                            if H1 > 1:
                                struct_mask = struct_mask.expand(B, H1, Q, K)
                            attention_mask = attention_mask + struct_mask
                            fig = visualize_mask_heatmap(
                                 attention_mask[0, 0], title="Question Struct Mask", show=False, save=False
                             )
                return old_fwd(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )
            return patched_forward
        
        layer.forward = make_patched_forward(old_forward, layer_idx)
    
    model._llama_patch_info = patch_info
    return patch_info


def uninstall_llama_struct_mask(model, patch_info=None):
    """
    Restore original functions for LLaMA model.
    
    Args:
        model: HuggingFace LlamaForCausalLM model
        patch_info: Patch info dict (optional, can be retrieved from model._llama_patch_info)
    """
    if patch_info is None:
        if hasattr(model, '_llama_patch_info'):
            patch_info = model._llama_patch_info
        else:
            raise ValueError(
                "Cannot restore original functions: patch_info not provided "
                "and model._llama_patch_info not found"
            )
    
    base = model.model
    
    for layer_idx, old_forward in patch_info['originals'].items():
        if isinstance(layer_idx, int):
            base.layers[layer_idx].forward = old_forward
    
    # Clean up stored reference
    if hasattr(model, '_llama_patch_info'):
        del model._llama_patch_info


# ==============================================================================
# Qwen model patching placeholder
# ==============================================================================

def install_qwen_struct_mask(model):
    """
    Install custom structure mask for Qwen model.
    
    Placeholder for future Qwen support. Currently not implemented.
    
    Args:
        model: HuggingFace Qwen model
    """
    raise NotImplementedError(
        "Qwen model support is not yet implemented. "
        "Please use LLaMA models for now."
    )


def uninstall_qwen_struct_mask(model, original_forwards):
    """
    Restore original forwards for Qwen model.
    
    Placeholder for future Qwen support.
    
    Args:
        model: HuggingFace Qwen model
        original_forwards: Dict of original forward functions
    """
    raise NotImplementedError(
        "Qwen model support is not yet implemented."
    )


# ==============================================================================
# NEW - Custom attention generation function
# ==============================================================================

@torch.no_grad()
def myriadlama_custom_attention_generation(
    model,
    tokenizer,
    prompt_parts,
    max_new_tokens=50,
    debug_info=None,
    use_normal_attention=False
):
    """
    Generate text using custom attention masking for MyriadLAMA.
    
    This uses HuggingFace's native attention mask with custom structure mask
    overlay to implement:
    - Causal mask always applied
    - Shared part uses normal causal attention
    - Each para part can see itself and shared part
    - Para parts are isolated from each other
    
    Args:
        model: HuggingFace model (LLaMA)
        tokenizer: HuggingFace tokenizer
        prompt_parts: List of strings [shared_part, para_part_1, ...]
        max_new_tokens: Maximum tokens to generate (default: 10)
        debug_info: Optional dict for debug output
        
    Returns:
        Generated text string
    """
    # Set model config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # Tokenize and get positions (now returns input_ids directly)
    input_ids, segment_positions, question_group_ids, total_length = \
        tokenize_prompt_parts(prompt_parts, tokenizer, add_special_tokens=True)
    
    # Move to device
    input_ids = input_ids.to(model.device)
    
    # Store debug info if requested
    if debug_info is not None:
        debug_info['input_ids'] = input_ids.tolist()
        debug_info['segment_positions'] = segment_positions
        debug_info['question_group_ids'] = question_group_ids.tolist()
        debug_info['total_length'] = total_length
    
    inputs = {"input_ids": input_ids}
    
    # Set question_group_ids on model for mask generation
    # Extend to cover potential generated tokens (set as group 0 to allow full attention)
    max_length = inputs["input_ids"].shape[1] + max_new_tokens
    # 
    extended_group_ids = torch.zeros(max_length, dtype=torch.long, device=model.device)
    extended_group_ids[:len(question_group_ids)] = question_group_ids.to(model.device)
    # 
    if not use_normal_attention:
        model.question_group_ids = extended_group_ids
    # 
    generated = None
    
    # Generation loop
    for step in range(max_new_tokens):
        # Forward pass
        
        logits = model(inputs["input_ids"]).logits[:, -1, :]
        
        # Token selection (greedy)
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        
        # Check for EOS or newline
        if next_token.item() == tokenizer.eos_token_id:
            break
        decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
        if '\n' in decoded_token and step > 0:
            # break
            pass
    
    # Clear model's question_group_ids and cache
    model.question_group_ids = None
    if hasattr(model, '_struct_mask_cache'):
        model._struct_mask_cache.clear()
    
    # Decode output
    if generated is None:
        return ""
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated_texts[0].strip()


# ==============================================================================
# Debug visualization
# ==============================================================================

def visualize_custom_attention_mask(segment_positions, question_group_ids, tokenizer, input_ids=None):
    """
    Visualize the custom attention mask structure for debugging.
    
    Args:
        segment_positions: List of (start, end) tuples
        question_group_ids: Tensor of group IDs
        tokenizer: Tokenizer for decoding
        input_ids: Token IDs (list or tensor)
        
    Returns:
        String representation of the mask structure
    """
    # Get sequence length
    if input_ids is not None:
        if isinstance(input_ids, list):
            seq_len = len(input_ids)
        else:
            seq_len = len(input_ids) if input_ids.dim() == 1 else input_ids.shape[1]
    else:
        seq_len = len(question_group_ids)
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"CUSTOM ATTENTION MASK VISUALIZATION")
    lines.append(f"{'='*80}")
    lines.append(f"Total tokens: {seq_len}")
    
    # Show segment structure
    lines.append(f"\nSegment Structure:")
    for i, (start, end) in enumerate(segment_positions):
        group_name = "shared_part" if i == 0 else f"para_{i}"
        lines.append(f"  {group_name}: tokens {start}-{end} (group {i})")
    
    # Show question_group_ids distribution
    lines.append(f"\nQuestion Group IDs Distribution:")
    unique_groups = torch.unique(question_group_ids[:seq_len])
    for g in unique_groups.tolist():
        count = (question_group_ids[:seq_len] == g).sum().item()
        lines.append(f"  Group {g}: {count} tokens")
    
    # Build sample mask visualization for a few key positions
    lines.append(f"\n{'='*80}")
    lines.append(f"SAMPLE MASK VALUES")
    lines.append(f"{'='*80}")
    lines.append(f"Legend: ✓ = can attend, ✗ = blocked")
    
    # Sample query positions from each group
    sample_positions = []
    for i, (start, end) in enumerate(segment_positions):
        mid = (start + end) // 2
        sample_positions.append((mid, f"group_{i}"))
    
    # Create mask tensor for visualization
    dtype = torch.float32
    device = question_group_ids.device
    struct_mask = build_question_struct_mask(question_group_ids, seq_len, seq_len, dtype, device)
    
    for q_pos, q_label in sample_positions:
        lines.append(f"\nQuery position {q_pos} ({q_label}):")
        row = "  "
        # Sample key positions
        key_samples = []
        for i, (start, end) in enumerate(segment_positions):
            key_samples.append((start, f"g{i}_start"))
            key_samples.append(((start + end) // 2, f"g{i}_mid"))
        
        for k_pos, k_label in key_samples:
            if k_pos < seq_len:
                mask_val = struct_mask[0, 0, q_pos, k_pos].item()
                can_attend = mask_val == 0
                row += f"[{k_label}: {'✓' if can_attend else '✗'}] "
        lines.append(row)
    
    lines.append(f"{'='*80}\n")
    
    return "\n".join(lines)


# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MyriadLAMA Custom Attention generation"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2_3b_it",
        help="Model name from constants.MODEL_PATHs (must be LLaMA model)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for model (default: auto)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=f"Output directory for results (default: {DATASET_ROOT}/myriadlama/{{model}})"
    )
    parser.add_argument(
        "--lemmaize", action="store_true",
        help="Normalize predictions and answers to lemmas"
    )
    parser.add_argument(
        "--num_paraphrases", type=int, default=2,
        help="Number of paraphrases to use (default: 2)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=2,
        help="Maximum number of samples to generate (default: None, process all)"
    )
    parser.add_argument(
        "--rewrite", action="store_true",default=True,
        help="Regenerate results even if output file already exists"
    )
    parser.add_argument(
        "--disable_p2p", action="store_true",
        help="Disable GPU peer-to-peer (P2P) access"
    )
    parser.add_argument(
        "-n", "--normal_attention", action="store_true",
        help="Use normal causal attention instead of custom structured attention"
    )
    args = parser.parse_args()
    
    # Validate model is LLaMA
    if args.model not in MODEL_PATHs:
        raise ValueError(
            f"Model {args.model} not supported. "
            f"Choose from {list(MODEL_PATHs.keys())}"
        )
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    # Check if this is a LLaMA model
    is_llama = "llama" in args.model.lower() or "llama" in model_path.lower()
    if not is_llama:
        print(f"⚠️  Warning: This script is optimized for LLaMA models.")
        print(f"   Model '{args.model}' may not work correctly.")
        print(f"   Qwen support is planned for future implementation.")
    
    # Load MyriadLAMA dataset
    from dataset import MyriadLamaDataset
    dataset = MyriadLamaDataset(model_name=args.model)
    
    # Use batch_size=1 for sequential processing
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    
    # Output file setup - use provided output_dir or default
    if args.output_dir:
        local_output_dir = args.output_dir
    else:
        # Default output directory (may need to be adjusted for your environment)
        local_output_dir = f"{DATASET_ROOT}/myriadlama/{args.model}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Determine file name
    dump_file = f"{local_output_dir}/myriadlama_custom_{args.num_paraphrases}paras.feather"
    
    print(f"Output file: {dump_file}")
    
    # Lemmatization mode
    if args.lemmaize:
        if not os.path.exists(dump_file):
            raise FileNotFoundError(
                f"File {dump_file} does not exist. Run without --lemmaize first."
            )
        
        df = pd.read_feather(dump_file)
        if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
            print(f"Lemmatized data already exists in {dump_file}")
            exit(0)
        
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        
        df = append_lemmas(df, results)
        df.to_feather(dump_file)
        
        # Convert to CSV automatically
        csv_file = dump_file.replace('.feather', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"✅ CSV file saved to {csv_file}")
        
        exit(0)
    
    # Check if file exists
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"File {dump_file} already exists, skipping generation.")
        print("Use --rewrite flag to regenerate.")
        exit(0)
    
    # Optional: disable P2P before any CUDA allocations
    if args.disable_p2p:
        os.environ["TORCH_CUDA_DISABLE_P2P"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        print("🔧 Disabled CUDA/NCCL peer-to-peer")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if args.device == "auto":
        print("📦 Loading model with device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype="auto"
        )
    else:
        target_device = args.device
        print(f"📦 Loading full model onto single device: {target_device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto"
        )
        try:
            model.to(target_device)
        except Exception as e:
            print(f"❌ Failed to move model to {target_device}: {type(e).__name__}: {e}")
            fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
            model.to(fallback)
            print(f"   Moved model to {fallback}")
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Force eager attention for explicit mask usage
    # Note: _attn_implementation is a private attribute that may change in future
    # versions of transformers. This is the recommended way to force eager attention
    # as of transformers 4.30.0+. If this breaks, check the transformers documentation.
    model.config._attn_implementation = "eager"
    print(f"🔧 Set attention implementation to 'eager'")
    
    # Install custom attention mask hook
    if not args.normal_attention:
        print(f"🔧 Installing custom structure mask for LLaMA model...")
        patch_info = install_llama_struct_mask(model)
        print(f"   Patching method: {patch_info['method']}")
    else:
        print(f"🔧 Using normal causal attention (custom attention disabled)")
    
    # Print model info
    print(f"🔍 Model: {args.model}")
    print(f"   PyTorch version: {torch.__version__}")
    attention_type = "Normal Causal Attention" if args.normal_attention else "Custom Structured Attention"
    print(f"   Attention Type: {attention_type}")
    print(f"   Using {args.num_paraphrases} paraphrases per question")
    if hasattr(model.config, 'num_attention_heads'):
        print(f"   Attention heads: {model.config.num_attention_heads}")
    
    # Get few-shot examples
    few_shot_examples = get_few_shot_examples_with_paraphrases(
        dataset, 
        k=5, 
        num_fs_paraphrases=args.num_paraphrases,
        seed=42
    )
    instruction = dataset.instruction
    
    # DataFrame initialization
    df = pd.DataFrame(columns=[
        "uuid", "answers", "prediction", "generation", "templates",
        "debug_prompt", "debug_mask"
    ])
    
    print(f"\nMyriadLAMA generation with Custom Attention")
    if args.max_samples:
        print(f"Processing maximum {args.max_samples} samples")
    
    # Main generation loop
    sample_count = 0
    total_iterations = args.max_samples if args.max_samples else len(dataset.ds)
    
    for uuids, answers, all_paraphrases in tqdm(dataloader, total=total_iterations):
        batch_predictions = []
        batch_generations = []
        batch_templates = []
        batch_debug_prompt = []
        batch_debug_mask = []
        
        # Process each question in batch
        for i, paraphrases in enumerate(zip(*all_paraphrases)):
            # Select paraphrases
            all_templates = list(paraphrases)
            selected_templates = all_templates[:args.num_paraphrases]
            
            # Construct prompt parts
            prompt_parts = construct_prompt_new_format(
                instruction,
                few_shot_examples,
                selected_templates
            )
            
            # Prepare debug info container
            debug_info = {} if sample_count < 5 else None
            
            # Generate
            generation = myriadlama_custom_attention_generation(
                model,
                tokenizer,
                prompt_parts,
                max_new_tokens=50,
                debug_info=debug_info,
                use_normal_attention=args.normal_attention
            )
            
            # Extract prediction (first word only)
            prediction = generation.strip().split()[0] if generation.strip() else ""
            
            batch_predictions.append(prediction)
            batch_generations.append(generation)
            batch_templates.append(selected_templates)
            
            # Collect debug info for first 5 samples
            if debug_info:
                # Full prompt
                full_prompt = prompt_parts[0] + "\n".join(prompt_parts[1:])
                batch_debug_prompt.append(full_prompt)
                
                # Create attention mask visualization
                mask_viz = visualize_custom_attention_mask(
                    debug_info['segment_positions'],
                    torch.tensor(debug_info['question_group_ids']),
                    tokenizer,
                    input_ids=debug_info['input_ids'][0] if debug_info['input_ids'] else None
                )
                batch_debug_mask.append(mask_viz)
                
                # Print for first 5 samples
                print(f"\n{'='*80}")
                print(f"SAMPLE {sample_count + 1} DEBUG OUTPUT")
                print(f"{'='*80}")
                print(f"UUID: {uuids[i] if isinstance(uuids, list) else uuids}")
                print(f"Prompt Parts: {len(prompt_parts)}")
                print(f"Generation: {generation}")
                print(mask_viz)
            else:
                batch_debug_prompt.append("")
                batch_debug_mask.append("")
        
        # Store results
        items = {
            "uuid": uuids,
            "templates": batch_templates,
            "answers": answers,
            "prediction": batch_predictions,
            "generation": batch_generations,
            "debug_prompt": batch_debug_prompt,
            "debug_mask": batch_debug_mask,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
        
        # Check if we've reached max_samples
        sample_count += len(uuids)
        if args.max_samples and sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples}), stopping generation")
            break
    
    # Save results
    df.to_feather(dump_file)
    print(f"✅ Results saved to {dump_file}")
    
    # Convert to CSV automatically
    csv_file = dump_file.replace('.feather', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV file saved to {csv_file}")
