"""
MyriadLama-specific FlexAttention generation.

This script implements a modified FlexAttention-based ensemble generation specifically
designed for the MyriadLAMA dataset, with custom prompt construction and mask logic.

Key differences from flex_attention_generate.py:
- Custom prompt formatting for [MASK] token prediction
- Modified mask logic: each segment is isolated during encoding
- Segments include: instruction, each few-shot example, and each question paraphrase
- All paraphrases are manually generated (no distinction between manual/auto)
- Optimized for MyriadLAMA's fill-in-the-blank task structure

Features:
- Parses prompts to identify instruction, few-shot examples, and questions
- Each few-shot example is isolated (cannot attend to other few-shot examples)
- Each question paraphrase is isolated (cannot attend to other paraphrases)
- Within each segment, only causal attention (later tokens attend to earlier tokens)
- Allows generated tokens to attend to all segments for fusion
- Specifically designed for one-word prediction tasks
"""

import itertools
import multiprocessing as mp
import os
import random
import warnings
from pdb import set_trace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

torch.set_printoptions(profile="full", linewidth=200)

from constants import MODEL_PATHs  # noqa: E402
from utils import append_lemmas, init_spacy, lemmaize_chunk  # noqa: E402

torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

FLEX_ATTENTION_AVAILABLE = True
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

nlp = None
num_parts = 8

# concatenate_paraphrases_with_positions
def tokenize_with_segment(prompt, tokenizer, segment_metadata, has_bos=True):
    segment_positions = []
    context = prompt[: segment_metadata["len_context"]]
    paraphrases = [
        prompt[segment_metadata["len_context"] + sum(segment_metadata["len_paras"][:i]): 
               segment_metadata["len_context"] + sum(segment_metadata["len_paras"][:i+1])] 
        for i in range(len(segment_metadata["len_paras"]))]

    full_tokens = tokenizer.encode(context)
    segment_positions.append({"start": 0, "end": len(full_tokens), "type": "context"})
    for p in paraphrases:
        p_tokens = tokenizer.encode(p)
        if has_bos:
            p_tokens = p_tokens[1:]  # Exclude BOS for llama tokenizers

        start = len(full_tokens)
        end = start + len(p_tokens)
        segment_positions.append({"start": start, "end": end, "type": "paraphrase"})
        full_tokens.extend(p_tokens)

    seq_length_for_flexattn_scoremod = len(full_tokens)
    if "len_answer" in segment_metadata:
        answer_start = segment_metadata["len_context"] + sum(
            segment_metadata["len_paras"]
        )
        answer_end = answer_start + segment_metadata["len_answer"]
        ans_tokens = tokenizer.encode(prompt[answer_start:answer_end])
        if has_bos:
            ans_tokens = ans_tokens[1:]  # Exclude BOS for llama tokenizers
        segment_positions.append(
            {
                "start": len(full_tokens),
                "end": len(full_tokens) + len(ans_tokens),
                "type": "answer",
            }
        )
        full_tokens.extend(ans_tokens)
    # Decode back to text
    concatenated_text = tokenizer.decode(full_tokens, skip_special_tokens=False)

    return (
        concatenated_text,
        full_tokens,
        segment_positions,
        seq_length_for_flexattn_scoremod,
    )

def create_myriadlama_mask_mod(segment_positions, prefix_len):
    """
    Create attention mask for MyriadLAMA with complex few-shot and paraphrase masking.

    Mask logic for MyriadLAMA:
    - Paraphrases of the same few-shot example CANNOT attend to each other
    - Paraphrases from different few-shot examples CAN attend to each other
    - Answer parts of few-shot examples have normal causal mask
    - Question paraphrases are isolated from each other
    - During generation: new tokens attend to all segments for fusion

    This ensures that:
    1. Few-shot example 1 from paraphrase A can attend to few-shot example 1 from paraphrase B
       (but only the question parts are isolated by paraphrase)
    2. Few-shot example 1 from paraphrase A can attend to few-shot example 2 from paraphrase A
    3. Answer parts can attend normally (causal)
    4. Question paraphrases remain isolated

    Args:
        segment_positions: List of (start, end) tuples defining all segment boundaries
        segment_metadata: List of dicts with 'type', 'paraphrase_idx', 'few_shot_idx'
        prefix_len: Length of the context sequence

    Returns:
        mask_mod: Function (b, h, q_idx, kv_idx) -> Tensor[bool]
    """
    # Convert segment positions to tensors
    segment_starts = torch.tensor(
        [segment["start"] for segment in segment_positions], dtype=torch.int64
    )
    segment_ends = torch.tensor(
        [segment["end"] for segment in segment_positions], dtype=torch.int64
    )
    
    def mask_mod(b, h, q_idx, kv_idx):
        """
        Mask function for MyriadLAMA FlexAttention with complex rules.

        IMPORTANT: Must use only tensor operations (no .item() or Python if on tensors)
        to avoid vmap compilation errors.

        Logic (PRIORITY ORDER):
        1. HIGHEST PRIORITY: Causal constraint (cannot attend to future)
        2. Generated tokens (>= prefix_len) attend to all previous tokens
        3. Within encoding phase, apply complex rules based on segment types
        """        
        # Move segment tensors to same device as indices
        device = q_idx.device
        seg_starts = segment_starts.to(device)
        seg_ends = segment_ends.to(device)
        
        # 1) HIGHEST PRIORITY: Causal constraint - cannot attend to future
        causal_mask = q_idx >= kv_idx

        # 2) If query is in generation phase, allow attention to all previous tokens (with causal)
        is_generated = q_idx >= prefix_len

        # 3) Find which segment the query and key belong to
        q_in_segment = (q_idx >= seg_starts) & (q_idx < seg_ends)
        kv_in_segment = (kv_idx >= seg_starts) & (kv_idx < seg_ends)
        
        # 4) Mark the self-attention within the same segment as one based on context, para1, para2, ...
        q_seg_id = q_in_segment.to(torch.int32).argmax()
        kv_seg_id = kv_in_segment.to(torch.int32).argmax()
        same_segment = (q_seg_id == kv_seg_id)
        
        kv_is_context = (kv_seg_id == 0)
        intra_segment_or_context = same_segment | kv_is_context

        valid_topology = torch.where(
            is_generated,
            torch.tensor(True, device=device), # Generation: attend to all history
            intra_segment_or_context           # Encoding: isolate paraphrases
        )
        final_mask = valid_topology & causal_mask
        return final_mask

    return mask_mod


def create_myriadlama_score_mod(
    segment_positions,
    prefix_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    # Collect paraphrase spans
    para_spans = [
        (seg["start"], seg["end"])
        for seg in segment_positions
        if seg.get("type") == "paraphrase"
    ]
    assert len(para_spans) > 0, "No paraphrase spans found in segment_positions"
    
    # weight = 1 + scale_factor/len(para_spans)
    # Calculate length-based weight
    para_segs = [seg for seg in segment_positions if seg.get("type") == "paraphrase"]
    shared_segs = [seg for seg in segment_positions if seg.get("type") != "paraphrase"]
    len_para = sum(para_segs[i]["end"] - para_segs[i]["start"] for i in range(len(para_segs)))
    len_share = sum(shared_segs[i]["end"] - shared_segs[i]["start"] for i in range(len(shared_segs)))
    len_para_avg = len_para / len(para_segs)
    weight = len_para_avg * (len_share + len_para_avg) / ((len_para_avg + len_share) * len_para)

    # Build a boolean mask over *prefix positions* [0, prefix_len)
    # True where kv is inside a paraphrase span (restricted to prefix)
    para_mask = torch.zeros(prefix_len, device=device, dtype=torch.bool)
    for start, end in para_spans:
        s = max(0, min(prefix_len, int(start)))
        e = max(0, min(prefix_len, int(end)))
        if e > s:
            para_mask[s:e] = True

    # Precompute constants once
    logw = torch.tensor(float(torch.log(torch.tensor(weight))), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)

    def score_mod(score, b, h, q_idx, kv_idx):
        is_decode = q_idx >= prefix_len
        is_prefix_key = kv_idx < prefix_len
        kv_safe = torch.clamp(kv_idx, 0, prefix_len - 1)
        in_para = para_mask[kv_safe] & is_prefix_key  # safe + correct
        apply = is_decode & in_para
        return score + torch.where(apply, logw.to(score.dtype), zero.to(score.dtype))

    return score_mod


class FlexAttentionWrapper:
    """
    Wrapper that patches model attention layers to use FlexAttention.
    Reused from flex_attention_generate.py with no modifications.
    """

    def __init__(self, model):
        self.model = model
        self.original_forwards = {}
        self.is_patched = False
        self.current_mask_mod = None
        self.current_score_mod = None

    def create_patched_forward_for_llama(
        self,
        layer_idx, 
        original_attn,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, original_attn.head_dim)

        # Llama3: no q_norm / k_norm here (unlike Qwen3)
        query_states = original_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = original_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)

        # ↑ Identical to original up to here
        # ↓ Replace attention_interface with flex_attention

        bsz = hidden_states.shape[0]
        num_heads = query_states.shape[1]
        q_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        # GQA expansion
        num_key_value_heads = key_states.shape[1]
        if num_key_value_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            value_states = value_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)

        if self.current_mask_mod is not None and q_len > 1:
            block_mask = create_block_mask(
                self.current_mask_mod,
                B=bsz,
                H=num_heads,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=query_states.device,
            )
        else:
            block_mask = None

        # Llama3: no explicit self.scaling attribute, use default (None = 1/sqrt(head_dim))
        scale = getattr(original_attn, "scaling", None)
        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            score_mod=self.current_score_mod,
            scale=scale,
        )

        # ↓ Identical to original from here
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = original_attn.o_proj(attn_output)
        return attn_output, None

    def create_patched_forward_for_qwen3(
        self,
        layer_idx, 
        original_attn,        
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        if (self.current_mask_mod is None and self.current_score_mod is None):
            return self.original_forwards[layer_idx](
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
        
        # Qwen3-specific attention logic in original code: 
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, original_attn.head_dim)

        query_states = original_attn.q_norm(original_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = original_attn.k_norm(original_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, original_attn.layer_idx, cache_kwargs)
        
        # ↑ Everything above is unchanged
        # ↓ Replace attention_interface with flex_attention

        bsz = hidden_states.shape[0]
        q_len = query_states.shape[2]   # after KV cache update, Q len may differ from KV len
        kv_len = key_states.shape[2]
        num_heads = query_states.shape[1]
        
        # GQA expansion — flex_attention requires Q and KV head counts to match
        num_key_value_heads = key_states.shape[1]
        if num_key_value_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            value_states = value_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)

        if self.current_mask_mod is not None and q_len > 1:
            block_mask = create_block_mask(
                self.current_mask_mod,
                B=bsz,
                H=num_heads,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=query_states.device,
            )
        else:
            block_mask = None
        
        # scale=self.scaling keeps Qwen3's explicit head_dim scaling instead of the default 1/sqrt(head_dim)
        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            score_mod=self.current_score_mod,
            scale=original_attn.scaling,
        )
        
        # ↑ flex_attention returns (bsz, num_heads, q_len, head_dim), no attn_weights
        # ↓ Everything below is unchanged except attn_weights is None
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = original_attn.o_proj(attn_output)
        return attn_output, None  # flex_attention doesn't return weights

    def create_patched_forward(self, layer_idx, original_attn):
        """Create a patched forward function for an attention layer."""

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            # set_trace()
            if self.model.config.model_type.startswith("llama"):
                return self.create_patched_forward_for_llama(
                    layer_idx, 
                    original_attn, 
                    hidden_states, 
                    position_embeddings, 
                    attention_mask, 
                    past_key_values, 
                    cache_position, 
                    **kwargs
                )
            elif self.model.config.model_type.startswith("qwen3"):
                return self.create_patched_forward_for_qwen3(
                    layer_idx, 
                    original_attn, 
                    hidden_states, 
                    position_embeddings, 
                    attention_mask, 
                    past_key_values, 
                    cache_position, 
                    **kwargs
                )
            else:
                raise NotImplementedError("FlexAttention patching only implemented for LLaMA-based models currently.")

        return patched_forward

    def patch_model(self, mask_mod, score_mod):
        """Patch all attention layers with FlexAttention."""
        if self.is_patched:
            self.unpatch_model()

        self.current_mask_mod = mask_mod
        self.current_score_mod = score_mod

        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self.original_forwards[i] = attn.forward
            attn.forward = self.create_patched_forward(i, attn)

        self.is_patched = True

    def unpatch_model(self):
        """Restore original attention implementation."""
        if not self.is_patched:
            return

        for i, layer in enumerate(self.model.model.layers):
            if i in self.original_forwards:
                layer.self_attn.forward = self.original_forwards[i]

        self.original_forwards = {}
        self.current_mask_mod = None
        self.current_score_mod = None
        self.is_patched = False

@torch.no_grad()
def flex_generation(prompt, segment_metadata, max_new_tokens=10, modify_rope=False, has_bos=True):
    """
    Generate text using FlexAttention.
    - Accepts a SINGLE prompt with ALL paraphrases
    - Shorter max_new_tokens (10 instead of 20) for one-word answers
    - All paraphrases are treated equally (all are manually generated)
    - Each paraphrase segment is isolated during encoding
    
    Args:
        prompt: Single prompt string with ALL paraphrases
        segment_metadata: Metadata dict with segment lengths and types
            E.g., `{'len_context': 397, 'len_paras': [25, 34, 19]}`
        max_new_tokens: Maximum tokens to generate (default: 10 for one-word answers)

    Returns:
        Generated text string
    """

    # Set model config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    paraphrases = []
    for i, length in enumerate(segment_metadata["len_paras"]):
        start = segment_metadata["len_context"] + sum(segment_metadata["len_paras"][:i])
        end = start + length
        paraphrases.append(prompt[start:end])

    # Process prompt with position tracking and metadata
    concatnate_text, full_tokens, segment_positions, attn_mod_len = (
        tokenize_with_segment(prompt, tokenizer, segment_metadata, has_bos=has_bos)
    )

    inputs = {
        "input_ids": torch.tensor([full_tokens]), 
        "attention_mask": torch.ones(1, len(full_tokens)), 
    }
    inputs = BatchEncoding(data=inputs).to(model.device)

    if modify_rope:
        # The segmented position should follows the order of `context`, `paraphrases`, `answer`
        position_ids = torch.arange(len(full_tokens), dtype=torch.long, device=model.device)
        context_end = segment_positions[0]['end']
        start_generation_token_id = context_end + max(
            segment["end"] - segment["start"]
            for segment in segment_positions[1:]
            if segment["type"] == "paraphrase"
        )
        for segment in segment_positions[1:]:
            if segment["type"] != "paraphrase":
                continue
            position_ids[segment['start']:segment['end']] = torch.arange(
                0, segment['end'] - segment['start'], dtype=torch.long, device=model.device
            ) + position_ids[context_end - 1] + 1

        if segment_metadata.get("len_answer", 0) > 0:
            answer_segment = segment_positions[-1]
            start_generation_token_id += answer_segment["end"] - answer_segment["start"]
            position_ids[answer_segment["start"] : answer_segment["end"]] = (
                torch.arange(
                    0,
                    answer_segment["end"] - answer_segment["start"],
                    dtype=torch.long,
                    device=model.device,
                )
                + position_ids[: answer_segment["start"]].max()
                + 1
            )
        position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"]) 
    else:
        position_ids = torch.arange(len(full_tokens), dtype=torch.long, device=model.device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs["input_ids"]) 
        start_generation_token_id = len(full_tokens)

    # Create FlexAttention wrapper
    if args.modify_attn or args.scale_factor:
        flex_wrapper = FlexAttentionWrapper(model)

    mask_mod, score_mod = None, None
    if args.modify_attn:
        mask_mod = create_myriadlama_mask_mod(
            segment_positions, attn_mod_len
        )
    if args.scale_factor:
        score_mod = create_myriadlama_score_mod(
            segment_positions, attn_mod_len, 
            device=model.device, dtype=model.dtype
        )

    # Generation loop
    generated = None
    for step in range(max_new_tokens):
        if args.modify_attn or args.scale_factor:
            flex_wrapper.patch_model(mask_mod, score_mod)
        try:
            logits = model(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                position_ids=position_ids).logits[:, -1, :]
        except Exception as e:
            raise RuntimeError(f"Generation failed at step {step}: {type(e).__name__}: {e}")
        finally:
            if args.modify_attn or args.scale_factor:
                flex_wrapper.unpatch_model()
    
        # Token selection
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones(1, 1, device=model.device)], dim=1
        )

        new_pos_id = torch.tensor([[start_generation_token_id + step]], device=model.device)
        position_ids = torch.cat([position_ids, new_pos_id], dim=1)

        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)

        decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
        
        # Check for EOS or newline (likely end of one-word answer)
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Also check if we generated a newline or space (end of word)
        if "\n" in decoded_token and step > 0:  # Allow at least one token
            break

    # Decode output
    if generated is None:
        print("⚠️  Warning: No tokens generated")
        return ""
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated_texts[0].strip()


# ==============================================================================
# Main script
# ==============================================================================

if __name__ == "__main__":
    import argparse

    from dataset import MyriadLamaDataset

    parser = argparse.ArgumentParser(
        description="MyriadLAMA-specific FlexAttention generation"
    )
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Model name from constants.MODEL_PATHs")
    parser.add_argument("--device", type=str, default="auto", help="Device for model (default: auto)")
    parser.add_argument(
        "--dataset", type=str, required=True, 
        choices=["webqa", "myriadlama", "commonsense", "mmlu", "logiqa", "hotpot"], 
        help="Dataset to use for generating paraphrases.")
    parser.add_argument("--lemmaize", action="store_true", help="Normalize predictions and answers to lemmas")
    parser.add_argument("--modify_rope", action="store_true", help="Modify RoPE embeddings during generation")
    parser.add_argument("--modify_attn", action="store_true", help="Modify attention masks using FlexAttention")
    parser.add_argument("--scale_factor", action="store_true", help="Scale attention scores using FlexAttention")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate for testing (default: 5)")
    parser.add_argument("--num_fewshots", type=int, default=5, help="Number of few-shot examples to use (default: 5)")
    parser.add_argument("--num_paraphrases", type=int, default=2, help="Number of paraphrases to use (same for main question and few-shot examples, default: 5)")
    parser.add_argument("--single_para_qapair", action="store_true", help="Use only one Q&A section for the target paraphrase")
    parser.add_argument("--explicit_prompts", action="store_true", help="Use explicit prompt construction without few-shot examples and Q&A pairs")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to generate (default: None, process all)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (default: 1)")
    parser.add_argument("--repeat_paras", action="store_true", help="Repeating the same paraphrase multiple times")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing output file")
    args = parser.parse_args()

    assert int(args.explicit_prompts) + int(args.single_para_qapair) <= 1, \
        "Cannot use both --explicit_prompts and --single_para_qapair together."

    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "commonsense":
        from dataset import CommonsenseParaphraseDataset
        dataset = CommonsenseParaphraseDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "mmlu":
        from dataset import MMLUParaphraseDataset
        dataset = MMLUParaphraseDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "logiqa":
        from dataset import LogiQAParaphraseDataset
        dataset = LogiQAParaphraseDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "hotpot":
        from dataset import HotpotDataset
        dataset = HotpotDataset(model_name=args.model, debug=args.debug)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa', 'myriadlama', 'commonsense', 'mmlu', 'logiqa', or 'hotpot'.")
    
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} not supported. Choose from {list(MODEL_PATHs.keys())}")
    model_path = MODEL_PATHs.get(args.model, args.model)

    # Determine file name based on number of paraphrases
    dump_file = f"{dataset.dataset_root}/"
    if args.modify_attn:
        dump_file += "modifyattn."
    if args.modify_rope:
        dump_file += "modifyrope."
    if args.repeat_paras:
        dump_file += "repeatparas."
    if args.scale_factor:
        dump_file += "scalescore."
    if args.single_para_qapair:
        dump_file += "singleparaqapair."
    if args.explicit_prompts:
        dump_file += "explicitprompts."
    if args.num_fewshots != 5:
        dump_file += f"{args.num_fewshots}fshots."

    dump_file += f"{args.num_samples}samples.{args.num_paraphrases}paras.feather"
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"✅ File {dump_file} already exists, skipping generation.")
        exit(0)
    
    max_new_tokens = 10 if args.num_fewshots > 0 else 20
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    
    print(f"🔄 Starting generation to {dump_file}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token
    has_bos = "llama" in args.model.lower()

    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation", "templates"])
    if args.max_samples:
        print(f"Processing maximum {args.max_samples} samples")

    # Get few-shot examples with multiple paraphrases (new format)
    # Use same number of paraphrases for few-shot as for main question
    few_shot_examples = dataset.get_few_shot_examples(k=args.num_fewshots) if args.num_fewshots > 0 else ""

    sample_count = 0
    samples = []
    for batch_data in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
        if dataset.is_multi_choice:
            uuids, answers, all_paraphrases, choices_labels, choices_texts, answer_labels, _ = batch_data
        else:
            uuids, answers, all_paraphrases, _ = batch_data
            choices_labels = [None] * len(uuids)
            choices_texts = [None] * len(uuids)
            answer_labels = [None] * len(uuids)
            
        assert len(uuids) == 1, "Batch size for data preparation must be 1 for MyriadLAMA generation"
        uuid, answer = uuids[0], answers[0]
        choices_labels = choices_labels[0]
        choices_texts = choices_texts[0]
        answer_labels = answer_labels[0]
        all_paraphrases = list(zip(*all_paraphrases))[0]
        
        all_indices = list(range(len(all_paraphrases)))
        
        if args.repeat_paras:
            all_sampled_paras = list([[n] * args.num_paraphrases for n in all_indices])
        else:
            all_sampled_paras = itertools.permutations(
                all_indices, args.num_paraphrases
            )

        random.seed(uuids[0])
        for paraids in random.sample(list(all_sampled_paras), k=args.num_samples):
            sampled_paraphrases = [all_paraphrases[i] for i in paraids]
            samples.append((uuid, answer, sampled_paraphrases, choices_labels, choices_texts, answer_labels))

    print(f"Total samples to generate: {len(samples)}")
    sample_dataloader = torch.utils.data.DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
    )

    for batch in tqdm(sample_dataloader, dynamic_ncols=True):
        uuids, answers, sampled_paraphrases, choices_labels, choices_texts, answer_labels = zip(*batch)
        batch_predictions = []
        batch_generations = []
        batch_templates = []
        batch_prompts = []

        if args.explicit_prompts:
            prompt, segment_metadata = dataset.construct_explicit_prompts(paraphrases=sampled_paraphrases[0])
        else:
            if args.single_para_qapair:
                prompt, segment_metadata = dataset.construct_prompts_single_para_qapair(
                    few_shot_examples, paraphrases=sampled_paraphrases[0]
                )
            else:
                prompt, segment_metadata = dataset.construct_prompts_with_paraphrases(
                    few_shot_examples, paraphrases=sampled_paraphrases[0]
                )
        
        # Generate using MyriadLAMA-specific FlexAttention
        generation = flex_generation(
            prompt, segment_metadata, max_new_tokens=max_new_tokens, modify_rope=args.modify_rope, has_bos=has_bos
        )

        # Extract prediction (first word only for MyriadLAMA)
        prediction = generation.strip().split()[0] if generation.strip() else ""
        batch_predictions.append(prediction)
        batch_generations.append(generation)
        batch_templates.append(sampled_paraphrases)
        batch_prompts.append(prompt)
        items = {
            "uuid": uuids,
            "paraphrases": sampled_paraphrases,
            "prompts": batch_prompts,
            "templates": batch_templates,
            "answers": answers,
            "prediction": batch_predictions,
            "generation": batch_generations,
        }

        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

        sample_count += len(uuids)
        if args.max_samples and sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples}), stopping generation")
            break

    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_chunk, chunks)
    try:
        df = append_lemmas(df, results)
    except Exception as e:
        print(f"❌ Lemmatization failed: {type(e).__name__}: {e}")
        set_trace()
    finally:
        df.to_feather(dump_file)
        print(f"✅ Results saved to {dump_file}")