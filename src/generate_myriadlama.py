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

from pdb import set_trace
import os
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.constants import MODEL_PATHs
from mask_visualization import visualize_mask_heatmap
from utils import DATASET_ROOT

# Try to import FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    FLEX_ATTENTION_AVAILABLE = True
    print("✅ FlexAttention is available")
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("⚠️  FlexAttention not available. This script requires PyTorch 2.5+ or nightly.")
    print("    Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

# ==============================================================================
# REUSED FROM generate.py - Lemmatization functions
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
# NEW - Helper functions for new prompt format
# ==============================================================================

def get_few_shot_examples_with_paraphrases(dataset, k=5, num_fs_paraphrases=3, seed=42):
    """
    Get few-shot examples formatted with multiple paraphrase questions + one answer.
    
    New format per user requirement:
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
    Construct a prompt in the new format with ALL question paraphrases.
    
    Format:
    {instruction}
    
    Q: {fs1_para1}
    Q: {fs1_para2}
    Q: {fs1_para3}
    A: {fs1_answer}
    
    Q: {fs2_para1}
    Q: {fs2_para2}
    Q: {fs2_para3}
    A: {fs2_answer}
    
    Q: {main_question_paraphrase1}
    Q: {main_question_paraphrase2}
    Q: {main_question_paraphrase3}
    A:
    
    Args:
        instruction: Instruction string
        few_shot_examples: List of {'paraphrases': [...], 'answer': ...}
        question_paraphrases: List of all main question paraphrase strings
        
    Returns:
        Single prompt string
    """
    prompt_parts = [instruction]
    
    # Add few-shot examples
    for fs_example in few_shot_examples:
        fs_parts = []
        # Add all paraphrase questions
        for para in fs_example['paraphrases']:
            fs_parts.append(f"Q: {para}")
        # Add answer
        fs_parts.append(f"A: {fs_example['answer']}")
        prompt_parts.append("\n".join(fs_parts))
    
    # Add ALL main question paraphrases
    main_q_parts = []
    for para in question_paraphrases:
        main_q_parts.append(f"Q: {para}")
    main_q_parts.append("A:")
    prompt_parts.append("\n".join(main_q_parts))
    
    return "\n\n".join(prompt_parts)


# ==============================================================================
# MODIFIED - MyriadLama-specific paraphrase concatenation with few-shot masking
# ==============================================================================

def parse_prompt_segments_with_metadata_new_format(prompt):
    """
    Parse a prompt in the NEW format into segments with metadata.
    
    NEW MyriadLAMA prompt structure:
    {instruction}
    
    Q: {fs1_para1}
    Q: {fs1_para2}
    Q: {fs1_para3}
    A: {fs1_answer}
    
    Q: {fs2_para1}
    Q: {fs2_para2}
    Q: {fs2_para3}
    A: {fs2_answer}
    
    Q: {main_question_para1}
    Q: {main_question_para2}
    Q: {main_question_para3}
    A:
    
    Returns segments with metadata to enable proper masking:
    - Each Q paraphrase in a few-shot example is a separate segment
    - The A in a few-shot example is a separate segment
    - Each main question paraphrase is a separate segment
    
    Args:
        prompt: Single prompt string (with all main question paraphrases)
        
    Returns:
        List of tuples: (segment_text, metadata_dict)
        metadata_dict contains:
            - 'type': 'instruction', 'few_shot_q', 'few_shot_a', or 'question'
            - 'paraphrase_idx': which main question paraphrase (for 'question' type)
            - 'few_shot_idx': which few-shot example (for few_shot type)
            - 'fs_q_para_idx': which paraphrase within a few-shot (for few_shot_q type)
    """
    segments = []
    
    # Split by double newline to get sections
    sections = prompt.split("\n\n")
    
    # First section is instruction
    if sections[0].strip():
        segments.append((
            sections[0].strip(),
            {
                'type': 'instruction', 
                'paraphrase_idx': None, 
                'few_shot_idx': None,
                'fs_q_para_idx': None
            }
        ))
    
    # Process remaining sections
    # Count few-shot examples vs main question
    few_shot_count = 0
    
    for section_idx, section in enumerate(sections[1:]):
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        
        # Check if this section ends with "A:" (main question) or "A: {answer}" (few-shot)
        has_answer_value = any(line.startswith("A:") and len(line) > 2 for line in lines)
        
        if has_answer_value:
            # This is a few-shot example with multiple Q paraphrases + one A
            q_lines = [line for line in lines if line.startswith("Q:")]
            a_line = [line for line in lines if line.startswith("A:")][0]
            
            # Add each Q paraphrase as a segment
            for q_para_idx, q_line in enumerate(q_lines):
                segments.append((
                    q_line.strip(),
                    {
                        'type': 'few_shot_q',
                        'paraphrase_idx': None,  # FS doesn't belong to a main para
                        'few_shot_idx': few_shot_count,
                        'fs_q_para_idx': q_para_idx
                    }
                ))
            
            # Add the answer as a segment
            segments.append((
                a_line.strip(),
                {
                    'type': 'few_shot_a',
                    'paraphrase_idx': None,
                    'few_shot_idx': few_shot_count,
                    'fs_q_para_idx': None
                }
            ))
            
            few_shot_count += 1
        else:
            # This is the main question section with MULTIPLE Q paraphrases + A:
            q_lines = [line for line in lines if line.startswith("Q:")]
            
            # Add each main question paraphrase as a segment
            for main_q_para_idx, q_line in enumerate(q_lines):
                segments.append((
                    q_line.strip(),
                    {
                        'type': 'question',
                        'paraphrase_idx': main_q_para_idx,  # Track which main Q para
                        'few_shot_idx': None,
                        'fs_q_para_idx': None
                    }
                ))
    
    return segments


def parse_prompt_segments_with_metadata(prompt, paraphrase_idx):
    """
    Parse a prompt into segments with metadata for proper masking.
    
    A MyriadLAMA prompt has the structure:
    {instruction}
    
    {few-shot example 1: Q: ... A: ...}
    
    {few-shot example 2: Q: ... A: ...}
    ...
    
    Q: {question}
    A:
    
    Returns segments with metadata to enable proper masking:
    - Paraphrases of same few-shot example cannot attend to each other
    - Paraphrases from different few-shot can attend to each other
    - Answer parts have normal causal mask
    - Question paraphrases are isolated
    
    Args:
        prompt: Single prompt string
        paraphrase_idx: Which paraphrase this prompt represents (0, 1, 2, ...)
        
    Returns:
        List of tuples: (segment_text, metadata_dict)
        metadata_dict contains:
            - 'type': 'instruction', 'few_shot_q', 'few_shot_a', or 'question'
            - 'paraphrase_idx': which paraphrase this belongs to
            - 'few_shot_idx': which few-shot example (for few_shot type)
    """
    segments = []
    
    # Split by Q: to find all Q-A pairs
    parts = prompt.split("Q: ")
    
    # First part is the instruction (before any Q:)
    if parts[0].strip():
        segments.append((
            parts[0].strip(),
            {'type': 'instruction', 'paraphrase_idx': paraphrase_idx, 'few_shot_idx': None}
        ))
    
    # Process Q-A pairs
    few_shot_count = 0
    for i, part in enumerate(parts[1:]):
        # Each part starts after "Q: " and may contain "A: "
        if "A:" in part:
            # This is a Q-A pair (few-shot example or the question with answer prompt)
            # Split by "A:" to separate question and answer
            q_and_a = part.split("A:", 1)
            question_text = q_and_a[0].strip()
            answer_text = q_and_a[1].strip() if len(q_and_a) > 1 else ""
            
            # Check if this is a few-shot example (has non-empty answer) or the final question
            is_few_shot = (i < len(parts[1:]) - 1) or (answer_text and answer_text != "")
            
            if is_few_shot:
                # This is a few-shot example - split into question and answer segments
                segments.append((
                    f"Q: {question_text}".strip(),
                    {'type': 'few_shot_q', 'paraphrase_idx': paraphrase_idx, 'few_shot_idx': few_shot_count}
                ))
                segments.append((
                    f"A: {answer_text}".strip(),
                    {'type': 'few_shot_a', 'paraphrase_idx': paraphrase_idx, 'few_shot_idx': few_shot_count}
                ))
                few_shot_count += 1
            else:
                # This is the actual question (final Q with empty A:)
                segments.append((
                    f"Q: {question_text}\nA:".strip(),
                    {'type': 'question', 'paraphrase_idx': paraphrase_idx, 'few_shot_idx': None}
                ))
        else:
            # Just a question without "A:" (shouldn't happen in normal prompts)
            segments.append((
                f"Q: {part}".strip(),
                {'type': 'question', 'paraphrase_idx': paraphrase_idx, 'few_shot_idx': None}
            ))
    
    return segments

def concatenate_paraphrases_with_positions(prompt, tokenizer, separator="\n\n"):
    """
    Process a SINGLE prompt with ALL paraphrases and segment-level position tracking for MyriadLAMA.
    
    Modified for MyriadLAMA with complex few-shot masking (NEW FORMAT):
    - Parses the prompt to identify instruction, few-shot Q/A pairs (with multi-para Qs), and main Q paras
    - Tracks metadata: paraphrase index, few-shot index, segment type, fs_q_para_idx
    - Enables masking where:
      * Paraphrases of same few-shot cannot attend to each other
      * Paraphrases from different few-shot can attend to each other
      * Answer parts have normal causal mask
      * Main question paraphrases are isolated
    
    Args:
        prompt: Single prompt string with ALL paraphrases
        tokenizer: HuggingFace tokenizer
        separator: Separator token between segments (default: double newline)
        
    Returns:
        concatenated_text: Single concatenated string
        segment_positions: List of (start, end) tuples for each segment
        segment_metadata: List of metadata dicts for each segment
        total_length: Total number of tokens
    """
    # Parse the prompt to extract segments with metadata (using NEW format parser)
    segments_with_meta = parse_prompt_segments_with_metadata_new_format(prompt)
    
    all_segments = []
    all_metadata = []
    
    for seg_text, meta in segments_with_meta:
        all_segments.append(seg_text)
        all_metadata.append(meta)
    
    # Tokenize each segment individually
    tokenized_segments = []
    sep_tokens = tokenizer.encode(separator, add_special_tokens=False)
    
    for segment in all_segments:
        tokens = tokenizer.encode(segment, add_special_tokens=False)
        tokenized_segments.append(tokens)
    
    # Build full token sequence and track positions
    full_tokens = []
    segment_positions = []
    current_pos = 0
    
    for i, tokens in enumerate(tokenized_segments):
        if i > 0:
            # Add separator
            full_tokens.extend(sep_tokens)
            current_pos += len(sep_tokens)
        
        start_pos = current_pos
        full_tokens.extend(tokens)
        current_pos += len(tokens)
        end_pos = current_pos
        
        segment_positions.append((start_pos, end_pos))
    
    # Decode back to text
    concatenated_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    
    return concatenated_text, segment_positions, all_metadata, len(full_tokens)

# ==============================================================================
# MODIFIED - MyriadLama-specific mask creation
# ==============================================================================

def create_myriadlama_mask(segment_positions, segment_metadata, original_length):
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
        original_length: Length of the original concatenated sequence
        
    Returns:
        mask_mod: Function (b, h, q_idx, kv_idx) -> Tensor[bool]
    """
    import torch
    
    # Convert segment positions to tensors
    segment_starts = torch.tensor([start for start, _ in segment_positions], dtype=torch.int64)
    segment_ends = torch.tensor([end for _, end in segment_positions], dtype=torch.int64)
    num_segments = len(segment_positions)
    
    def mask_mod(b, h, q_idx, kv_idx):
        """
        Mask function for MyriadLAMA FlexAttention with complex rules.
        
        IMPORTANT: Must use only tensor operations (no .item() or Python if on tensors)
        to avoid vmap compilation errors.
        
        Logic (PRIORITY ORDER):
        1. HIGHEST PRIORITY: Causal constraint (cannot attend to future)
        2. Generated tokens (>= original_length) attend to all previous tokens
        3. Within encoding phase, apply complex rules based on segment types
        """
        # Move segment tensors to same device as indices
        device = q_idx.device
        seg_starts = segment_starts.to(device)
        seg_ends = segment_ends.to(device)
        
        # HIGHEST PRIORITY: Causal constraint - cannot attend to future
        causal_mask = q_idx >= kv_idx
        
        # If query is in generation phase, allow attention to all previous tokens (with causal)
        is_generated = q_idx >= original_length
        
        # Find which segment the query and key belong to
        q_in_segment = (q_idx >= seg_starts) & (q_idx < seg_ends)
        kv_in_segment = (kv_idx >= seg_starts) & (kv_idx < seg_ends)
        
        # Check if both are in same segment
        same_segment = (q_in_segment & kv_in_segment).any()
        
        # For complex rules, we need to build a mask using tensor operations
        # Strategy: Build up conditions as tensors, combine with logical ops
        
        # Create metadata lookup tensors (convert segment_metadata to tensors)
        # This avoids .item() calls and Python control flow
        num_segs = len(segment_metadata)
        
        # Build metadata tensors for all segments
        types_list = []
        paras_list = []
        fs_list = []
        fs_q_para_list = []
        
        for seg_meta in segment_metadata:
            # Map types to integers for tensor operations
            type_map = {'instruction': 0, 'few_shot_q': 1, 'few_shot_a': 2, 'question': 3}
            types_list.append(type_map.get(seg_meta['type'], -1))
            paras_list.append(seg_meta['paraphrase_idx'] if seg_meta['paraphrase_idx'] is not None else -1)
            fs_list.append(seg_meta['few_shot_idx'] if seg_meta['few_shot_idx'] is not None else -1)
            fs_q_para_list.append(seg_meta.get('fs_q_para_idx', -1) if seg_meta.get('fs_q_para_idx') is not None else -1)
        
        seg_types = torch.tensor(types_list, dtype=torch.int64, device=device)
        seg_paras = torch.tensor(paras_list, dtype=torch.int64, device=device)
        seg_fs = torch.tensor(fs_list, dtype=torch.int64, device=device)
        seg_fs_q_para = torch.tensor(fs_q_para_list, dtype=torch.int64, device=device)
        
        # Get query and kv segment metadata using tensor indexing
        # For each position, find which segment it belongs to using argmax on q_in_segment
        # If no segment found, will be index 0 but we'll handle with validity check
        q_seg_idx = torch.argmax(q_in_segment.to(torch.int64))
        kv_seg_idx = torch.argmax(kv_in_segment.to(torch.int64))
        
        # Check if indices are valid (actually found a segment)
        q_valid = q_in_segment.any()
        kv_valid = kv_in_segment.any()
        both_valid = q_valid & kv_valid
        
        # Get metadata for query and kv segments
        q_type = seg_types[q_seg_idx]
        kv_type = seg_types[kv_seg_idx]
        q_para = seg_paras[q_seg_idx]
        kv_para = seg_paras[kv_seg_idx]
        q_fs = seg_fs[q_seg_idx]
        kv_fs = seg_fs[kv_seg_idx]
        q_fs_q_para_idx = seg_fs_q_para[q_seg_idx]
        kv_fs_q_para_idx = seg_fs_q_para[kv_seg_idx]
        
        # Build up the masking logic using pure tensor operations
        # Type constants for comparison
        INST = torch.tensor(0, device=device, dtype=torch.int64)
        FS_Q = torch.tensor(1, device=device, dtype=torch.int64)
        FS_A = torch.tensor(2, device=device, dtype=torch.int64)
        QUES = torch.tensor(3, device=device, dtype=torch.int64)
        
        # Rule 1: Instruction <-> Instruction
        inst_to_inst = (q_type == INST) & (kv_type == INST)
        
        # Rule 2: Any -> Instruction
        any_to_inst = (kv_type == INST)
        
        # Rule 3: Few-shot answer rules
        fs_a_to_own_qs = (q_type == FS_A) & (kv_type == FS_Q) & (q_para == kv_para) & (q_fs == kv_fs)
        fs_a_to_diff_fs_q = (q_type == FS_A) & (kv_type == FS_Q) & (q_fs != kv_fs)
        fs_a_to_diff_fs_a = (q_type == FS_A) & (kv_type == FS_A) & (q_fs != kv_fs)
        fs_a_to_same_seg_a = (q_type == FS_A) & (kv_type == FS_A) & same_segment
        
        # Rule 4: Few-shot question paraphrase rules
        fs_q_same_all = (q_type == FS_Q) & (kv_type == FS_Q) & (q_fs == kv_fs) & (q_para == kv_para) & (q_fs_q_para_idx == kv_fs_q_para_idx)
        fs_q_diff_fs = (q_type == FS_Q) & (kv_type == FS_Q) & (q_fs != kv_fs)
        
        # Rule 5: Question paraphrase rules
        ques_same_para = (q_type == QUES) & (kv_type == QUES) & (q_para == kv_para)
        
        # Rule 6: Question -> Few-shot
        ques_to_fs = (q_type == QUES) & ((kv_type == FS_Q) | (kv_type == FS_A))
        
        # Combine all allow rules
        custom_allow = (
            inst_to_inst |
            any_to_inst |
            fs_a_to_own_qs |
            fs_a_to_diff_fs_q |
            fs_a_to_diff_fs_a |
            fs_a_to_same_seg_a |
            fs_q_same_all |
            fs_q_diff_fs |
            ques_same_para |
            ques_to_fs |
            same_segment  # Default: same segment allowed
        )
        
        # Final result: causal AND (generated OR (valid custom_allow))
        # If query is generated, allow all causal attention
        # Otherwise, only allow if both positions are valid AND custom rules allow
        result = causal_mask & (is_generated | (both_valid & custom_allow))
        
        return result
    
    return mask_mod

# ==============================================================================
# REUSED FROM flex_attention_generate.py - Attention wrapper
# ==============================================================================

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
    
    def create_patched_forward(self, layer_idx, original_attn):
        """Create a patched forward function for an attention layer."""
        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs
        ):
            # If no custom mask or sequence is too short, use original
            bsz, q_len, _ = hidden_states.size()
            if self.current_mask_mod is None or q_len == 1:
                return self.original_forwards[layer_idx](
                    hidden_states, position_embeddings, attention_mask,
                    past_key_value, cache_position, **kwargs
                )
            
            # Extract position embeddings
            cos, sin = position_embeddings
            
            # Compute Q, K, V projections
            query_states = original_attn.q_proj(hidden_states)
            key_states = original_attn.k_proj(hidden_states)
            value_states = original_attn.v_proj(hidden_states)
            
            # Reshape to multi-head format
            num_heads = original_attn.config.num_attention_heads
            num_key_value_heads = original_attn.config.num_key_value_heads
            head_dim = original_attn.head_dim
            
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            
            # Apply rotary position embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            
            # Expand key and value states for GQA
            if num_key_value_heads != num_heads:
                key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)
                value_states = value_states.repeat_interleave(num_heads // num_key_value_heads, dim=1)
            
            # Create block mask and use FlexAttention
            try:
                block_mask = create_block_mask(
                    self.current_mask_mod,
                    B=bsz,
                    H=num_heads,
                    Q_LEN=q_len,
                    KV_LEN=q_len,
                    device=query_states.device
                )
                
                attn_output = flex_attention(
                    query_states,
                    key_states,
                    value_states,
                    block_mask=block_mask
                )
            except Exception as e:
                # Fallback to standard SDPA
                import traceback
                print(f"⚠️  FlexAttention failed in layer {layer_idx}: {type(e).__name__}: {e}")
                print(f"    Falling back to standard attention")
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True
                )
            
            # Reshape output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            
            # Output projection
            attn_output = original_attn.o_proj(attn_output)
            
            return attn_output, attn_output
        
        return patched_forward
    
    def patch_model(self, mask_mod):
        """Patch all attention layers with FlexAttention."""
        if self.is_patched:
            self.unpatch_model()
        
        self.current_mask_mod = mask_mod
        
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
        self.is_patched = False

# ==============================================================================
# MODIFIED - MyriadLama-specific generation function
# ==============================================================================

@torch.no_grad()
def myriadlama_flex_generation(prompt, max_new_tokens=10):
    """
    Generate text using FlexAttention for MyriadLAMA.
    
    Modified for MyriadLAMA (NEW FORMAT):
    - Accepts a SINGLE prompt with ALL paraphrases
    - Shorter max_new_tokens (10 instead of 20) for one-word answers
    - All paraphrases are treated equally (all are manually generated)
    - Each paraphrase segment is isolated during encoding
    - Optimized for fill-in-the-blank task
    
    Args:
        prompt: Single prompt string with ALL paraphrases
        max_new_tokens: Maximum tokens to generate (default: 10 for one-word answers)
        
    Returns:
        Generated text string
    """
    # Set model config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # Process prompt with position tracking and metadata
    concatenated_text, segment_positions, segment_metadata, original_length = \
        concatenate_paraphrases_with_positions(prompt, tokenizer)
    
    # Tokenize input
    inputs = tokenizer(
        concatenated_text,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True
    ).to(model.device)
    
    # Create FlexAttention wrapper
    flex_wrapper = FlexAttentionWrapper(model)
    
    generated = None
    
    # Generation loop
    for step in range(max_new_tokens):
        current_length = inputs["input_ids"].shape[1]
        
        # Create mask for MyriadLAMA with complex few-shot paraphrase rules
        mask_mod = create_myriadlama_mask(
            segment_positions,
            segment_metadata,
            original_length
        )
        
        # Patch model with FlexAttention
        flex_wrapper.patch_model(mask_mod)
        
        try:
            # Forward pass
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        except Exception as e:
            import traceback
            print(f"⚠️  Generation step {step} failed: {type(e).__name__}: {e}")
            print(f"    Traceback:")
            traceback.print_exc()
            print(f"    Falling back to unpatched model...")
            flex_wrapper.unpatch_model()
            logits = model(inputs["input_ids"]).logits[:, -1, :]
        finally:
            # Always unpatch after each step
            flex_wrapper.unpatch_model()
        
        # Token selection
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        
        # Debug: show what was generated
        decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
        print(f"  Step {step}: generated token '{decoded_token}' (id: {next_token.item()})")
        
        # Check for EOS or newline (likely end of one-word answer)
        if next_token.item() == tokenizer.eos_token_id:
            print(f"  Stopped: EOS token")
            break
        # Also check if we generated a newline or space (end of word)
        if '\n' in decoded_token and step > 0:  # Allow at least one token
            print(f"  Stopped: newline detected")
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
    
    parser = argparse.ArgumentParser(
        description="MyriadLAMA-specific FlexAttention generation"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2_3b_it",
        help="Model name from constants.MODEL_PATHs"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for model (default: auto)"
    )
    parser.add_argument(
        "--lemmaize", action="store_true",
        help="Normalize predictions and answers to lemmas"
    )
    parser.add_argument(
        "--num_paraphrases", type=int, default=5,
        help="Number of paraphrases to use (same for main question and few-shot examples, default: 5)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to generate (default: None, process all)"
    )
    args = parser.parse_args()
    
    # Check FlexAttention availability
    if not FLEX_ATTENTION_AVAILABLE:
        print("❌ FlexAttention is required for this script.")
        print("   Please install PyTorch 2.5+ or nightly:")
        print("   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")
        exit(1)
    
    # Load MyriadLAMA dataset
    from core.dataset import MyriadLamaDataset
    dataset = MyriadLamaDataset(model_name=args.model)
    
    # Use batch_size=1 for sequential processing
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    
    if args.model not in MODEL_PATHs:
        raise ValueError(
            f"Model {args.model} not supported. "
            f"Choose from {list(MODEL_PATHs.keys())}"
        )
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    # Output file setup
    local_output_dir = f"{DATASET_ROOT}/myriadlama/{args.model}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Determine file name based on number of paraphrases
    dump_file = f"{local_output_dir}/myriadlama_flex_{args.num_paraphrases}paras.feather"
    
    print(f"Output file: {dump_file}")
    
    # Lemmatization mode
    if args.lemmaize:
        assert os.path.exists(dump_file), \
            f"File {dump_file} does not exist. Run without --lemmaize first."
        
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
    
    if os.path.exists(dump_file):
        print(f"File {dump_file} already exists, skipping generation.")
        exit(0)
    
    # Model loading
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=args.device, torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Print model info
    print(f"🔍 Model: {args.model}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Using MyriadLAMA-specific FlexAttention")
    print(f"   Using {args.num_paraphrases} paraphrases per question")
    if hasattr(model.config, 'num_attention_heads'):
        print(f"   Attention heads: {model.config.num_attention_heads}")
    
    # DataFrame initialization
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation", "templates"])
    print(f"\nMyriadLAMA FlexAttention generation")
    if args.max_samples:
        print(f"Processing maximum {args.max_samples} samples")
    
    # Get few-shot examples with multiple paraphrases (new format)
    # Use same number of paraphrases for few-shot as for main question
    few_shot_examples = get_few_shot_examples_with_paraphrases(
        dataset, 
        k=5, 
        num_fs_paraphrases=args.num_paraphrases,
        seed=42
    )
    instruction = dataset.instruction
    
    # Main generation loop
    sample_count = 0
    for uuids, answers, all_paraphrases in tqdm(dataloader):
        batch_predictions = []
        batch_generations = []
        batch_templates = []
        
        # Process each question in batch
        for i, paraphrases in enumerate(zip(*all_paraphrases)):
            # All paraphrases in MyriadLAMA are manually generated
            # Simply select the first N paraphrases
            all_templates = list(paraphrases)
            selected_templates = all_templates[:args.num_paraphrases]
            
            # Construct ONE prompt with ALL question paraphrases (NEW FORMAT)
            # Prompt has: instruction + few-shot examples + ALL main question paraphrases
            prompt = construct_prompt_new_format(
                instruction,
                few_shot_examples,
                selected_templates  # Pass ALL paraphrases, not just one
            )
            
            # Generate using MyriadLAMA-specific FlexAttention
            generation = myriadlama_flex_generation(
                prompt,  # Single prompt with all paraphrases
                max_new_tokens=10
            )
            
            # Extract prediction (first word only for MyriadLAMA)
            prediction = generation.strip().split()[0] if generation.strip() else ""
            
            batch_predictions.append(prediction)
            batch_generations.append(generation)
            batch_templates.append(selected_templates)
        
        # Store results
        items = {
            "uuid": uuids,
            "templates": batch_templates,
            "answers": answers,
            "prediction": batch_predictions,
            "generation": batch_generations,
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
