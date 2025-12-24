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
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_PATHs
from utils import append_lemmas, init_spacy, lemmaize_chunk

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

num_parts = 8

@torch.no_grad()
def ensemble_generation(
    prompt_sets, 
    integration_method="max", 
    weights=None, 
    max_new_tokens=10,
    ensemble_method="logits", 
    multilayer=False, 
    token_mode="last",
    ensemble_layer_idx=10, 
    ensemble_alpha=1.0):

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generated = None

    prompts = [prompt[0] for prompt in prompt_sets]
    inputs = tokenizer(
        prompts, return_tensors="pt", 
        padding=True, truncation=True,
        padding_side='left', return_attention_mask=True).to(model.device)

    max_layer = len(_get_blocks(model))
    if multilayer:
        layer_indices = list(range(ensemble_layer_idx, max_layer))
    else:
        layer_indices = [ensemble_layer_idx]
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            if ensemble_method is None:
                logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, -1, :]
            elif ensemble_method == "layer_output_avg":
                logits = next_token_logits_with_weighted_layer_outavg(
                    model, inputs['input_ids'], inputs['attention_mask'],
                    layer_indices=layer_indices, alpha=ensemble_alpha, 
                    weights=None, token_mode=token_mode)
            elif ensemble_method.startswith("ffn_activation"):
                logits = next_token_logits_with_weighted_ffn_midavg(
                    model, inputs['input_ids'], inputs['attention_mask'],
                    layer_indices=layer_indices, alpha=ensemble_alpha, 
                    weights=None, token_mode=token_mode, use_max=(ensemble_method=="ffn_activation_max"))
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        if integration_method == "avg":
            avg_logits = logits.mean(dim=0)
            next_token = torch.argmax(avg_logits, dim=-1).unsqueeze(0).unsqueeze(1)
        elif integration_method == "max":    
            max_probs = logits.softmax(dim=-1).max(dim=0).values
            next_token = torch.argmax(max_probs, dim=-1).unsqueeze(0).unsqueeze(1)
        elif integration_method == "weighted_avg":
            if weights is None:
                raise ValueError("Weights must be provided for weighted_avg integration.")
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits.device)
            weights = weights / weights.sum(dim=0).unsqueeze(0)
            weighted_logits = (logits * weights.unsqueeze(-1)).sum(dim=0)
            next_token = torch.argmax(weighted_logits, dim=-1).unsqueeze(0).unsqueeze(1)
        elif integration_method == "weighted_max":
            if weights is None:
                raise ValueError("Weights must be provided for weighted_max integration.")
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits.device)
            argmax = weights.argmax(dim=0)
            max_logits = logits[argmax, torch.arange(logits.shape[1])]
            next_token = max_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(1)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        # Take the element-wise min across the two distributions
        # Append next token to input_ids for next round
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.expand(inputs["input_ids"].size(0), -1)], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].size(0), 1, device=model.device)], dim=1)

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



def sample_paraphrases_per_item(uuids, all_paraphrases, num_paraphrases, num_samples, repeat_paras=False):
    """
    Sample paraphrases using the same logic as series_ensemble.py:
    1. Generate all permutations of paraphrase indices (or repeated patterns if repeat_paras=True)
    2. Randomly sample num_samples permutations from all possible combinations
    
    Each UUID gets deterministic random sampling using uuid as seed.
    Same uuid will always produce the same sampling results, matching series_ensemble.py.
    
    Args:
        all_paraphrases: List of paraphrase lists, where all_paraphrases[i][j] is the i-th paraphrase version for the j-th item
        num_paraphrases: Number of paraphrases to select in each sample
        num_samples: Number of different paraphrase combinations to generate per uuid
        uuids: List of uuids for each item in the batch, used as random seeds
        repeat_paras: If True, repeat the same paraphrase multiple times instead of using permutations
    
    Returns:
        List of samples, where each sample is (uuid, sampled_paraphrases_list) --
    """
    batch_size = len(all_paraphrases[0])
    num_paraphrase_versions = len(all_paraphrases)
    
    all_samples = []
    
    # Process each item in the batch separately with deterministic sampling
    for item_idx in range(batch_size):
        uuid = uuids[item_idx]
        # Get all paraphrases for this item
        item_paraphrases = [all_paraphrases[i][item_idx] for i in range(num_paraphrase_versions)]
        
        # Generate all possible combinations
        all_indices = list(range(len(item_paraphrases)))
        if repeat_paras:
            # Repeat same paraphrase: [[0,0], [1,1], [2,2], ...]
            all_sampled_paras = list([[n] * num_paraphrases for n in all_indices])
        else:
            # Use permutations: all ordered selections of num_paraphrases from available paraphrases
            all_sampled_paras = itertools.permutations(all_indices, num_paraphrases)
        
        # Set random seed based on uuid to ensure deterministic sampling (same as series_ensemble.py)
        random.seed(uuid)
        
        # Sample num_samples different combinations
        all_sampled_paras_list = list(all_sampled_paras)
        sampled_combinations = random.sample(all_sampled_paras_list, k=min(num_samples, len(all_sampled_paras_list)))
        
        # For each combination, extract the actual paraphrases
        for paraids in sampled_combinations:
            sampled_paraphrases = [item_paraphrases[i] for i in paraids]
            all_samples.append((uuid, sampled_paraphrases))
    
    return all_samples


def _get_blocks(model):
    # Covers many HF causal LMs (LLaMA/Mistral/Qwen2/GPTNeoX/Falcon variations need small tweaks)
    if hasattr(model, "model") and hasattr(model.model, "layers"):      # LLaMA/Mistral/Qwen2
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"): # GPT-NeoX
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"): # GPT-2 style
        return model.transformer.h
    raise ValueError("Unsupported architecture: can't locate transformer blocks.")

def _get_ffn_out_proj(block):
    """
    Return the FFN output projection module where we can pre-hook
    to edit the *middle* FFN activations (the input to this projection).

    - LLaMA/Mistral/Qwen2: block.mlp.down_proj
    - GPT-2:              block.mlp.c_proj
    - GPT-NeoX:           block.mlp.dense_4h_to_h
    """
    mlp = getattr(block, "mlp", None)

    # Some models name it "feed_forward" or "ffn"
    if mlp is None:
        mlp = getattr(block, "feed_forward", None)
    if mlp is None:
        mlp = getattr(block, "ffn", None)

    if mlp is None:
        raise ValueError(f"Can't find MLP/FFN module inside block: {type(block)}")

    for attr in ("down_proj", "c_proj", "dense_4h_to_h", "fc2"):
        if hasattr(mlp, attr):
            return getattr(mlp, attr)

    raise ValueError(f"Unsupported FFN structure in block: {type(block)} / mlp: {type(mlp)}")


def _weighted_batch_average(x_last: torch.Tensor, weights: torch.Tensor | None):
    """
    x_last: [B, D]
    weights: [B] or [B,1] or None
    returns: [1, D] weighted mean (keepdim on batch)
    """
    if weights is None:
        return x_last.mean(dim=0, keepdim=True)

    w = weights.to(device=x_last.device, dtype=x_last.dtype)
    if w.dim() == 2 and w.size(1) == 1:
        w = w.squeeze(1)
    assert w.dim() == 1 and w.numel() == x_last.size(0), (w.shape, x_last.shape)

    denom = w.sum().clamp_min(torch.finfo(x_last.dtype).eps)
    return (x_last * w[:, None]).sum(dim=0, keepdim=True) / denom

def ensemble_transformer_layer_output(
        alpha=1, token_mode="last", weights=None):
    """
    attention_mask: [B,T] int/bool tensor (on same device)
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        if weights is not None:
            assert weights.dim() == 1 and weights.numel() == hidden_states.shape, (weights.shape, hidden_states.shape)

        if token_mode == "last":
            x_last = hidden_states[:, -1, :]
            mean_last = _weighted_batch_average(x_last, weights)  # [1, D]
            hidden_states[:, -1, :] = x_last * (1 - alpha) + mean_last * alpha
        elif token_mode == "all":
            if weights is None:
                mean_emb = hidden_states.mean(dim=0, keepdim=True)
            else:
                denom = weights.sum().clamp_min(torch.finfo(hidden_states.dtype).eps)
                mean_emb = (hidden_states * weights[:, None, None]).sum(dim=0, keepdim=True) / denom
            hidden_states = hidden_states * (1 - alpha) + mean_emb * alpha

        if rest is None:
            return hidden_states
        return (hidden_states, *rest)
    return hook

@torch.no_grad()
def next_token_logits_with_weighted_layer_outavg(
    model, input_ids, attention_mask,
    layer_indices: list[int],
    alpha: float = 1.0, 
    weights: torch.Tensor | None = None,
    token_mode: str = "last"):

    blocks = _get_blocks(model)
    assert all(0 <= idx < len(blocks) for idx in layer_indices), (layer_indices, len(blocks))

    hook = ensemble_transformer_layer_output(alpha=alpha, token_mode=token_mode, weights=weights)
    handles = []
    for idx in layer_indices:
        handles.append(blocks[idx].register_forward_hook(hook))

    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    logits = out.logits
    T = attention_mask.size(1)
    idx = torch.arange(T, device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
    last_pos = (idx * attention_mask).max(dim=1).values.long()
    next_logits = logits[torch.arange(logits.size(0), device=logits.device), last_pos]  # [B,V]
    return next_logits

def make_ffn_mid_activation_hook(
    attention_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    alpha: float = 1.0,
    token_mode: str = "last", # "last" or "all"
    use_max: bool = False
):
    """
    Returns a forward *pre*-hook that edits the input to the FFN output projection.
    - token_mode="last": only edits the last real token per sample (using attention_mask)
    - token_mode="all": edits all time positions (more aggressive)
    """
    assert token_mode in ("last", "all")

    def pre_hook(module, inputs):
        (x, *rest) = inputs  # x is the input to the projection: typically [B,T,Hff] or [T,B,Hff] depending on model
        if not torch.is_tensor(x):
            return inputs

        # Assume [B,T,D]
        B, T, D = x.shape
        am = attention_mask.to(device=x.device)
        assert am.shape[0] == B and am.shape[1] == T, (am.shape, x.shape)

        if token_mode == "all":
            # For each time step, compute weighted mean over batch and blend
            # x[:, t, :] <- (1-alpha)*x[:, t, :] + alpha*mean_t
            # mean_t: [1,D]
            # This is heavier but simple.
            for t in range(T):
                if not use_max:
                    mean_t = _weighted_batch_average(x[:, t, :], weights)  # [1,D]
                else:
                    mean_t = x[:, t, :].max(dim=0, keepdim=True).values  # [1,D]
                x[:, t, :] = x[:, t, :] * (1 - alpha) + mean_t * alpha
        else:
            # Only last real token per sample
            idx = torch.arange(T, device=x.device).unsqueeze(0).expand_as(am)  # [B,T]
            last_pos = (idx * am).max(dim=1).values.long()  # [B]

            rows = torch.arange(B, device=x.device)
            x_last = x[rows, last_pos, :]  # [B,D]
            if not use_max:
                mean_last = _weighted_batch_average(x_last, weights)
            else:
                mean_last = x_last.max(dim=0, keepdim=True).values
            x[rows, last_pos, :] = x_last * (1 - alpha) + mean_last * alpha
        # set_trace()
        return (x, *rest)

    return pre_hook

@torch.no_grad()
def next_token_logits_with_weighted_ffn_midavg(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: list[int] | tuple[int, ...],
    alpha: float = 1.0,
    weights: torch.Tensor | None = None,
    token_mode: str = "last",  # "last" or "all"
    use_max: bool = False,
):
    """
    Apply weighted batch-averaging to the FFN middle activations (input to FFN out proj)
    at the specified transformer layers, then return next-token logits at each sample's
    last real position.

    weights: None -> simple mean
             Tensor [B] (or [B,1]) -> weighted mean across batch
    alpha:   blend alpha (0=no change, 1=replace with mean)
    """
    blocks = _get_blocks(model)
    L = len(blocks)
    for li in layer_indices:
        assert 0 <= li < L, (li, L)

    pre_hook = make_ffn_mid_activation_hook(
        attention_mask=attention_mask,
        weights=weights,
        alpha=alpha,
        token_mode=token_mode,
        use_max=use_max,
    )

    handles = []
    try:
        for li in layer_indices:
            proj = _get_ffn_out_proj(blocks[li])
            handles.append(proj.register_forward_pre_hook(pre_hook))

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    logits = out.logits  # [B,T,V]
    B, T, V = logits.shape

    idx = torch.arange(T, device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
    last_pos = (idx * attention_mask).max(dim=1).values.long()  # [B]
    next_logits = logits[torch.arange(B, device=logits.device), last_pos]  # [B,V]
    return next_logits

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama"], help="Dataset to use for generating paraphrases.")    
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda).")
    parser.add_argument("--num_paraphrases", type=int, default=5, help="Number of paraphrases to use in each sample (default: 2)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of different paraphrase combinations to generate per question (default: 5)")
    parser.add_argument("--num_fewshots", type=int, default=5, help="Number of few-shot examples to use (default: 5)")
    parser.add_argument("--repeat_paras", action="store_true", help="Repeat the same paraphrase multiple times instead of using permutations")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to generate (default: None, process all)")
    
    parser.add_argument("--logits_ensemble_method", type=str, default="avg", choices=["max", "avg", "weighted_avg", "weighted_max"],
                        help="Integration method for ensemble generation")
    
    parser.add_argument("--ensemble_method", type=str, default=None, 
                        choices=["layer_output_avg", "ffn_activation_avg", "ffn_activation_max"], 
                        help="Method for ensemble internal states within Transformer layers, by either using layer outputs or FFN activations")
    parser.add_argument("--ensemble_layer", type=int, default=16, help="Transformer layer index to apply ensemble merging")
    parser.add_argument("--ensemble_alpha", type=float, default=1.0, help="alpha for ensemble merging of transformer outputs")
    parser.add_argument("--multilayer", action="store_true", help="Use only a single layer's output for ensemble (not used currently)")
    parser.add_argument("--token_mode", type=str, default="last", choices=["last", "all"], help="Token mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing output files")
    args = parser.parse_args()    

    if args.ensemble_method is None:
        assert args.multilayer is False, "multilayer option not applicable for logits ensemble"

    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model, debug=args.debug)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")
    
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    dump_file = f"{dataset.dataset_root}/myriadlama.logits.{args.logits_ensemble_method}."
    if args.repeat_paras:
        dump_file += "repeatparas."
    
    if args.ensemble_method == "layer_output_avg":
        dump_file += f"avglayer.layer{args.ensemble_layer}.alpha{int(args.ensemble_alpha*100)}.token-{args.token_mode}."
    elif args.ensemble_method == "ffn_activation_avg":
        dump_file += f"avgffn.layer{args.ensemble_layer}.alpha{int(args.ensemble_alpha*100)}.token-{args.token_mode}."
    elif args.ensemble_method == "ffn_activation_max":
        dump_file += f"maxffn.layer{args.ensemble_layer}.alpha{int(args.ensemble_alpha*100)}.token-{args.token_mode}."
    if args.multilayer:
        dump_file += "multilayer."
    if args.num_fewshots != 5:
        dump_file += f"{args.num_fewshots}fshots."
    
    dump_file += f"{args.num_samples}samples.{args.num_paraphrases}paras.feather"
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"File {dump_file} already exists, skipping generation.")
        exit(0)

    max_new_tokens = 10 if args.num_fewshots > 0 else 30    
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)

    print(f"🔄 Starting {args.logits_ensemble_method} logits ensembling to {dump_file}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, dtype="auto")

    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    if args.logits_ensemble_method.startswith("weighted_"):
        conf_df = pd.read_feather(os.path.join(dataset.dataset_root, "confidence.feather"))

    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation","correctness"])
    if args.max_samples:
        print(f"Processing maximum {args.max_samples} samples")
    
    few_shot_examples = dataset.get_few_shot_examples(k=args.num_fewshots) if args.num_fewshots > 0 else ""

    sample_count = 0
    all_samples = []
    for uuids, answers, all_paraphrases in tqdm(dataloader, desc="Preparing samples"):
        # Use sampling function to select paraphrases
        # Same uuid will always produce the same sampling results (matching series_ensemble.py)
        samples = sample_paraphrases_per_item(
            uuids=uuids,
            all_paraphrases=all_paraphrases, 
            num_paraphrases=args.num_paraphrases, 
            num_samples=args.num_samples,
            repeat_paras=args.repeat_paras
        )
        
        sample_count += len(uuids)
        for uuid, sampled_paraphrases in samples:
            idx = uuids.index(uuid)
            all_samples.append((uuid, answers[idx], sampled_paraphrases))
        
        if args.max_samples and len(all_samples) >= args.max_samples:
            all_samples = all_samples[:args.max_samples]
            break
    
    print(f"Total samples to process: {len(all_samples)}")
    
    # Process each sample
    for uuid, answer, sampled_paraphrases in tqdm(all_samples, desc="Generating"):
        all_prompts = []
        confidences = [] if args.logits_ensemble_method.startswith("weighted_") else None
        
        # For ensemble generation, treat each paraphrase as a separate prompt
        for para in sampled_paraphrases:
            if args.logits_ensemble_method.startswith("weighted_"):
                if confidences is None:
                    confidences = []
                _sdf = conf_df[conf_df["paraphrase"] == para]
                if len(_sdf) > 0:
                    confidences.append(float(_sdf["confidence"].values[0]))
                else:
                    confidences.append(1.0)  # Default confidence
            prompts = dataset.construct_prompts(few_shot_examples, [para])
            all_prompts.append(prompts)
        
        generation = ensemble_generation(
            all_prompts, 
            integration_method=args.logits_ensemble_method, 
            weights=[confidences] if confidences else None, 
            max_new_tokens=max_new_tokens, 
            ensemble_method=args.ensemble_method,
            ensemble_layer_idx=args.ensemble_layer - 1,
            ensemble_alpha=args.ensemble_alpha, 
            token_mode=args.token_mode,
            multilayer=args.multilayer)
        prediction = generation.strip().split()[0] if generation.strip() else ""
        
        items = {
            "uuid": [uuid],
            "paraphrases": [sampled_paraphrases],
            "prompts": [all_prompts],
            "answers": [answer],
            "prediction": [prediction],
            "generation": [generation],
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