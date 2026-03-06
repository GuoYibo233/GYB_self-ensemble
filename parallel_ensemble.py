import itertools
import multiprocessing as mp
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import get_dataset_instance
from utils import (
    append_lemmas,
    get_label_prob,
    init_spacy,
    lemmaize_chunk,
    load_model_tokenizer,
    load_tokenizer,
)

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

num_parts = 8

@torch.no_grad()
def ensemble_generation(
    model,
    tokenizer,
    prompts,
    integration_method="max",
    weights=None,
    max_new_tokens=32,
    choice_labels=None):

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generated = None
    past_key_values = None
    inputs = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True,
        padding_side='left', return_attention_mask=True).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    current_model_input = input_ids

    label_probs = None
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_model_input,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
        
        if integration_method == "avg":
            logits = logits.mean(dim=0)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(1)
        elif integration_method == "max":    
            logits = logits.max(dim=0)
            max_probs = logits.softmax(dim=-1).max(dim=0).values
            next_token = torch.argmax(max_probs, dim=-1).unsqueeze(0).unsqueeze(1)
        elif integration_method == "weighted_avg":
            if weights is None:
                raise ValueError("Weights must be provided for weighted_avg integration.")
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits.device)
            weights = weights / weights.sum(dim=0).unsqueeze(0)
            logits = (logits * weights.unsqueeze(-1)).sum(dim=0)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(1)
        elif integration_method == "weighted_max":
            if weights is None:
                raise ValueError("Weights must be provided for weighted_max integration.")
            weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits.device)
            argmax = weights.argmax(dim=0)
            logits = logits[argmax, torch.arange(logits.shape[1])]
            next_token = logits.argmax(dim=-1).unsqueeze(0).unsqueeze(1)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
        
        if step == 0 and choice_labels is not None:
            label_probs = get_label_prob(tokenizer, logits, choice_labels)
        
        # inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.expand(inputs["input_ids"].size(0), -1)], dim=1)
        # inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].size(0), 1, device=model.device)], dim=1)
        input_ids = torch.cat([input_ids, next_token.expand(input_ids.size(0), -1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), 1, device=model.device)], dim=1)
        
        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        
        current_model_input = next_token.repeat(input_ids.size(0), 1)
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
    return generated_texts[0].strip(), label_probs

def sample_paraphrases_per_item(
        uuid, 
        paraphrases, 
        is_origs,
        messages, 
        num_paraphrases, 
        num_samples, 
        repeat_paras=False):
    """
    Sample paraphrases using the same logic as series_ensemble.py:
    1. Generate all permutations of paraphrase indices (or repeated patterns if repeat_paras=True)
    2. Randomly sample num_samples permutations from all possible combinations
    
    Each UUID gets deterministic random sampling using uuid as seed.
    Same uuid will always produce the same sampling results, matching series_ensemble.py.
    
    Args:
        paraphrases: List of paraphrase lists, where paraphrases[i][j] is the i-th paraphrase version for the j-th item
        num_paraphrases: Number of paraphrases to select in each sample
        num_samples: Number of different paraphrase combinations to generate per uuid
        uuid: UUID for the current batch, used as random seed
        repeat_paras: If True, repeat the same paraphrase multiple times instead of using permutations
    
    Returns:
        List of samples, where each sample is (uuid, sampled_paraphrases_list) --
    """
    all_samples = []    
    # Handle special case: use all paraphrases
    if num_paraphrases == -1: 
        return (uuid, paraphrases, is_origs, messages)
    
    # Generate all possible combinations
    all_indices = list(range(len(paraphrases)))
    effective_num_paraphrases = num_paraphrases if num_paraphrases <= len(all_indices) else len(all_indices)
    if repeat_paras: # Repeat same paraphrase: [[0,0], [1,1], [2,2], ...]
        all_sampled_paras = list([[n] * effective_num_paraphrases for n in all_indices])
    else: # Use permutations: all ordered selections of num_paraphrases from available paraphrases
        all_sampled_paras = itertools.permutations(all_indices, effective_num_paraphrases)

    random.seed(uuid)
    all_sampled_paras_list = list(all_sampled_paras)
    sampled_combinations = random.sample(all_sampled_paras_list, k=min(num_samples, len(all_sampled_paras_list)))
    for paraids in sampled_combinations:
        sampled_paraphrases = [paraphrases[i] for i in paraids]
        sammpled_is_origs = [is_origs[i] for i in paraids]
        sampled_messages = [messages[i] for i in paraids]
        all_samples.append((uuid, sampled_paraphrases, sammpled_is_origs, sampled_messages))
    
    return all_samples



def get_parallel_ensemble_dumpfile(dataset, args):
    dump_file = f"{dataset.dataset_root}/parallel.{args.logits_ensemble_method}."
    dump_file += f"{args.num_samples}samples.{args.num_paraphrases}paras.feather"
    return dump_file

def craft_prompts_from_baseline_file(baseline_file, thinking):
    df = pd.read_feather(baseline_file)
    for uuid, subdf in df.groupby('uuid'):
        subdf = subdf.reset_index(drop=True)
        if thinking:
            subdf = subdf[subdf['thinking'].str.len() > 0]
        if len(subdf) == 0:
            print(f"⚠️  Warning: No valid thinking entries for uuid {uuid}, skipping.")
            continue

        
        if thinking:
            messages = [
                (paraphrase, is_orig, f"{prompt}{thinking}")
                for paraphrase, is_orig, prompt, thinking in zip(
                    subdf["paraphrase"].tolist(), 
                    subdf["is_origs"].tolist(), 
                    subdf["prompt"].tolist(), 
                    subdf["thinking"].tolist())
                if thinking.strip().endswith("</think>")
            ]
            paraphrases, is_origs, messages = zip(*messages) if messages else ([], [], [])
        else:
            paraphrases = subdf["paraphrase"].tolist()
            is_origs = subdf["is_orig"].tolist()
            messages = subdf["prompt"].tolist()
        
        yield (
            uuid, 
            subdf['answers'].tolist()[0].tolist(),
            paraphrases,
            is_origs,
            messages
        ) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    # General parameters
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama", "commonsense", "mmlu", "logiqa", "hotpot"], help="Dataset to use for generating paraphrases.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda).")    
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite existing output files")
    
    # Prompts sampling/construction parameters
    parser.add_argument("--baseline_file", type=str, required=True, help="Path to baseline generations file (required for reasoning mode)")
    parser.add_argument("--additional_paraphrases_file", type=str, default=None, help="Path to additional paraphrases file (for datasets that support it)")
    parser.add_argument("--repeat_paras", action="store_true", help="Whether to repeat the same paraphrase multiple times instead of using permutations (for small number of paraphrases)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of different paraphrase combinations to generate per question (default: 5)")
    parser.add_argument("--num_paraphrases", type=int, default=-1, help="Number of paraphrases to use in each sample (default: 2)")    

    # Ensemble parameters
    parser.add_argument("--logits_ensemble_method", type=str, default="avg",
                        choices=["max", "avg", "weighted_avg", "weighted_max"],
                        help="Integration method for ensemble generation")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    
    args = parser.parse_args()    

    # Load dataset
    dataset = get_dataset_instance(
        dataset_name=args.dataset,
        model_name=args.model,
        debug=args.debug,
        thinking=args.thinking,
        additional_paraphrases_file=args.additional_paraphrases_file,
    )

    # Load model and tokenizer
    tokenizer = load_tokenizer(args.model)
    max_new_tokens = 32
    
    group_by_uuid = craft_prompts_from_baseline_file(args.baseline_file, args.thinking)
    
    # Determine dump file path
    dataset.dataset_root = os.path.dirname(args.baseline_file)
    os.makedirs(dataset.dataset_root, exist_ok=True)
    dump_file = get_parallel_ensemble_dumpfile(dataset, args)
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"✅ File {dump_file} already exists, skipping generation.")
        exit(0)

    model, tokenizer = load_model_tokenizer(args.model)
    print(f"🔄 Starting {args.logits_ensemble_method} logits ensembling to {dump_file}")
    if args.logits_ensemble_method.startswith("weighted_"):
        conf_df = pd.read_feather(os.path.join(dataset.dataset_root, "confidence.feather"))
    
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation", "prompt", "paraphrases", "is_orig"])
    
    all_samples = []
    uuid_count = 0
    for batch_data in tqdm(group_by_uuid, desc="Preparing samples", dynamic_ncols=True):
        if dataset.is_multi_choice:
            uuid, answers, paraphrases, choices_labels, choices_texts, answer_labels, is_origs = batch_data
        else:
            uuid, answers, paraphrases, is_origs, messages = batch_data
            choices_labels = None
            choices_texts = None
            answer_labels = None

        samples = sample_paraphrases_per_item(
            uuid=uuid,
            paraphrases=paraphrases, 
            is_origs=is_origs,
            messages=messages,
            num_paraphrases=args.num_paraphrases, 
            num_samples=args.num_samples,
            repeat_paras=args.repeat_paras
        )
        
        for uuid, sampled_paraphrases, sampled_is_origs, sampled_messages in samples:
            if dataset.is_multi_choice:
                all_samples.append((
                    uuid, answers, 
                    sampled_paraphrases, sampled_is_origs, sampled_messages,
                    choices_labels, choices_texts, answer_labels))
            else:
                all_samples.append((
                    uuid, answers, 
                    sampled_paraphrases, sampled_is_origs, sampled_messages, 
                    None, None, None))
                
    print(f"Total samples to process: {len(all_samples)}")
    
    # Process each sample
    for sample_data in tqdm(all_samples, desc="Generating", dynamic_ncols=True):
        if dataset.is_multi_choice:
            uuid, answer, sampled_paraphrases, sampled_is_origs, sampled_messages, choices_label, choices_text, answer_label = sample_data
        else:
            uuid, answer, sampled_paraphrases, sampled_is_origs, sampled_messages, _, _, _ = sample_data
            choices_label = None
            choices_text = None
            answer_label = None
        
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
        
        generation, label_probs = ensemble_generation(
            model,
            tokenizer,
            prompts=sampled_messages,
            integration_method=args.logits_ensemble_method,
            weights=[confidences] if confidences else None,
            max_new_tokens=max_new_tokens,
            choice_labels=dataset.choice_labels)
        
        labels, label_probs = zip(*label_probs) if label_probs else ([], [])
        
        # Extract prediction - for multi-choice, extract first capital letter
        if dataset.is_multi_choice:
            import re
            match = re.search(r'[A-E]', generation.strip())
            prediction = match.group(0) if match else ""
        else:
            prediction = generation.strip().split()[0] if generation.strip() else ""
        
        items = {
            "uuid": [uuid],
            "paraphrases": [sampled_paraphrases],
            "is_orig": [sampled_is_origs],
            "prompts": [sampled_messages],
            "answers": [answer],
            "prediction": [prediction],
            "generation": [generation],
            "labels": [labels], 
            "label_probs": [label_probs],
        }
        
        # Add multi-choice specific fields
        if dataset.is_multi_choice:
            items["choices_label"] = [choices_label]
            items["choices_text"] = [choices_text]
            items["answer_label"] = [answer_label]
        
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_chunk, chunks)
    df = append_lemmas(df, results)
    df.to_feather(dump_file)
    print(f"✅ Parallel ensemble results saved to {dump_file}")