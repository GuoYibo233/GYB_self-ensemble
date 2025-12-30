#!/usr/bin/env python3
"""
Baseline generation script for self-ensemble experiments.

This script generates baseline results for comparison with ensemble methods:
1. Baseline 1 (origin): Uses only the original question (attention mode baseline)
2. Baseline 2 (per_prompt): Generates with each paraphrase separately
   (second baseline for attention mode when using auto-generated prompts)

Based on generate.py but focused specifically on baseline generation.

Usage:
    # Baseline 1: Original questions only
    python baseline_generate.py --method origin --dataset webqa --model llama3.2_3b_it

    # Baseline 2: Per-prompt generation
    python baseline_generate.py --method per_prompt --dataset webqa --model llama3.2_3b_it
"""

import multiprocessing as mp
import os
import warnings
from pdb import set_trace

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_PATHs
from utils import append_lemmas, init_spacy, lemmaize_chunk, single_generation

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

nlp = None
num_parts = 8


def generate_baseline_origin(dataset, dataloader, args):
    """
    Baseline 1: Generate using only original questions.

    This is the baseline for attention-based ensemble methods.
    Uses only the original question without any paraphrases.

    Output: datasets/{dataset}/{model}/baseline_origin.feather
    """
    df = pd.DataFrame(
        columns=["uuid", "answers", "question", "prompt", "prediction", "generation"]
    )
    few_shot_context = dataset.get_few_shot_examples()
    
    max_new_tokens = 10 if args.num_fewshots > 0 else 20
    for uuids, answers, all_paraphrases in tqdm(
        dataloader, desc="Generating baseline (origin)", dynamic_ncols=True
    ):
        # Use only the original questions (paraphrase0)
        original_questions = all_paraphrases[0]
        prompts = dataset.construct_prompts(few_shot_context, original_questions)
        generations = single_generation(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        predictions = [gen.strip().split("\n")[0] for gen in generations]

        items = {
            "uuid": uuids,
            "answers": answers,
            "question": original_questions,
            "prompt": prompts,
            "prediction": predictions,
            "generation": generations,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
    return df

def generate_baseline_per_prompt(dataset, dataloader, args):
    """
    Baseline 2: Generate with each paraphrase separately.

    This is the second baseline for attention mode when using auto-generated prompts.
    Generates a response for each paraphrase independently (no ensemble).

    Output: datasets/{dataset}/{model}/baseline_per_prompt.feather
    """
    df = pd.DataFrame(
        columns=["uuid", "answers", "paraphrase", "prompt", "prediction", "generation"]
    )

    max_new_tokens = 10 if args.num_fewshots > 0 else 20

    for batch_data in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
        if flag_multi_choice:
            uuids, answers, all_paraphrases, choices_labels, choices_texts, answer_labels = batch_data
        else:
            uuids, answers, all_paraphrases = batch_data
        
        preds_in_batch = []
        prompts_in_batch = []
        paraphrases_in_batch = []
        generations_in_batch = []
        predictions_in_batch = []
        choices_labels_in_batch = []
        choices_texts_in_batch = []
        answer_labels_in_batch = []

        few_shot_context = dataset.get_few_shot_examples(k=args.num_fewshots)
        for paraphrases in all_paraphrases:
            paraphrases_in_batch.extend(paraphrases)
            if flag_multi_choice:
                prompts = dataset.construct_multi_choice_prompts(few_shot_context, paraphrases, choices_labels[0], choices_texts[0])
            else:
                prompts = dataset.construct_prompts(few_shot_context, paraphrases)
            generations = single_generation(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            predictions = [gen.strip().split("\n")[0] for gen in generations]
            prompts_in_batch.extend(prompts)
            preds_in_batch.extend(predictions)
            generations_in_batch.extend(generations)
            predictions_in_batch.extend(predictions)
            if flag_multi_choice:
                choices_labels_in_batch.extend(choices_labels)
                choices_texts_in_batch.extend(choices_texts)
                answer_labels_in_batch.extend(answer_labels)

        if flag_multi_choice:
            items = {
                "uuid": uuids * len(all_paraphrases),
                "answers": answers * len(all_paraphrases),
                "paraphrase": paraphrases_in_batch,
                "prompt": prompts_in_batch,
                "prediction": predictions_in_batch,
                "generation": generations_in_batch,
                "choices_label": choices_labels_in_batch,
                "choices_text": choices_texts_in_batch,
                "answer_label": answer_labels_in_batch,
            }
        else:
            items = {
                "uuid": uuids * len(all_paraphrases),
                "answers": answers * len(all_paraphrases),
                "paraphrase": paraphrases_in_batch,
                "prompt": prompts_in_batch,
                "prediction": predictions_in_batch,
                "generation": generations_in_batch,
            }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
    return df


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate baseline results for self-ensemble experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Generate baseline 1 (origin)
    python baseline_generate.py --method origin --dataset webqa --model llama3.2_3b_it
    
    # Generate baseline 2 (per_prompt)
    python baseline_generate.py --method per_prompt --dataset webqa --model llama3.2_3b_it
    
    # Generate both baselines
    python baseline_generate.py --method all --dataset webqa --model llama3.2_3b_it
    
    # Regenerate existing baseline
    python baseline_generate.py --method origin --dataset webqa --model llama3.2_3b_it --rewrite
        """,
    )
    parser.add_argument(
        "--method", type=str, required=True, choices=["origin", "per_prompt", "all", "ppl"], 
        help="Baseline method: 'origin' (original questions), 'per_prompt' (each paraphrase), or 'all' (both)")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Model name (default: llama3.2_3b_it)")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama", "commonsense", "mmlu", "logiqa", "hotpot"], help="Dataset: 'webqa' or 'myriadlama'")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--num_fewshots", type=int, default=5, help="Number of few-shot examples to use in prompts (default: 5)")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Regenerate baseline even if file already exists",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output",
    )

    args = parser.parse_args()

    # Load dataset
    flag_multi_choice = False
    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "commonsense":
        from dataset import CommonsenseParaphraseDataset
        dataset = CommonsenseParaphraseDataset(model_name=args.model, debug=args.debug)
        flag_multi_choice = True
    elif args.dataset == "mmlu":
        from dataset import MMLUParaphraseDataset
        dataset = MMLUParaphraseDataset(model_name=args.model, debug=args.debug)
        flag_multi_choice = True
    elif args.dataset == "logiqa":
        from dataset import LogiQAParaphraseDataset
        dataset = LogiQAParaphraseDataset(model_name=args.model, debug=args.debug)
        flag_multi_choice = True
    elif args.dataset == "hotpot":
        from dataset import HotpotDataset
        dataset = HotpotDataset(model_name=args.model, debug=args.debug)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa', 'myriadlama', 'commonsense', 'mmlu', or 'logiqa'.")
    
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)

    # Validate model
    if args.model not in MODEL_PATHs:
        raise ValueError(
            f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}."
        )

    model_path = MODEL_PATHs.get(args.model, args.model)

    print("=" * 70)
    print("Baseline Generation for Self-Ensemble Experiments")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Rewrite: {args.rewrite}")
    print()
    
    
    if args.method == "origin":
        dump_file = f"{dataset.dataset_root}/baseline_origin.{args.num_fewshots}shots.feather"
    elif args.method == "per_prompt":
        dump_file = f"{dataset.dataset_root}/baseline_per_prompt.{args.num_fewshots}shots.feather"
    elif args.method == "ppl":
        dump_file = f"{dataset.dataset_root}/baseline_ppl.{args.num_fewshots}shots.feather"
    else:  # args.method == "all"
        raise NotImplementedError("Method 'all' is not implemented in this script.")    

    if os.path.exists(dump_file) and not args.rewrite:
        print(f"File {dump_file} already exists, skipping generation.")
        print("Use --rewrite to regenerate.")
        sys.exit(0)
    print(f"🔄 Output to: {dump_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token

    if args.method == "origin":
        df = generate_baseline_origin(dataset, dataloader, args)
    elif args.method == "per_prompt":
        df = generate_baseline_per_prompt(dataset, dataloader, args)
    
    # Lemmaize predictions and answers
    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_chunk, chunks)
    df = append_lemmas(df, results)
    df.to_feather(dump_file)
    print(f"\n✅ Baseline (per_prompt) results saved to: {dump_file}")
    