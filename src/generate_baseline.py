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

import os
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM

from core.constants import MODEL_PATHs
from utils import init_spacy, lemmaize_chunk, append_lemmas, single_generation

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

nlp = None
num_parts = 8

def init_spacy():
    global nlp
    nlp = spacy.load("en_core_web_lg")

def lemmaize_predicts(predict):
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    predict_lemmas = []
    answer_lemmas = []
    for prediction, answers in tqdm(zip(chunk["prediction"], chunk["answers"]), total=len(chunk)):
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
    return predict_lemmas, answer_lemmas

def append_lemmas(df, results):
    all_predict_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df

def generate_baseline_origin(dataset, dataloader, model_path, args):
    """
    Baseline 1: Generate using only original questions.
    
    This is the baseline for attention-based ensemble methods.
    Uses only the original question without any paraphrases.
    
    Output: datasets/{dataset}/{model}/baseline_origin.feather
    """
    dump_file = f"{dataset.dataset_root}/baseline_origin.feather"
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"File {dump_file} already exists, skipping generation.")
        print("Use --rewrite to regenerate.")
        return dump_file

    print("\n" + "="*70)
    print("Baseline 1: Origin (Attention Mode Baseline)")
    print("="*70)
    print(f"Method: Uses only original questions (no paraphrases)")
    print(f"Output: {dump_file}")
    print()

    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.DataFrame(columns=["uuid", "answers", "question", "prompt", "prediction", "generation"])
    few_shot_context = dataset.get_few_shot_examples()

    for uuids, answers, all_paraphrases in tqdm(dataloader, desc="Generating baseline (origin)"):
        # Use only the original questions (paraphrase0)
        original_questions = all_paraphrases[0]
        prompts = dataset.construct_prompts(few_shot_context, original_questions)
        generations = single_generation(model, tokenizer, prompts)
        predictions = [gen.strip().split('\n')[0] for gen in generations]

        items = {
            "uuid": uuids,
            "answers": answers,
            "question": original_questions,
            "prompt": prompts,
            "prediction": predictions,
            "generation": generations,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

    # Lemmaize predictions and answers
    print("\nLemmatizing predictions and answers...")
    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_chunk, chunks)
    df = append_lemmas(df, results)

    df.to_feather(dump_file)
    print(f"\n✅ Baseline 1 (origin) results saved to: {dump_file}")
    print(f"   Total samples: {len(df)}")
    return dump_file


def generate_baseline_per_prompt(dataset, dataloader, model_path, args):
    """
    Baseline 2: Generate with each paraphrase separately.
    
    This is the second baseline for attention mode when using auto-generated prompts.
    Generates a response for each paraphrase independently (no ensemble).
    
    Output: datasets/{dataset}/{model}/baseline_per_prompt.feather
    """
    dump_file = f"{dataset.dataset_root}/baseline_per_prompt.feather"
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"File {dump_file} already exists, skipping generation.")
        print("Use --rewrite to regenerate.")
        return dump_file

    print("\n" + "="*70)
    print("Baseline 2: Per-Prompt (Attention Mode Second Baseline)")
    print("="*70)
    print(f"Method: Generate with each paraphrase separately")
    print(f"Output: {dump_file}")
    print()

    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=args.device, torch_dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.DataFrame(columns=["uuid", "answers", "paraphrase", "prompt", "prediction", "generation"])

    for uuids, answers, all_paraphrases in tqdm(dataloader, desc="Generating baseline (per_prompt)"):
        preds_in_batch = []
        prompts_in_batch = []
        paraphrases_in_batch = []
        generations_in_batch = []
        predictions_in_batch = []

        few_shot_context = dataset.get_few_shot_examples()
        for paraphrases in all_paraphrases:
            paraphrases_in_batch.extend(paraphrases)
            prompts = dataset.construct_prompts(few_shot_context, paraphrases)
            generations = single_generation(model, tokenizer, prompts)
            predictions = [gen.strip().split('\n')[0] for gen in generations]
            prompts_in_batch.extend(prompts)
            preds_in_batch.extend(predictions)
            generations_in_batch.extend(generations)
            predictions_in_batch.extend(predictions)

        items = {
            "uuid": [uuid[0] for uuid in uuids] * len(all_paraphrases),
            "answers": [ans[0] for ans in answers] * len(all_paraphrases),
            "paraphrase": paraphrases_in_batch,
            "prompt": prompts_in_batch,
            "prediction": predictions_in_batch,
            "generation": generations_in_batch,
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

    # Lemmaize predictions and answers
    print("\nLemmatizing predictions and answers...")
    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_chunk, chunks)
    df = append_lemmas(df, results)

    df.to_feather(dump_file)
    print(f"\n✅ Baseline 2 (per_prompt) results saved to: {dump_file}")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique questions: {df['uuid'].nunique()}")
    return dump_file


if __name__ == "__main__":
    import argparse
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
        """
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["origin", "per_prompt", "all"],
        help="Baseline method: 'origin' (original questions), 'per_prompt' (each paraphrase), or 'all' (both)"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2_3b_it",
        help="Model name (default: llama3.2_3b_it)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["webqa", "myriadlama"],
        help="Dataset: 'webqa' or 'myriadlama'"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run the model on (default: cuda)"
    )
    parser.add_argument(
        "--rewrite", action="store_true",
        help="Regenerate baseline even if file already exists"
    )
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == "webqa":
        from core.dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from core.dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")
    
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    
    # Validate model
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    print("="*70)
    print("Baseline Generation for Self-Ensemble Experiments")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Rewrite: {args.rewrite}")
    print()
    
    # Generate baselines
    output_files = []
    
    if args.method in ["origin", "all"]:
        output_file = generate_baseline_origin(dataset, dataloader, model_path, args)
        output_files.append(output_file)
    
    if args.method in ["per_prompt", "all"]:
        output_file = generate_baseline_per_prompt(dataset, dataloader, model_path, args)
        output_files.append(output_file)
    
    print("\n" + "="*70)
    print("Baseline Generation Complete")
    print("="*70)
    print(f"Generated {len(output_files)} baseline(s):")
    for f in output_files:
        print(f"  ✅ {f}")
    print()
    print("Next steps:")
    print("  1. Analyze results: python analysis/analyze_baseline.py --dataset {} --model {}".format(args.dataset, args.model))
    print("  2. Compare with ensemble methods")
    print("="*70)
