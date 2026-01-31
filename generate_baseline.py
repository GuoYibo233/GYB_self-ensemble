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
from dataset import get_dataset_instance
from utils import (
    append_lemmas,
    create_batches,
    get_baseline_dumpfile,
    init_spacy,
    lemmaize_chunk,
    reasoning_generation,
    single_generation,
)

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
        columns=["uuid", "answers", "question", "prompt", "generation"]
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
        columns=["uuid", "answers", "paraphrase", "prompt", "generation"]
    )

    few_shot_context = dataset.get_few_shot_examples(k=args.num_fewshots)

    assert not (dataset.is_multi_choice and args.temperature > 0), \
        "Sampling not supported for multi-choice tasks in baseline generation."
    
    if not dataset.is_multi_choice:
        all_uuids, all_answers, all_paraphrases, all_is_origs = [], [], [], []
        for batch_data in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
            _uuids, _answers, _all_paraphrases, _is_origs = batch_data
            assert len(_uuids) == 1, "Batch size must be 1 for baseline per_prompt generation."
            assert len(_all_paraphrases) == len(_is_origs), "Mismatch in paraphrases and is_orig lengths."
            for paraphrase, is_orig in zip(_all_paraphrases, _is_origs):
                all_uuids.extend([_uuids[0]] * args.repeat)
                all_answers.extend([_answers[0]] * args.repeat)
                all_paraphrases.extend([paraphrase[0]] * args.repeat)
                all_is_origs.extend([is_orig[0]] * args.repeat)

        batch_iterator = create_batches(list(zip(all_uuids, all_answers, all_paraphrases, all_is_origs)), batch_size=args.batch_size)
    else:
        all_uuids, all_answers, all_paraphrases, all_choices_labels, all_choices_texts, all_answer_labels, all_is_origs = [], [], [], [], [], [], []
        for batch_data in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
            _uuids, _answers, _all_paraphrases, _choices_labels, _choices_texts, _answer_labels, _is_origs = batch_data
            assert len(_uuids) == 1, "Batch size must be 1 for baseline per_prompt generation."
            assert len(_all_paraphrases) == len(_is_origs), "Mismatch in paraphrases and is_orig lengths."
            choices_labels = _choices_labels[0]
            choices_texts = _choices_texts[0]
            answer_labels = _answer_labels[0]
            for paraphrase, is_orig in zip(_all_paraphrases, _is_origs):
                all_uuids.extend([_uuids[0]] * args.repeat)
                all_answers.extend([_answers[0]] * args.repeat)
                all_paraphrases.extend([paraphrase[0]] * args.repeat)
                all_is_origs.extend([is_orig[0]] * args.repeat)
                all_choices_labels.extend([choices_labels] * args.repeat)
                all_choices_texts.extend([choices_texts] * args.repeat)
                all_answer_labels.extend([answer_labels] * args.repeat)
        
        batch_iterator = create_batches(
            list(zip(
                all_uuids, all_answers, all_paraphrases, 
                all_choices_labels, all_choices_texts, 
                all_answer_labels, all_is_origs)), 
            batch_size=args.batch_size)

    for batch in tqdm(batch_iterator, desc="Generating baseline (per_prompt)", dynamic_ncols=True, total=len(all_uuids)//args.batch_size + 1):
        if dataset.is_multi_choice:
            (uuids, answers, paraphrases, 
             choices_labels, choices_texts, 
             answer_labels, is_origs) = zip(*batch)
        else:
            uuids, answers, paraphrases, is_origs = zip(*batch)
        
        if dataset.is_multi_choice:
            prompts = []
            for paraphrase, _choice_labels, _choice_texts in zip(paraphrases, choices_labels, choices_texts):
                prompt = dataset.construct_multi_choice_prompts(few_shot_context, [paraphrase], _choice_labels, _choice_texts)
                prompts.append(prompt[0])
        elif not dataset.reasoning:
            prompts = dataset.construct_prompts(few_shot_context, paraphrases)
        else:
            prompts = dataset.construct_prompts_for_reasoning(few_shot_context, paraphrases)
        
        if args.temperature == 0:
            generations, label_probs = single_generation(
                model, tokenizer, prompts, 
                choice_labels=dataset.choice_labels, 
                max_new_tokens=args.max_new_tokens)
        
            generations = [gen.strip().split("\n")[0] for gen in generations]
            label_strs, label_probs_ = zip(*label_probs) if label_probs is not None else ([], [])
            label_probs_ = list(zip(*label_probs_))
        else:
            messages, thinkings, generations = reasoning_generation(
                model, tokenizer, 
                dataset.instruction, prompts, 
                temperature=args.temperature, 
                top_p=args.top_p, 
                max_new_tokens=args.max_new_tokens
            )

        items = {
            "uuid": uuids,
            "answers": answers,
            "paraphrase": paraphrases,
            "is_orig": is_origs,
            "prompt": prompts,
            "generation": generations,
            "thinking": thinkings if dataset.reasoning else None,
            "messages": messages if dataset.reasoning else None,
        }

        # set_trace()
        if dataset.is_multi_choice:
            items.update({
                "choices_label": choices_labels,
                "choices_text": choices_texts,
                "answer_label": answer_labels,
                "labels": [label_strs] * len(paraphrases),
                "label_probs": label_probs_})
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing (default: 5)")
    parser.add_argument("--additional_paraphrases_file", type=str, default=None, help="Path to additional paraphrases file (for datasets that support it)")
    parser.add_argument("--num_paraphrases", type=int, default=4, help="Number of paraphrases to use (default: 4, only for datasets that support paraphrases)")
    parser.add_argument("--rewrite", action="store_true", help="Regenerate baseline even if file already exists",)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 for greedy generation)",)
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling value (default: 0.95)",)
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum number of new tokens to generate (default: 1024)",)
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat generation for each input (default: 1)",)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output",)
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode for datasets that support it (e.g., HotpotQA)",)
    args = parser.parse_args()

    # Load dataset
    dataset = get_dataset_instance(
        dataset_name=args.dataset,
        model_name=args.model,
        debug=args.debug,
        reasoning=args.reasoning,
        num_paraphrases=args.num_paraphrases,
        additional_paraphrases_file=args.additional_paraphrases_file,
    )
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)

    if args.model.startswith("phi3") and dataset.is_multi_choice:
        dataset.choice_labels = [label.strip() for label in dataset.choice_labels]

    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    model_path = MODEL_PATHs.get(args.model, args.model)
    dump_file = get_baseline_dumpfile(dataset, args)
    print("Baseline Generation for Self-Ensemble Experiments")    

    if os.path.exists(dump_file) and not args.rewrite:
        print(f"✅ File {dump_file} already exists, skipping generation. Use --rewrite to regenerate.")
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
    