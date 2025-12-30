import multiprocessing as mp
import os
import random
import warnings
from pdb import set_trace

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_PATHs
from parallel_ensemble import ensemble_generation
from utils import (
    init_spacy,
    lemmaize_predicts,
    single_generation,
)

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

num_parts = 8

def lemmaize_multiple_chunk(chunk):
    predict_lemmas = []
    ensemble_generation_lemmas = []
    single_generation_lemmas = []
    answer_lemmas = []

    for idx, row in chunk.iterrows():
        prediction = row["prediction"]
        ensemble_generation = row["ensemble_generation"]
        single_generations = row["single_generations"]
        answers = row["answers"]

        ensemble_generation = str(ensemble_generation).strip().split(".")[0] if "." in str(ensemble_generation) else str(ensemble_generation)
        single_generations = [str(gen).strip().split(".")[0] if "." in str(gen) else str(gen) for gen in single_generations]
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
        ensemble_generation_lemmas.append(lemmaize_predicts(ensemble_generation))
        single_generation_lemmas.append([lemmaize_predicts(gen) for gen in single_generations])
    return predict_lemmas, ensemble_generation_lemmas, single_generation_lemmas, answer_lemmas

def append_multiple_lemmas(df, results):
    all_predict_lemmas = []
    all_ensemble_generation_lemmas = []
    all_single_generation_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, ensemble_generation_lemmas, single_generation_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_ensemble_generation_lemmas.extend(ensemble_generation_lemmas)
        all_single_generation_lemmas.extend(single_generation_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["ensemble_generation_lemmas"] = pd.Series(all_ensemble_generation_lemmas, dtype=object)
    df["single_generation_lemmas"] = pd.Series(all_single_generation_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama"], help="Dataset to use for generating paraphrases.")    
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda).")
    parser.add_argument("--num_fewshots", type=int, default=5, help="Number of few-shot examples to use (default: 5)")
    
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
    
    dump_file = f"{dataset.dataset_root}/myriadlama.multiple_instr.logits.{args.logits_ensemble_method}."
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
    
    dump_file += f"feather"
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"✅ File {dump_file} already exists, skipping generation.")
        exit(0)

    max_new_tokens = 10 if args.num_fewshots > 0 else 30    
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)

    print(f"🔄 Starting {args.logits_ensemble_method} logits ensembling to {dump_file}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")

    if args.logits_ensemble_method.startswith("weighted_"):
        conf_df = pd.read_feather(os.path.join(dataset.dataset_root, "confidence.feather"))

    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
    few_shot_examples = dataset.get_few_shot_examples(k=args.num_fewshots) if args.num_fewshots > 0 else ""

    sample_count = 0
    all_samples = []
    for uuids, answers, all_paraphrases in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
        target_prompt = all_paraphrases[0]
        sample_count += len(uuids)
        for item_idx in range(len(uuids)):
            uuid = uuids[item_idx]
            random.seed(uuid)  # Ensure reproducibility
            tgt_paraphrase = random.sample(all_paraphrases[item_idx][:5], 1)[0]
            all_samples.append((uuid, answers[item_idx], tgt_paraphrase))
            
    print(f"Total samples to process: {len(all_samples)}")
    
    # Process each sample
    instructions = [
        dataset.instruction, 
        "Your task is to replace [MASK] with the correct word based on the context.",
        "Which word best fills the [MASK] in the sentence based on the context?"
    ]
    for uuid, answer, prompt in tqdm(all_samples, desc="Generating", dynamic_ncols=True):
        all_prompts = []
        confidences = [] if args.logits_ensemble_method.startswith("weighted_") else None
        
        # For ensemble generation, treat each paraphrase as a separate prompt
        single_instr_generations = []
        for instr in instructions:
            if args.logits_ensemble_method.startswith("weighted_"):
                raise NotImplementedError("Weighted ensemble not implemented for diverse instructions yet.")
            prompts = dataset.construct_prompts(few_shot_examples, [prompt], instruction=instr)
            single_instr_generations.append(single_generation(model, tokenizer, prompts, max_new_tokens))
            all_prompts.append(prompts)
        
        generation = ensemble_generation(
            model,
            tokenizer,
            prompt_sets=all_prompts, 
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
            "paraphrases": [prompt],
            "prompts": [all_prompts],
            "answers": [answer],
            "prediction": [prediction],
            "ensemble_generation": [generation],
            "single_generations": [single_instr_generations],
        }
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

        sample_count += len(uuids)
        
    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(lemmaize_multiple_chunk, chunks)
    try:
        df = append_multiple_lemmas(df, results)
    except Exception as e:
        print(f"❌ Lemmatization failed: {type(e).__name__}: {e}")
        set_trace()
    finally:
        df.to_feather(dump_file)
        print(f"✅ Results saved to {dump_file}")