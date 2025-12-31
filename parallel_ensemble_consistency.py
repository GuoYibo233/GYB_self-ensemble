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
from notebooks._utils import get_parallel_ensemble_filename
from parallel_ensemble import ensemble_generation
from utils import append_lemmas, init_spacy, lemmaize_chunk

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

num_parts = 8

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda).")
    parser.add_argument("--num_paraphrases", type=int, default=5, help="Number of paraphrases to use in each sample (default: 2)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of different paraphrase combinations to generate per question (default: 5)")
    parser.add_argument("--num_fewshots", type=int, default=5, help="Number of few-shot examples to use (default: 5)")
    parser.add_argument("--repeat_paras", action="store_true", help="Repeat the same paraphrase multiple times instead of using permutations")
    
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

    from dataset import MyriadLama100Dataset
    dataset = MyriadLama100Dataset(model_name=args.model, debug=args.debug)
    
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    # Use dataset name for output file prefix
    dataset_name = getattr(dataset, 'name', None) or getattr(dataset, '__class__', type(dataset)).__name__.replace('Dataset', '').lower()

    dump_file = get_parallel_ensemble_filename(
        dump_file_prefix=f"{dataset.dataset_root}/consistency.",
        logits_ensemble_method=args.logits_ensemble_method,
        repeat_paras=args.repeat_paras,
        ensemble_method=args.ensemble_method,
        ensemble_layer=args.ensemble_layer,
        ensemble_alpha=args.ensemble_alpha,
        token_mode=args.token_mode,
        multilayer=args.multilayer,
        num_fewshots=args.num_fewshots,
        num_paraphrases=args.num_paraphrases,
        num_samples=args.num_samples
    )
    
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"✅ File {dump_file} already exists, skipping generation.")
        exit(0)

    max_new_tokens = 10 if args.num_fewshots > 0 else 20
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)
    
    print(f"🔄 Starting {args.logits_ensemble_method} logits ensembling to {dump_file}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
    few_shot_examples = dataset.get_few_shot_examples(k=args.num_fewshots) if args.num_fewshots > 0 else ""

    all_samples = []
    all_samples = []
    for uuids, answers, paraphrases, _ in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
        paraphrases = list(zip(*paraphrases))
        for uuid, answers, paraphrases_ in zip(uuids, answers, paraphrases):
            random.seed(uuid)
            paraphrases_ = list(paraphrases_)
            random.shuffle(paraphrases_)
            assert args.num_samples * args.num_paraphrases <= len(paraphrases_)

            para_groups = [paraphrases_[i * args.num_paraphrases:(i + 1) * args.num_paraphrases] for i in range(args.num_samples)]
            for group_id, para_group in enumerate(para_groups):
                all_samples.append((uuid, answers, para_group, group_id))
        
    print(f"Total samples to process: {len(all_samples)}")
    
    # Process each sample
    for sample_data in tqdm(all_samples, desc="Generating", dynamic_ncols=True):
        uuid, answer, sampled_paraphrases, group_id = sample_data
        all_prompts = dataset.construct_prompts(few_shot_examples, sampled_paraphrases)

        generation, label_probs = ensemble_generation(
            model,
            tokenizer,
            prompts=all_prompts, 
            integration_method=args.logits_ensemble_method,
            weights=None, 
            max_new_tokens=max_new_tokens, 
            ensemble_method=args.ensemble_method,
            ensemble_layer_idx=args.ensemble_layer - 1,
            ensemble_alpha=args.ensemble_alpha, 
            token_mode=args.token_mode,
            multilayer=args.multilayer, 
            choice_labels=dataset.choice_labels)
        
        labels, label_probs = zip(*label_probs) if label_probs else ([], [])
        prediction = generation.strip().split()[0] if generation.strip() else ""
        
        items = {
            "uuid": [uuid],
            "paraphrases": [sampled_paraphrases],
            "group_id": [group_id],
            "prompts": [all_prompts],
            "answers": [answer],
            "prediction": [prediction],
            "generation": [generation],
            "labels": [labels], 
            "label_probs": [label_probs],
        }
        
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

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