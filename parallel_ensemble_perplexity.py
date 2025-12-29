import os
import warnings
from functools import partial
from pdb import set_trace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from constants import MODEL_PATHs
from parallel_ensemble import (
    _get_blocks,
    next_token_logits_with_weighted_ffn_midavg,
    next_token_logits_with_weighted_layer_outavg,
    sample_paraphrases_per_item,
)

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

num_parts = 8

def construct_perplexity_prompt(question, choice, few_shot_context=""):
    instruction = "Answer the given question.\n"
    if choice is None:
         return instruction + f"{few_shot_context}\n\nQ: {question}\nA:"
    return instruction + f"{few_shot_context}\n\nQ: {question}\nA: {choice}"

def get_ensemble_logits(model, inputs, ensemble_method, integration_method, weights, layer_indices, ensemble_alpha, token_mode):
    if inputs["input_ids"].size(0) == 1:
        # Expand to multiple inputs for ensemble
        final_logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, -1, :]
        return final_logits[0]

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
        final_logits = logits.mean(dim=0)
    elif integration_method == "max":    
        final_logits = logits.softmax(dim=-1).max(dim=0).values
    elif integration_method == "weighted_avg":
        if weights is None:
            raise ValueError("Weights must be provided for weighted_avg integration.")
        weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits.device)
        weights = weights / weights.sum(dim=0).unsqueeze(0)
        final_logits = (logits * weights.unsqueeze(-1)).sum(dim=0)
    elif integration_method == "weighted_max":
        if weights is None:
            raise ValueError("Weights must be provided for weighted_max integration.")
        weights = torch.tensor(weights).clone().detach().requires_grad_(False).to(logits.device)
        argmax = weights.argmax(dim=0)
        final_logits = logits[argmax, torch.arange(logits.shape[1])]
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")
    return final_logits

@torch.no_grad()
def measure_per_token_perplexity(
    model,
    tokenizer,
    paraphrases: list[str], 
    choices: list[str],
    few_shot_context: str,
    integration_method="max", 
    weights=None, 
    ensemble_method=None, 
    multilayer=False, 
    token_mode="last",
    ensemble_layer_idx=10, 
    ensemble_alpha=1.0):

    """ Measure per-token perplexity with ensemble methods. 
    Given the paraphrases, we first gather the logits after ensembling on them, 
    then we measure the per-token perplexity on the given multiple choices. 
    The choice with the lowest perplexity is selected as the final answer.
    """
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    paraphrases = [paraphrase for paraphrase in paraphrases]
    qonly_prompt = [construct_perplexity_prompt(paraphrase, choice=None, few_shot_context=few_shot_context) for paraphrase in paraphrases]
    qonly_inputs = tokenizer(
        qonly_prompt, return_tensors="pt", 
        padding=True, truncation=True,
        padding_side='left', return_attention_mask=True).to(model.device)
    qonly_len = qonly_inputs["input_ids"].size(1)
    
    choice_ppls = []
    for choice in choices:
        qchoice_prompts = [
            construct_perplexity_prompt(paraphrase, choice, few_shot_context=few_shot_context) 
            for paraphrase in paraphrases
        ]
        
        qchoice_inputs = tokenizer(
            qchoice_prompts, return_tensors="pt", 
            padding=True, truncation=True, 
            return_attention_mask=True).to(model.device)
        
        choice_len = qchoice_inputs["input_ids"].size(1) - qonly_len

        all_ids, all_logits = [], []
        for idx in range(choice_len):
            inputs = BatchEncoding({
                "input_ids": qchoice_inputs["input_ids"][:, :qonly_len + idx], 
                "attention_mask": qchoice_inputs["attention_mask"][:, :qonly_len + idx]
            })
            target_id = qchoice_inputs["input_ids"][:, qonly_len + idx]
            assert len(set(target_id.cpu().tolist())) == 1, "All paraphrases should have the same next token to examine"
            target_id = target_id[0].unsqueeze(0)
            
            logits = get_ensemble_logits(
                model, inputs, ensemble_method, integration_method, weights, 
                layer_indices=list(range(ensemble_layer_idx, len(_get_blocks(model)))) if multilayer else [ensemble_layer_idx], 
                ensemble_alpha=ensemble_alpha, token_mode=token_mode)
            
            all_ids.append(target_id)
            all_logits.append(logits.unsqueeze(0))

        all_ids = torch.cat(all_ids, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        
        log_probs = F.log_softmax(all_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=all_ids.unsqueeze(-1)).squeeze(-1)
        avg_nll = -token_log_probs.mean()
        choice_ppl = torch.exp(avg_nll)
        choice_ppls.append(choice_ppl.item())
    
    choice_ppls = np.array(choice_ppls)
    return choice_ppls


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["commonsense", "mmlu", "logiqa"], help="Dataset to use for generating paraphrases.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda).")
    parser.add_argument("--num_paraphrases", type=int, default=5, help="Number of paraphrases to use in each sample (default: 2)")
    parser.add_argument("--num_fewshots", type=int, default=5, help="Number of few-shot examples to use (default: 5)")
    parser.add_argument("--repeat_paras", action="store_true", help="Repeat the same paraphrase multiple times instead of using permutations")
    parser.add_argument("--is_baseline", action="store_true", help="Use single paraphrase as baseline")

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

    if args.dataset == "commonsense":
        from dataset import CommonsenseParaphraseDataset
        dataset = CommonsenseParaphraseDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "mmlu":
        from dataset import MMLUParaphraseDataset
        dataset = MMLUParaphraseDataset(model_name=args.model, debug=args.debug)
    elif args.dataset == "logiqa":
        from dataset import LogiQAParaphraseDataset
        dataset = LogiQAParaphraseDataset(model_name=args.model, debug=args.debug)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa', 'myriadlama', 'commonsense', 'mmlu', or 'logiqa'.")
    
    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    model_path = MODEL_PATHs.get(args.model, args.model)
    
    # Use dataset name for output file prefix
    dataset_name = getattr(dataset, 'name', None) or getattr(dataset, '__class__', type(dataset)).__name__.replace('Dataset', '').lower()
    dump_file = f"{dataset.dataset_root}/{dataset_name}.ppl.logits.{args.logits_ensemble_method}."
    
    if args.is_baseline:
        dump_file = f"{dataset.dataset_root}/{dataset_name}.ppl.baseline."
    else:
        dump_file = f"{dataset.dataset_root}/{dataset_name}.ppl.logits.{args.logits_ensemble_method}."
    
        if args.ensemble_method == "layer_output_avg":
            dump_file += f"avglayer.layer{args.ensemble_layer}.alpha{int(args.ensemble_alpha*100)}.token-{args.token_mode}."
        elif args.ensemble_method == "ffn_activation_avg":
            dump_file += f"avgffn.layer{args.ensemble_layer}.alpha{int(args.ensemble_alpha*100)}.token-{args.token_mode}."
        elif args.ensemble_method == "ffn_activation_max":
            dump_file += f"maxffn.layer{args.ensemble_layer}.alpha{int(args.ensemble_alpha*100)}.token-{args.token_mode}."
        if args.multilayer:
            dump_file += "multilayer."
        
    if args.repeat_paras:
        dump_file += "repeatparas."
    if args.num_fewshots != 5:
        dump_file += f"{args.num_fewshots}fshots."
        
    dump_file += f"{args.num_paraphrases}paras.feather"
    if os.path.exists(dump_file) and not args.rewrite:
        print(f"✅ File {dump_file} already exists, skipping generation.")
        exit(0)

    max_new_tokens = 10 if args.num_fewshots > 0 else 30    
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False)

    print(f"🔄 Starting {args.logits_ensemble_method} logits ensembling to {dump_file}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    if args.logits_ensemble_method.startswith("weighted_"):
        conf_df = pd.read_feather(os.path.join(dataset.dataset_root, "confidence.feather"))

    df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation","correctness"])
    
    few_shot_context = dataset.get_few_shot_examples(k=args.num_fewshots, is_ppl_format=True) if args.num_fewshots > 0 else ""
    all_samples = []

    uuid_count = 0
    for batch_data in tqdm(dataloader, desc="Preparing samples", dynamic_ncols=True):
        uuids, answers, all_paraphrases, choices_labels, choices_texts, answer_labels = batch_data
        len(all_paraphrases) >= 3, 'Each question must have at least 3 paraphrases.'
        
        samples = sample_paraphrases_per_item(
            uuids=uuids,
            all_paraphrases=all_paraphrases, 
            num_paraphrases=args.num_paraphrases, 
            num_samples=1,
            repeat_paras=args.repeat_paras
        )
        
        for uuid, sampled_paraphrases in samples:
            idx = uuids.index(uuid)
            all_samples.append((uuid, answers[idx], sampled_paraphrases, 
                                choices_labels[idx], choices_texts[idx], answer_labels[idx]))
        uuid_count += len(uuids)
    
    print(f"Total samples to process: {len(all_samples)}")
    
    measure_per_token_perplexity_partial = partial(
        measure_per_token_perplexity, model=model, tokenizer=tokenizer,
        integration_method=args.logits_ensemble_method,
        ensemble_method=args.ensemble_method,
        ensemble_layer_idx=args.ensemble_layer - 1,
        ensemble_alpha=args.ensemble_alpha,
        token_mode=args.token_mode,
        multilayer=args.multilayer,
        weights=None, 
    )

    def _get_iter():
        if args.is_baseline:
            for uuid, answer, sampled_paraphrases, choices_label, choices_text, answer_label in \
                tqdm(all_samples, desc="Generating", dynamic_ncols=True):
                for paraphrase in sampled_paraphrases:
                    ppls = measure_per_token_perplexity_partial(paraphrases=[paraphrase], choices=choices_text, few_shot_context=few_shot_context)
                    yield uuid, answer, [paraphrase], choices_label, choices_text, answer_label, ppls
        else:
            for uuid, answer, sampled_paraphrases, choices_label, choices_text, answer_label in \
                tqdm(all_samples, desc="Generating", dynamic_ncols=True):
                ppls = measure_per_token_perplexity_partial(paraphrases=sampled_paraphrases, choices=choices_text, few_shot_context=few_shot_context)
                yield uuid, answer, sampled_paraphrases, choices_label, choices_text, answer_label, ppls
    
    for uuid, answer, sampled_paraphrases, choices_label, choices_text, answer_label, ppls in _get_iter():
        items = {
            "uuid": [uuid],
            "paraphrases": [sampled_paraphrases],
            "answers": [answer],
            "ppls": [ppls],
            "best_choice_idx": [choices_text[ppls.argmin()]],
            "prediction": ["ABCDEF"[ppls.argmin()]],
            "generation": ["ABCDEF"[ppls.argmin()]],
        }
        
        items["choices_label"] = [choices_label]
        items["choices_text"] = [choices_text]
        items["answer_label"] = [answer_label]
    
        df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

    df.to_feather(dump_file)
    # chunks = np.array_split(df, num_parts)
    # with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
    #     results = pool.map(lemmaize_chunk, chunks)
    # try:
    #     df = append_lemmas(df, results)
    # except Exception as e:
    #     print(f"❌ Lemmatization failed: {type(e).__name__}: {e}")
    #     set_trace()
    # finally:
    #     df.to_feather(dump_file)
    #     print(f"✅ Results saved to {dump_file}")