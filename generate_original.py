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

from constants import MODEL_PATHs
from utils import init_spacy, lemmaize_chunk, append_lemmas, single_generation

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

nlp = None
num_parts = 8


@torch.no_grad()
def ensemble_generation(prompt_sets, integration_method="max", weights=None):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generated = None
    max_new_tokens = 20

    all_inputs = []
    for prompts in prompt_sets:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
            return_attention_mask=True,
        ).to(model.device)
        all_inputs.append(inputs)

    all_input_ids = [inputs["input_ids"] for inputs in all_inputs]
    all_attention_masks = [inputs["attention_mask"] for inputs in all_inputs]
    for _ in range(max_new_tokens):
        logits_set = []
        with torch.no_grad():
            for idx, prompts in enumerate(prompt_sets):
                input_ids = all_input_ids[idx]
                attention_mask = all_attention_masks[idx]
                logits = model(input_ids, attention_mask=attention_mask).logits[
                    :, -1, :
                ]
                logits_set.append(logits)
            logits_set = torch.stack(logits_set)

        if integration_method == "avg":
            avg_logits = logits_set.mean(dim=0)
            next_token = torch.argmax(avg_logits, dim=-1).unsqueeze(1)
        elif integration_method == "max":
            max_probs = logits_set.softmax(dim=-1).max(dim=0).values
            next_token = torch.argmax(max_probs, dim=-1).unsqueeze(1)
        elif integration_method == "weighted_avg":
            if weights is None:
                raise ValueError(
                    "Weights must be provided for weighted_avg integration."
                )
            weights = (
                torch.tensor(weights)
                .clone()
                .detach()
                .requires_grad_(False)
                .to(logits_set.device)
            )
            weights = weights / weights.sum(dim=0).unsqueeze(0)
            weighted_logits = (logits_set * weights.unsqueeze(-1)).sum(dim=0)
            next_token = torch.argmax(weighted_logits, dim=-1).unsqueeze(1)
        elif integration_method == "weighted_max":
            if weights is None:
                raise ValueError(
                    "Weights must be provided for weighted_max integration."
                )
            weights = (
                torch.tensor(weights)
                .clone()
                .detach()
                .requires_grad_(False)
                .to(logits_set.device)
            )
            argmax = weights.argmax(dim=0)
            max_logits = logits_set[argmax, torch.arange(logits_set.shape[1])]
            next_token = max_logits.argmax(dim=-1).unsqueeze(1)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

        # Take the element-wise min across the two distributions
        # Append next token to input_ids for next round
        for i in range(len(all_input_ids)):
            all_input_ids[i] = torch.cat([all_input_ids[i], next_token], dim=1)
            all_attention_masks[i] = torch.cat(
                [all_attention_masks[i], torch.ones_like(next_token)], dim=1
            )

        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)
        generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        new_generated_texts = [gen.strip() for gen in generated_texts]
    return new_generated_texts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble generation")
    parser.add_argument(
        "--method",
        type=str,
        default="per_prompt",
        choices=["origin", "per_prompt", "max", "avg", "weighted_avg", "weighted_max"],
        help="Integration method for ensemble generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2_3b_it",
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["webqa", "myriadlama"],
        help="Dataset to use for generating paraphrases.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (default: cuda).",
    )
    parser.add_argument(
        "--lemmaize",
        action="store_true",
        help="normalize predictions and answers to lemmas.",
    )
    parser.add_argument(
        "--indexs",
        type=str,
        default=None,
        help="Indexs of the dataset to use for generation. If None, use all.",
    )
    parser.add_argument(
        "--num_ensemble",
        type=int,
        default=6,
        help="Number of models to ensemble. Only used for 'max' and 'avg' methods.",
    )
    args = parser.parse_args()

    if args.dataset == "webqa":
        from dataset import WebQADataset

        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset

        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")

    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)
    if args.model not in MODEL_PATHs:
        raise ValueError(
            f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}."
        )

    model_path = MODEL_PATHs.get(args.model, args.model)

    if args.method == "origin":
        dump_file = f"{dataset.dataset_root}/origin.feather"
        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=args.device, torch_dtype="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token

        df = pd.DataFrame(
            columns=[
                "uuid",
                "answers",
                "question",
                "prompt",
                "prediction",
                "generation",
            ]
        )
        few_shot_context = dataset.get_few_shot_examples()

        for uuids, answers, all_paraphrases in tqdm(dataloader):
            # Use only the original questions (paraphrase0)
            original_questions = all_paraphrases[0]
            prompts = dataset.construct_prompts(few_shot_context, original_questions)
            generations = single_generation(model, tokenizer, prompts)
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

        # Lemmaize predictions and answers
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        df = append_lemmas(df, results)

        df.to_feather(dump_file)
        print(f"Baseline results saved to {dump_file}")
        exit(0)

    if args.method == "per_prompt":
        dump_file = f"{dataset.dataset_root}/per_prompt.feather"
        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=args.device, torch_dtype="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token

        df = pd.DataFrame(columns=["uuid", "answers", "prompts", "predictions"])
        for uuids, answers, all_paraphrases in tqdm(dataloader):
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
                predictions = [gen.strip().split("\n")[0] for gen in generations]
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
        df.to_feather(dump_file)
        exit(0)
    else:
        if args.indexs is not None:
            _root = os.path.join(dataset.dataset_root, "diversity")
            os.makedirs(_root, exist_ok=True)
            dump_file = f"{_root}/ensemble_{args.method}-{args.indexs}.feather"
        else:
            dump_file = f"{dataset.dataset_root}/ensemble_{args.method}-{args.num_ensemble}.feather"

        print(f"Dump file: {dump_file}")

        if args.lemmaize:
            assert os.path.exists(
                dump_file
            ), f"Confidence scores file {dump_file} does not exist. Please run the script with --rewrite to create it."
            df = pd.read_feather(dump_file)
            if "predict_lemma" in df.columns and "answer_lemmas" in df.columns:
                print(
                    f"Confidence scores already exist in {dump_file}. Use --rewrite to overwrite."
                )
                exit(0)

            chunks = np.array_split(df, num_parts)
            with mp.get_context("spawn").Pool(
                num_parts, initializer=init_spacy
            ) as pool:
                results = pool.map(lemmaize_chunk, chunks)

            df = append_lemmas(df, results)
            df.to_feather(dump_file)
            exit(0)

        if os.path.exists(dump_file):
            print(f"File {dump_file} already exists, skipping generation.")
            exit(0)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=args.device, torch_dtype="auto"
        )
        tokenizer.pad_token = tokenizer.eos_token

        if args.method.startswith("weighted_"):
            conf_df = pd.read_feather(
                os.path.join(dataset.dataset_root, "confidence.feather")
            )

        df = pd.DataFrame(columns=["uuid", "answers", "prediction", "generation"])
        print(f"Ensemble generation with method: {args.method}")
        few_shot_context = dataset.get_few_shot_examples()
        for uuids, answers, all_paraphrases in tqdm(dataloader):
            if args.indexs is not None:
                all_paraphrases = [
                    all_paraphrases[int(idx)] for idx in args.indexs.split(",")
                ]
            else:
                assert (
                    len(all_paraphrases) >= args.num_ensemble
                ), f"Expected at least {args.num_ensemble} paraphrases, found {len(all_paraphrases)}"
                all_paraphrases = all_paraphrases[: args.num_ensemble]
                assert (
                    len(all_paraphrases) == args.num_ensemble
                ), f"Expected {args.num_ensemble} paraphrases, found {len(all_paraphrases)}"
            all_paraphrases = [list(paras) for paras in all_paraphrases]
            all_prompts = []
            confidences = [] if args.method.startswith("weighted_") else None
            for paraphrases in all_paraphrases:
                if args.method.startswith("weighted_"):
                    confidences.append([])
                    for para in paraphrases:
                        _sdf = conf_df[conf_df["paraphrase"] == para]
                        # assert len(_sdf) == 1, f"Expected one confidence score for paraphrase '{para}', found {_sdf.shape[0]}"
                        confidences[-1].append(float(_sdf["confidence"].values[0]))
                prompts = dataset.construct_prompts(few_shot_context, paraphrases)
                all_prompts.append(prompts)
            generations = ensemble_generation(
                all_prompts, integration_method=args.method, weights=confidences
            )
            predictions = [gen.strip().split("\n")[0] for gen in generations]
            items = {
                "uuid": uuids,
                "paraphrases": list(zip(*all_paraphrases)),
                "prompts": list(zip(*all_prompts)),
                "answers": answers,
                "prediction": predictions,
                "generation": generations,
            }
            df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

        # Split the DataFrame into chunks for multiprocessing
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        df = append_lemmas(df, results)

        # Save the DataFrame to a Feather file
        df.to_feather(dump_file)
