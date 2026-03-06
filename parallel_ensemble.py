import itertools
import logging
import multiprocessing as mp
import os
import random
import warnings
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
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

        input_ids = torch.cat([input_ids, next_token.expand(input_ids.size(0), -1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.size(0), 1, device=model.device)], dim=1)

        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)

        current_model_input = next_token.repeat(input_ids.size(0), 1)
        decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=False)
        if next_token.item() == tokenizer.eos_token_id:
            break
        if "\n" in decoded_token and step > 0:
            break

    if generated is None:
        logging.warning("No tokens generated")
        return "", None
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
        paraphrases: List of paraphrase strings for the current item
        num_paraphrases: Number of paraphrases to select in each sample (-1 = all)
        num_samples: Number of different paraphrase combinations to generate per uuid
        uuid: UUID for the current batch, used as random seed
        repeat_paras: If True, repeat the same paraphrase multiple times instead of permutations

    Returns:
        List of (uuid, sampled_paraphrases, sampled_is_origs, sampled_messages) tuples
    """
    if num_paraphrases == -1:
        return [(uuid, paraphrases, is_origs, messages)]

    all_indices = list(range(len(paraphrases)))
    effective_num_paraphrases = min(num_paraphrases, len(all_indices))
    if repeat_paras:
        all_sampled_paras = [[n] * effective_num_paraphrases for n in all_indices]
    else:
        all_sampled_paras = list(itertools.permutations(all_indices, effective_num_paraphrases))

    random.seed(uuid)
    sampled_combinations = random.sample(all_sampled_paras, k=min(num_samples, len(all_sampled_paras)))

    result = []
    for paraids in sampled_combinations:
        result.append((
            uuid,
            [paraphrases[i] for i in paraids],
            [is_origs[i] for i in paraids],
            [messages[i] for i in paraids],
        ))
    return result


def get_parallel_ensemble_dumpfile(dataset_root, cfg, timestamp):
    filename = (
        f"parallel.{cfg.logits_ensemble_method}"
        f".{cfg.num_samples}samples"
        f".{cfg.num_paraphrases}paras"
        f".{timestamp}.feather"
    )
    return os.path.join(dataset_root, filename)


def craft_prompts_from_baseline_file(
        baseline_file, thinking,
        dataset=None, instruction=None, few_shot_context=None):
    """
    Yield per-UUID prompt batches from a baseline feather file.

    If `instruction` or `few_shot_context` is provided (and dataset is not None),
    prompts are rebuilt from the raw `paraphrase` column using the dataset's
    construct_prompts() method.  Otherwise the stored `prompt` column is used as-is.

    Yields:
        (uuid, answers, paraphrases, is_origs, messages)
    """
    df = pd.read_feather(baseline_file)
    rebuild = (instruction is not None or few_shot_context is not None) and dataset is not None

    for uuid, subdf in df.groupby('uuid'):
        subdf = subdf.reset_index(drop=True)
        if thinking:
            subdf = subdf[subdf['thinking'].str.len() > 0]
        if len(subdf) == 0:
            logging.warning(f"No valid thinking entries for uuid {uuid}, skipping.")
            continue

        paraphrases = subdf["paraphrase"].tolist()
        is_origs = subdf["is_orig"].tolist()

        if rebuild:
            _inst = instruction if instruction is not None else dataset.instruction
            _fsc = few_shot_context if few_shot_context is not None else ""
            if dataset.is_multi_choice and "choices_label" in subdf.columns:
                messages = []
                for para, cl, ct in zip(
                        paraphrases,
                        subdf["choices_label"].tolist(),
                        subdf["choices_text"].tolist()):
                    msg = dataset.construct_multi_choice_prompts(
                        _inst, _fsc, [para], cl, ct)[0]
                    messages.append(msg)
            else:
                messages = dataset.construct_prompts(
                    _inst, _fsc, paraphrases, series_ensemble=False)
        else:
            messages = subdf["prompt"].tolist()

        if thinking:
            combined = [
                (para, is_orig, f"{prompt}{think}")
                for para, is_orig, prompt, think in zip(
                    paraphrases, is_origs, messages, subdf["thinking"].tolist())
                if think.strip().endswith("</think>")
            ]
            if not combined:
                continue
            paraphrases, is_origs, messages = zip(*combined)

        yield (
            uuid,
            subdf['answers'].tolist()[0].tolist(),
            list(paraphrases),
            list(is_origs),
            list(messages),
        )


def _load_config(config_path):
    """Load YAML config and return a SimpleNamespace with defaults filled in."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = SimpleNamespace(**cfg_dict)
    # Optional fields with defaults
    cfg.device = getattr(cfg, "device", "cuda")
    cfg.max_uuids = getattr(cfg, "max_uuids", None)          # None = run all
    cfg.additional_paraphrases_file = getattr(cfg, "additional_paraphrases_file", None)
    cfg.repeat_paras = getattr(cfg, "repeat_paras", False)
    cfg.num_samples = getattr(cfg, "num_samples", 1)
    cfg.num_paraphrases = getattr(cfg, "num_paraphrases", -1)
    cfg.logits_ensemble_method = getattr(cfg, "logits_ensemble_method", "avg")
    cfg.thinking = getattr(cfg, "thinking", False)
    cfg.few_shot_context = getattr(cfg, "few_shot_context", None)  # literal string
    cfg.instruction = getattr(cfg, "instruction", None)            # literal string
    return cfg, cfg_dict


def _setup_logging(log_dir, timestamp):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"parallel_ensemble_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return log_file


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Parallel ensemble generation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg, cfg_dict = _load_config(args.config)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dataset root comes from directory of baseline_file
    dataset_root = os.path.dirname(os.path.abspath(cfg.baseline_file))
    os.makedirs(dataset_root, exist_ok=True)

    log_file = _setup_logging(os.path.join(dataset_root, "logs"), run_ts)
    log = logging.getLogger(__name__)

    dump_file = get_parallel_ensemble_dumpfile(dataset_root, cfg, run_ts)

    log.info("=" * 60)
    log.info("parallel_ensemble.py  START")
    log.info(f"Config file : {os.path.abspath(args.config)}")
    log.info(f"Config      : {cfg_dict}")
    log.info(f"Output file : {dump_file}")
    log.info(f"Log file    : {log_file}")
    log.info("=" * 60)

    try:
        dataset = get_dataset_instance(
            dataset_name=cfg.dataset,
            model_name=cfg.model,
            debug=False,
            thinking=cfg.thinking,
            additional_paraphrases_file=cfg.additional_paraphrases_file,
        )
        dataset.dataset_root = dataset_root

        group_by_uuid = craft_prompts_from_baseline_file(
            cfg.baseline_file,
            cfg.thinking,
            dataset=dataset,
            instruction=cfg.instruction,
            few_shot_context=cfg.few_shot_context,
        )

        if cfg.logits_ensemble_method.startswith("weighted_"):
            conf_df = pd.read_feather(
                os.path.join(dataset_root, "confidence.feather"))

        model, tokenizer = load_model_tokenizer(cfg.model)
        log.info(f"Model loaded: {cfg.model}")
        log.info(f"Ensemble method: {cfg.logits_ensemble_method}")
        if cfg.max_uuids is not None:
            log.info(f"Limiting run to {cfg.max_uuids} UUIDs")

        max_new_tokens = 32

        # ---- Collect all samples ----
        all_samples = []
        uuid_count = 0
        for batch_data in tqdm(group_by_uuid, desc="Preparing samples", dynamic_ncols=True):
            if cfg.max_uuids is not None and uuid_count >= cfg.max_uuids:
                break

            uuid, answers, paraphrases, is_origs, messages = batch_data
            choices_labels = choices_texts = answer_labels = None

            samples = sample_paraphrases_per_item(
                uuid=uuid,
                paraphrases=paraphrases,
                is_origs=is_origs,
                messages=messages,
                num_paraphrases=cfg.num_paraphrases,
                num_samples=cfg.num_samples,
                repeat_paras=cfg.repeat_paras,
            )

            for s_uuid, s_paras, s_is_origs, s_msgs in samples:
                all_samples.append((
                    s_uuid, answers,
                    s_paras, s_is_origs, s_msgs,
                    choices_labels, choices_texts, answer_labels,
                ))
            uuid_count += 1

        log.info(f"UUIDs prepared : {uuid_count}")
        log.info(f"Total samples  : {len(all_samples)}")

        # ---- Generate ----
        df = pd.DataFrame(columns=[
            "uuid", "answers", "prediction", "generation",
            "prompts", "paraphrases", "is_orig",
        ])

        for sample_data in tqdm(all_samples, desc="Generating", dynamic_ncols=True):
            (uuid, answer,
             sampled_paraphrases, sampled_is_origs, sampled_messages,
             choices_label, choices_text, answer_label) = sample_data

            confidences = [] if cfg.logits_ensemble_method.startswith("weighted_") else None
            if confidences is not None:
                for para in sampled_paraphrases:
                    _sdf = conf_df[conf_df["paraphrase"] == para]
                    confidences.append(
                        float(_sdf["confidence"].values[0]) if len(_sdf) > 0 else 1.0)

            generation, label_probs = ensemble_generation(
                model,
                tokenizer,
                prompts=sampled_messages,
                integration_method=cfg.logits_ensemble_method,
                weights=[confidences] if confidences else None,
                max_new_tokens=max_new_tokens,
                choice_labels=dataset.choice_labels,
            )

            labels, label_probs = zip(*label_probs) if label_probs else ([], [])

            if dataset.is_multi_choice:
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
            if dataset.is_multi_choice:
                items["choices_label"] = [choices_label]
                items["choices_text"] = [choices_text]
                items["answer_label"] = [answer_label]

            df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

        # ---- Lemmatize and save ----
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)
        df = append_lemmas(df, results)
        df.to_feather(dump_file)

        log.info(f"SUCCESS — {len(df)} rows saved")
        log.info(f"Output file : {dump_file}")

    except Exception:
        log.exception("FAILED — unhandled exception")
        raise
    finally:
        log.info("=" * 60)
        log.info(f"parallel_ensemble.py  END  [{datetime.now().strftime('%Y%m%d_%H%M%S')}]")
        log.info("=" * 60)
