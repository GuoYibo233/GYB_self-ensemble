import multiprocessing as mp
import os
from pdb import set_trace

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import greedy_generation, is_matched_str, multinormal_generation

nlp = None

def init_spacy():
    global nlp
    nlp = spacy.load("en_core_web_lg")

def normalize_predicts(predict):
    global nlp
    doc = nlp(predict)
    return [token.lemma_.lower() for token in doc]

def normalize_chunk(chunk):
    confis = []
    greedy_lemmas = []
    sample_lemmas = []
    answer_lemmas = []
    for answers, greedy_predict, sample_predicts in tqdm(zip(chunk["answers"], chunk["greedy_predict"], chunk["sample_predicts"]), total=len(chunk)):
        greedy = normalize_predicts(greedy_predict)
        samples = [normalize_predicts(sample) for sample in sample_predicts]
        answers = [normalize_predicts(ans) for ans in answers]
        matches = [is_matched_str(greedy, sample) for sample in samples]
        confi = sum(matches) / len(matches)
        confis.append(confi)
        greedy_lemmas.append(greedy)
        sample_lemmas.append(samples)
        answer_lemmas.append(answers)

    return answer_lemmas, greedy_lemmas, sample_lemmas, confis

def append_lemmas(df, results):
    all_greedy_lemmas = []
    all_sample_lemmas = []
    all_answer_lemmas = []
    all_confis = []
    for answer_lemmas, greedy_lemmas, sample_lemmas, confis in results:
        all_answer_lemmas.extend(answer_lemmas)
        all_greedy_lemmas.extend(greedy_lemmas)
        all_sample_lemmas.extend(sample_lemmas)
        all_confis.extend(confis)

    df["greedy_lemma"] = all_greedy_lemmas
    df["sample_lemmas"] = pd.Series(all_sample_lemmas, dtype=object)
    df["sample_lemmas"] = df["sample_lemmas"].apply(lambda xs: [list(x) for x in xs])
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    df["answer_lemmas"] = df["answer_lemmas"].apply(lambda xs: [list(x) for x in xs])
    df["confidence"] = all_confis
    return df 

if __name__ == "__main__":
    import argparse

    from constants import MODEL_PATHs

    parser = argparse.ArgumentParser(description="Generate confidence scores for paraphrases.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on.")
    parser.add_argument("--model", type=str, default="llama3.2_3b_it", help="Path to the pre-trained model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["webqa", "myriadlama"], help="Dataset to use for generating paraphrases.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite the dataset if it already exists.")
    parser.add_argument("--write_confidence", action="store_true", help="Write confidence scores to a file.")
    args = parser.parse_args()

    num_parts = 32
    
    if args.dataset == "webqa":
        from dataset import WebQADataset
        dataset = WebQADataset(model_name=args.model)
    elif args.dataset == "myriadlama":
        from dataset import MyriadLamaDataset
        dataset = MyriadLamaDataset(model_name=args.model)
    else:
        raise ValueError("Unsupported dataset. Please use 'webqa' or 'myriadlama'.")

    dump_path = os.path.join(dataset.dataset_root, "confidence.feather")

    print("Dump path: ", dump_path)
    if args.write_confidence:
        assert os.path.exists(dump_path), f"Confidence scores file {dump_path} does not exist. Please run the script with --rewrite to create it."
        df = pd.read_feather(dump_path)
        if "confidence" in df.columns and "greedy_lemma" in df.columns and "sample_lemmas" in df.columns and "answer_lemmas" in df.columns:
            print(f"Confidence scores already exist in {dump_path}. Use --rewrite to overwrite.")
            exit(0)
        
        chunks = np.array_split(df, num_parts)
        with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(normalize_chunk, chunks)

        df = append_lemmas(df, results)
        df.to_feather(dump_path)
        exit(0)

    if os.path.exists(dump_path) and not args.rewrite:
        print(f"Confidence scores already exist at {dump_path}. Use --rewrite to overwrite.")
        exit(0)
        
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=False)

    if args.model not in MODEL_PATHs:
        raise ValueError(f"Model {args.model} is not supported. Please choose from {list(MODEL_PATHs.keys())}.")
    model_path = MODEL_PATHs.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
    tokenizer.pad_token = tokenizer.eos_token

    # Do sampling and generation
    items = []
    few_shot_prompt = dataset.get_few_shot_examples()
    for uuids, answers, all_paraphrases in tqdm(dataloader):
        for paraphrases in all_paraphrases:
            prompts = dataset.construct_prompts(few_shot_prompt, paraphrases)
            multinormal_samples = multinormal_generation(model, tokenizer, prompts, num_samples=100)
            greedy_samples = greedy_generation(model, tokenizer, prompts)
            
            greedy_predicts = [sample.strip().split('\n')[0] for sample in greedy_samples]
            sample_predicts = [[sample.strip().split('\n')[0] for sample in samples] for samples in multinormal_samples]
            for uuid, prompt, answer, paraphrase, greedy_predict, _sample_predicts in zip(uuids, prompts, answers, paraphrases, greedy_predicts, sample_predicts):
                item = {
                    "uuid": uuid,
                    "paraphrase": paraphrase,
                    "prompt": prompt,
                    "answers": answer,
                    "greedy_predict": greedy_predict,
                    "sample_predicts": _sample_predicts
                }
                items.append(item)
    df = pd.DataFrame(items)

    # Split the DataFrame into chunks for multiprocessing
    chunks = np.array_split(df, num_parts)
    with mp.get_context("spawn").Pool(num_parts, initializer=init_spacy) as pool:
        results = pool.map(normalize_chunk, chunks)
    df = append_lemmas(df, results)
    
    # Save the DataFrame to a Feather file
    df.to_feather(dump_path)