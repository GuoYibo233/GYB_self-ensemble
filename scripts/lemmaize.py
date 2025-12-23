import multiprocessing as mp
from ast import arg
from pdb import set_trace

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm


def init_spacy():
    global nlp
    nlp = spacy.load("en_core_web_lg")

def lemmaize_predicts(predict):
    global nlp
    if not predict or pd.isna(predict):
        return []
    doc = nlp(str(predict))
    return [token.lemma_.lower() for token in doc]

def lemmaize_chunk(chunk):
    predict_lemmas = []
    generation_lemmas = []
    answer_lemmas = []

    for idx, row in chunk.iterrows():
        prediction = row["prediction"]
        generation = row["generation"]
        answers = row["answers"]
        generation = str(generation).strip().split(".")[0] if "." in str(generation) else str(generation)
        predict_lemmas.append(lemmaize_predicts(prediction))
        answer_lemmas.append([lemmaize_predicts(ans) for ans in answers])
        generation_lemmas.append(lemmaize_predicts(generation))
    return predict_lemmas, generation_lemmas, answer_lemmas


def append_lemmas(df, results):
    all_predict_lemmas = []
    all_generation_lemmas = []
    all_answer_lemmas = []
    for predict_lemmas, generation_lemmas, answer_lemmas in results:
        all_predict_lemmas.extend(predict_lemmas)
        all_generation_lemmas.extend(generation_lemmas)
        all_answer_lemmas.extend(answer_lemmas)
    df["predict_lemma"] = pd.Series(all_predict_lemmas, dtype=object)
    df["generation_lemmas"] = pd.Series(all_generation_lemmas, dtype=object)
    df["answer_lemmas"] = pd.Series(all_answer_lemmas, dtype=object)
    return df

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name, e.g., llama3.2_1b_it"
    )
    args = parser.parse_args()

    num_parts = 20
    data_root = f"/home/xzhao/workspace/GYB_self-ensemble/datasets/myriadlama/{args.model}/"
    for filename in tqdm(os.listdir(data_root)):
        if not filename.endswith(".feather"):
            continue

        filename = os.path.join(data_root, filename)
        print(f"🔁 Processing {filename}...")
    
        df = pd.read_feather(filename)

        if "predict_lemma" in df.columns and "generation_lemmas" in df.columns and "answer_lemmas" in df.columns:
            print("✅ Lemmas already exist!")
            continue
        
        chunks = np.array_split(df, num_parts)
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_parts, initializer=init_spacy) as pool:
            results = pool.map(lemmaize_chunk, chunks)

        df = append_lemmas(df, results)
        df.to_feather(filename)
        print("✅ Done!")
        