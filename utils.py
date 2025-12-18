import re
import string
import spacy
import torch
import random 
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_ROOT = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/self-ensemble"
PROJECT_DATASET_ROOT = "/home/xzhao/workspace/GYB_self-ensemble/datasets"


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
    for prediction, answers in tqdm(
        zip(chunk["prediction"], chunk["answers"]), total=len(chunk)
    ):
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def webqa_collate_fn(batch):
    prompt0 = [item["question"] for item in batch]
    prompt1 = [item["paraphrase1"] for item in batch]
    prompt2 = [item["paraphrase2"] for item in batch]
    prompt3 = [item["paraphrase3"] for item in batch]
    prompt4 = [item["paraphrase4"] for item in batch]
    prompt5 = [item["paraphrase5"] for item in batch]
    answers = [item["answers"] for item in batch]
    all_prompts = [prompt0, prompt1, prompt2, prompt3, prompt4, prompt5]
    return prompt0, answers, all_prompts

def myriadlama_collate_fn(batch):
    uuids = [item["uuid"] for item in batch]
    answers = [item["answers"] for item in batch]
    manual_paraphrase = [item["manual_paraphrases"] for item in batch]
    manual_paraphrase = zip(*manual_paraphrase)
    return uuids, answers, manual_paraphrase

def format_example(example):
    question = example["question"]
    # Pick the first gold answer (you can change this logic if needed)
    answer = example["answers"][0]
    return f"Q: {question}\nA: {answer}"

def get_few_shot_examples(dataset, k=5, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), k)
    return "\n\n".join(format_example(dataset[i]) for i in indices)

def construct_prompts(questions, few_shot_examples, instruction):
    prompts = [f"{instruction}\n\n{few_shot_examples}\n\nQ: {question}\nA:" for question in questions]
    return prompts

def multinormal_generation(model, tokenizer, prompts, num_samples):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side='left',
        truncation=True, return_token_type_ids=False).to(model.device)
    
    set_seed(100)
    outputs = model.generate(
        **inputs, 
        do_sample=True, 
        num_beams=1,
        num_return_sequences=num_samples,
        return_dict_in_generate=False,
        output_scores=False,
        output_hidden_states=False,
        max_new_tokens=20, 
        pad_token_id=tokenizer.eos_token_id)
    
    generated_token_ids = outputs[:, inputs.input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    generated_texts = [text.strip() for text in generated_texts]
    # generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # new_generated_texts = [gen[len(prompt):] for gen, prompt in zip(generated_texts, [prompt for prompt in prompts for _ in range(100)])]
    split_generated_texts = [generated_texts[i:i+100] for i in range(0, len(generated_texts), 100)]
    return split_generated_texts

def greedy_generation(model, tokenizer, prompts):
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side='left',
        truncation=True, return_token_type_ids=False).to(model.device)
    
    outputs = model.generate(
        **inputs, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        max_new_tokens=20)
    
    generated_token_ids = outputs[:, inputs.input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    generated_texts = [text.strip() for text in generated_texts]
    # new_generated_texts = [gen[len(prompt):] for gen, prompt in zip(generated_texts, [prompt for prompt in prompts for _ in range(100)])]
    return generated_texts


def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    if isinstance(s, str):
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    elif isinstance(s, float):
        return str(s).strip()
    else:
        return ""


def single_generation(model, tokenizer, prompts, max_new_tokens=20):
    """Generate responses using greedy decoding."""
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)

    generated = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(next_token)], dim=1
        )

        if generated is None:
            generated = next_token
        else:
            generated = torch.cat([generated, next_token], dim=1)

    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    new_generated_texts = [gen.strip() for gen in generated_texts]
    return new_generated_texts


def is_matched_str(pred_tokens, gold_tokens, birdirectional=True):
    if any(" ".join(gold_tokens) == " ".join(pred_tokens[i:i+len(gold_tokens)]) for i in range(len(pred_tokens))):
        return True
    elif birdirectional and any(" ".join(pred_tokens) == " ".join(gold_tokens[i:i+len(pred_tokens)]) for i in range(len(gold_tokens))):
        return True
    return False

def partial_match(pred, golds, birdirectional=True):
    return any(is_matched_str(pred, gold, birdirectional) for gold in golds)

def partial_match_scores(predictions, gold_answers, birdirect=False):
    scores = []
    for prediction, _gold_answers in zip(predictions, gold_answers):
        score = partial_match(prediction, _gold_answers, birdirect)
        scores.append(int(score))
    return sum(scores)/len(scores)
