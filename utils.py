import os
import random
import re
import string
from pdb import set_trace

import numpy as np
import pandas as pd
import spacy
import torch

# Dynamic path configuration based on current user
_current_user = os.environ.get('USER', 'unknown')
if _current_user == 'y-guo':
    DATASET_ROOT = "/home/y-guo/self-ensemble"
    PROJECT_DATASET_ROOT = "/home/y-guo/self-ensemble"
else:
    DATASET_ROOT = "/net/tokyo100-10g/data/str01_01/xzhao/datasets/self-ensemble"
    PROJECT_DATASET_ROOT = "/home/xzhao/workspace/GYB_self-ensemble/datasets"


nlp = None

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

def get_label_prob(tokenizer, logits, choice_labels):
    """Get the probabilities of choice labels [A, B, C, D, E]"""
    probs = logits.softmax(dim=-1)
    label_probs = []
    for label in choice_labels:
        label_id = tokenizer.encode(label, add_special_tokens=False)
        assert len(label_id) == 1, \
            f"Only single-token labels are supported, got label '{label}' with token ids {label_id}"
        
        if len(logits.shape) == 1:
            label_logit = probs[label_id[0]].cpu().item()
            label_probs.append((label, label_logit))
        elif len(logits.shape) == 2:
            ### Batch size > 1
            batch_label_probs = probs[:, label_id[0]].cpu().tolist()
            label_probs.append((label, batch_label_probs))
    return label_probs

def single_generation(model, tokenizer, prompts, choice_labels=None, max_new_tokens=10):
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
    
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    bsz = input_ids.size(0)
    generated = torch.empty((bsz, max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)

    past_key_values = None
    label_probs = None
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        for step in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1)  # [B]
            generated[:, step] = next_token

            out = model(
                input_ids=next_token.unsqueeze(1),  # only 1 token
                attention_mask=attn_mask,           # see note below
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values

            if choice_labels is not None and step == 0:
                label_probs = get_label_prob(tokenizer, logits, choice_labels)
    
    generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    new_generated_texts = [gen.strip() for gen in generated_texts]
    return new_generated_texts, label_probs

def multinormal_generation(model, tokenizer, prompts, num_samples):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side='left',
        truncation=True, return_token_type_ids=False).to(model.device)
    
    # Get newline token id for early stopping
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)
    if newline_token_id:
        eos_token_id = [tokenizer.eos_token_id] + newline_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    
    set_seed(100)
    outputs = model.generate(
        **inputs, 
        do_sample=True, 
        num_beams=1,
        num_return_sequences=num_samples,
        return_dict_in_generate=False,
        output_scores=False,
        output_hidden_states=False,
        max_new_tokens=10, 
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id)
    
    generated_token_ids = outputs[:, inputs.input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    # Stop at newline to get only the first line
    generated_texts = [text.split('\n')[0].strip() for text in generated_texts]
    # generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # new_generated_texts = [gen[len(prompt):] for gen, prompt in zip(generated_texts, [prompt for prompt in prompts for _ in range(100)])]
    split_generated_texts = [generated_texts[i:i+100] for i in range(0, len(generated_texts), 100)]
    return split_generated_texts

def greedy_generation(model, tokenizer, prompts):
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Get newline token id for early stopping
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)
    if newline_token_id:
        eos_token_id = [tokenizer.eos_token_id] + newline_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side='left',
        truncation=True, return_token_type_ids=False).to(model.device)
    
    outputs = model.generate(
        **inputs, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        eos_token_id=eos_token_id,
        max_new_tokens=10)
    
    generated_token_ids = outputs[:, inputs.input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    # Stop at newline to get only the first line
    generated_texts = [text.split('\n')[0].strip() for text in generated_texts]
    # new_generated_texts = [gen[len(prompt):] for gen, prompt in zip(generated_texts, [prompt for prompt in prompts for _ in range(100)])]
    return generated_texts


import torch.nn.functional as F


def prompt_ppl(model, tokenizer, q_len, prompts):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]          # [B, L]
    attention_mask = inputs["attention_mask"]# [B, L]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits               # [B, L, V]

    shift_logits = logits[:, q_len-1:-1, :]
    shift_labels = input_ids[:, q_len:]
    shift_mask   = attention_mask[:, q_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_log_probs = token_log_probs * shift_mask
    avg_nll = -token_log_probs.sum(dim=1) / shift_mask.sum(dim=1)
    ppl = torch.exp(avg_nll)
    return ppl



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
    
# def partial_match(prediction, gold_answers, birdirect=False):
#     """Return 1 if the prediction matches any gold answer after normalization."""
#     pred_norm = normalize_answer(prediction)
#     answer_norms = [normalize_answer(answer) for answer in gold_answers]

#     def is_match(pred, ans):
#         if birdirect:
#             return pred in ans or ans in pred
#         else:
#             return ans in pred

#     matches = any([is_match(pred_norm, ans) for ans in answer_norms])
#     return matches

def take_until_punct_or_space(tokens: list[str]) -> list[str]:
    """
    Return the prefix of tokens until the next token is
    punctuation or whitespace.
    """
    result = []
    for tok in tokens:
        if tok.isspace() or tok in [":", ";", ",", ".", "!", "?"]:
            break
        result.append(tok)
    return result

# def partial_match_scores(predictions, gold_answers, birdirect=False):
#     scores = []
#     for prediction, _gold_answers in zip(predictions, gold_answers):
#         try:
#             prediction = prediction.tolist()
#         except Exception:
#             assert isinstance(prediction, str)
#             prediction = [prediction]

#         if len(prediction) == 0:
#             scores.append(0)
#             continue
        
#         score = partial_match(prediction, _gold_answers, birdirect)        
#         scores.append(int(score))
#     return sum(scores)/len(scores)

def partial_match_scores(predictions, gold_answers, birdirect=False):
    scores = []
    for prediction, _gold_answers in zip(predictions, gold_answers):
        score = partial_match(prediction, _gold_answers, birdirect)
        scores.append(int(score))
    return scores

def is_list_of_str(generations):
    if not isinstance(generations, list):
        raise ValueError(f"generations should be a list, got {type(generations)}")
    for generation in generations:
        if not isinstance(generation, list):
            raise ValueError(f"each generation should be a list, got {type(generation)}")
        for lemma in generation:
            if not isinstance(lemma, str):
                raise ValueError(f"each lemma should be a str, got {type(lemma)}")
            
def _get_first_unspace_lemma(generation):
    assert isinstance(generation, list), f"generation should be a list of lemmas, got {generation} with type {type(generation)}"
    if len(generation) == 0:
        return ""
    if isinstance(generation[0], str):
        for lemma in generation:
            if lemma.strip() != "":
                return lemma.strip()
    elif isinstance(generation[0], list):
        assert len(generation) == 1, f"generation should be a list of lemmas with only one generation, got {generation} with length {len(generation)}"
        for lemma in generation[0]:
            if lemma.strip() != "":
                return lemma.strip()
    else:
        raise ValueError(f"generation[0] should be either str or list, got {generation[0]} with type {type(generation[0])}")
    return ""


def partial_match_scores_use_generation(
        predictions, gold_answers, 
        birdirect=False, is_multichoice=False):
    scores = []
    for generations, _gold_answers in zip(predictions, gold_answers):
        is_list_of_str(generations)
        if is_multichoice:
            generations_ = _get_first_unspace_lemma(generations[0])
        else:
            generations_ = take_until_punct_or_space(generations[0])
        if len(generations_) == 0:
            scores.append(0)
            continue
        score = partial_match(generations_, _gold_answers, birdirect)
        scores.append(int(score))
    return scores

def is_matched_str(pred_tokens, gold_tokens, birdirectional=True):
    if any(" ".join(gold_tokens) == " ".join(pred_tokens[i:i+len(gold_tokens)]) for i in range(len(pred_tokens))):
        return True
    elif birdirectional and any(" ".join(pred_tokens) == " ".join(gold_tokens[i:i+len(pred_tokens)]) for i in range(len(gold_tokens))):
        return True
    return False

def partial_match(pred, golds, birdirectional=True):
    return any(is_matched_str(pred, gold, birdirectional) for gold in golds)
