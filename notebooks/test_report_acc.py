from functools import partial
from pdb import set_trace

import pandas

from notebooks._utils import report_accuracy_by_nshot

ds_name = "myriadlama"
import os
_current_user = os.environ.get('USER', 'unknown')
if _current_user == 'y-guo':
    dataset_root = "/home/y-guo/self-ensemble"
else:
    dataset_root = "/home/xzhao/workspace/GYB_self-ensemble/datasets"

model_name = "llama3.2_3b_it"
# model_name = "qwen2.5_3b_it"

dump_file_prefix = f"{dataset_root}/{ds_name}/{model_name}/myriadlama."

report_accuracy = partial(
    report_accuracy_by_nshot, 
    dump_file_prefix=dump_file_prefix,
    single_para_qapair=True,
    explicit_prompts=False,
    repeat_paras=False, 
    num_paraphrases=5, 
    use_generation=True)


report_accuracy(modifyattn=False, modifyrope=False, scale_score=False)


# import string


# def take_until_punct_or_space(tokens: list[str]) -> list[str]:
#     """
#     Return the prefix of tokens until the next token is
#     punctuation or whitespace.
#     """
#     result = []
#     for tok in tokens:
#         if tok.isspace() or tok in string.punctuation:
#             break
#         result.append(tok)
#     return result

# def is_matched_str(pred_tokens, gold_tokens, birdirectional=True):
#     if any(" ".join(gold_tokens) == " ".join(pred_tokens[i:i+len(gold_tokens)]) for i in range(len(pred_tokens))):
#         return True
#     elif birdirectional and any(" ".join(pred_tokens) == " ".join(gold_tokens[i:i+len(pred_tokens)]) for i in range(len(gold_tokens))):
#         return True
#     return False

# def partial_match(pred, golds, birdirectional=True):
#     return any(is_matched_str(pred, gold, birdirectional) for gold in golds)

# def partial_match_scores(predictions, gold_answers, birdirect=False):
#     scores = []
#     for prediction, _gold_answers in zip(predictions, gold_answers):
#         prediction = prediction.tolist()
#         _gold_answers = [gold.tolist() for gold in _gold_answers]
#         score = partial_match(prediction, _gold_answers, birdirect)        
#         scores.append(int(score))
#     return sum(scores)/len(scores)

# def partial_match_scores_use_generation(predictions, gold_answers, birdirect=False):
#     scores = []
#     for generations, _gold_answers in zip(predictions, gold_answers):
#         generations = take_until_punct_or_space(generations[0])
#         _gold_answers = [gold.tolist() for gold in _gold_answers]
#         score = partial_match(generations, _gold_answers, birdirect)
#         scores.append(int(score))
#     return sum(scores)/len(scores)


# def _calculate_accuracy(df, label, use_generation=True):
#     answers = [answers for answers in df["answer_lemmas"]]
#     if use_generation:
#         generations = [[pred.tolist()] for pred in df["generation_lemmas"].tolist()]
#         acc = partial_match_scores_use_generation(generations, answers, birdirect=True)
#     else:
#         predicts = [preds for preds in df["predict_lemma"]]
#         acc = partial_match_scores(predicts, answers, birdirect=True)
    
#     print(f"Acc: {acc:.4f} ==> 🏷️ {label}")

# # baseline_fn = f"{dataset_root}/{ds_name}/{model_name}/baseline_per_prompt.feather"
# # baseline = pandas.read_feather(baseline_fn)
# # # baseline["predict_lemma"] = baseline["predict_lemma"].apply(lambda xs: xs[0])
# # _calculate_accuracy(baseline, "MyriadLlama Baseline Per Prompt", use_generation=True)
