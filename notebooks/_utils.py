import os

import numpy as np
import pandas
from numpy import argmax

from utils import partial_match_scores, partial_match_scores_use_generation


def get_layers(model: str):
    model2layers = {
        "llama3.2_1b": 12,
        "llama3.2_3b": 21,
        "llama3.1_8b": 24,
        "qwen2.5_3b": 27,
        "qwen2.5_7b": 21,
        "qwen2.5_14b": 36,
        "qwen3_4b": 27,
        "qwen3_30b": 36,
        "qwen3_235b": 71,
        "pythia_2.8b": 24,
        "phi4_mini": 24,
        "bloom_3b": 25,
        "gpt_20b": 18,
    }

    for key in model2layers:
        if model.startswith(key):
            return model2layers[key]
    raise NotImplementedError(f"Layers not defined for model {model}")

def get_series_ensemble_filename(
        dump_file_prefix, 
        modifyattn, modifyrope, 
        scale_score, num_fewshots,
        single_para_qapair, explicit_prompts, repeat_paras, 
        num_paraphrases, num_samples):
    # dump_file = f"{dataset_root}/{ds_name}/{model_name}/myriadlama."
    if modifyattn:
        dump_file_prefix += "modifyattn."
    if modifyrope:
        dump_file_prefix += "modifyrope."
    if repeat_paras:
        dump_file_prefix += "repeatparas."
    if scale_score:
        dump_file_prefix += "scalescore."
    if single_para_qapair:
        dump_file_prefix += "singleparaqapair."
    if explicit_prompts:
        dump_file_prefix += "explicitprompts."
    if num_fewshots != 5:
        dump_file_prefix += f"{num_fewshots}fshots."


    dump_file = f"{dump_file_prefix}{num_samples}samples.{num_paraphrases}paras.feather"
    # print(f"Loading from {dump_file}")
    return dump_file

def calculate_accuracy(df, label, is_multichoice, use_generation=True, by_probs=False, verbose=True):
    """
    by_probs: whether to calculate accuracy based on label probabilities (for multi-choice tasks)
    use_generation: whether to use generation_lemmas (True) or predict_lemma (False)
    
    by_probs is only for multi-choice tasks and has higher priority than use_generation
    """
    if not by_probs:
        answers = [[answer.tolist() for answer in answers.tolist()] for answers in df["answer_lemmas"]]
        try:
            if use_generation:
                generations = [[pred.tolist()] for pred in df["generation_lemmas"].tolist()]
                scores = partial_match_scores_use_generation(generations, answers, birdirect=True, is_multichoice=is_multichoice)
            else:
                predicts = df["predict_lemma"].tolist()
                scores = partial_match_scores(predicts, answers, birdirect=True)
            df["correct"] = scores
            acc = sum(scores)/len(scores)
            if verbose:
                print(f"Acc: {acc:.4f} ==> 🏷️ {label}")
        except KeyError as e:
            print(f"KeyError: {e} ==> 🏷️ {label}")
    else:
        labels = df['labels'].tolist()[0]
        assert isinstance(labels, list) or isinstance(labels, np.ndarray) and len(labels) > 0, f"labels should be a list of label strings and not empty: got {labels} with type {type(labels)}"
        pred_labels = [labels[argmax(probs).item()].strip() for probs in df['label_probs'].tolist()]
        scores = [pred == answer for pred, answer in zip(pred_labels, df['answer_label'].tolist())]
        df["correct"] = scores
        acc = sum(scores)/len(scores)
        if verbose:
            print(f"Acc: {acc:.4f} ==> 🏷️ {label}")
    return acc

# def calculate_accuracy(df, label, use_generation=True, is_multichoice=False, verbose=True):
#     # answers = [answers for answers in df["answer_lemmas"]]
#     if not is_multichoice:
#         answers = [[answer.tolist() for answer in answers.tolist()] for answers in df["answer_lemmas"]]
#         try:
#             if use_generation:
#                 generations = [[pred.tolist()] for pred in df["generation_lemmas"].tolist()]    
#                 scores = partial_match_scores_use_generation(generations, answers, birdirect=True)
#             else:
#                 predicts = df["predict_lemma"].tolist()
#                 scores = partial_match_scores(predicts, answers, birdirect=True)
#             df["correct"] = scores
#             acc = sum(scores)/len(scores)
#             if verbose:
#                 print(f"Acc: {acc:.4f} ==> 🏷️ {label}")
#         except KeyError as e:
#             print(f"KeyError: {e} ==> 🏷️ {label}")
#     else:
#         answer_labels = df["answer_label"].tolist()
#         predict_labels = df["prediction"].tolist()
#         scores = [1 if a == p else 0 for a, p in zip(answer_labels, predict_labels)]
#         df["correct"] = scores
#         acc = sum(scores)/len(scores)
#         if verbose:
#             print(f"Multichoice Acc: {acc:.4f} ==> 🏷️ {label}")
#     return acc

def _read_per_prompt_dataframe(dataset_root, ds_name, model_name, num_fewshots):
    # filename = os.path.join(dataset_root,  ds_name, model_name, f"baseline_per_prompt.{num_fewshots}shots.feather")
    # print(f"Reading baseline per-prompt dataframe from {filename}")
    try:
        baseline_df = pandas.read_feather(
            os.path.join(
                dataset_root,  ds_name, model_name, 
                f"baseline_per_prompt.{num_fewshots}shots.feather"))
        return baseline_df    
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return None

def calculate_baseline_accuracy(dataset_root, ds_name, model_name, num_fewshots, is_multichoice=False, by_probs=False):
    baseline_df = _read_per_prompt_dataframe(
        dataset_root, ds_name, model_name, num_fewshots)
    if baseline_df is None:
        return None
    calculate_accuracy(
        baseline_df, "baseline of average accuracy per-paraphrase", 
        by_probs=by_probs, is_multichoice=is_multichoice)

def calculate_oracle_accuracy(dataset_root, ds_name, model_name, num_fewshots, is_multichoice=False, by_probs=False):
    baseline_df = _read_per_prompt_dataframe(
        dataset_root, ds_name, model_name, num_fewshots)
    if baseline_df is None:
        return None
    
    calculate_accuracy(baseline_df, "", verbose=False, by_probs=by_probs, is_multichoice=is_multichoice)
    
    scores = []
    for uuid, gdf in baseline_df.groupby('uuid'):
        scores.append(gdf['correct'].sum() > 0)
    oracle_acc = sum(scores)/len(scores)
    print(f"Acc: {oracle_acc:.4f} ==> 🏷️ Oracle accuracy per-paraphrase")
        
def calculate_series_ensemble_accuracy(
        dump_file_prefix,
        single_para_qapair, explicit_prompts, repeat_paras,
        modifyattn, modifyrope, scale_score,
        num_paraphrases, num_fewshots, num_samples, 
        use_generation=True):
    filename = get_series_ensemble_filename(
        dump_file_prefix=dump_file_prefix,
        modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score,
        single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, 
        repeat_paras=repeat_paras, num_fewshots=num_fewshots,
        num_paraphrases=num_paraphrases, num_samples=num_samples)
    if os.path.exists(filename) is False:
        basename = filename.replace(dump_file_prefix, "./")
        print(f"File {basename} does not exist!")
        return None
    
    try:
        df = pandas.read_feather(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        os.remove(filename)
        return None
    label = f"{num_paraphrases}paras {num_fewshots}shots "
    label += f"{'1QA' if single_para_qapair else ''} "
    label += f"{'+Explicit' if explicit_prompts else ''} "
    label += f"{'+Repeat' if repeat_paras else ''} "
    label += f"{'+Attn' if modifyattn else ''} {'+Rope' if modifyrope else ''} {'+ScaleScore' if scale_score else ''}"
    
    if modifyattn is False and modifyrope is False and scale_score == 0 and num_paraphrases == 1:
        label += " (Baseline)"
    try:
        calculate_accuracy(df, label, use_generation=use_generation)
    except KeyError as e:
        print(f"KeyError: {e} ==> 🏷️ {label}")
    return df

def report_series_ensemble_accuracy_by_nparas(
        dump_file_prefix, 
        single_para_qapair, explicit_prompts, 
        repeat_paras, num_fewshots,
        modifyattn, modifyrope, scale_score):
    
    # for num_paraphrases in [2, 3, 4, 5]:
    for num_paraphrases in [5]:
        calculate_series_ensemble_accuracy(
            dump_file_prefix=dump_file_prefix, 
            single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, repeat_paras=repeat_paras, 
            modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score, 
            num_paraphrases=num_paraphrases, num_fewshots=num_fewshots)
    
def report_series_ensemble_accuracy_by_nshot(
        dump_file_prefix, 
        single_para_qapair, explicit_prompts, 
        repeat_paras, num_paraphrases,
        modifyattn, modifyrope, scale_score, 
        use_generation):
    for num_fewshots in [0, 1, 2, 3, 4, 5]:
    # for num_fewshots in [0]:
        calculate_series_ensemble_accuracy(
            dump_file_prefix=dump_file_prefix, 
            single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, repeat_paras=repeat_paras, 
            modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score, 
            num_paraphrases=num_paraphrases, num_fewshots=num_fewshots, 
            use_generation=use_generation)
        

def get_parallel_ensemble_filename(
        dump_file_prefix, repeat_paras, 
        logits_ensemble_method,
        ensemble_method, ensemble_layer,
        multilayer, ensemble_alpha, token_mode,
        num_fewshots, num_paraphrases, num_samples):
    # dump_file = f"{dataset_root}/{ds_name}/{model_name}/myriadlama."
    dump_file_prefix += f"logits.{logits_ensemble_method}."
    if repeat_paras:
        dump_file_prefix += "repeatparas."
    
    if ensemble_method == "layer_output_avg":
        dump_file_prefix += f"avglayer.layer{ensemble_layer}.alpha{int(ensemble_alpha*100)}.token-{token_mode}."
    elif ensemble_method == "ffn_activation_avg":
        dump_file_prefix += f"avgffn.layer{ensemble_layer}.alpha{int(ensemble_alpha*100)}.token-{token_mode}."
    elif ensemble_method == "ffn_activation_max":
        dump_file_prefix += f"maxffn.layer{ensemble_layer}.alpha{int(ensemble_alpha*100)}.token-{token_mode}."
    if multilayer:
        dump_file_prefix += "multilayer."
    if num_fewshots != 5:
        dump_file_prefix += f"{num_fewshots}fshots."
        
    dump_file = f"{dump_file_prefix}{num_samples}samples.{num_paraphrases}paras.feather"
    return dump_file

def calculate_parallel_ensemble_accuracy(
        dump_file_prefix, repeat_paras,
        num_paraphrases, num_fewshots, num_samples,
        logits_ensemble_method,
        ensemble_method=None, ensemble_layer=None, 
        multilayer=False, ensemble_alpha=1.0, 
        token_mode="all", use_generation=True, 
        by_probs=False, is_multichoice=False):
    filename = get_parallel_ensemble_filename(
        dump_file_prefix=dump_file_prefix, repeat_paras=repeat_paras, 
        logits_ensemble_method=logits_ensemble_method,
        ensemble_method=ensemble_method, ensemble_layer=ensemble_layer,
        multilayer=multilayer, ensemble_alpha=ensemble_alpha, token_mode=token_mode,
        num_fewshots=num_fewshots, num_paraphrases=num_paraphrases, num_samples=num_samples)
    if os.path.exists(filename) is False:
        basename = filename.replace(dump_file_prefix, "./")
        print(f"File {basename} does not exist!")
        return None
    
    try:
        df = pandas.read_feather(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        os.remove(filename)
        return None
    label = f"{num_paraphrases}paras {num_fewshots}shots "
    label += f"{'+Repeat' if repeat_paras else ''} "
    label += f"{ensemble_method} layer{ensemble_layer} "
    label += f"{'Multilayer' if multilayer else ''} "
    label += f"alpha{ensemble_alpha} token-{token_mode}"
    
    calculate_accuracy(df, label, use_generation=use_generation, by_probs=by_probs, is_multichoice=is_multichoice)
    return df
