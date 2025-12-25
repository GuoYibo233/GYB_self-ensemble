import os

import pandas

from utils import partial_match_scores, partial_match_scores_use_generation


def get_parallel_ensemble_filename(
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

def calculate_accuracy(df, label, use_generation=False):
    # answers = [answers for answers in df["answer_lemmas"]]
    answers = [[answer.tolist() for answer in answers.tolist()] for answers in df["answer_lemmas"]]
    try:
        if use_generation:
            generations = [[pred.tolist()] for pred in df["generation_lemmas"].tolist()]
            acc = partial_match_scores_use_generation(generations, answers, birdirect=True)
        else:
            predicts = df["predict_lemma"].tolist()
            acc = partial_match_scores(predicts, answers, birdirect=True)
    
        print(f"Acc: {acc:.4f} ==> 🏷️ {label}")
    except KeyError as e:
        print(f"KeyError: {e} ==> 🏷️ {label}")

    
def calculate_series_ensemble_accuracy(
        dump_file_prefix,
        single_para_qapair, explicit_prompts, repeat_paras,
        modifyattn, modifyrope, scale_score,
        num_paraphrases, num_fewshots, use_generation=True):
    filename = get_parallel_ensemble_filename(
        dump_file_prefix=dump_file_prefix,
        modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score,
        single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, 
        repeat_paras=repeat_paras, num_fewshots=num_fewshots,
        num_paraphrases=num_paraphrases, num_samples=5)
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
    calculate_accuracy(df, label, use_generation=use_generation)
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
        

def get_series_ensemble_filename(
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
        num_paraphrases, num_fewshots,
        logits_ensemble_method,
        ensemble_method=None, ensemble_layer=None, 
        multilayer=False, ensemble_alpha=1.0, 
        token_mode="all", use_generation=False):
    filename = get_series_ensemble_filename(
        dump_file_prefix=dump_file_prefix, repeat_paras=repeat_paras, 
        logits_ensemble_method=logits_ensemble_method,
        ensemble_method=ensemble_method, ensemble_layer=ensemble_layer,
        multilayer=multilayer, ensemble_alpha=ensemble_alpha, token_mode=token_mode,
        num_fewshots=num_fewshots, num_paraphrases=num_paraphrases, num_samples=5)
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
    
    calculate_accuracy(df, label, use_generation=use_generation)
    return df
