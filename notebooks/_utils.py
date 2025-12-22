import os
from pdb import set_trace

import pandas

from utils import partial_match_scores, partial_match_scores_use_generation


def get_filenames(
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
        dump_file_prefix += "scalescore20."
    if single_para_qapair:
        dump_file_prefix += "singleparaqapair."
    if explicit_prompts:
        dump_file_prefix += "explicitprompts."
    if num_fewshots != 5:
        dump_file_prefix += f"{num_fewshots}fshots."


    dump_file = f"{dump_file_prefix}{num_samples}samples.{num_paraphrases}paras.feather"
    # print(f"Loading from {dump_file}")
    return dump_file

def _calculate_accuracy(df, label, use_generation=False):
    answers = [answers for answers in df["answer_lemmas"]]
    if use_generation:
        generations = [[pred.tolist()] for pred in df["generation_lemmas"].tolist()]
        acc = partial_match_scores_use_generation(generations, answers, birdirect=True)
    else:
        predicts = [preds for preds in df["predict_lemma"]]
        acc = partial_match_scores(predicts, answers, birdirect=True)
    
    print(f"Acc: {acc:.4f} ==> 🏷️ {label}")

    
def calculate_accuracy(
        dump_file_prefix,
        single_para_qapair, explicit_prompts, repeat_paras,
        modifyattn, modifyrope, scale_score,
        num_paraphrases, num_fewshots, use_generation=False):
    filename = get_filenames(
        dump_file_prefix=dump_file_prefix,
        modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score,
        single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, 
        repeat_paras=repeat_paras, num_fewshots=num_fewshots,
        num_paraphrases=num_paraphrases, num_samples=5)
    if os.path.exists(filename) is False:
        basename = filename.replace(dump_file_prefix, "./")
        print(f"File {basename} does not exist!")
        return None
    

    df = pandas.read_feather(filename)
    label = f"{num_paraphrases}paras {num_fewshots}shots "
    label += f"{'1QA' if single_para_qapair else ''} "
    label += f"{'+Explicit' if explicit_prompts else ''} "
    label += f"{'+Repeat' if repeat_paras else ''} "
    label += f"{'+Attn' if modifyattn else ''} {'+Rope' if modifyrope else ''} {'+ScaleScore' if scale_score else ''}"
    
    if modifyattn is False and modifyrope is False and scale_score == 0 and num_paraphrases == 1:
        label += " (Baseline)"
    _calculate_accuracy(df, label, use_generation=use_generation)
    return df

def report_accuracy_by_nparas(
        dump_file_prefix, 
        single_para_qapair, explicit_prompts, 
        repeat_paras, num_fewshots,
        modifyattn, modifyrope, scale_score):
    calculate_accuracy(
            dump_file_prefix=dump_file_prefix, 
            single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, repeat_paras=repeat_paras, 
            modifyattn=False, modifyrope=False, scale_score=0, 
            num_paraphrases=1, num_fewshots=num_fewshots)
    
    for num_paraphrases in [2, 3, 4, 5]:
        calculate_accuracy(
            dump_file_prefix=dump_file_prefix, 
            single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, repeat_paras=repeat_paras, 
            modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score, 
            num_paraphrases=num_paraphrases, num_fewshots=num_fewshots)
    
def report_accuracy_by_nshot(
        dump_file_prefix, 
        single_para_qapair, explicit_prompts, 
        repeat_paras, num_paraphrases,
        modifyattn, modifyrope, scale_score, 
        use_generation):
    for num_fewshots in [0, 1, 2, 3, 4, 5]:
    # for num_fewshots in [0]:
        calculate_accuracy(
            dump_file_prefix=dump_file_prefix, 
            single_para_qapair=single_para_qapair, explicit_prompts=explicit_prompts, repeat_paras=repeat_paras, 
            modifyattn=modifyattn, modifyrope=modifyrope, scale_score=scale_score, 
            num_paraphrases=num_paraphrases, num_fewshots=num_fewshots, 
            use_generation=use_generation)