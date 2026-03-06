from notebooks._utils import (
    calculate_baseline_accuracy,
    calculate_oracle_accuracy,
    calculate_parallel_ensemble_accuracy,
)

num_fewshots = 0
dataset_root = "/home/xzhao/workspace/GYB_self-ensemble/datasets"
all_models = ["qwen2.5_3b"]


# for ds_name in ["commonsense", "mmlu", "logiqa"]:
for ds_name in ["commonsense"]:

    for model_name in all_models:
        print(f"\n---------- Model: {model_name} ----------")
        dump_file_prefix = f"{dataset_root}/{ds_name}/{model_name}/{ds_name}paraphrase."
        print("---- 📜 GENERATION: Calculating baseline ----")
        calculate_baseline_accuracy(dataset_root, ds_name, model_name, num_fewshots, by_probs=False, is_multichoice=True)
        
        print("---- 📜 GENERATION: Calculating Oracle ----")
        calculate_oracle_accuracy(dataset_root, ds_name, model_name, num_fewshots, by_probs=False, is_multichoice=True)

        print("---- 📜 GENERATION: Logits Ensemble ----")
        calculate_parallel_ensemble_accuracy(dump_file_prefix, repeat_paras=False,
            num_paraphrases=5, num_fewshots=num_fewshots, num_samples=1,
            logits_ensemble_method="avg",
            use_generation=True, by_probs=False, is_multichoice=True)
        
        
        print("---- 🎲 PROBABILITY: Calculating baseline ----")
        calculate_baseline_accuracy(dataset_root, ds_name, model_name, num_fewshots, by_probs=True)
        
        print("---- 🎲 PROBABILITY: Calculating Oracle ----")
        calculate_oracle_accuracy(dataset_root, ds_name, model_name, num_fewshots, by_probs=True)
    
        print("---- 🎲 PROBABILITY: Logits Ensemble ----")
        calculate_parallel_ensemble_accuracy(dump_file_prefix, repeat_paras=False,
            num_paraphrases=5, num_fewshots=num_fewshots, num_samples=1,
            logits_ensemble_method="avg",
            use_generation=True, by_probs=True, is_multichoice=True)
    