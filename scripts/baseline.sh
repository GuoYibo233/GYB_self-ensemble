#!/bin/bash

DEVICE=$1
# NUM_FEWSHOTS=$2
echo "Running baseline methods on device $DEVICE."

# for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.1_8b_it llama3.2_3b llama3.2_1b llama3.1_8b qwen2.5_3b qwen2.5_7b qwen2.5_14b qwen2.5_3b_it qwen2.5_7b_it qwen2.5_14b_it; doz
for MODEL in llama3.2_3b qwen2.5_3b pythia_2.8b qwen3_4b qwen3_30b; do
    for NUM_FEWSHOTS in 0 5; do
        CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
            --model $MODEL \
            --method per_prompt \
            --dataset myriadlama \
            --num_fewshots $NUM_FEWSHOTS

        # CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        #     --logits_ensemble_method avg \
        #     --model $MODEL \
        #     --dataset myriadlama \
        #     --num_paraphrases 5 \
        #     --num_fewshots $NUM_FEWSHOTS \
        #     --num_samples 5 \
        #     --ensemble_method layer_output_avg \
        #     --ensemble_layer $LAYER \
        #     --ensemble_alpha 1 \
        #     --token_mode last \
        #     --multilayer
    done
done