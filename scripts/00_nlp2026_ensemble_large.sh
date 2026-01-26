#!/bin/bash

DEVICE=$1
DATASET=${2:-"myriadlama"} # myriadlama hotpot commonsense, mmlu, logiqa
# MODEL=${4:-"qwen3_30b"}
echo "Running baseline methods on device $DEVICE "

# MODELS=("qwen3_4b" "qwen3_0.6b" "qwen3_1.7b" "qwen3_8b" "qwen3_30b")
# MODELS=("qwen3_4b" "qwen3_0.6b" "qwen3_1.7b" "qwen3_8b")
MODELS=("qwen3_14b")
for NUM_FEWSHOTS in 0 5; do
    for MODEL in ${MODELS[@]}; do
        
        CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
            --debug \
            --model $MODEL \
            --method per_prompt \
            --dataset $DATASET \
            --batch_size 2 \
            --num_fewshots $NUM_FEWSHOTS \
            --additional_paraphrases_file paraphrases.num_para2.max_rounds2.temp1.5.topp0.95.jsonl

        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
            --debug \
            --logits_ensemble_method avg \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrases -1 \
            --num_fewshots $NUM_FEWSHOTS \
            --num_samples 1 \
            --additional_paraphrases_file paraphrases.num_para2.max_rounds2.temp1.5.topp0.95.jsonl
    done
done
