#!/bin/bash

DEVICE=$1
DATASET=${2:-"hotpot"}
NUM_PARAPHRASE=${3:-4}

MODELS=("qwen3_0.6b" "qwen3_1.7b" "qwen3_4b" "qwen3_8b" "qwen3_14b" "qwen3_32b")
# MODELS=("qwen3_32b")

# for DATASET in "myriadlama" "hotpot" "commonsense" "mmlu"; do
for DATASET in "hotpot"; do
    for MODEL in ${MODELS[@]}; do
        CUDA_VISIBLE_DEVICES=$DEVICE python3 self_paraphrase.py \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrase $NUM_PARAPHRASE \
            --debug \
            --batch_size 8 \
            --temperature 1.5 \
            --top_p 0.95 \
            --max_rounds 5
    done
done    
wait