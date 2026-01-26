#!/bin/bash

DEVICE=$1
DATASET=${2:-"myriadlama"}
NUM_PARAPHRASE=${3:-2}

# MODELS=("qwen3_0.6b" "qwen3_1.7b" "qwen3_4b" "qwen3_8b" "qwen3_30b")
MODELS=("qwen3_30b_it")

for DATASET in "myriadlama" "hotpot" "commonsense" "mmlu"; do
    for MODEL in ${MODELS[@]}; do
        CUDA_VISIBLE_DEVICES=$DEVICE python3 self_paraphrase.py \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrase $NUM_PARAPHRASE \
            --debug \
            --batch_size 8 \
            --temperature 1.5 \
            --top_p 0.95 \
            --max_rounds 2
    done
done    
wait