#!/bin/bash

DEVICE=$1
NUM_FEWSHOTS=$2
for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.1_8b_it llama3.2_3b llama3.2_1b llama3.1_8b qwen2.5_3b qwen2.5_7b qwen2.5_14b qwen2.5_3b_it qwen2.5_7b_it qwen2.5_14b_it; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_fewshots $NUM_FEWSHOTS \
        --single_para_qapair \
        --rewrite \
        --num_paraphrases 1 
done