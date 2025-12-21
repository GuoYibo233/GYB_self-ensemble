#!/bin/bash

DEVICE=$1
NUM_PARAS=$2

for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.1_8b_it llama3.2_3b llama3.2_1b llama3.1_8b ; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1

    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --scale_factor 2

    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --modify_attn
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --modify_attn \
        --scale_factor 2

    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --modify_attn \
        --modify_rope
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --modify_attn \
        --modify_rope \
        --scale_factor 2

    CUDA_VISIBLE_DzEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --modify_attn \
        --modify_rope \
        --scale_factor 2 \
        --repeat_paras
    
done