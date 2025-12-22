#!/bin/bash

DEVICE=$1
MODEL=$2

# qwen2.5_3b qwen2.5_3b_it llama3.2_3b llama3.2_3b_it

for NUM_FWSHOTS in 0 1 2 3 4 5 ; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_fewshots $NUM_FWSHOTS \
        --single_para_qapair \
        --num_paraphrases 5

    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_fewshots $NUM_FWSHOTS \
        --single_para_qapair \
        --num_paraphrases 5 \
        --modify_attn
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_fewshots $NUM_FWSHOTS \
        --single_para_qapair \
        --num_paraphrases 5 \
        --modify_attn \
        --modify_rope
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --num_fewshots $NUM_FWSHOTS \
        --single_para_qapair \
        --num_paraphrases 5 \
        --modify_attn \
        --modify_rope \
        --scale_factor 2
done