#!/bin/bash

DEVICE=$1
for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.1_8b_it llama3.2_3b llama3.2_1b llama3.1_8b qwen2.5_3b qwen2.5_7b qwen2.5_14b qwen2.5_3b_it qwen2.5_7b_it qwen2.5_14b_it; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases 5 \
        --num_fewshots 5 \
        --modify_rope
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases 5 \
        --num_fewshots 3 \
        --modify_rope        
done