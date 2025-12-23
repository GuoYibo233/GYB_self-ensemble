#!/bin/bash

DEVICE=$1
MODEL_TYPE=$2

if [ "$MODEL_TYPE" == "base" ]; then
    MODELS=("llama3.2_1b" "llama3.2_3b" "qwen2.5_3b" "llama3.1_8b" "qwen2.5_7b" "qwen2.5_14b")
elif [ "$MODEL_TYPE" == "it" ]; then
    MODELS=("llama3.2_1b_it" "llama3.2_3b_it" "qwen2.5_3b_it" "llama3.1_8b_it" "qwen2.5_7b_it" "qwen2.5_14b_it")
else
    echo "Unknown MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --rewrite \
        --num_fewshots 0 \
        --single_para_qapair \
        --num_paraphrases 5

    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --rewrite \
        --num_fewshots 0 \
        --single_para_qapair \
        --num_paraphrases 5 \
        --modify_attn

    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --rewrite \
        --num_fewshots 0 \
        --single_para_qapair \
        --num_paraphrases 5 \
        --modify_rope
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --model $MODEL \
        --rewrite \
        --num_fewshots 0 \
        --single_para_qapair \
        --num_paraphrases 5 \
        --modify_attn \
        --modify_rope        
done