#!/bin/bash

DEVICE=$1
DATASET=${2}
MODEL_TYPE=${3:-base}
NUM_FEWSHOTS=${4:-0}

if [ "$MODEL_TYPE" == "base" ]; then
    MODELS="llama3.2_1b llama3.2_3b llama3.1_8b qwen2.5_3b qwen2.5_7b qwen2.5_14b"
elif [ "$MODEL_TYPE" == "it" ]; then
    MODELS="llama3.2_1b_it llama3.2_3b_it llama3.1_8b_it qwen2.5_3b_it qwen2.5_7b_it qwen2.5_14b_it"
elif [ "$MODEL_TYPE" == "others" ]; then
    MODELS="qwen3_30b pythia_2.8b qwen3_4b"
fi

for MODEL in $MODELS; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --dataset $DATASET \
        --num_fewshots $NUM_FEWSHOTS \
        --single_para_qapair \
        --num_samples 1 \
        --num_paraphrases 5
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --dataset $DATASET \
        --num_fewshots $NUM_FEWSHOTS \
        --single_para_qapair \
        --num_samples 1 \
        --num_paraphrases 5 \
        --modify_attn \
        --modify_rope \
        --scale_factor
done