#!/bin/bash

DEVICE=$1
MODEL_TYPE=${2:-base}
NUM_FEWSHOTS=${3:-5}
DATASET="myriadlama"
NUM_PARAS=5

if [ "$MODEL_TYPE" == "base" ]; then
    MODELS="llama3.2_1b llama3.2_3b llama3.1_8b qwen2.5_3b qwen2.5_7b qwen2.5_14b"
elif [ "$MODEL_TYPE" == "it" ]; then
    MODELS="llama3.2_1b_it llama3.2_3b_it llama3.1_8b_it qwen2.5_3b_it qwen2.5_7b_it qwen2.5_14b_it"
elif [ "$MODEL_TYPE" == "llama" ]; then
    MODELS="llama3.2_1b llama3.2_3b llama3.1_8b"
elif [ "$MODEL_TYPE" == "qwen3" ]; then
    MODELS="qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b qwen3_32b"
elif [ "$MODEL_TYPE" == "others" ]; then
    MODELS="qwen3_30b pythia_2.8b qwen3_4b"
fi

for MODEL in $MODELS; do
    # CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    #     --model $MODEL \
    #     --single_para_qapair \
    #     --num_paraphrases $NUM_PARAS \
    #     --num_fewshots $NUM_FEWSHOTS

    # CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    #     --model $MODEL \
    #     --single_para_qapair \
    #     --num_paraphrases $NUM_PARAS \
    #     --num_fewshots $NUM_FEWSHOTS \
    #     --modify_attn
    
    # CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    #     --model $MODEL \
    #     --single_para_qapair \
    #     --num_paraphrases $NUM_PARAS \
    #     --num_fewshots $NUM_FEWSHOTS \
    #     --modify_attn \
    #     --scale_factor

    # CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    #     --model $MODEL \
    #     --single_para_qapair \
    #     --num_paraphrases $NUM_PARAS \
    #     --num_fewshots $NUM_FEWSHOTS \
    #     --modify_rope

    # CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    #     --model $MODEL \
    #     --single_para_qapair \
    #     --num_paraphrases $NUM_PARAS \
    #     --num_fewshots $NUM_FEWSHOTS \
    #     --modify_attn \
    #     --modify_rope
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --debug \
        --dataset $DATASET \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_attn \
        --modify_rope \
        --scale_factor
done