#!/bin/bash

DEVICE=$1
NUM_FEWSHOTS=$2

NUM_PARAS=5
for MODEL in qwen2.5_3b_it qwen2.5_14b_it qwen2.5_7b_it qwen2.5_3b qwen2.5_7b qwen2.5_14b ; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS

    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_attn
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_attn \
        --scale_factor 2

    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_rope

    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_attn \
        --modify_rope
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_attn \
        --modify_rope \
        --scale_factor 2
done