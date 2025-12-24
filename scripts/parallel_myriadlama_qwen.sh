#!/bin/bash

DEVICE=$1
NUM_FEWSHOTS=5
NUM_PARAS=5

for MODEL in qwen2.5_3b_it qwen2.5_14b_it qwen2.5_7b_it qwen2.5_3b qwen2.5_7b qwen2.5_14b ; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --debug \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method max \
        --model $MODEL \
        --debug \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5
done