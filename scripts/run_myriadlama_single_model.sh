#!/bin/bash

DEVICE=$1
NUM_PARAS=$2
MODEL=$3

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1 \
    --scale_factor 2

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1 \
    --modify_attn
    
CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1 \
    --modify_attn \
    --scale_factor 2

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1 \
    --modify_attn \
    --modify_rope
    
CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1 \
    --modify_attn \
    --modify_rope \
    --scale_factor 2

CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
    --model $MODEL \
    --single_para_qapair \
    --num_paraphrases $NUM_PARAS \
    --batch_size 1 \
    --modify_attn \
    --modify_rope \
    --scale_factor 2 \
    --repeat_paras