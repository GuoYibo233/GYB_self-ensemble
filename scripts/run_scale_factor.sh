#!/bin/bash

DEVICE=$1
NUM_PARAS=$2
MODEL_NAME=$3

# for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.1_8b_it llama3.2_3b llama3.2_1b llama3.1_8b ; do

for SCALE_FACTOR in 0.5 1.0 1.5 2.0 2.5 3.0 ; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_myriadlama2.py \
        --debug \
        --model $MODEL_NAME \
        --single_para_qapair \
        --num_paraphrases $NUM_PARAS \
        --batch_size 1 \
        --modify_attn \
        --modify_rope \
        --scale_factor $SCALE_FACTOR
done