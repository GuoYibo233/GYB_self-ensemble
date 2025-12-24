#!/bin/bash

DEVICE=$1
MODEL=$2


CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    --model $MODEL \
    --debug \
    --single_para_qapair \
    --num_paraphrases 1 \
    --num_fewshots 5

CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    --model $MODEL \
    --debug \
    --single_para_qapair \
    --num_paraphrases 5 \
    --num_fewshots 5

CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    --model $MODEL \
    --debug \
    --rewrite \
    --single_para_qapair \
    --num_paraphrases 5 \
    --num_fewshots 5 \
    --modify_attn \
    --modify_rope

CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
    --model $MODEL \
    --debug \
    --rewrite \
    --single_para_qapair \
    --num_paraphrases 5 \
    --num_fewshots 5 \
    --modify_attn \
    --modify_rope \
    --rewrite \
    --scale_factor
