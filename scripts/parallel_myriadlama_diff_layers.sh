#!/bin/bash

NUM_FEWSHOTS=5
NUM_PARAS=5

DEVICE=$1
MODEL=$2
ENSEMBLE_METHOD=${3:-"ffn_activation_max"}
TOKEN_MODELE=${4:-"all"}

if [ "$MODEL" == "llama3.2_1b_it" ] || [ "$MODEL" == "llama3.2_1b" ]; then
    LAYERS="4 8 12 16"  # e.g., "6 12 18 24 30 36" or "4 16 24 28"
elif [ "$MODEL" == "qwen2.5_3b_it" ] || [ "$MODEL" == "qwen2.5_3b" ]; then
    LAYERS="9 18 27 36"
elif [ "$MODEL" == "llama3.2_3b_it" ] || [ "$MODEL" == "llama3.2_3b" ]; then
    LAYERS="7 14 21 28"
else
    echo "Unsupported model: $MODEL"
    exit 1
fi

for ALPHA in 0.6 0.8 1 ; do
    for LAYER in $LAYERS; do
        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
            --debug \
            --model $MODEL \
            --dataset myriadlama \
            --num_paraphrases $NUM_PARAS \
            --num_fewshots $NUM_FEWSHOTS \
            --num_samples 5 \
            --logits_ensemble_method avg \
            --ensemble_method $ENSEMBLE_METHOD \
            --ensemble_layer $LAYER \
            --ensemble_alpha $ALPHA \
            --token_mode $TOKEN_MODELE \
            --multilayer

    done
done