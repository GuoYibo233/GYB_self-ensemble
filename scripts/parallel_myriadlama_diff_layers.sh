#!/bin/bash

DEVICE=$1
MODEL=$2
NUM_FEWSHOTS=${3:-5}
if [ "$MODEL" == "llama3.2_1b" ] || [ "$MODEL" == "llama3.2_1b_it" ]; then
    LAYERS="1 2 4 6 8 10 12 14 16"
elif [ "$MODEL" == "llama3.2_3b" ] || [ "$MODEL" == "llama3.2_3b_it" ]; then
    LAYERS="1 4 6 8 10 12 14 16 18 20 22 24 26 28"
elif [ "$MODEL" == "llama3.1_8b" ] || [ "$MODEL" == "llama3.1_8b_it" ]; then
    LAYERS="1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32"
elif [ "$MODEL" == "qwen2.5_3b" ] || [ "$MODEL" == "qwen2.5_3b_it" ]; then
    LAYERS="1 4 8 12 16 20 24 28 30 32 34 36"
else
    echo "Unknown MODEL: $MODEL"
    exit 1
fi

for LAYER in $LAYERS; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset myriadlama \
        --num_paraphrases 5 \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5 \
        --ensemble_method layer_output_avg \
        --ensemble_layer $LAYER \
        --ensemble_alpha 1 \
        --token_mode last \
        --multilayer
done