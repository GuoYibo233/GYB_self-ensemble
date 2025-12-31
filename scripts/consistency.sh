#!/bin/bash

DEVICE=$1
NUM_FEWSHOTS=${2:-0}

MODELS="llama3.2_1b llama3.2_3b llama3.1_8b llama3.2_1b_it llama3.2_3b_it llama3.1_8b_it"

for MODEL in $MODELS ; do
    for NUM_PARAPHRASES in 1 2 4 6 8 12 16 20; do
        echo "===== NUM_PARAPHRASES: $NUM_PARAPHRASES on $MODEL ====="
        if [ "$MODEL" == "llama3.2_1b" ] || [ "$MODEL" == "llama3.2_1b_it" ]; then
            LAYER=12
        elif [ "$MODEL" == "llama3.2_3b" ] || [ "$MODEL" == "llama3.2_3b_it" ]; then
            LAYER=21
        elif [ "$MODEL" == "llama3.1_8b" ] || [ "$MODEL" == "llama3.1_8b_it" ]; then
            LAYER=24
        else
            echo "Unknown MODEL: $MODEL"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_consistency.py \
            --logits_ensemble_method avg \
            --model $MODEL \
            --num_paraphrases $NUM_PARAPHRASES \
            --num_fewshots $NUM_FEWSHOTS \
            --num_samples 5 \
            --ensemble_method layer_output_avg \
            --ensemble_layer $LAYER \
            --ensemble_alpha 1 \
            --token_mode last \
            --multilayer
    done
done
