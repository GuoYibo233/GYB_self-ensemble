#!/bin/bash

DEVICE=$1
METHOD=${2:-"layer_output_avg"} # layer_output_avg ffn_activation_avg ffn_activation_max
TOKEN_MODEL=${3:-"last"}  # last all


# if [ "$MODEL_TYPE" == "base" ]; then
#     MODELS="llama3.2_1b llama3.2_3b llama3.1_8b qwen2.5_3b qwen2.5_7b "
# else
#     MODELS="llama3.2_1b_it llama3.2_3b_it llama3.1_8b_it qwen2.5_3b_it qwen2.5_7b_it"
# fi

MODELS="llama3.2_1b llama3.2_3b llama3.1_8b qwen2.5_3b qwen2.5_7b"

for MODEL in $MODELS; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --model $MODEL \
        --debug \
        --dataset myriadlama \
        --num_paraphrases 5 \
        --num_fewshots 5 \
        --num_samples 5 \
        --logits_ensemble_method avg
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --model $MODEL \
        --debug \
        --dataset myriadlama \
        --num_paraphrases 5 \
        --num_fewshots 5 \
        --num_samples 5 \
        --logits_ensemble_method max

    echo "Processing MODEL: $MODEL, LAYER: $LAYER, METHOD: $METHOD, TOKEN_MODEL: $TOKEN_MODEL"
    if [ "$MODEL" == "qwen2.5_14b" ] || [ "$MODEL" == "qwen2.5_14b_it" ]; then
        LAYERS="1 12 24 36 48"
    elif [ "$MODEL" == "llama3.1_8b" ] || [ "$MODEL" == "llama3.1_8b_it" ]; then
        LAYERS="1 8 16 24 32"
    elif [ "$MODEL" == "qwen2.5_7b" ] || [ "$MODEL" == "qwen2.5_7b_it" ]; then
        LAYERS="1 7 14 21 28"
    elif [ "$MODEL" == "llama3.2_3b" ] || [ "$MODEL" == "llama3.2_3b_it" ]; then
        LAYERS="1 7 14 21 28"
    elif [ "$MODEL" == "qwen2.5_3b" ] || [ "$MODEL" == "qwen2.5_3b_it" ]; then
        LAYERS="1 9 18 27 36"
    elif [ "$MODEL" == "llama3.2_1b" ] || [ "$MODEL" == "llama3.2_1b_it" ]; then
        LAYERS="1 4 8 12 16"
    else
        echo "Unknown MODEL: $MODEL"
        exit 1
    fi
    for LAYER in $LAYERS; do    
        for METHOD in layer_output_avg ffn_activation_avg ffn_activation_max; do
            CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
                --model $MODEL \
                --debug \
                --dataset myriadlama \
                --num_paraphrases 5 \
                --num_fewshots 5 \
                --num_samples 5 \
                --logits_ensemble_method avg \
                --ensemble_method $METHOD \
                --ensemble_layer $LAYER \
                --ensemble_alpha 0.5 \
                --token_mode $TOKEN_MODEL \
                --multilayer
        done
    done
done