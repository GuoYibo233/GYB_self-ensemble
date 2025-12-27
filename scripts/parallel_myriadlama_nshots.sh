#!/bin/bash

DEVICE=$1
MODEL_SETS=$2

if [ "$MODEL_SETS" == "llama1" ]; then
    MODELS="llama3.2_1b llama3.2_1b_it"
elif [ "$MODEL_SETS" == "llama3" ]; then
    MODELS="llama3.2_3b llama3.2_3b_it"
elif [ "$MODEL_SETS" == "qwen" ]; then
    MODELS="qwen2.5_3b qwen2.5_3b_it"
else
    echo "Unknown MODEL_SETS: $MODEL_SETS"
    exit 1
fi

for MODEL in $MODELS ; do
    if [ "$MODEL" == "llama3.2_1b" ] || [ "$MODEL" == "llama3.2_1b_it" ]; then
        # LAYERS="1 4 8 12 16"
        LAYER=12
    elif [ "$MODEL" == "llama3.2_3b" ] || [ "$MODEL" == "llama3.2_3b_it" ]; then
        # LAYER="1 7 14 21 28"
        LAYER=21
    elif [ "$MODEL" == "llama3.1_8b" ] || [ "$MODEL" == "llama3.1_8b_it" ]; then
        # LAYER="1 8 16 24 32"
        LAYER=24
    elif [ "$MODEL" == "qwen2.5_3b" ] || [ "$MODEL" == "qwen2.5_3b_it" ]; then
        # LAYER="1 9 18 27 36"
        LAYER=27
    elif [ "$MODEL" == "qwen2.5_7b" ] || [ "$MODEL" == "qwen2.5_7b_it" ]; then
        # LAYER="1 7 14 21 28"
        LAYER=21
    elif [ "$MODEL" == "qwen2.5_14b" ] || [ "$MODEL" == "qwen2.5_14b_it" ]; then
        # LAYER="1 12 24 36 48"
        LAYER=36
    else
        echo "Unknown MODEL: $MODEL"
        exit 1
    fi
    
    for NUM_FEWSHOTS in 2 4 6 8 0 10; do
        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
            --logits_ensemble_method avg \
            --model $MODEL \
            --dataset myriadlama \
            --num_paraphrases 5 \
            --num_samples 5 \
            --num_fewshots $NUM_FEWSHOTS
        
        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
            --logits_ensemble_method max \
            --model $MODEL \
            --dataset myriadlama \
            --num_paraphrases 5 \
            --num_samples 5 \
            --num_fewshots $NUM_FEWSHOTS
        
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
        
        # CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        #     --logits_ensemble_method max \
        #     --model $MODEL \
        #     --dataset myriadlama \
        #     --num_paraphrases 5 \
        #     --num_fewshots $NUM_FEWSHOTS \
        #     --num_samples 5 \
        #     --ensemble_method layer_output_avg \
        #     --ensemble_layer $LAYER \
        #     --ensemble_alpha 1 \
        #     --token_mode last \
        #     --multilayer
    done
done