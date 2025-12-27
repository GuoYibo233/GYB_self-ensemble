#!/bin/bash


DEVICE=$1
MODEL_TYPE=${2:-base}
NUM_FEWSHOTS=${3:-5}
NUM_PARAS=5

if [ "$MODEL_TYPE" == "base" ]; then
    MODELS="llama3.2_1b llama3.2_3b llama3.1_8b"
elif [ "$MODEL_TYPE" == "it" ]; then
    MODELS="llama3.2_1b_it llama3.2_3b_it llama3.1_8b_it"
elif [ "$MODEL_TYPE" == "others" ]; then
    MODELS="qwen3_30b pythia_2.8b qwen3_4b"
fi

for MODEL in $MODELS ; do

    if [ "$MODEL" == "llama3.2_1b" ] || [ "$MODEL" == "llama3.2_1b_it" ]; then
        LAYER=12
    elif [ "$MODEL" == "llama3.2_3b" ] || [ "$MODEL" == "llama3.2_3b_it" ]; then
        LAYER=21
    elif [ "$MODEL" == "llama3.1_8b" ] || [ "$MODEL" == "llama3.1_8b_it" ]; then
        LAYER=24
    elif [ "$MODEL" == "qwen2.5_3b" ] || [ "$MODEL" == "qwen2.5_3b_it" ]; then
        LAYER=27
    elif [ "$MODEL" == "qwen2.5_7b" ] || [ "$MODEL" == "qwen2.5_7b_it" ]; then
        LAYER=21
    elif [ "$MODEL" == "qwen2.5_14b" ] || [ "$MODEL" == "qwen2.5_14b_it" ]; then
        LAYER=36
    elif [ "$MODEL" == "qwen3_30b" ]; then
        LAYER=36
    elif [ "$MODEL" == "qwen3_235b" ]; then
        LAYER=71
    elif [ "$MODEL" == "pythia_2.8b" ]; then
        LAYER=24
    else
        echo "Unknown MODEL: $MODEL"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method max \
        --model $MODEL \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5 \
        --ensemble_method layer_output_avg \
        --ensemble_layer $LAYER \
        --ensemble_alpha 1 \
        --token_mode last \
        --multilayer
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5 \
        --ensemble_method ffn_activation_avg \
        --ensemble_layer $LAYER \
        --ensemble_alpha 1 \
        --token_mode last \
        --multilayer
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset myriadlama \
        --num_paraphrases $NUM_PARAS \
        --num_fewshots $NUM_FEWSHOTS \
        --num_samples 5 \
        --ensemble_method ffn_activation_max \
        --ensemble_layer $LAYER \
        --ensemble_alpha 1 \
        --token_mode last \
        --multilayer
done