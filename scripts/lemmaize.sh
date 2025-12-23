#!/bin/bash
REVERSE_ORDER=${1:-false}

MODELS=("llama3.2_1b" "llama3.2_1b_it" "llama3.2_3b_it" "llama3.1_8b_it" "llama3.2_3b" "llama3.1_8b" "qwen2.5_3b" "qwen2.5_7b" "qwen2.5_14b" "qwen2.5_3b_it" "qwen2.5_7b_it" "qwen2.5_14b_it")

if [ "$REVERSE_ORDER" = true ]; then
    for (( idx=${#MODELS[@]}-1 ; idx>=0 ; idx-- )) ; do
        MODEL=${MODELS[idx]}
        python3 ./lemmaize.py --model $MODEL 
    done
else
    for MODEL in "${MODELS[@]}"; do
        python3 ./lemmaize.py --model $MODEL 
    done
fi