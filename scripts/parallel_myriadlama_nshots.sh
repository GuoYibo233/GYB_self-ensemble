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
    for NUM_FEWSHOTS in 0 2 4 6 8 10; do
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
    done
done