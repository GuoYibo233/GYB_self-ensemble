#!/bin/bash

DEVICE=$1
DATASET=${2:-"myriadlama"} # myriadlama hotpot commonsense, mmlu, logiqa
# MODEL=${4:-"qwen3_30b"}
echo "Running baseline methods on device $DEVICE "


if [ "$DATASET" == "myriadlama" ]; then
    NUM_SAMPLES=5
else
    NUM_SAMPLES=1
fi

# MODELS="llama3.2_3b qwen2.5_3b qwen3_4b pythia_2.8b qwen3_30b"
# MODELS="qwen3_30b"
MODELS="phi3.5_mini"
# MODEL="llama3.2_3b"

for NUM_FEWSHOTS in 0 5; do
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
        elif [ "$MODEL" == "qwen3_4b" ]; then
            LAYER=27
        elif [ "$MODEL" == "qwen3_30b" ]; then
            LAYER=36
        elif [ "$MODEL" == "bloom_3b" ]; then
            LAYER=25
        elif [ "$MODEL" == "pythia_2.8b" ]; then
            LAYER=24
        elif [ "$MODEL" == "phi3.5_mini" ]; then
            LAYER=24
        elif [ "$MODEL" == "gpt_20b" ]; then
            LAYER=18
        else
            echo "Unknown MODEL: $MODEL"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
            --model $MODEL \
            --method per_prompt \
            --dataset $DATASET \
            --num_fewshots $NUM_FEWSHOTS

        CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
            --logits_ensemble_method avg \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrases 5 \
            --num_fewshots $NUM_FEWSHOTS \
            --num_samples $NUM_SAMPLES \
            --ensemble_method layer_output_avg \
            --ensemble_layer $LAYER \
            --ensemble_alpha 1 \
            --token_mode last \
            --multilayer
    done
done
