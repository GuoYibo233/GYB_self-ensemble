#!/bin/bash

DEVICE=$1
MODEL=$2


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
elif [ "$MODEL" == "qwen3_235b" ]; then
    LAYER=71
elif [ "$MODEL" == "pythia_2.8b" ]; then
    LAYER=24
else
    echo "Unknown MODEL: $MODEL"
    exit 1
fi


for NUM_FEWSHOTS in 0 1 2 3 4 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
        --model $MODEL \
        --method per_prompt \
        --dataset myriadlama \
        --num_fewshots $NUM_FEWSHOTS

    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases 1 \
        --num_fewshots $NUM_FEWSHOTS
        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --model $MODEL \
        --single_para_qapair \
        --num_paraphrases 5 \
        --num_fewshots $NUM_FEWSHOTS \
        --modify_attn \
        --modify_rope \
        --scale_factor    
    
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