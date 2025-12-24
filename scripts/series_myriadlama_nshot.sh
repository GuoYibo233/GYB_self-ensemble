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
        CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --model $MODEL \
            --single_para_qapair \
            --num_paraphrases 1 \
            --num_fewshots $NUM_FEWSHOTS

        CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --model $MODEL \
            --single_para_qapair \
            --num_paraphrases 5 \
            --num_fewshots $NUM_FEWSHOTS

        CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --model $MODEL \
            --single_para_qapair \
            --num_paraphrases 5 \
            --num_fewshots $NUM_FEWSHOTS \
            --modify_attn \
            --scale_factor
        
        CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --model $MODEL \
            --single_para_qapair \
            --num_paraphrases 5 \
            --num_fewshots $NUM_FEWSHOTS \
            --modify_attn \
            --modify_rope
            
        CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
            --model $MODEL \
            --single_para_qapair \
            --num_paraphrases 5 \
            --num_fewshots $NUM_FEWSHOTS \
            --modify_attn \
            --modify_rope \
            --scale_factor    
    done
done