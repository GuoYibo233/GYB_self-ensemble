#!/bin/bash

DEVICE=$1
DATASET=${2:-"myriadlama"} # myriadlama hotpot commonsense, mmlu, logiqa
MODEL_TYPE=${3:-base}
NUM_PARAS=${4:-5}
NUM_FEWSHOTS=${5:-5}
THINKING=${6:-false}

THINKING_FLAG=""
if [ "$THINKING" = true ] ; then
    THINKING_FLAG="--thinking"
    NUM_PARAS=2
fi

echo "Running baseline methods on device $DEVICE, with dataset $DATASET, model type $MODEL_TYPE, num_paras $NUM_PARAS, num_fewshots $NUM_FEWSHOTS, thinking_flag $THINKING_FLAG"

if [ "$DATASET" == "myriadlama" ]; then
    NUM_SAMPLES=5
else
    NUM_SAMPLES=1
fi

if [ "$MODEL_TYPE" == "base" ]; then
    MODELS="llama3.2_1b llama3.2_3b llama3.1_8b"
elif [ "$MODEL_TYPE" == "it" ]; then
    MODELS="llama3.2_1b_it llama3.2_3b_it llama3.1_8b_it"
elif [ "$MODEL_TYPE" == "llama" ]; then
    MODELS="llama3.2_1b llama3.2_3b llama3.1_8b"
elif [ "$MODEL_TYPE" == "qwen3" ]; then
    MODELS="qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b"
elif [ "$MODEL_TYPE" == "others" ]; then
    MODELS="qwen3_30b pythia_2.8b qwen3_4b"
else
    MODELS="$MODEL_TYPE"
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
    elif [ "$MODEL" == "qwen3_1.7b" ]; then
        LAYER=21
    elif [ "$MODEL" == "qwen3_4b" ]; then
        LAYER=27
    elif [ "$MODEL" == "qwen3_8b" ]; then
        LAYER=27
    elif [ "$MODEL" == "qwen3_14b" ]; then
        LAYER=30
    elif [ "$MODEL" == "qwen3_30b" ]; then
        LAYER=36
    elif [ "$MODEL" == "qwen3_32b" ]; then
        LAYER=48
    elif [ "$MODEL" == "qwen3_235b" ]; then
        LAYER=71
    elif [ "$MODEL" == "pythia_2.8b" ]; then
        LAYER=24
    else
        echo "Unknown MODEL: $MODEL"
        exit 1
    fi
    
    output=$(
        CUDA_VISIBLE_DEVICES=$DEVICE python3 generate_baseline.py \
            --debug \
            --model $MODEL \
            --method per_prompt \
            --dataset $DATASET \
            --num_fewshots $NUM_FEWSHOTS \
            $THINKING_FLAG
    )
    baseline_file=$(echo "$output" | grep -oP '(?<=Output to: ).*|(?<=results saved to: ).*|(?<=File ).*(?= already exists)' | head -1)

    echo "Baseline results saved to: $baseline_file"

    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --debug \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset $DATASET \
        --num_paraphrases $NUM_PARAS \
        --num_samples $NUM_SAMPLES \
        --baseline_file $baseline_file \
        $THINKING_FLAG

        
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble.py \
        --debug \
        --logits_ensemble_method avg \
        --model $MODEL \
        --dataset $DATASET \
        --num_paraphrases $NUM_PARAS \
        --num_samples $NUM_SAMPLES \
        --ensemble_method layer_output_avg \
        --ensemble_layer $LAYER \
        --ensemble_alpha 1 \
        --token_mode last \
        --multilayer \
        --baseline_file $baseline_file \
        $THINKING_FLAG

    CUDA_VISIBLE_DEVICES=$DEVICE python3 series_ensemble.py \
        --debug \
        --model $MODEL \
        --dataset $DATASET \
        --num_samples $NUM_SAMPLES \
        --num_paraphrases $NUM_PARAS \
        --modify_attn \
        --modify_rope \
        --scale_factor \
        --baseline_file $baseline_file \
        $THINKING_FLAG
done
