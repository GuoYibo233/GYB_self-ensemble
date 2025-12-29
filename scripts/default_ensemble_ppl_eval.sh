DEVICE=$1
NUM_FEWSHOTS=${2:-5}
DATASET=${3:-"commonsense"}

MODELS="llama3.2_3b qwen2.5_3b qwen3_4b pythia_2.8b"

for MODEL in $MODELS ; do
    if [ "$MODEL" == "llama3.2_3b" ]; then
        LAYER=21
    elif [ "$MODEL" == "qwen2.5_3b" ]; then
        LAYER=27
    elif [ "$MODEL" == "qwen3_4b" ]; then
        LAYER=27
    elif [ "$MODEL" == "pythia_2.8b" ]; then
        LAYER=24
    elif [ "$MODEL" == "qwen3_30b" ]; then
        LAYER=36
    else
        echo "Unknown MODEL: $MODEL"
        exit 1
    fi

    # Baseline
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_perplexity.py \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrases 3 \
            --num_fewshots $NUM_FEWSHOTS \
            --is_baseline

    # Ensemble
    CUDA_VISIBLE_DEVICES=$DEVICE python3 parallel_ensemble_perplexity.py \
            --model $MODEL \
            --dataset $DATASET \
            --num_paraphrases 3 \
            --num_fewshots $NUM_FEWSHOTS \
            --logits_ensemble_method avg \
            --ensemble_method layer_output_avg \
            --ensemble_layer $LAYER \
            --ensemble_alpha 1 \
            --token_mode all \
            --multilayer
done